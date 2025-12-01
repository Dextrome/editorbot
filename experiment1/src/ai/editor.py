"""AI-powered audio editor that automates editing decisions."""

from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..audio.processor import AudioProcessor
from ..audio.analyzer import AudioAnalyzer
from ..audio.effects import AudioEffects
from .arranger import SongArranger, Arrangement, Section
from .remixatron_adapter import RemixatronAdapter
from .trainer import SongTrainer, StyleProfile


class EditDecision:
    """Represents a single edit decision."""
    
    def __init__(
        self,
        action: str,
        start_time: float,
        end_time: float,
        reason: str,
        confidence: float = 1.0
    ):
        self.action = action  # "cut", "keep", "crossfade", "replace"
        self.start_time = start_time
        self.end_time = end_time
        self.reason = reason
        self.confidence = confidence
    
    def __repr__(self):
        return f"EditDecision({self.action}, {self.start_time:.2f}s-{self.end_time:.2f}s, '{self.reason}')"


logger = logging.getLogger(__name__)


class AIEditor:
    """
    AI-powered audio editor that analyzes recordings and applies
    intelligent edits to create polished songs.
    """

    def __init__(self, sample_rate: int = 44100, demucs_device: str | None = None, remixatron_max_jump: int | None = None, remixatron_gap_heal_ms: int | None = None, remixatron_gap_heal_threshold: float | None = None, remixatron_gap_mode: str = 'heal', remixatron_truncate: bool = False, remixatron_truncate_min_ms: int = 100, remixatron_truncate_max_ms: int = 300, remixatron_truncate_threshold: float = 0.01, remixatron_truncate_crossfade_ms: int = 20, remixatron_truncate_adaptive_factor: float = 0.0, remixatron_truncate_mode: str = 'remove', remixatron_truncate_compress_ms: int = 20, remixatron_truncate_sample_pct: float = 0.95, remixatron_phrase_beats: int = 4, remixatron_mode_str: str = 'off'):
        """
        Initialize the AI editor.

        Args:
            sample_rate: Sample rate for audio processing.
        """
        self.sample_rate = sample_rate
        self.processor = AudioProcessor(sample_rate)
        self.analyzer = AudioAnalyzer(sample_rate)
        self.effects = AudioEffects(sample_rate)
        self.arranger = SongArranger(sample_rate)
        self.trainer = SongTrainer(sample_rate)
        self.edit_log: List[EditDecision] = []
        self.arrangement: Optional[Arrangement] = None
        self.detected_sections: List[Section] = []
        self.style_profile: Optional[StyleProfile] = None
        # Optional Demucs device override. If None, the Demucs wrapper will auto-detect from torch
        self.demucs_device = demucs_device
        self.remixatron_max_jump = remixatron_max_jump
        # optional small gap healing in ms applied to remix output
        self.remixatron_gap_heal_ms = remixatron_gap_heal_ms
        self.remixatron_gap_heal_threshold = remixatron_gap_heal_threshold
        self.remixatron_gap_mode = remixatron_gap_mode
        # Truncate mid length silence runs in the final output (100-300ms by default)
        self.remixatron_truncate = remixatron_truncate
        self.remixatron_truncate_min_ms = remixatron_truncate_min_ms
        self.remixatron_truncate_max_ms = remixatron_truncate_max_ms
        self.remixatron_truncate_threshold = remixatron_truncate_threshold
        self.remixatron_truncate_crossfade_ms = remixatron_truncate_crossfade_ms
        self.remixatron_truncate_adaptive_factor = remixatron_truncate_adaptive_factor
        self.remixatron_truncate_mode = remixatron_truncate_mode
        self.remixatron_truncate_compress_ms = remixatron_truncate_compress_ms
        self.remixatron_truncate_sample_pct = remixatron_truncate_sample_pct
        # Phrase grouping for Remixatron (number of beats to concatenate into a phrase)
        self.remixatron_phrase_beats = int(remixatron_phrase_beats) if remixatron_phrase_beats and remixatron_phrase_beats > 0 else 4
        # Wire demucs device to arranger for downstream blending choices
        try:
            setattr(self.arranger, 'demucs_device', demucs_device)
        except Exception:
            pass
        # Store remix mode string for advanced pipelines (e.g., 'creative_pipeline')
        self.remixatron_mode_str = remixatron_mode_str
    
    def load_style(self, style_name: str) -> bool:
        """
        Load a trained style profile to use for arrangements.
        
        Args:
            style_name: Name of the style to load.
            
        Returns:
                remix_mode: bool = False,
                remix_audio_path: Optional[str] = None,
                remix_clusters: int = 0,
            True if style was loaded successfully.
        """
        try:
            self.trainer.load_style(style_name)
            self.style_profile = self.trainer.style_profiles.get(style_name)
            return self.style_profile is not None
        except Exception as e:
            logger.warning("Could not load style '%s': %s", style_name, e)
            return False

    def analyze_recording(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a recording to determine what edits are needed.
                    remix_mode: Whether to enable remix mode.
                    remix_audio_path: Path to the audio file for remixing.
                    remix_clusters: Number of clusters for remixing.

        Args:
            audio_data: Raw audio data to analyze.

        Returns:
            Analysis results with recommendations.
        """
        analysis = {}

        # Basic analysis
        analysis["tempo"] = self.analyzer.detect_tempo(audio_data)
        analysis["key"] = self.analyzer.detect_key(audio_data)
        analysis["loudness"] = self.analyzer.get_loudness(audio_data)
        analysis["beats"] = self.analyzer.detect_beats(audio_data)
        analysis["duration"] = len(audio_data) / self.sample_rate

        # Extract features for ML processing
        analysis["features"] = self.analyzer.extract_features(audio_data)
        # Optionally use allin1 for improved structure/beat detection if available
        try:
            allin1_results = self.analyzer.analyze_with_allin1(audio_data=audio_data, sample_rate=self.sample_rate)
            if allin1_results:
                analysis["allin1"] = allin1_results
        except Exception:
            # allin1 not available or failed; continue with existing analysis
            analysis["allin1"] = None
        # NATTEN-based contextual features (optional; guarded)
        try:
            analysis["natten_features"] = self.analyzer.extract_attention_features(
                audio_data, frame_size=256, hop_size=128, proj_dim=8, kernel_size=7
            )
        except Exception:
            analysis["natten_features"] = None

        # Detect sections
        sections, labels = self.analyzer.detect_sections(audio_data)
        analysis["sections"] = sections
        analysis["section_labels"] = labels

        # NEW: Structural analysis for editing
        analysis["silence_regions"] = self.analyzer.detect_silence_regions(audio_data)
        analysis["anomalies"] = self.analyzer.detect_anomalies(audio_data)
        analysis["repeated_sections"] = self.analyzer.detect_repeated_sections(audio_data)

        # Generate edit decisions
        analysis["edit_decisions"] = self._generate_edit_decisions(audio_data, analysis)
        
        # Generate processing recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_edit_decisions(
        self, audio_data: np.ndarray, analysis: Dict[str, Any]
    ) -> List[EditDecision]:
        """
        Generate structural edit decisions based on analysis.

        Args:
            audio_data: Raw audio data.
            analysis: Analysis results.

        Returns:
            List of edit decisions.
        """
        decisions = []
        duration = analysis["duration"]

        # 1. Cut silence at start (except brief natural pause)
        if analysis["silence_regions"]:
            for start, end in analysis["silence_regions"]:
                if start < 0.1:  # Silence at very beginning
                    # Keep a tiny bit of room tone, cut the rest
                    if end > 0.3:
                        decisions.append(EditDecision(
                            "cut", 0.1, end - 0.1,
                            "Remove excess silence at start",
                            confidence=0.95
                        ))
                elif end > duration - 0.1:  # Silence at end
                    decisions.append(EditDecision(
                        "cut", start + 0.1, duration,
                        "Remove trailing silence",
                        confidence=0.95
                    ))
                elif end - start > 2.0:  # Long silence in middle
                    # Keep some silence for natural pause, cut excess
                    decisions.append(EditDecision(
                        "cut", start + 0.5, end - 0.5,
                        f"Trim long silence ({end-start:.1f}s -> 1.0s)",
                        confidence=0.85
                    ))

        # 2. Handle repeated sections (false starts/retakes)
        if analysis["repeated_sections"]:
            # Group overlapping repeated sections
            for t1_start, t1_end, t2_start, t2_end in analysis["repeated_sections"]:
                # Find the best take
                sections = [(t1_start, t1_end), (t2_start, t2_end)]
                best_idx = self.analyzer.find_best_take(audio_data, sections)
                
                # Cut the worse take
                worse_idx = 1 - best_idx
                worse_start, worse_end = sections[worse_idx]
                decisions.append(EditDecision(
                    "cut", worse_start, worse_end,
                    f"Remove inferior take (keeping take at {sections[best_idx][0]:.1f}s)",
                    confidence=0.7
                ))

        # 3. Handle anomalies
        for start, end, anomaly_type in analysis["anomalies"]:
            if anomaly_type == "clipping":
                # Move clipping log to debug/verbose logger
                logger.debug("Clipping detected - may need re-recording")
            elif anomaly_type == "sudden_increase" and start < 3.0:
                # Sudden volume increase at start might be a false start
                decisions.append(EditDecision(
                    "cut", 0, start,
                    "Remove potential false start",
                    confidence=0.6
                ))

        # 4. Sort decisions and resolve conflicts
        decisions = self._resolve_edit_conflicts(decisions)

        return decisions

    def _resolve_edit_conflicts(
        self, decisions: List[EditDecision]
    ) -> List[EditDecision]:
        """
        Resolve conflicting edit decisions.

        Args:
            decisions: List of edit decisions.

        Returns:
            Resolved list of non-overlapping decisions.
        """
        if not decisions:
            return []

        # Sort by start time
        decisions.sort(key=lambda d: d.start_time)

        # Remove overlapping cuts, keeping higher confidence ones
        resolved = []
        for decision in decisions:
            if decision.action == "flag":
                resolved.append(decision)
                continue

            # Check for overlap with existing decisions
            overlaps = False
            for existing in resolved:
                if existing.action == "flag":
                    continue
                # Check if ranges overlap
                if (decision.start_time < existing.end_time and 
                    decision.end_time > existing.start_time):
                    overlaps = True
                    # Keep the one with higher confidence
                    if decision.confidence > existing.confidence:
                        resolved.remove(existing)
                        resolved.append(decision)
                    break

            if not overlaps:
                resolved.append(decision)

        return sorted(resolved, key=lambda d: d.start_time)

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate editing recommendations based on analysis.

        Args:
            analysis: Analysis results.

        Returns:
            List of recommended actions.
        """
        recommendations = []

        # Loudness recommendations
        if analysis["loudness"] < -20:
            recommendations.append("normalize_audio")
            recommendations.append("apply_compression")
        elif analysis["loudness"] > -6:
            recommendations.append("reduce_gain")

        # Check for noise (using RMS variance)
        rms = analysis["features"]["rms"]
        rms_variance = np.var(rms)
        if rms_variance > 0.1:
            recommendations.append("apply_noise_gate")

        # Always recommend these for polish
        recommendations.append("apply_eq")
        recommendations.append("apply_reverb")
        recommendations.append("trim_silence")

        return recommendations

    def auto_edit(
        self,
        audio_data: np.ndarray,
        analysis: Optional[Dict[str, Any]] = None,
        preset: str = "balanced",
        arrange: bool = True,
        arrangement_template: str = "auto",
        allow_rearrange: bool = False,
        min_section_duration: Optional[float] = None,
        max_section_duration: Optional[float] = None,
        remix_mode: bool = False,
    ) -> np.ndarray:
        """
        Automatically edit audio based on analysis.

        Args:
            audio_data: Raw audio data.
            analysis: Pre-computed analysis (computed if not provided).
            preset: Editing preset ("balanced", "warm", "bright", "aggressive").
            arrange: Whether to arrange the song structure.
            arrangement_template: Template for arrangement ("auto", "pop", "edm", "rock", "simple", "minimal").
            allow_rearrange: Whether to allow reordering sections for better flow.
            min_section_duration: Minimum section duration in seconds.
            max_section_duration: Maximum section duration in seconds.

        Returns:
            Edited audio data.
        """
        if remix_mode:
            # Import RemixatronAdapter for all remix modes
            from src.ai.remixatron_adapter import RemixatronAdapter
            import tempfile, soundfile as sf
            
            # Two options: normal remix pass, or an advanced 'creative_pipeline' two-pass flow
            if getattr(self, 'remixatron_mode_str', None) == 'creative_pipeline':
                logger.info("[AIEditor] Remix: running creative two-pass pipeline")
                # PASS 1: aggressive arranger to create a creative rearranged bed
                # Use "full" template to keep all good material for Remixatron to remix
                logger.info("[AIEditor] Creative pass 1: aggressive arranger (full material)")
                arranged_audio, arrangement, sections = self.arranger.auto_arrange(
                    audio_data,
                    template="full",  # Keep all good material, not trimmed to target duration
                    style_profile=self.style_profile,
                    allow_rearrange=True,  # Allow creative reordering
                    min_section_duration=1.0 if allow_rearrange else (min_section_duration or 2.0),
                    max_section_duration=max_section_duration,
                    external_sections=None,
                )

                # Write the arranged audio to a temp file for Remixatron to operate on
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t1:
                    sf.write(t1.name, arranged_audio, self.sample_rate)
                    t1_path = t1.name

                # PASS 2: run Remixatron twice (short phrases and long phrases) and mix
                logger.info("[AIEditor] Creative pass 2: Remixatron mixing short and large chunks")
                # Short-phrase adapter
                adapter_short = RemixatronAdapter(
                    sample_rate=self.sample_rate,
                    demucs_device=self.demucs_device,
                    max_jump=self.remixatron_max_jump,
                    gap_heal_ms=self.remixatron_gap_heal_ms,
                    gap_heal_threshold=self.remixatron_gap_heal_threshold or 1e-4,
                    gap_mode='heal',
                    truncate_enabled=False,
                    phrase_beats=max(1, int(self.remixatron_phrase_beats // 2) or 1),
                )
                # Long-phrase adapter
                adapter_long = RemixatronAdapter(
                    sample_rate=self.sample_rate,
                    demucs_device=self.demucs_device,
                    max_jump=self.remixatron_max_jump,
                    gap_heal_ms=self.remixatron_gap_heal_ms,
                    gap_heal_threshold=self.remixatron_gap_heal_threshold or 1e-4,
                    gap_mode='stem',
                    truncate_enabled=False,
                    phrase_beats=max(4, int(self.remixatron_phrase_beats * 2)),
                )
                # Run adapters
                short_audio = adapter_short.rearrange(t1_path)
                long_audio = adapter_long.rearrange(t1_path)

                # Mix by alternating segments of short and long results
                seg_s = int(4.0 * self.sample_rate)  # 4s segments for alternating mix
                L = max(short_audio.shape[0], long_audio.shape[0])
                # pad shorter arrays
                def pad_to(a, n):
                    if a.shape[0] >= n:
                        return a[:n]
                    pad = np.zeros((n - a.shape[0], a.shape[1]), dtype=a.dtype)
                    return np.concatenate([a, pad], axis=0)
                short_audio = pad_to(short_audio, L)
                long_audio = pad_to(long_audio, L)
                mixed = np.zeros((L, 2), dtype=short_audio.dtype)
                for i in range(0, L, seg_s):
                    seg_end = min(i + seg_s, L)
                    if ((i // seg_s) % 2) == 0:
                        mixed[i:seg_end] = short_audio[i:seg_end]
                    else:
                        mixed[i:seg_end] = long_audio[i:seg_end]

                # Skip Pass 3 (conservative arranger) to avoid OOM on long remix results
                # The mixed output from Pass 2 is already a creative remix
                logger.info("[AIEditor] Creative pipeline complete - returning mixed result")
                return mixed
            else:
                logger.info("[AIEditor] Remix mode enabled: using RemixatronAdapter!")
                adapter = RemixatronAdapter(
                    sample_rate=self.sample_rate,
                    demucs_device=self.demucs_device,
                    max_jump=self.remixatron_max_jump,
                    gap_heal_ms=self.remixatron_gap_heal_ms,
                    gap_heal_threshold=self.remixatron_gap_heal_threshold or 1e-4,
                    gap_mode=self.remixatron_gap_mode,
                    truncate_enabled=self.remixatron_truncate,
                    truncate_min_ms=self.remixatron_truncate_min_ms,
                    truncate_max_ms=self.remixatron_truncate_max_ms,
                    truncate_threshold=self.remixatron_truncate_threshold,
                    truncate_crossfade_ms=self.remixatron_truncate_crossfade_ms,
                    truncate_adaptive_factor=self.remixatron_truncate_adaptive_factor,
                    truncate_mode=self.remixatron_truncate_mode,
                    truncate_compress_ms=self.remixatron_truncate_compress_ms,
                    truncate_sample_pct=self.remixatron_truncate_sample_pct,
                    phrase_beats=self.remixatron_phrase_beats,
                )
                # Save to temp wav if needed
                import tempfile, soundfile as sf
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, audio_data, self.sample_rate)
                    remix_audio = adapter.rearrange(f.name)
                return remix_audio

        if analysis is None:
            analysis = self.analyze_recording(audio_data)

        recommendations = analysis["recommendations"]
        edit_decisions = analysis.get("edit_decisions", [])
        # Store edit log
        self.edit_log = edit_decisions
        # Get preset parameters
        params = self._get_preset_params(preset)

        # PHASE 1: Song arrangement (if enabled)
        if arrange:
            logger.info("\n🎼 Analyzing song structure...")
            
            # Use style profile if loaded and applicable
            style_to_use = None
            if self.style_profile and arrangement_template in ["doom", "stoner"]:
                logger.info("   Using learned style: %s", self.style_profile.name)
                style_to_use = self.style_profile
            
            # Use allin1 segments/labels if available
            external_sections = None
            if analysis.get("allin1") and analysis["allin1"].get("segments"):
                # allin1 segments: list of dicts with start, end, label (best effort)
                external_sections = []
                segments = analysis["allin1"]["segments"]
                labels = analysis["allin1"].get("labels", [])
                for i, seg in enumerate(segments):
                    # seg: [start, end] or dict
                    if isinstance(seg, dict):
                        external_sections.append(seg)
                    elif isinstance(seg, (list, tuple)) and len(seg) == 2:
                        label = labels[i] if i < len(labels) else "section"
                        external_sections.append({
                            "start_time": float(seg[0]),
                            "end_time": float(seg[1]),
                            "label": label
                        })
            arranged_audio, arrangement, sections = self.arranger.auto_arrange(
                audio_data, 
                template=arrangement_template,
                style_profile=style_to_use,
                allow_rearrange=allow_rearrange,
                min_section_duration=min_section_duration,
                max_section_duration=max_section_duration,
                external_sections=external_sections
            )
            self.arrangement = arrangement
            self.detected_sections = sections
            
            logger.info("   Found %d sections", len(sections))
            logger.info("   Structure: %s", ' → '.join(arrangement.structure))
            logger.info("   Duration: %.1fs → %.1fs", len(audio_data)/self.sample_rate, arrangement.total_duration)
            
            edited = arranged_audio
        else:
            # PHASE 1 (alt): Just structural edits (cuts, trims)
            edited = self._apply_structural_edits(audio_data, edit_decisions)
            
            # Print edit summary
            cuts_made = [d for d in edit_decisions if d.action == "cut" and d.confidence >= 0.6]
            if cuts_made:
                logger.info("\n📝 Structural edits made: %d", len(cuts_made))
                for decision in cuts_made:
                    logger.info("   • %s", decision.reason)

        # PHASE 2: Processing edits (EQ, compression, etc.)
        if "trim_silence" in recommendations:
            edited = self.processor.trim_silence(edited)

        if "normalize_audio" in recommendations:
            edited = self.processor.normalize(edited)

        if "apply_noise_gate" in recommendations:
            edited = self.effects.apply_noise_gate(
                edited, threshold_db=params["gate_threshold"]
            )

        if "apply_compression" in recommendations:
            edited = self.effects.apply_compression(
                edited,
                threshold=params["comp_threshold"],
                ratio=params["comp_ratio"],
            )

        if "apply_eq" in recommendations:
            edited = self.effects.apply_eq(
                edited,
                low_gain=params["eq_low"],
                mid_gain=params["eq_mid"],
                high_gain=params["eq_high"],
            )

        if "apply_reverb" in recommendations:
            edited = self.effects.apply_reverb(
                edited,
                room_size=params["reverb_size"],
                wet_level=params["reverb_wet"],
            )

        # Final normalization
        edited = self.processor.normalize(edited)

        return edited

    def _apply_structural_edits(
        self, audio_data: np.ndarray, decisions: List[EditDecision]
    ) -> np.ndarray:
        """
        Apply structural edits (cuts) to the audio.

        Args:
            audio_data: Raw audio data.
            decisions: List of edit decisions.

        Returns:
            Audio with cuts applied.
        """
        # Filter to only cut decisions with sufficient confidence
        cuts = [d for d in decisions if d.action == "cut" and d.confidence >= 0.6]
        
        if not cuts:
            return audio_data

        # Sort cuts by start time (descending) so we can cut from end to start
        cuts.sort(key=lambda d: d.start_time, reverse=True)

        edited = audio_data.copy()
        
        for cut in cuts:
            start_sample = int(cut.start_time * self.sample_rate)
            end_sample = int(cut.end_time * self.sample_rate)
            
            # Ensure valid bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(edited), end_sample)
            
            if start_sample < end_sample:
                # Apply short crossfade to avoid clicks
                crossfade_samples = min(int(0.01 * self.sample_rate), start_sample, len(edited) - end_sample)
                
                if crossfade_samples > 0 and end_sample < len(edited):
                    # Create crossfade
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    
                    before = edited[:start_sample]
                    after = edited[end_sample:]
                    
                    # Apply fades
                    if len(before) >= crossfade_samples:
                        before[-crossfade_samples:] *= fade_out
                    if len(after) >= crossfade_samples:
                        after[:crossfade_samples] *= fade_in
                    
                    edited = np.concatenate([before, after])
                else:
                    # Simple cut
                    edited = np.concatenate([edited[:start_sample], edited[end_sample:]])

        return edited

    def get_edit_summary(self) -> str:
        """
        Get a human-readable summary of edits made.

        Returns:
            String summary of all edits.
        """
        lines = []
        
        # Arrangement summary
        if self.arrangement:
            lines.append(self.arranger.get_arrangement_summary(
                self.arrangement, self.detected_sections
            ))
            lines.append("")
        
        # Structural edit summary
        if self.edit_log:
            lines.append("Edit Decisions:")
            lines.append("-" * 40)
            
            for decision in self.edit_log:
                status = "✓" if decision.confidence >= 0.6 else "○"
                lines.append(
                    f"{status} [{decision.action.upper()}] "
                    f"{decision.start_time:.2f}s - {decision.end_time:.2f}s: "
                    f"{decision.reason} (confidence: {decision.confidence:.0%})"
                )
        
        if not lines:
            return "No structural edits were made."

        return "\n".join(lines)

    def _get_preset_params(self, preset: str) -> Dict[str, float]:
        """
        Get editing parameters for a preset.

        Args:
            preset: Preset name.

        Returns:
            Dictionary of parameter values.
        """
        presets = {
            "balanced": {
                "gate_threshold": -40.0,
                "comp_threshold": -18.0,
                "comp_ratio": 3.0,
                "eq_low": 1.0,
                "eq_mid": 0.5,
                "eq_high": 1.5,
                "reverb_size": 0.4,
                "reverb_wet": 0.2,
            },
            "warm": {
                "gate_threshold": -45.0,
                "comp_threshold": -15.0,
                "comp_ratio": 2.5,
                "eq_low": 3.0,
                "eq_mid": 1.0,
                "eq_high": -1.0,
                "reverb_size": 0.5,
                "reverb_wet": 0.25,
            },
            "bright": {
                "gate_threshold": -35.0,
                "comp_threshold": -20.0,
                "comp_ratio": 3.5,
                "eq_low": -1.0,
                "eq_mid": 1.0,
                "eq_high": 4.0,
                "reverb_size": 0.3,
                "reverb_wet": 0.15,
            },
            "aggressive": {
                "gate_threshold": -30.0,
                "comp_threshold": -12.0,
                "comp_ratio": 6.0,
                "eq_low": 2.0,
                "eq_mid": -1.0,
                "eq_high": 3.0,
                "reverb_size": 0.2,
                "reverb_wet": 0.1,
            },
        }

        return presets.get(preset, presets["balanced"])

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        preset: str = "balanced",
        arrange: bool = True,
        arrangement_template: str = "auto",
        allow_rearrange: bool = False,
        min_section_duration: Optional[float] = None,
        max_section_duration: Optional[float] = None,
        remix_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Process an audio file from start to finish.

        Args:
            input_path: Path to input audio file.
            output_path: Path for output audio file.
            preset: Editing preset to use.
            arrange: Whether to arrange the song structure.
            arrangement_template: Template for arrangement.
            allow_rearrange: Whether to allow reordering sections for better flow.
            min_section_duration: Minimum section duration in seconds.
            max_section_duration: Maximum section duration in seconds.

        Returns:
            Processing results including analysis and output path.
        """
        # Load audio
        audio_data, sr = self.processor.load(input_path)

        # Analyze
        analysis = self.analyze_recording(audio_data)

        # Edit
        edited = self.auto_edit(
            audio_data, analysis, preset, 
            arrange=arrange, 
            arrangement_template=arrangement_template,
            allow_rearrange=allow_rearrange,
            min_section_duration=min_section_duration,
            max_section_duration=max_section_duration,
            remix_mode=remix_mode
        )

        # Save
        self.processor.save(output_path, edited)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "analysis": analysis,
            "preset": preset,
            "arrangement": self.arrangement,
            "sections": self.detected_sections,
        }

    def batch_process(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        preset: str = "balanced",
        extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg"],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in a directory.

        Args:
            input_dir: Directory containing input files.
            output_dir: Directory for output files.
            preset: Editing preset to use.
            extensions: List of file extensions to process.

        Returns:
            List of processing results for each file.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for ext in extensions:
            for input_file in input_dir.glob(f"*{ext}"):
                output_file = output_dir / f"{input_file.stem}_edited{input_file.suffix}"
                try:
                    result = self.process_file(input_file, output_file, preset)
                    result["status"] = "success"
                except Exception as e:
                    result = {
                        "input_path": str(input_file),
                        "status": "error",
                        "error": str(e),
                    }
                results.append(result)

        return results

    def analyze_with_demucs_and_natten(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio using Demucs for stem separation and NATTEN for feature extraction.

        Args:
            audio_data: Raw audio data to analyze.

        Returns:
            Dictionary containing analysis results.
        """
        # Step 1: Separate stems using Demucs
        stems = self.processor.separate_stems(audio_data, device=self.demucs_device)

        # Step 2: Analyze sections using stems
        sections = self.arranger.analyze_sections_with_stems(audio_data, stems)

        # Step 3: Extract NATTEN-based features for each section
        for section in sections:
            natten_features = self.arranger._analyze_section_features_with_natten(audio_data)
            section.features.update(natten_features)

        return {
            "sections": sections,
            "stems": stems,
        }

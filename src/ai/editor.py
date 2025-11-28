"""AI-powered audio editor that automates editing decisions."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..audio.processor import AudioProcessor
from ..audio.analyzer import AudioAnalyzer
from ..audio.effects import AudioEffects
from .arranger import SongArranger, Arrangement, Section


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


class AIEditor:
    """
    AI-powered audio editor that analyzes recordings and applies
    intelligent edits to create polished songs.
    """

    def __init__(self, sample_rate: int = 44100):
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
        self.edit_log: List[EditDecision] = []
        self.arrangement: Optional[Arrangement] = None
        self.detected_sections: List[Section] = []

    def analyze_recording(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a recording to determine what edits are needed.

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
                # Can't cut clipping, but flag it
                decisions.append(EditDecision(
                    "flag", start, end,
                    "Clipping detected - may need re-recording",
                    confidence=0.9
                ))
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
    ) -> np.ndarray:
        """
        Automatically edit audio based on analysis.

        Args:
            audio_data: Raw audio data.
            analysis: Pre-computed analysis (computed if not provided).
            preset: Editing preset ("balanced", "warm", "bright", "aggressive").
            arrange: Whether to arrange the song structure.
            arrangement_template: Template for arrangement ("auto", "pop", "edm", "rock", "simple", "minimal").

        Returns:
            Edited audio data.
        """
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
            print("\n🎼 Analyzing song structure...")
            arranged_audio, arrangement, sections = self.arranger.auto_arrange(
                audio_data, 
                template=arrangement_template
            )
            self.arrangement = arrangement
            self.detected_sections = sections
            
            print(f"   Found {len(sections)} sections")
            print(f"   Structure: {' → '.join(arrangement.structure)}")
            print(f"   Duration: {len(audio_data)/self.sample_rate:.1f}s → {arrangement.total_duration:.1f}s")
            
            edited = arranged_audio
        else:
            # PHASE 1 (alt): Just structural edits (cuts, trims)
            edited = self._apply_structural_edits(audio_data, edit_decisions)
            
            # Print edit summary
            cuts_made = [d for d in edit_decisions if d.action == "cut" and d.confidence >= 0.6]
            if cuts_made:
                print(f"\n📝 Structural edits made: {len(cuts_made)}")
                for decision in cuts_made:
                    print(f"   • {decision.reason}")

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
    ) -> Dict[str, Any]:
        """
        Process an audio file from start to finish.

        Args:
            input_path: Path to input audio file.
            output_path: Path for output audio file.
            preset: Editing preset to use.
            arrange: Whether to arrange the song structure.
            arrangement_template: Template for arrangement.

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
            arrangement_template=arrangement_template
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

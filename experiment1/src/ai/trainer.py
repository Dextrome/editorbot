"""Training module to learn song structures from reference tracks."""

import json
import pickle
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa

from .arranger import SongArranger, Section


@dataclass
class TransitionPattern:
    """Detailed pattern for a single transition between sections."""
    from_label: str
    to_label: str
    energy_ratio: float  # to_energy / from_energy
    chroma_similarity: float  # harmonic continuity (0-1)
    timbre_similarity: float  # timbral continuity (0-1)
    brightness_change: float  # to_brightness - from_brightness
    tempo_ratio: float  # local tempo change if any
    transition_duration: float  # duration of transition region in seconds


@dataclass
class SongProfile:
    """Profile of a single analyzed song."""
    name: str
    duration: float
    tempo: float
    key: str
    
    # Structure
    num_sections: int
    section_labels: List[str]
    section_durations: List[float]
    section_ratios: List[float]  # Duration as percentage of total
    
    # Energy curve
    energy_curve: List[float]  # Normalized energy over time (10 points)
    energy_mean: float
    energy_variance: float
    
    # Transitions
    transition_types: List[str]  # "rise", "fall", "steady"
    avg_section_duration: float
    transition_patterns: List[TransitionPattern]  # Detailed transition features
    
    # Audio features
    avg_brightness: float
    avg_harmonicity: float


@dataclass
class StyleProfile:
    """Learned profile for a genre/style based on multiple songs."""
    name: str
    num_songs: int
    
    # Typical structure
    common_structures: List[Tuple[List[str], int]]  # (structure, count)
    avg_num_sections: float
    typical_section_order: List[str]  # Most common section sequence
    
    # Timing
    avg_duration: float
    avg_tempo: float
    avg_section_duration: float
    section_duration_ranges: Dict[str, Tuple[float, float]]  # label -> (min, max)
    
    # Energy profile
    target_energy_curve: List[float]  # Ideal energy shape (normalized)
    energy_variance_range: Tuple[float, float]
    
    # Section ratios (what % of song each section type takes)
    section_ratios: Dict[str, float]
    
    # Transitions
    preferred_transitions: Dict[str, List[str]]  # section -> likely next sections
    
    # Learned transition patterns (NEW)
    transition_patterns: Dict[str, Dict[str, float]]  # "label1_to_label2" -> {energy_ratio, chroma_sim, etc}
    typical_energy_jumps: List[float]  # Distribution of energy ratios at transitions
    typical_chroma_continuity: float  # Average harmonic continuity (0-1)
    typical_timbre_continuity: float  # Average timbral continuity (0-1)
    
    # Section timing (NEW) - learned target durations per section type
    section_timing: Dict[str, Dict[str, float]]  # label -> {"mean": 45.0, "std": 10.0, "min": 30.0, "max": 60.0}


class SongTrainer:
    """
    Learn song structures and styles from reference tracks.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.arranger = SongArranger(sample_rate)
        self.song_profiles: Dict[str, List[SongProfile]] = {}  # style -> profiles
        self.style_profiles: Dict[str, StyleProfile] = {}
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Optional encoder (projection + NATTEN) that can be used during training
        self.encoder = None

    def set_encoder(self, encoder):
        """Register a PyTorch encoder (e.g., NattenFrameEncoder) for use in training routines."""
        self.encoder = encoder
    
    def analyze_reference_song(
        self, 
        audio_path: str,
        song_name: Optional[str] = None
    ) -> SongProfile:
        """
        Analyze a reference song and create a profile.
        
        Args:
            audio_path: Path to the reference audio file.
            song_name: Optional name for the song.
            
        Returns:
            SongProfile with extracted characteristics.
        """
        logger = logging.getLogger(__name__)
        logger.info("📊 Analyzing: %s", audio_path)
        
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(audio_data) / self.sample_rate
        
        if song_name is None:
            song_name = Path(audio_path).stem
        
        # Basic analysis
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        tempo = float(tempo)
        
        key = self._detect_key(audio_data)
        
        # Analyze sections
        sections = self.arranger.analyze_sections(audio_data)
        
        section_labels = [s.label for s in sections]
        section_durations = [s.duration for s in sections]
        section_ratios = [d / duration for d in section_durations]
        
        # Energy curve (sample at 10 points through the song)
        energy_curve = self._compute_energy_curve(audio_data, num_points=10)
        energy_mean = float(np.mean(energy_curve))
        energy_variance = float(np.var(energy_curve))
        
        # Transition analysis - basic types
        transition_types = []
        for i in range(len(sections) - 1):
            e1 = sections[i].energy
            e2 = sections[i + 1].energy
            if e2 > e1 * 1.2:
                transition_types.append("rise")
            elif e2 < e1 * 0.8:
                transition_types.append("fall")
            else:
                transition_types.append("steady")
        
        # Detailed transition pattern extraction (NEW)
        transition_patterns = self._extract_transition_patterns(audio_data, sections)
        
        avg_section_duration = float(np.mean(section_durations)) if section_durations else 30.0
        
        # Average features
        avg_brightness = float(np.mean([s.features.get("brightness", 2000) for s in sections]))
        avg_harmonicity = float(np.mean([s.features.get("harmonicity", 0.5) for s in sections]))
        
        profile = SongProfile(
            name=song_name,
            duration=duration,
            tempo=tempo,
            key=key,
            num_sections=len(sections),
            section_labels=section_labels,
            section_durations=section_durations,
            section_ratios=section_ratios,
            energy_curve=energy_curve,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            transition_types=transition_types,
            avg_section_duration=avg_section_duration,
            transition_patterns=transition_patterns,
            avg_brightness=avg_brightness,
            avg_harmonicity=avg_harmonicity
        )
        
        logger.info("   ✓ Tempo: %.0f BPM, Key: %s", tempo, key)
        logger.info("   ✓ Sections: %d (%s)", len(sections), ' → '.join(section_labels))
        logger.info("   ✓ Duration: %.0fs (%.1f min)", duration, duration/60)
        logger.info("   ✓ Transitions: %d patterns extracted", len(transition_patterns))
        
        return profile
    
    def _extract_transition_patterns(
        self, 
        audio_data: np.ndarray, 
        sections: List
    ) -> List[TransitionPattern]:
        """
        Extract detailed transition patterns between consecutive sections.
        
        Analyzes the musical characteristics of each transition:
        - Energy ratio (how much energy changes)
        - Chroma similarity (harmonic continuity)
        - Timbre similarity (tonal continuity)
        - Brightness change
        """
        patterns = []
        
        if len(sections) < 2:
            return patterns
        
        # Use adaptive hop_length for memory efficiency
        target_frames = 5000
        hop_length = max(512, int(len(audio_data) / target_frames))
        
        for i in range(len(sections) - 1):
            s1 = sections[i]
            s2 = sections[i + 1]
            
            # Get audio for each section (last 2s of s1, first 2s of s2)
            transition_window = 2.0  # seconds
            
            # End of section 1
            s1_end_start = max(s1.start_time, s1.end_time - transition_window)
            s1_start_sample = int(s1_end_start * self.sample_rate)
            s1_end_sample = int(s1.end_time * self.sample_rate)
            s1_audio = audio_data[s1_start_sample:s1_end_sample]
            
            # Start of section 2
            s2_start_sample = int(s2.start_time * self.sample_rate)
            s2_end_sample = int(min(s2.start_time + transition_window, s2.end_time) * self.sample_rate)
            s2_audio = audio_data[s2_start_sample:s2_end_sample]
            
            if len(s1_audio) < self.sample_rate * 0.5 or len(s2_audio) < self.sample_rate * 0.5:
                continue  # Skip if sections are too short
            
            # Compute features for each transition region
            try:
                # Energy ratio
                rms1 = np.sqrt(np.mean(s1_audio ** 2))
                rms2 = np.sqrt(np.mean(s2_audio ** 2))
                energy_ratio = float(rms2 / max(rms1, 1e-6))
                
                # Chroma (harmonic content)
                chroma1 = librosa.feature.chroma_cqt(y=s1_audio, sr=self.sample_rate, hop_length=hop_length)
                chroma2 = librosa.feature.chroma_cqt(y=s2_audio, sr=self.sample_rate, hop_length=hop_length)
                chroma1_avg = np.mean(chroma1, axis=1)
                chroma2_avg = np.mean(chroma2, axis=1)
                # Normalize
                chroma1_avg = chroma1_avg / (np.linalg.norm(chroma1_avg) + 1e-6)
                chroma2_avg = chroma2_avg / (np.linalg.norm(chroma2_avg) + 1e-6)
                chroma_similarity = float(np.dot(chroma1_avg, chroma2_avg))
                
                # MFCC (timbre)
                mfcc1 = librosa.feature.mfcc(y=s1_audio, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
                mfcc2 = librosa.feature.mfcc(y=s2_audio, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
                mfcc1_avg = np.mean(mfcc1, axis=1)
                mfcc2_avg = np.mean(mfcc2, axis=1)
                # Normalize
                mfcc1_avg = mfcc1_avg / (np.linalg.norm(mfcc1_avg) + 1e-6)
                mfcc2_avg = mfcc2_avg / (np.linalg.norm(mfcc2_avg) + 1e-6)
                timbre_similarity = float(np.dot(mfcc1_avg, mfcc2_avg))
                
                # Brightness (spectral centroid)
                bright1 = float(np.mean(librosa.feature.spectral_centroid(y=s1_audio, sr=self.sample_rate, hop_length=hop_length)))
                bright2 = float(np.mean(librosa.feature.spectral_centroid(y=s2_audio, sr=self.sample_rate, hop_length=hop_length)))
                brightness_change = bright2 - bright1
                
                pattern = TransitionPattern(
                    from_label=s1.label,
                    to_label=s2.label,
                    energy_ratio=energy_ratio,
                    chroma_similarity=chroma_similarity,
                    timbre_similarity=timbre_similarity,
                    brightness_change=brightness_change,
                    tempo_ratio=1.0,  # Could detect local tempo changes
                    transition_duration=transition_window * 2
                )
                patterns.append(pattern)
                
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning("Error extracting transition %s→%s: %s", s1.label, s2.label, e)
                continue
        
        return patterns
    
    def _detect_key(self, audio_data: np.ndarray) -> str:
        """Detect musical key."""
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_index = int(np.argmax(chroma_avg))
        return f"{keys[key_index]} major"
    
    def _compute_energy_curve(self, audio_data: np.ndarray, num_points: int = 10) -> List[float]:
        """Compute normalized energy curve across the song."""
        rms = librosa.feature.rms(y=audio_data)[0]
        
        # Sample at num_points evenly spaced positions
        indices = np.linspace(0, len(rms) - 1, num_points).astype(int)
        samples = rms[indices]
        
        # Normalize to 0-1
        max_val = np.max(samples)
        if max_val > 0:
            samples = samples / max_val
        
        return samples.tolist()
    
    def add_to_style(self, style_name: str, profile: SongProfile):
        """Add a song profile to a style category."""
        if style_name not in self.song_profiles:
            self.song_profiles[style_name] = []
        
        self.song_profiles[style_name].append(profile)
        logger = logging.getLogger(__name__)
        logger.info("   ✓ Added to style '%s' (%d songs)", style_name, len(self.song_profiles[style_name]))
    
    def train_style(
        self, 
        style_name: str,
        audio_paths: List[str],
        song_names: Optional[List[str]] = None
    ) -> StyleProfile:
        """
        Train a style profile from multiple reference songs.
        
        Args:
            style_name: Name for this style (e.g., "doom", "pop", "edm").
            audio_paths: List of paths to reference audio files.
            song_names: Optional names for the songs.
            
        Returns:
            StyleProfile learned from the reference songs.
        """
        logger = logging.getLogger(__name__)
        logger.info("\n🎓 Training style: %s", style_name)
        logger.info("   Using %d reference tracks\n", len(audio_paths))
        
        profiles = []
        for i, path in enumerate(audio_paths):
            # Check if file still exists
            if not Path(path).exists():
                logger.warning("   File not found (skipping): %s", path)
                continue
                
            name = song_names[i] if song_names and i < len(song_names) else None
            try:
                profile = self.analyze_reference_song(path, name)
                profiles.append(profile)
                self.add_to_style(style_name, profile)
            except Exception as e:
                logger.warning("   Error analyzing %s: %s", path, e)
        
        if not profiles:
            raise ValueError("No songs were successfully analyzed")
        
        # Build style profile from collected data
        style_profile = self._build_style_profile(style_name, profiles)
        self.style_profiles[style_name] = style_profile
        
        logger.info("\n✅ Style '%s' trained successfully!", style_name)
        logger.info("   Typical structure: %s", ' → '.join(style_profile.typical_section_order))
        logger.info("   Avg duration: %.1f min", style_profile.avg_duration/60)
        logger.info("   Avg tempo: %.0f BPM", style_profile.avg_tempo)
        
        return style_profile
    
    def _build_style_profile(self, name: str, profiles: List[SongProfile]) -> StyleProfile:
        """Build a style profile from multiple song profiles."""
        
        # Collect structures
        structures = [tuple(p.section_labels) for p in profiles]
        structure_counts = {}
        for s in structures:
            structure_counts[s] = structure_counts.get(s, 0) + 1
        common_structures = sorted(structure_counts.items(), key=lambda x: x[1], reverse=True)
        common_structures = [(list(s), c) for s, c in common_structures[:5]]
        
        # Average values
        avg_duration = float(np.mean([p.duration for p in profiles]))
        avg_tempo = float(np.mean([p.tempo for p in profiles]))
        avg_num_sections = float(np.mean([p.num_sections for p in profiles]))
        avg_section_duration = float(np.mean([p.avg_section_duration for p in profiles]))
        
        # Most common section order
        # Use the most common structure, or build from transitions
        if common_structures:
            typical_section_order = common_structures[0][0]
        else:
            typical_section_order = ["intro", "verse", "chorus", "outro"]
        
        # Section duration ranges
        section_durations: Dict[str, List[float]] = {}
        section_ratios_all: Dict[str, List[float]] = {}
        
        for profile in profiles:
            for label, duration, ratio in zip(
                profile.section_labels, 
                profile.section_durations,
                profile.section_ratios
            ):
                if label not in section_durations:
                    section_durations[label] = []
                    section_ratios_all[label] = []
                section_durations[label].append(duration)
                section_ratios_all[label].append(ratio)
        
        section_duration_ranges = {
            label: (min(durations), max(durations))
            for label, durations in section_durations.items()
        }
        
        # Compute detailed section timing stats (NEW)
        section_timing = {}
        for label, durations in section_durations.items():
            section_timing[label] = {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)) if len(durations) > 1 else 5.0,
                "min": float(min(durations)),
                "max": float(max(durations)),
                "count": len(durations)
            }
        
        section_ratios = {
            label: float(np.mean(ratios))
            for label, ratios in section_ratios_all.items()
        }
        
        # Target energy curve (average of all songs)
        energy_curves = np.array([p.energy_curve for p in profiles])
        target_energy_curve = np.mean(energy_curves, axis=0).tolist()
        
        energy_variances = [p.energy_variance for p in profiles]
        energy_variance_range = (min(energy_variances), max(energy_variances))
        
        # Preferred transitions
        preferred_transitions: Dict[str, List[str]] = {}
        for profile in profiles:
            for i in range(len(profile.section_labels) - 1):
                current = profile.section_labels[i]
                next_section = profile.section_labels[i + 1]
                if current not in preferred_transitions:
                    preferred_transitions[current] = []
                preferred_transitions[current].append(next_section)
        
        # Keep most common transitions
        for label in preferred_transitions:
            transitions = preferred_transitions[label]
            # Count and sort
            counts = {}
            for t in transitions:
                counts[t] = counts.get(t, 0) + 1
            preferred_transitions[label] = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        
        # Aggregate learned transition patterns (NEW)
        transition_patterns, typical_energy_jumps, typical_chroma, typical_timbre = \
            self._aggregate_transition_patterns(profiles)
        
        return StyleProfile(
            name=name,
            num_songs=len(profiles),
            common_structures=common_structures,
            avg_num_sections=avg_num_sections,
            typical_section_order=typical_section_order,
            avg_duration=avg_duration,
            avg_tempo=avg_tempo,
            avg_section_duration=avg_section_duration,
            section_duration_ranges=section_duration_ranges,
            target_energy_curve=target_energy_curve,
            energy_variance_range=energy_variance_range,
            section_ratios=section_ratios,
            preferred_transitions=preferred_transitions,
            transition_patterns=transition_patterns,
            typical_energy_jumps=typical_energy_jumps,
            typical_chroma_continuity=typical_chroma,
            typical_timbre_continuity=typical_timbre,
            section_timing=section_timing
        )
    
    def _aggregate_transition_patterns(
        self, 
        profiles: List[SongProfile]
    ) -> Tuple[Dict[str, Dict[str, float]], List[float], float, float]:
        """
        Aggregate transition patterns from all song profiles into style-level patterns.
        
        Returns:
            - transition_patterns: Dict mapping "label1_to_label2" -> {avg features}
            - typical_energy_jumps: List of all energy ratios (for distribution)
            - typical_chroma_continuity: Average chroma similarity across all transitions
            - typical_timbre_continuity: Average timbre similarity across all transitions
        """
        # Collect all transitions by type
        transitions_by_type: Dict[str, List[TransitionPattern]] = {}
        all_energy_ratios = []
        all_chroma_sims = []
        all_timbre_sims = []
        
        for profile in profiles:
            for pattern in profile.transition_patterns:
                key = f"{pattern.from_label}_to_{pattern.to_label}"
                if key not in transitions_by_type:
                    transitions_by_type[key] = []
                transitions_by_type[key].append(pattern)
                
                all_energy_ratios.append(pattern.energy_ratio)
                all_chroma_sims.append(pattern.chroma_similarity)
                all_timbre_sims.append(pattern.timbre_similarity)
        
        # Aggregate into averages per transition type
        transition_patterns: Dict[str, Dict[str, float]] = {}
        for key, patterns in transitions_by_type.items():
            transition_patterns[key] = {
                "energy_ratio_mean": float(np.mean([p.energy_ratio for p in patterns])),
                "energy_ratio_std": float(np.std([p.energy_ratio for p in patterns])),
                "chroma_similarity": float(np.mean([p.chroma_similarity for p in patterns])),
                "timbre_similarity": float(np.mean([p.timbre_similarity for p in patterns])),
                "brightness_change": float(np.mean([p.brightness_change for p in patterns])),
                "count": len(patterns)
            }
        
        # Global statistics
        typical_energy_jumps = all_energy_ratios if all_energy_ratios else [1.0]
        typical_chroma = float(np.mean(all_chroma_sims)) if all_chroma_sims else 0.5
        typical_timbre = float(np.mean(all_timbre_sims)) if all_timbre_sims else 0.5
        
        return transition_patterns, typical_energy_jumps, typical_chroma, typical_timbre
    
    def save_style(self, style_name: str, filepath: Optional[str] = None):
        """Save a trained style to disk."""
        if style_name not in self.style_profiles:
            raise ValueError(f"Style '{style_name}' not found")
        
        if filepath is None:
            filepath = self.data_dir / f"style_{style_name}.pkl"
        
        data = {
            "style_profile": self.style_profiles[style_name],
            "song_profiles": self.song_profiles.get(style_name, [])
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger = logging.getLogger(__name__)
        logger.info("💾 Saved style '%s' to %s", style_name, filepath)
    
    def load_style(self, style_name: str, filepath: Optional[str] = None):
        """Load a trained style from disk."""
        if filepath is None:
            filepath = self.data_dir / f"style_{style_name}.pkl"
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.style_profiles[style_name] = data["style_profile"]
        self.song_profiles[style_name] = data["song_profiles"]
        
        logger = logging.getLogger(__name__)
        logger.info("📂 Loaded style '%s' (%d songs)", style_name, data['style_profile'].num_songs)
    
    def list_styles(self) -> List[str]:
        """List available trained styles."""
        return list(self.style_profiles.keys())
    
    def get_style_summary(self, style_name: str) -> str:
        """Get a summary of a trained style."""
        if style_name not in self.style_profiles:
            return f"Style '{style_name}' not found"
        
        sp = self.style_profiles[style_name]
        
        lines = [
            f"🎵 Style: {style_name}",
            "=" * 40,
            f"Trained on: {sp.num_songs} songs",
            f"",
            f"Structure:",
            f"  Typical: {' → '.join(sp.typical_section_order)}",
            f"  Avg sections: {sp.avg_num_sections:.1f}",
            f"",
            f"Timing:",
            f"  Avg duration: {sp.avg_duration/60:.1f} min",
            f"  Avg tempo: {sp.avg_tempo:.0f} BPM",
            f"  Avg section: {sp.avg_section_duration:.0f}s",
            f"",
            f"Section ratios:"
        ]
        
        for label, ratio in sorted(sp.section_ratios.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {label}: {ratio*100:.1f}%")
        
        lines.extend([
            f"",
            f"Energy curve: {'↗' if sp.target_energy_curve[5] > sp.target_energy_curve[0] else '↘'}"
            f" (builds {'up' if sp.target_energy_curve[5] > sp.target_energy_curve[2] else 'down'})"
        ])
        
        # Add transition pattern info (NEW)
        if sp.transition_patterns:
            lines.extend([
                f"",
                f"Learned transition patterns: {len(sp.transition_patterns)}"
            ])
            # Show top patterns by frequency
            sorted_patterns = sorted(
                sp.transition_patterns.items(), 
                key=lambda x: x[1].get("count", 0), 
                reverse=True
            )[:5]
            for key, data in sorted_patterns:
                from_label, to_label = key.replace("_to_", " → ").split(" → ")
                energy_change = "↗" if data.get("energy_ratio_mean", 1.0) > 1.1 else \
                               ("↘" if data.get("energy_ratio_mean", 1.0) < 0.9 else "→")
                chroma_sim = data.get("chroma_similarity", 0.5)
                lines.append(f"  {from_label} → {to_label}: {energy_change} (harmonic: {chroma_sim:.2f})")
            
            lines.extend([
                f"",
                f"Typical continuity:",
                f"  Harmonic: {sp.typical_chroma_continuity:.2f}",
                f"  Timbral: {sp.typical_timbre_continuity:.2f}"
            ])
        
        # Show section timing (NEW)
        if hasattr(sp, 'section_timing') and sp.section_timing:
            lines.extend([
                f"",
                f"Learned section timing:"
            ])
            for label, timing in sorted(sp.section_timing.items(), key=lambda x: x[1].get("mean", 0), reverse=True):
                mean_dur = timing.get("mean", 0)
                std_dur = timing.get("std", 0)
                lines.append(f"  {label}: {mean_dur:.0f}s ± {std_dur:.0f}s")
        
        return "\n".join(lines)


class LearnedArranger(SongArranger):
    """
    Arranger that uses learned style profiles to create arrangements.
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.trainer = SongTrainer(sample_rate)
        self.current_style: Optional[StyleProfile] = None
    
    def load_style(self, style_name: str):
        """Load a trained style to use for arrangement."""
        self.trainer.load_style(style_name)
        self.current_style = self.trainer.style_profiles.get(style_name)
        logger = logging.getLogger(__name__)
        logger.info("🎨 Using style: %s", style_name)
    
    def arrange_with_style(
        self,
        audio_data: np.ndarray,
        style_name: Optional[str] = None
    ) -> Tuple[np.ndarray, 'Arrangement', List[Section]]:
        """
        Arrange audio using a learned style profile.
        
        Args:
            audio_data: Raw audio to arrange.
            style_name: Style to use (uses current if not specified).
            
        Returns:
            Tuple of (arranged_audio, arrangement, sections).
        """
        if style_name:
            self.load_style(style_name)
        
        if not self.current_style:
            logger.warning("No style loaded, using default arrangement")
            return self.auto_arrange(audio_data)
        
        # Analyze input sections
        sections = self.analyze_sections(audio_data)
        
        # Create arrangement following the learned style
        arrangement = self._create_styled_arrangement(audio_data, sections)
        
        # Render
        arranged_audio = self.render_arrangement(audio_data, arrangement)
        
        return arranged_audio, arrangement, sections
    
    def _create_styled_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section]
    ) -> 'Arrangement':
        """Create arrangement following the learned style profile with smooth transitions."""
        from .arranger import Arrangement
        
        style = self.current_style
        target_duration = style.avg_duration
        target_structure = style.typical_section_order.copy()
        
        # Analyze section boundaries for better transitions
        section_transitions = self._analyze_section_transitions(audio_data, sections)
        
        # Map our sections to the target structure
        sections_by_energy = sorted(sections, key=lambda s: s.energy)
        
        # Categorize our sections
        num_sections = len(sections)
        quiet_sections = sections_by_energy[:num_sections//3]
        mid_sections = sections_by_energy[num_sections//3:2*num_sections//3]
        loud_sections = sections_by_energy[2*num_sections//3:]
        
        # Map style sections to energy categories
        section_mapping = {
            "intro": quiet_sections,
            "outro": quiet_sections,
            "verse": mid_sections or quiet_sections,
            "chorus": loud_sections or mid_sections,
            "bridge": mid_sections,
            "breakdown": quiet_sections,
            "buildup": mid_sections,
            "drop": loud_sections,
            "riff": loud_sections,
            "build": mid_sections,
        }
        
        arrangement_sections = []
        current_time = 0.0
        used = set()
        previous_section = None
        
        for i, target_label in enumerate(target_structure):
            if current_time >= target_duration:
                break
            
            # Get target ratio for this section type
            target_ratio = style.section_ratios.get(target_label, 0.1)
            section_duration = target_duration * target_ratio
            
            # Find matching section that flows well from previous
            candidates = section_mapping.get(target_label, sections)
            
            # Score candidates by how well they connect to previous section
            if previous_section and candidates:
                scored_candidates = []
                for c in candidates:
                    if id(c) not in used:
                        flow_score = self._score_transition(
                            audio_data, previous_section, c, section_transitions
                        )
                        scored_candidates.append((c, flow_score))
                
                # Sort by flow score (higher is better)
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                if scored_candidates:
                    selected = scored_candidates[0][0]
                    used.add(id(selected))
                else:
                    # All used, pick best flowing even if used
                    all_scored = [(c, self._score_transition(audio_data, previous_section, c, section_transitions)) 
                                  for c in candidates]
                    all_scored.sort(key=lambda x: x[1], reverse=True)
                    selected = all_scored[0][0] if all_scored else candidates[0]
            else:
                # First section - just pick first unused
                selected = None
                for c in candidates:
                    if id(c) not in used:
                        selected = c
                        used.add(id(c))
                        break
                
                if not selected and candidates:
                    selected = candidates[0]
                elif not selected:
                    selected = sections[0]
            
            use_duration = min(section_duration, selected.duration)
            use_duration = max(use_duration, 10.0)
            
            arrangement_sections.append((
                Section(
                    start_time=selected.start_time,
                    end_time=selected.start_time + use_duration,
                    label=target_label,
                    energy=selected.energy,
                    features=selected.features
                ),
                current_time,
                current_time + use_duration
            ))
            current_time += use_duration
            previous_section = selected
        
        return Arrangement(
            sections=arrangement_sections,
            total_duration=current_time,
            structure=[s[0].label for s in arrangement_sections]
        )
    
    def _analyze_section_transitions(
        self, 
        audio_data: np.ndarray, 
        sections: List[Section]
    ) -> Dict:
        """
        Analyze the start and end characteristics of each section for better transitions.
        """
        import librosa
        
        transitions = {}
        analysis_duration = 1.0  # Analyze 1 second at start/end
        analysis_samples = int(analysis_duration * self.sample_rate)
        
        for section in sections:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            
            # Get the start and end segments
            start_segment = audio_data[start_sample:start_sample + analysis_samples]
            end_segment = audio_data[max(start_sample, end_sample - analysis_samples):end_sample]
            
            if len(start_segment) < analysis_samples // 2 or len(end_segment) < analysis_samples // 2:
                continue
            
            # Analyze characteristics
            section_id = id(section)
            transitions[section_id] = {
                # Energy at boundaries
                "start_energy": float(np.sqrt(np.mean(start_segment**2))),
                "end_energy": float(np.sqrt(np.mean(end_segment**2))),
                
                # Spectral characteristics at boundaries
                "start_brightness": self._get_brightness(start_segment),
                "end_brightness": self._get_brightness(end_segment),
                
                # Harmonic content (for key matching)
                "start_chroma": self._get_chroma_vector(start_segment),
                "end_chroma": self._get_chroma_vector(end_segment),
                
                # Is the ending "complete" (energy drops) or "continuing"
                "end_type": "complete" if np.mean(end_segment[-len(end_segment)//4:]**2) < np.mean(end_segment[:len(end_segment)//4]**2) * 0.7 else "continuing",
                
                # Is the start abrupt or gradual
                "start_type": "gradual" if np.mean(start_segment[:len(start_segment)//4]**2) < np.mean(start_segment[-len(start_segment)//4:]**2) * 1.3 else "abrupt",
            }
        
        return transitions
    
    def _get_brightness(self, audio_segment: np.ndarray) -> float:
        """Get spectral brightness of a segment."""
        import librosa
        if len(audio_segment) < 512:
            return 2000.0
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)
        return float(np.mean(centroid))
    
    def _get_chroma_vector(self, audio_segment: np.ndarray) -> np.ndarray:
        """Get normalized chroma vector for harmonic analysis."""
        import librosa
        if len(audio_segment) < 512:
            return np.zeros(12)
        chroma = librosa.feature.chroma_cqt(y=audio_segment, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        # Normalize
        norm = np.linalg.norm(chroma_avg)
        if norm > 0:
            chroma_avg = chroma_avg / norm
        return chroma_avg
    
    def _score_transition(
        self, 
        audio_data: np.ndarray,
        from_section: Section, 
        to_section: Section,
        transitions: Dict
    ) -> float:
        """
        Score how well two sections flow together.
        Higher score = better transition.
        """
        from_id = id(from_section)
        to_id = id(to_section)
        
        # If we don't have analysis, return neutral score
        if from_id not in transitions or to_id not in transitions:
            return 0.5
        
        from_trans = transitions[from_id]
        to_trans = transitions[to_id]
        
        score = 0.0
        
        # 1. Energy continuity (25% of score)
        # Prefer transitions where end energy is similar to start energy
        energy_diff = abs(from_trans["end_energy"] - to_trans["start_energy"])
        max_energy = max(from_trans["end_energy"], to_trans["start_energy"], 0.01)
        energy_score = 1.0 - min(energy_diff / max_energy, 1.0)
        score += energy_score * 0.25
        
        # 2. Brightness continuity (15% of score)
        brightness_diff = abs(from_trans["end_brightness"] - to_trans["start_brightness"])
        brightness_score = 1.0 - min(brightness_diff / 3000, 1.0)
        score += brightness_score * 0.15
        
        # 3. Harmonic compatibility (30% of score)
        # Compare chroma vectors - similar harmonics flow better
        chroma_similarity = np.dot(from_trans["end_chroma"], to_trans["start_chroma"])
        score += max(0, chroma_similarity) * 0.30
        
        # 4. End/start type compatibility (20% of score)
        # "complete" endings work with any start
        # "continuing" endings prefer "gradual" starts
        if from_trans["end_type"] == "complete":
            score += 0.20
        elif to_trans["start_type"] == "gradual":
            score += 0.15
        else:
            score += 0.05
        
        # 5. Not the same section (10% of score)
        # Avoid repeating the exact same section back-to-back
        if from_section.start_time != to_section.start_time:
            score += 0.10
        
        return score
    
    def train_with_demucs_and_natten(self, audio_paths: List[str], style_name: str):
        """
        Train a style profile using Demucs for stem separation and NATTEN for feature extraction.

        Args:
            audio_paths: List of paths to training audio files.
            style_name: Name of the style to train.
        """
        profiles = []

        for audio_path in audio_paths:
            # Load audio
            audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)

            # Separate stems using Demucs
            stems = self.arranger.processor.separate_stems(audio_data, device=self.arranger.demucs_device)

            # Analyze sections with stems
            sections = self.arranger.analyze_sections_with_stems(audio_data, stems)

            # Extract NATTEN-based features for transitions
            transition_patterns = self._extract_transition_patterns(audio_data, sections)

            # Create a song profile
            profile = SongProfile(
                name=Path(audio_path).stem,
                duration=librosa.get_duration(y=audio_data, sr=self.sample_rate),
                tempo=self.arranger.detect_beat_grid(audio_data)[0],
                key=self._detect_key(audio_data),
                num_sections=len(sections),
                section_labels=[section.label for section in sections],
                section_durations=[section.duration for section in sections],
                section_ratios=[section.duration / sum(section.duration for section in sections) for section in sections],
                energy_curve=self._compute_energy_curve(audio_data),
                energy_mean=np.mean([section.energy for section in sections]),
                energy_variance=np.var([section.energy for section in sections]),
                transition_types=["steady" for _ in sections],  # Placeholder
                avg_section_duration=np.mean([section.duration for section in sections]),
                transition_patterns=transition_patterns,
                avg_brightness=0.0,  # Placeholder
                avg_harmonicity=0.0,  # Placeholder
            )

            profiles.append(profile)

        # Build and save the style profile
        style_profile = self._build_style_profile(style_name, profiles)
        self.save_style(style_name)

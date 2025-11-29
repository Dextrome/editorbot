"""AI-powered song arranger that creates coherent song structures."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import librosa

if TYPE_CHECKING:
    from .trainer import StyleProfile


@dataclass
class Phrase:
    """Represents a musical phrase (typically 2-8 bars) within a section."""
    start_time: float
    end_time: float
    bar_start: int  # Which bar this phrase starts on
    num_bars: int  # How many bars in this phrase
    energy: float
    is_downbeat_aligned: bool = True  # Whether it starts on a downbeat
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"Phrase({self.start_time:.1f}s-{self.end_time:.1f}s, {self.num_bars} bars, energy={self.energy:.2f})"


@dataclass
class Section:
    """Represents a detected section of audio."""
    start_time: float
    end_time: float
    label: str  # "intro", "verse", "chorus", "bridge", "outro", "buildup", "drop"
    energy: float  # Average energy level
    features: Dict[str, float]  # Characteristic features
    confidence: float = 1.0
    phrases: List[Phrase] = field(default_factory=list)  # Sub-phrases within this section
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self):
        phrase_info = f", {len(self.phrases)} phrases" if self.phrases else ""
        return f"Section({self.label}, {self.start_time:.1f}s-{self.end_time:.1f}s, energy={self.energy:.2f}{phrase_info})"


@dataclass 
class Arrangement:
    """Represents a song arrangement plan."""
    sections: List[Tuple[Section, float, float]]  # (section, new_start, new_end)
    total_duration: float
    structure: List[str]  # e.g., ["intro", "verse", "chorus", "verse", "chorus", "outro"]


class SongArranger:
    """
    Analyzes raw recordings and arranges them into coherent song structures.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._beat_times = None  # Cached beat times
        self._tempo = None  # Cached tempo
        self._bar_times = None  # Cached bar (downbeat) times
        
        # Song structure templates
        # All now use simpler section-level arrangement for coherence
        self.templates = {
            "short": {"target_minutes": 3, "min_sections": 3, "max_sections": 5},
            "medium": {"target_minutes": 5, "min_sections": 4, "max_sections": 7},
            "long": {"target_minutes": 8, "min_sections": 5, "max_sections": 10},
            "full": {"target_minutes": None, "min_sections": None, "max_sections": None},  # Use all good material
            "content": None,  # Legacy: use actual detected sections as-is
        }
    
    def detect_beat_grid(self, audio_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Detect tempo and create a beat grid for the entire track.
        
        Returns:
            Tuple of (tempo_bpm, beat_times, bar_times)
            bar_times are downbeats (every 4 beats assuming 4/4 time)
        """
        if self._tempo is not None:
            return self._tempo, self._beat_times, self._bar_times
        
        # Detect tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        
        # Handle tempo as array (newer librosa) or scalar
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        
        # Create bar times (every 4 beats for 4/4 time)
        # Also try to detect actual downbeats using onset strength
        bar_times = beat_times[::4] if len(beat_times) >= 4 else beat_times
        
        # Cache results
        self._tempo = tempo
        self._beat_times = beat_times
        self._bar_times = bar_times
        
        return tempo, beat_times, bar_times
    
    def split_into_phrases(
        self, 
        audio_data: np.ndarray, 
        section: Section,
        bars_per_phrase: int = 8
    ) -> List[Phrase]:
        """
        Split a section into bar-aligned musical phrases.
        
        Args:
            audio_data: Full audio data
            section: Section to split
            bars_per_phrase: Target number of bars per phrase (default 8 = two 4-bar phrases, more coherent)
            
        Returns:
            List of Phrase objects
        """
        tempo, beat_times, bar_times = self.detect_beat_grid(audio_data)
        
        # Find bars within this section
        section_bars = [t for t in bar_times if section.start_time <= t < section.end_time]
        
        if len(section_bars) < 2:
            # Section too short for meaningful phrases, return as single phrase
            return [Phrase(
                start_time=section.start_time,
                end_time=section.end_time,
                bar_start=0,
                num_bars=1,
                energy=section.energy,
                is_downbeat_aligned=False
            )]
        
        # Group bars into phrases
        phrases = []
        bar_idx = 0
        phrase_num = 0
        
        while bar_idx < len(section_bars):
            phrase_start = section_bars[bar_idx]
            
            # Determine phrase end (bars_per_phrase bars later, or section end)
            end_bar_idx = min(bar_idx + bars_per_phrase, len(section_bars))
            
            if end_bar_idx < len(section_bars):
                phrase_end = section_bars[end_bar_idx]
            else:
                # Last phrase extends to section end
                phrase_end = section.end_time
            
            # Calculate phrase energy
            start_sample = int(phrase_start * self.sample_rate)
            end_sample = int(phrase_end * self.sample_rate)
            phrase_audio = audio_data[start_sample:end_sample]
            
            if len(phrase_audio) > 0:
                phrase_energy = float(np.sqrt(np.mean(phrase_audio**2)))
            else:
                phrase_energy = section.energy
            
            phrases.append(Phrase(
                start_time=phrase_start,
                end_time=phrase_end,
                bar_start=bar_idx,
                num_bars=end_bar_idx - bar_idx,
                energy=phrase_energy,
                is_downbeat_aligned=True
            ))
            
            bar_idx = end_bar_idx
            phrase_num += 1
        
        # Handle audio before first detected bar
        if section_bars[0] > section.start_time + 0.5:
            # There's significant audio before the first bar, add it as a pickup phrase
            pickup_start = section.start_time
            pickup_end = section_bars[0]
            
            start_sample = int(pickup_start * self.sample_rate)
            end_sample = int(pickup_end * self.sample_rate)
            pickup_audio = audio_data[start_sample:end_sample]
            pickup_energy = float(np.sqrt(np.mean(pickup_audio**2))) if len(pickup_audio) > 0 else section.energy
            
            phrases.insert(0, Phrase(
                start_time=pickup_start,
                end_time=pickup_end,
                bar_start=-1,
                num_bars=0,
                energy=pickup_energy,
                is_downbeat_aligned=False
            ))
        
        return phrases
    
    def snap_to_bar(self, time: float, bar_times: np.ndarray) -> float:
        """Snap a time to the nearest bar boundary."""
        if len(bar_times) == 0:
            return time
        
        idx = np.argmin(np.abs(bar_times - time))
        return float(bar_times[idx])
    
    def analyze_sections(
        self, 
        audio_data: np.ndarray,
        min_section_duration: Optional[float] = None,
        max_section_duration: Optional[float] = None
    ) -> List[Section]:
        """
        Analyze audio and identify distinct sections with characteristics.
        
        Args:
            audio_data: Raw audio data.
            min_section_duration: Minimum section duration in seconds (default: 8s).
            max_section_duration: Maximum section duration in seconds (default: ~30s based on audio length).
            
        Returns:
            List of detected sections with labels and phrases.
        """
        duration = len(audio_data) / self.sample_rate
        
        # Set defaults
        if min_section_duration is None:
            min_section_duration = 8.0
        if max_section_duration is None:
            max_section_duration = 30.0
        
        # Use adaptive hop_length for long files to avoid memory issues
        target_frames = 5000
        hop_length = max(512, int(len(audio_data) / target_frames))
        
        # Get multiple feature types for better segmentation
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate, hop_length=hop_length)
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate, hop_length=hop_length)
        
        # Stack features
        features = np.vstack([mfcc, chroma, spectral_contrast])
        
        # Detect segment boundaries using structural analysis
        # Number of segments based on max section duration
        num_segments = max(4, min(20, int(duration / max_section_duration) + 1))
        
        try:
            bounds = librosa.segment.agglomerative(features, num_segments)
            bound_times = librosa.frames_to_time(bounds, sr=self.sample_rate, hop_length=hop_length)
        except Exception:
            # Fallback to even segmentation
            bound_times = np.linspace(0, duration, num_segments + 1)
        
        # Add start and end if not present
        if bound_times[0] > 0.1:
            bound_times = np.insert(bound_times, 0, 0)
        if bound_times[-1] < duration - 0.1:
            bound_times = np.append(bound_times, duration)
        
        # Merge very short segments (using configurable min duration)
        bound_times = self._merge_short_segments(bound_times, min_duration=min_section_duration)
        
        # Detect beat grid for phrase alignment
        tempo, beat_times, bar_times = self.detect_beat_grid(audio_data)
        
        # Snap section boundaries to nearest bar
        snapped_bounds = []
        min_snap_duration = max(4.0, min_section_duration - 2)  # Allow slightly shorter after snapping
        for t in bound_times:
            snapped = self.snap_to_bar(t, bar_times)
            # Don't snap if it would create a very short section
            if len(snapped_bounds) == 0 or snapped - snapped_bounds[-1] >= min_snap_duration:
                snapped_bounds.append(snapped)
            elif t - snapped_bounds[-1] >= min_snap_duration:
                snapped_bounds.append(t)  # Keep original if snapping fails
        
        # Ensure we have end time
        if snapped_bounds[-1] < duration - 1.0:
            snapped_bounds.append(duration)
        
        bound_times = np.array(snapped_bounds)
        
        # Analyze each segment
        sections = []
        for i in range(len(bound_times) - 1):
            start_time = bound_times[i]
            end_time = bound_times[i + 1]
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < self.sample_rate * 2:  # Skip very short segments
                continue
            
            # Calculate section characteristics
            section_features = self._analyze_section_features(segment_audio)
            energy = section_features["energy"]
            
            # Classify section type based on features
            label = self._classify_section(section_features, start_time, end_time, duration)
            
            section = Section(
                start_time=start_time,
                end_time=end_time,
                label=label,
                energy=energy,
                features=section_features
            )
            
            # Split section into bar-aligned phrases
            section.phrases = self.split_into_phrases(audio_data, section)
            
            sections.append(section)
        
        # Refine labels based on context
        sections = self._refine_section_labels(sections)
        
        return sections
    
    def _merge_short_segments(self, bound_times: np.ndarray, min_duration: float = 8.0) -> np.ndarray:
        """Merge segments shorter than min_duration with neighbors."""
        if len(bound_times) < 3:
            return bound_times
        
        merged = [bound_times[0]]
        
        for i in range(1, len(bound_times) - 1):
            segment_duration = bound_times[i + 1] - bound_times[i]
            prev_segment_duration = bound_times[i] - merged[-1]
            
            # Only add boundary if both segments will be long enough
            if segment_duration >= min_duration and prev_segment_duration >= min_duration:
                merged.append(bound_times[i])
        
        merged.append(bound_times[-1])
        return np.array(merged)
    
    def _analyze_section_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract characteristic features from a section."""
        features = {}
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio_data)[0]
        features["energy"] = float(np.mean(rms))
        features["energy_variance"] = float(np.var(rms))
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features["brightness"] = float(np.mean(centroid))
        
        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        features["flatness"] = float(np.mean(flatness))
        
        # Zero crossing rate (percussion indicator)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features["percussiveness"] = float(np.mean(zcr))
        
        # Harmonic content
        harmonic = librosa.effects.harmonic(audio_data)
        features["harmonicity"] = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio_data)) + 1e-6))
        
        # Onset density (how "busy" the section is)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
        features["onset_density"] = float(np.mean(onset_env))
        
        return features
    
    def _classify_section(
        self, 
        features: Dict[str, float], 
        start_time: float, 
        end_time: float,
        total_duration: float
    ) -> str:
        """Classify a section based on its features and position."""
        
        position = start_time / total_duration
        energy = features["energy"]
        brightness = features["brightness"]
        onset_density = features["onset_density"]
        harmonicity = features.get("harmonicity", 0.5)
        
        # Position-based hints (only for very start/end)
        if position < 0.05:
            return "intro"
        elif position > 0.95:
            return "outro"
        
        # Energy-based classification with better thresholds
        # Normalize features for comparison
        if energy < 0.03:
            return "breakdown"
        
        # High energy + high brightness = chorus/drop
        if energy > 0.12 and brightness > 2500:
            return "chorus"
        
        # High onset density = busy section (drop or chorus)
        if onset_density > 0.6 and energy > 0.1:
            return "drop"
        
        # High energy but medium brightness = verse with energy
        if energy > 0.08:
            return "verse"
        
        # Building energy = buildup
        if features["energy_variance"] > 0.015:
            return "buildup"
        
        # Default to verse for medium energy
        if energy > 0.04:
            return "verse"
        
        # Low energy = breakdown or intro-like
        return "breakdown"
    
    def _refine_section_labels(self, sections: List[Section]) -> List[Section]:
        """Refine section labels based on context and patterns."""
        if len(sections) < 2:
            return sections
        
        # Calculate average energy to better classify
        avg_energy = np.mean([s.energy for s in sections])
        
        for i, section in enumerate(sections):
            # High energy sections after verse are likely chorus
            if section.energy > avg_energy * 1.3:
                if i > 0 and sections[i-1].label == "verse":
                    section.label = "chorus"
            
            # Low energy after chorus might be verse
            if section.energy < avg_energy * 0.8:
                if i > 0 and sections[i-1].label == "chorus":
                    section.label = "verse"
            
            # Identify bridges (different from both verse and chorus)
            if i > 2 and section.label == "verse":
                prev_verses = [s for s in sections[:i] if s.label == "verse"]
                if prev_verses:
                    avg_verse_brightness = np.mean([s.features["brightness"] for s in prev_verses])
                    if abs(section.features["brightness"] - avg_verse_brightness) > 500:
                        section.label = "bridge"
        
        return sections
    
    def find_similar_sections(
        self, 
        audio_data: np.ndarray, 
        sections: List[Section],
        similarity_threshold: float = 0.75
    ) -> Dict[str, List[Section]]:
        """
        Group sections by similarity to find repeated parts.
        
        Returns:
            Dictionary mapping group IDs to lists of similar sections.
        """
        groups: Dict[str, List[Section]] = {}
        
        for section in sections:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            # Get compact feature representation
            mfcc = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13)
            section_fingerprint = np.mean(mfcc, axis=1)
            
            # Try to match to existing group
            matched = False
            for group_id, group_sections in groups.items():
                # Compare to first section in group
                ref_section = group_sections[0]
                ref_start = int(ref_section.start_time * self.sample_rate)
                ref_end = int(ref_section.end_time * self.sample_rate)
                ref_segment = audio_data[ref_start:ref_end]
                
                ref_mfcc = librosa.feature.mfcc(y=ref_segment, sr=self.sample_rate, n_mfcc=13)
                ref_fingerprint = np.mean(ref_mfcc, axis=1)
                
                # Cosine similarity
                similarity = np.dot(section_fingerprint, ref_fingerprint) / (
                    np.linalg.norm(section_fingerprint) * np.linalg.norm(ref_fingerprint) + 1e-6
                )
                
                if similarity > similarity_threshold:
                    groups[group_id].append(section)
                    matched = True
                    break
            
            if not matched:
                groups[f"group_{len(groups)}"] = [section]
        
        return groups
    
    def create_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        template: str = "simple",
        target_duration: Optional[float] = None,
        style_profile: StyleProfile = None,
        allow_rearrange: bool = False
    ) -> Arrangement:
        """
        Create a song arrangement from detected sections.
        
        Args:
            audio_data: Raw audio data.
            sections: Detected sections.
            template: Song structure template to follow ("content" uses detected sections).
            target_duration: Target duration in seconds (None = auto).
            style_profile: Optional learned style profile for transition scoring.
            allow_rearrange: If True, allow reordering sections for better flow.
            
        Returns:
            Arrangement plan.
        """
        # Get template settings
        template_config = self.templates.get(template, self.templates["medium"])
        
        # "content" mode: use the best sections from the actual recording (legacy)
        if template == "content" or template_config is None:
            return self._create_content_based_arrangement(
                audio_data, sections, target_duration
            )
        
        # All templates now use simpler section-level arrangement for coherence
        # Template config determines target length and section count
        if target_duration is None and template_config.get("target_minutes"):
            target_duration = template_config["target_minutes"] * 60
        
        return self.create_simple_arrangement(
            audio_data, sections, target_duration, style_profile,
            min_sections=template_config.get("min_sections"),
            max_sections=template_config.get("max_sections"),
            allow_rearrange=allow_rearrange
        )
        
        # Legacy code below - keeping for reference but not used
        structure = self.templates.get(template, self.templates["medium"])
        
        # Group similar sections
        section_groups = self.find_similar_sections(audio_data, sections)
        
        # Map section types to available sections
        sections_by_type: Dict[str, List[Section]] = {}
        for section in sections:
            if section.label not in sections_by_type:
                sections_by_type[section.label] = []
            sections_by_type[section.label].append(section)
        
        # Sort sections by quality (energy consistency, no clipping)
        for label in sections_by_type:
            sections_by_type[label].sort(
                key=lambda s: s.features.get("energy_variance", 0),
                reverse=False  # Prefer consistent energy
            )
        
        # Build arrangement
        arrangement_sections = []
        current_time = 0.0
        used_sections: Dict[str, int] = {}  # Track which variations we've used
        
        # Calculate target duration if not specified
        # Aim for 2.5-4 minutes for most songs
        if target_duration is None:
            total_available = sum(s.duration for s in sections)
            target_duration = min(240, max(150, total_available * 0.6))
        
        current_duration = 0.0
        
        for part in structure:
            if current_duration >= target_duration:
                break
                
            # Find a suitable section for this part
            candidates = sections_by_type.get(part, [])
            
            # Fallback mappings
            if not candidates:
                if part in ["chorus", "drop"]:
                    candidates = sections_by_type.get("chorus", []) or sections_by_type.get("drop", [])
                    if not candidates:
                        # Use highest energy sections
                        all_sections = [s for ss in sections_by_type.values() for s in ss]
                        if all_sections:
                            candidates = sorted(all_sections, key=lambda s: s.energy, reverse=True)[:3]
                elif part in ["verse", "breakdown"]:
                    candidates = sections_by_type.get("verse", []) or sections_by_type.get("breakdown", [])
                    if not candidates:
                        all_sections = [s for ss in sections_by_type.values() for s in ss]
                        if all_sections:
                            candidates = sorted(all_sections, key=lambda s: s.energy)[:3]
                elif part == "bridge":
                    candidates = sections_by_type.get("verse", [])
                elif part == "buildup":
                    candidates = sections_by_type.get("verse", []) or sections_by_type.get("buildup", [])
                elif part in ["intro", "outro"]:
                    # Use quietest/first/last sections for intro/outro
                    all_sections = [s for ss in sections_by_type.values() for s in ss]
                    if all_sections:
                        if part == "intro":
                            candidates = sorted(all_sections, key=lambda s: s.start_time)[:2]
                        else:
                            candidates = sorted(all_sections, key=lambda s: s.energy)[:2]
            
            if not candidates:
                # Use any available section
                all_sections = [s for ss in sections_by_type.values() for s in ss]
                if all_sections:
                    candidates = all_sections
            
            if candidates:
                # Rotate through available sections of this type
                idx = used_sections.get(part, 0) % len(candidates)
                used_sections[part] = idx + 1
                
                selected = candidates[idx]
                
                # Ensure minimum section duration (use full section or extend)
                section_duration = max(selected.duration, 15.0)  # At least 15 seconds
                section_duration = min(section_duration, selected.duration)  # But not more than available
                
                new_start = current_time
                new_end = current_time + section_duration
                
                arrangement_sections.append((selected, new_start, new_end))
                current_time = new_end
                current_duration = new_end
        
        return Arrangement(
            sections=arrangement_sections,
            total_duration=current_time,
            structure=[s[0].label for s in arrangement_sections]
        )
    
    def _create_content_based_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        target_duration: Optional[float] = None
    ) -> Arrangement:
        """
        Create arrangement based on actual content, selecting the best sections.
        
        This mode picks the best-sounding sections and arranges them into
        a coherent flow based on energy curves.
        """
        if not sections:
            return Arrangement(sections=[], total_duration=0, structure=[])
        
        # Calculate target duration
        total_available = sum(s.duration for s in sections)
        if target_duration is None:
            target_duration = min(240, max(120, total_available * 0.5))
        
        # Score each section
        scored_sections = []
        for section in sections:
            score = self._score_section_quality(section)
            scored_sections.append((section, score))
        
        # Sort by score (best first)
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Build arrangement with energy curve: low -> high -> low
        # Divide into intro, rising, peak, falling, outro phases
        arrangement_sections = []
        current_time = 0.0
        
        # Categorize by energy
        low_energy = [s for s, _ in scored_sections if s.energy < 0.08]
        mid_energy = [s for s, _ in scored_sections if 0.08 <= s.energy < 0.15]
        high_energy = [s for s, _ in scored_sections if s.energy >= 0.15]
        
        # If categories are empty, distribute evenly
        if not low_energy:
            low_energy = scored_sections[:len(scored_sections)//3]
            low_energy = [s for s, _ in low_energy]
        if not high_energy:
            high_energy = scored_sections[:len(scored_sections)//3]
            high_energy = [s for s, _ in high_energy]
        if not mid_energy:
            mid_energy = scored_sections[len(scored_sections)//3:2*len(scored_sections)//3]
            mid_energy = [s for s, _ in mid_energy]
        
        # Build structure: intro -> build -> peak -> cool down -> outro
        phase_targets = [
            ("intro", low_energy, 0.12),      # 12% of song
            ("verse", mid_energy, 0.20),      # 20%
            ("chorus", high_energy, 0.25),    # 25%
            ("verse", mid_energy, 0.18),      # 18%
            ("chorus", high_energy, 0.15),    # 15%
            ("outro", low_energy, 0.10),      # 10%
        ]
        
        used_indices = set()
        
        for phase_name, candidates, duration_ratio in phase_targets:
            if current_time >= target_duration:
                break
            
            phase_duration = target_duration * duration_ratio
            
            # Find unused section from candidates
            selected = None
            for candidate in candidates:
                candidate_idx = sections.index(candidate) if candidate in sections else -1
                if candidate_idx not in used_indices:
                    selected = candidate
                    used_indices.add(candidate_idx)
                    break
            
            # Fallback to any unused section
            if selected is None:
                for s, _ in scored_sections:
                    s_idx = sections.index(s) if s in sections else -1
                    if s_idx not in used_indices:
                        selected = s
                        used_indices.add(s_idx)
                        break
            
            if selected:
                # Use up to phase_duration or section length
                use_duration = min(phase_duration, selected.duration)
                use_duration = max(use_duration, 10.0)  # At least 10 seconds
                use_duration = min(use_duration, selected.duration)
                
                arrangement_sections.append((
                    Section(
                        start_time=selected.start_time,
                        end_time=selected.start_time + use_duration,
                        label=phase_name,
                        energy=selected.energy,
                        features=selected.features
                    ),
                    current_time,
                    current_time + use_duration
                ))
                current_time += use_duration
        
        return Arrangement(
            sections=arrangement_sections,
            total_duration=current_time,
            structure=[s[0].label for s in arrangement_sections]
        )
    
    def _score_section_quality(self, section: Section) -> float:
        """Score a section's quality for selection."""
        score = 0.0
        
        # Prefer consistent energy (not too variable)
        variance = section.features.get("energy_variance", 0)
        score += 1.0 / (1.0 + variance * 10)
        
        # Prefer sections with good harmonic content
        harmonicity = section.features.get("harmonicity", 0.5)
        score += harmonicity * 0.5
        
        # Prefer longer sections (more content)
        score += min(section.duration / 30.0, 1.0) * 0.3
        
        # Slight preference for sections not at very start/end (avoid false starts)
        if section.start_time > 5.0:
            score += 0.2
        
        return score
    
    def _get_all_phrases(self, sections: List[Section]) -> List[Tuple[Phrase, Section]]:
        """Get all phrases from all sections with their parent section."""
        all_phrases = []
        for section in sections:
            if section.phrases:
                for phrase in section.phrases:
                    all_phrases.append((phrase, section))
            else:
                # Section has no phrases, treat the whole section as one phrase
                pseudo_phrase = Phrase(
                    start_time=section.start_time,
                    end_time=section.end_time,
                    bar_start=0,
                    num_bars=4,
                    energy=section.energy,
                    is_downbeat_aligned=False
                )
                all_phrases.append((pseudo_phrase, section))
        return all_phrases
    
    def _compute_phrase_chroma(self, audio_data: np.ndarray, phrase: Phrase) -> np.ndarray:
        """Compute chroma vector for a phrase (for harmonic similarity)."""
        start_sample = int(phrase.start_time * self.sample_rate)
        end_sample = int(phrase.end_time * self.sample_rate)
        phrase_audio = audio_data[start_sample:end_sample]
        
        if len(phrase_audio) < 2048:
            return np.zeros(12)
        
        chroma = librosa.feature.chroma_cqt(y=phrase_audio, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        norm = np.linalg.norm(chroma_avg)
        return chroma_avg / norm if norm > 0 else chroma_avg
    
    def _compute_phrase_timbre(self, audio_data: np.ndarray, phrase: Phrase) -> np.ndarray:
        """Compute MFCC-based timbre vector for a phrase."""
        start_sample = int(phrase.start_time * self.sample_rate)
        end_sample = int(phrase.end_time * self.sample_rate)
        phrase_audio = audio_data[start_sample:end_sample]
        
        if len(phrase_audio) < 2048:
            return np.zeros(13)
        
        mfcc = librosa.feature.mfcc(y=phrase_audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_avg = np.mean(mfcc, axis=1)
        norm = np.linalg.norm(mfcc_avg)
        return mfcc_avg / norm if norm > 0 else mfcc_avg
    
    def _precompute_phrase_features(
        self, 
        audio_data: np.ndarray, 
        phrases: List[Tuple[Phrase, Section]]
    ) -> Dict:
        """Precompute features for all phrases for efficient similarity comparison."""
        features = {}
        for phrase, section in phrases:
            phrase_id = (phrase.start_time, phrase.end_time)
            features[phrase_id] = {
                "chroma": self._compute_phrase_chroma(audio_data, phrase),
                "timbre": self._compute_phrase_timbre(audio_data, phrase),
                "energy": phrase.energy,
                "section_id": id(section),
                "original_time": phrase.start_time,
            }
        return features
    
    def _score_phrase_similarity(
        self,
        features: Dict,
        phrase1: Phrase,
        phrase2: Phrase
    ) -> float:
        """Score how similar two phrases are (for grouping related material)."""
        id1 = (phrase1.start_time, phrase1.end_time)
        id2 = (phrase2.start_time, phrase2.end_time)
        
        if id1 not in features or id2 not in features:
            return 0.0
        
        f1 = features[id1]
        f2 = features[id2]
        
        # Harmonic similarity (chroma)
        chroma_sim = np.dot(f1["chroma"], f2["chroma"])
        
        # Timbral similarity (MFCC)
        timbre_sim = np.dot(f1["timbre"], f2["timbre"])
        
        # Energy similarity
        energy_diff = abs(f1["energy"] - f2["energy"])
        max_energy = max(f1["energy"], f2["energy"], 0.01)
        energy_sim = 1.0 - min(energy_diff / max_energy, 1.0)
        
        # Combined similarity
        return chroma_sim * 0.4 + timbre_sim * 0.4 + energy_sim * 0.2
    
    def _score_phrase_transition(
        self,
        audio_data: np.ndarray,
        from_phrase: Phrase,
        to_phrase: Phrase,
        features: Dict = None,
        locality_weight: float = 0.3,
        style_profile: 'StyleProfile' = None,
        from_section: Section = None,
        to_section: Section = None
    ) -> float:
        """
        Score how well two phrases flow together.
        Optimized for musical coherence with locality bias.
        
        Args:
            audio_data: The audio data
            from_phrase: Source phrase
            to_phrase: Target phrase
            features: Precomputed phrase features
            locality_weight: Weight for locality bias (default 0.3)
            style_profile: Optional learned style profile with transition patterns
            from_section: Optional section that from_phrase belongs to
            to_section: Optional section that to_phrase belongs to
        """
        score = 0.0
        
        from_id = (from_phrase.start_time, from_phrase.end_time)
        to_id = (to_phrase.start_time, to_phrase.end_time)
        
        # Use precomputed features if available
        if features and from_id in features and to_id in features:
            f_from = features[from_id]
            f_to = features[to_id]
            
            # 1. LOCALITY BIAS (30%) - prefer phrases close in original time
            time_distance = abs(f_from["original_time"] - f_to["original_time"])
            max_distance = 120.0  # 2 minutes
            locality_score = 1.0 - min(time_distance / max_distance, 1.0)
            score += locality_score * locality_weight
            
            # 2. SAME SECTION BONUS (15%) - phrases from same section are coherent
            if f_from["section_id"] == f_to["section_id"]:
                score += 0.15
            
            # Compute actual transition characteristics
            chroma_sim = np.dot(f_from["chroma"], f_to["chroma"])
            timbre_sim = np.dot(f_from["timbre"], f_to["timbre"])
            energy_ratio = f_to["energy"] / max(f_from["energy"], 0.01)
            
            # If we have a learned style profile, score against learned patterns
            if style_profile is not None and from_section and to_section:
                style_bonus = self._score_against_style(
                    style_profile,
                    from_section.label,
                    to_section.label,
                    energy_ratio,
                    chroma_sim,
                    timbre_sim
                )
                # Style conformance bonus - 30% weight (increased from 20%)
                score += style_bonus * 0.30
                
                # Reduce other weights to make room for increased style influence
                chroma_weight = 0.12
                timbre_weight = 0.08
                energy_weight = 0.05
            else:
                chroma_weight = 0.20
                timbre_weight = 0.15
                energy_weight = 0.10
            
            # 3. HARMONIC CONTINUITY - similar key/chords
            score += max(0, chroma_sim) * chroma_weight
            
            # 4. TIMBRAL CONTINUITY - similar instrument/tone
            score += max(0, timbre_sim) * timbre_weight
            
            # 5. ENERGY CONTINUITY - smooth energy transitions
            energy_diff = abs(f_from["energy"] - f_to["energy"])
            max_energy = max(f_from["energy"], f_to["energy"], 0.01)
            energy_score = 1.0 - min(energy_diff / max_energy, 1.0)
            score += energy_score * energy_weight
        else:
            # Fallback to basic scoring
            score += 0.3  # Base score
        
        # 6. DOWNBEAT ALIGNMENT (5%) - landing on downbeat is cleaner
        if to_phrase.is_downbeat_aligned:
            score += 0.05
        
        # 7. NOT SAME PHRASE (5%) - avoid exact repetition
        if from_phrase.start_time != to_phrase.start_time:
            score += 0.05
        
        return score
    
    def _score_against_style(
        self,
        style_profile: 'StyleProfile',
        from_label: str,
        to_label: str,
        energy_ratio: float,
        chroma_sim: float,
        timbre_sim: float
    ) -> float:
        """
        Score a transition against learned style patterns.
        
        Returns a score 0-1 indicating how well this transition matches 
        the learned patterns from reference songs.
        """
        score = 0.0
        
        # Look for matching transition pattern
        transition_key = f"{from_label}_to_{to_label}"
        
        if hasattr(style_profile, 'transition_patterns') and style_profile.transition_patterns:
            if transition_key in style_profile.transition_patterns:
                pattern = style_profile.transition_patterns[transition_key]
                
                # Score energy ratio match (how close to learned pattern)
                learned_ratio = pattern.get("energy_ratio_mean", 1.0)
                learned_std = pattern.get("energy_ratio_std", 0.3)
                energy_diff = abs(energy_ratio - learned_ratio)
                # Score higher if within 1 std deviation
                if learned_std > 0:
                    energy_match = max(0, 1.0 - (energy_diff / (learned_std * 2)))
                else:
                    energy_match = 1.0 if energy_diff < 0.2 else 0.5
                score += energy_match * 0.4
                
                # Score chroma similarity match
                learned_chroma = pattern.get("chroma_similarity", 0.5)
                chroma_diff = abs(chroma_sim - learned_chroma)
                chroma_match = max(0, 1.0 - chroma_diff)
                score += chroma_match * 0.3
                
                # Score timbre similarity match
                learned_timbre = pattern.get("timbre_similarity", 0.5)
                timbre_diff = abs(timbre_sim - learned_timbre)
                timbre_match = max(0, 1.0 - timbre_diff)
                score += timbre_match * 0.3
                
            else:
                # No specific pattern learned, use general style metrics
                if hasattr(style_profile, 'typical_chroma_continuity'):
                    # Reward transitions that match typical continuity level
                    target_chroma = style_profile.typical_chroma_continuity
                    chroma_match = max(0, 1.0 - abs(chroma_sim - target_chroma))
                    score += chroma_match * 0.5
                
                if hasattr(style_profile, 'typical_timbre_continuity'):
                    target_timbre = style_profile.typical_timbre_continuity
                    timbre_match = max(0, 1.0 - abs(timbre_sim - target_timbre))
                    score += timbre_match * 0.5
        else:
            # No transition patterns available, return neutral score
            score = 0.5
        
        return score
    
    def _find_similar_phrases(
        self,
        target_phrase: Phrase,
        candidates: List[Tuple[Phrase, Section]],
        features: Dict,
        top_n: int = 5
    ) -> List[Tuple[Phrase, Section, float]]:
        """Find the most similar phrases to a target phrase."""
        similarities = []
        target_id = (target_phrase.start_time, target_phrase.end_time)
        
        for phrase, section in candidates:
            phrase_id = (phrase.start_time, phrase.end_time)
            if phrase_id == target_id:
                continue
            
            sim = self._score_phrase_similarity(features, target_phrase, phrase)
            similarities.append((phrase, section, sim))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_n]
    
    def _group_phrases_by_proximity(
        self,
        phrases: List[Tuple[Phrase, Section]],
        max_gap_seconds: float = 30.0
    ) -> List[List[Tuple[Phrase, Section]]]:
        """
        Group phrases by temporal proximity in the original recording.
        Phrases close together are likely from the same musical idea/section.
        """
        if not phrases:
            return []
        
        # Sort by start time
        sorted_phrases = sorted(phrases, key=lambda x: x[0].start_time)
        
        groups = []
        current_group = [sorted_phrases[0]]
        
        for i in range(1, len(sorted_phrases)):
            phrase, section = sorted_phrases[i]
            prev_phrase, _ = sorted_phrases[i - 1]
            
            # If this phrase is close to the previous one, same group
            gap = phrase.start_time - prev_phrase.end_time
            if gap < max_gap_seconds:
                current_group.append((phrase, section))
            else:
                # Start new group
                groups.append(current_group)
                current_group = [(phrase, section)]
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        # Sort groups by average energy (for structure building)
        groups.sort(key=lambda g: np.mean([p.energy for p, _ in g]))
        
        return groups
    
    def _group_phrases_by_similarity(
        self,
        phrases: List[Tuple[Phrase, Section]],
        features: Dict,
        similarity_threshold: float = 0.6
    ) -> List[List[Tuple[Phrase, Section]]]:
        """Group phrases into clusters of similar material."""
        if not phrases:
            return []
        
        groups = []
        used = set()
        
        for phrase, section in phrases:
            phrase_id = (phrase.start_time, phrase.end_time)
            if phrase_id in used:
                continue
            
            # Start a new group
            group = [(phrase, section)]
            used.add(phrase_id)
            
            # Find similar phrases
            for other_phrase, other_section in phrases:
                other_id = (other_phrase.start_time, other_phrase.end_time)
                if other_id in used:
                    continue
                
                sim = self._score_phrase_similarity(features, phrase, other_phrase)
                if sim >= similarity_threshold:
                    group.append((other_phrase, other_section))
                    used.add(other_id)
            
            groups.append(group)
        
        # Sort groups by average energy (for structure building)
        groups.sort(key=lambda g: np.mean([p.energy for p, _ in g]))
        
        return groups
    
    def create_simple_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        target_duration: Optional[float] = None,
        style_profile: StyleProfile = None,
        min_sections: Optional[int] = None,
        max_sections: Optional[int] = None,
        allow_rearrange: bool = False
    ) -> Arrangement:
        """
        Create a smart arrangement by scoring sections and keeping the best material.
        
        This approach:
        1. Scores every section for quality (tightness, energy, interest)
        2. Preserves original section order (natural flow of the jam) OR rearranges for better flow
        3. Removes truly weak sections (significantly below average)
        4. Only trims further if there's a big gap between good and mediocre content
        
        Args:
            audio_data: Raw audio data
            sections: Detected sections
            target_duration: Target output duration in seconds (flexible)
            style_profile: Optional learned style (not used currently)
            min_sections: Minimum number of sections to include
            max_sections: Maximum number of sections to include
            allow_rearrange: If True, reorder sections for better musical flow
        """
        if not sections:
            return Arrangement(sections=[], total_duration=0, structure=[])
        
        total_duration_available = sum(s.duration for s in sections)
        
        # If no target specified, keep all good material (just trim the fat)
        if target_duration is None:
            target_duration = total_duration_available  # No hard limit
            trim_to_target = False
            print(f"   Quality-based trimming (keeping all good material)...")
        else:
            trim_to_target = True
            print(f"   Quality-based trimming (preserving natural flow)...")
        
        print(f"   Target: ~{target_duration:.0f}s from {total_duration_available:.0f}s available")
        
        # Step 1: Score every section for quality
        scored_sections = []
        for section in sections:
            quality = self._score_section_deep(audio_data, section)
            scored_sections.append((section, quality))
        
        # Show quality distribution
        qualities = [q for _, q in scored_sections]
        avg_quality = np.mean(qualities)
        quality_range = max(qualities) - min(qualities)
        print(f"   Quality: min={min(qualities):.1f}, max={max(qualities):.1f}, avg={avg_quality:.1f}, range={quality_range:.1f}")
        
        # Step 2: Decide how aggressive to trim based on quality distribution
        # If quality is fairly uniform, don't trim much - everything is similarly good
        # Only trim if there's a clear gap between good and weak content
        
        # Check if there's a clear quality gap
        sorted_qualities = sorted(qualities)
        has_weak_outliers = False
        if len(sorted_qualities) >= 3:
            # Check if lowest sections are significantly worse (>15 points below median)
            median_q = np.median(sorted_qualities)
            if sorted_qualities[0] < median_q - 15:
                has_weak_outliers = True
                quality_threshold = median_q - 15
                print(f"   Found weak outliers below {quality_threshold:.1f}")
        
        # Step 3: Only remove truly weak sections
        good_sections = []
        for section, quality in scored_sections:
            if has_weak_outliers and quality < quality_threshold:
                print(f"   ✗ Removing weak section {section.start_time:.0f}s-{section.end_time:.0f}s (quality={quality:.1f})")
            else:
                good_sections.append((section, quality))
        
        print(f"   Keeping {len(good_sections)} of {len(scored_sections)} sections")
        
        # Step 4: Duration trimming - only if we have a target and significantly over it
        current_duration = sum(s.duration for s, _ in good_sections)
        
        if trim_to_target:
            overshoot_factor = 1.3  # Allow 30% overshoot before trimming
            min_keep = 3  # Keep at least 3 sections
            
            while current_duration > target_duration * overshoot_factor and len(good_sections) > min_keep:
                # Find lowest quality section that's not first or last
                min_idx = None
                min_quality = float('inf')
                for i in range(1, len(good_sections) - 1):
                    if good_sections[i][1] < min_quality:
                        min_quality = good_sections[i][1]
                        min_idx = i
                
                if min_idx is None:
                    break
                    
                removed = good_sections.pop(min_idx)
                current_duration -= removed[0].duration
                print(f"   ✗ Trimming {removed[0].start_time:.0f}s-{removed[0].end_time:.0f}s to reduce length (quality={removed[1]:.1f})")
        
        # Step 5: Optionally rearrange sections for better musical flow
        if allow_rearrange and len(good_sections) >= 3:
            good_sections = self._rearrange_for_flow(audio_data, good_sections)
            print(f"   Rearranged sections for better flow")
        
        # Step 6: Build final arrangement
        arrangement_sections = []
        arr_time = 0.0
        
        # Get energy distribution for labeling
        energies = [s.energy for s, _ in good_sections]
        energy_med = np.median(energies) if energies else 0.5
        
        for i, (section, quality) in enumerate(good_sections):
            # Assign labels based on position and energy
            if i == 0:
                label = "intro"
            elif i == len(good_sections) - 1:
                label = "outro"
            elif section.energy > energy_med * 1.2:
                label = "riff"
            elif section.energy < energy_med * 0.8:
                label = "breakdown"
            else:
                label = "verse"
            
            section_dur = section.duration
            
            new_section = Section(
                start_time=section.start_time,
                end_time=section.end_time,
                label=label,
                energy=section.energy,
                features=section.features
            )
            
            arrangement_sections.append((new_section, arr_time, arr_time + section_dur))
            arr_time += section_dur
        
        # Show what we kept
        print(f"   Final: {len(arrangement_sections)} sections, {arr_time:.0f}s")
        kept_times = [f"{s.start_time:.0f}-{s.end_time:.0f}s" for s, _, _ in arrangement_sections]
        print(f"   Keeping: {', '.join(kept_times)}")
        
        return Arrangement(
            sections=arrangement_sections,
            total_duration=arr_time,
            structure=[s[0].label for s in arrangement_sections]
        )
    
    def _rearrange_for_flow(
        self, 
        audio_data: np.ndarray, 
        scored_sections: List[Tuple[Section, float]]
    ) -> List[Tuple[Section, float]]:
        """
        Rearrange sections for better musical flow, but VERY conservatively.
        
        Since rearranged sections use hard cuts (crossfades between unrelated audio sounds bad),
        we can only swap sections that are:
        1. Originally adjacent or very close in the timeline
        2. Have similar energy levels
        3. Would clearly improve the arrangement
        
        Strategy:
        - Keep chronological order as the base
        - Only swap ADJACENT sections if it improves energy flow
        - Never move sections far from their original position
        """
        if len(scored_sections) < 3:
            return scored_sections
        
        # Analyze each section's energy
        sections_with_features = []
        for section, quality in scored_sections:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            if len(segment) < self.sample_rate:
                avg_energy = 0
            else:
                avg_energy = np.sqrt(np.mean(segment**2))
            
            sections_with_features.append({
                'section': section,
                'quality': quality,
                'avg_energy': avg_energy,
                'original_idx': len(sections_with_features)
            })
        
        # Start with original chronological order
        result = list(sections_with_features)
        
        # Only do ADJACENT swaps that improve energy flow
        # Goal: energy should generally build up, not jump around randomly
        made_swap = True
        max_iterations = len(result)  # Prevent infinite loops
        iterations = 0
        
        while made_swap and iterations < max_iterations:
            made_swap = False
            iterations += 1
            
            for i in range(len(result) - 1):
                curr = result[i]
                next_sec = result[i + 1]
                
                # Skip if this is first or last position (intro/outro should stay)
                if i == 0 or i == len(result) - 2:
                    continue
                
                # Check if swapping would improve energy flow
                # We want energy to generally increase toward the climax (position -2)
                curr_energy = curr['avg_energy']
                next_energy = next_sec['avg_energy']
                
                # If current has MORE energy than next, and we're before the climax,
                # consider swapping so energy builds up
                climax_pos = len(result) - 2
                if i < climax_pos and curr_energy > next_energy * 1.3:
                    # Swap would put lower energy first (building up)
                    # But only if it doesn't create too big a jump with neighbors
                    
                    prev_energy = result[i - 1]['avg_energy'] if i > 0 else curr_energy
                    after_energy = result[i + 2]['avg_energy'] if i + 2 < len(result) else next_energy
                    
                    # Check that swap doesn't create jarring transitions
                    # Current order: prev -> curr -> next -> after
                    # After swap:    prev -> next -> curr -> after
                    
                    curr_flow_ok = True
                    # Check prev -> next transition
                    if prev_energy > 0 and next_energy > 0:
                        ratio = next_energy / prev_energy
                        if ratio < 0.4 or ratio > 2.5:
                            curr_flow_ok = False
                    # Check curr -> after transition  
                    if curr_energy > 0 and after_energy > 0:
                        ratio = after_energy / curr_energy
                        if ratio < 0.4 or ratio > 2.5:
                            curr_flow_ok = False
                    
                    if curr_flow_ok:
                        # Do the swap
                        result[i], result[i + 1] = result[i + 1], result[i]
                        made_swap = True
                        print(f"   Swapped sections at positions {i} and {i+1} for better energy flow")
                        break  # Restart the loop after a swap
        
        if iterations == 1:
            print(f"   Keeping original order (no beneficial swaps found)")
        
        return [(s['section'], s['quality']) for s in result]
    
    def _score_section_deep(self, audio_data: np.ndarray, section: Section) -> float:
        """
        Deep quality scoring for a section - tries to find the "good parts".
        
        Scoring philosophy: A good section has:
        - Clear rhythmic pulse (locked in groove, not sloppy)
        - Musical structure (repeating motifs, not random noodling)
        - Peak moments / climaxes
        - Tension and release patterns
        - Harmonic interest (chord changes, not static)
        
        Returns score 0-100, aiming for more spread between good/bad sections.
        CRITICAL: Use continuous scoring, not just thresholds!
        """
        start_sample = int(section.start_time * self.sample_rate)
        end_sample = int(section.end_time * self.sample_rate)
        segment = audio_data[start_sample:end_sample]
        
        if len(segment) < self.sample_rate:
            return 0.0
        
        score = 0.0
        penalties = []
        bonuses = []
        metrics = {}  # Store raw metrics for debug
        
        # === 1. RHYTHMIC TIGHTNESS (0-20 points) ===
        # Use CONTINUOUS scoring based on variance
        try:
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=segment, sr=self.sample_rate, hop_length=hop_length)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate, hop_length=hop_length)
            
            if len(beats) > 4:
                beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=hop_length)
                beat_intervals = np.diff(beat_times)
                
                if len(beat_intervals) > 2:
                    interval_variance = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)
                    metrics['rhythm_var'] = interval_variance
                    # Continuous score: lower variance = higher score
                    # 0.05 variance = 20 pts, 0.4 variance = 0 pts
                    rhythm_score = max(0, min(20, 20 * (1 - interval_variance / 0.4)))
                    score += rhythm_score
                    if interval_variance < 0.12:
                        bonuses.append("tight")
                    elif interval_variance > 0.25:
                        penalties.append("sloppy")
                else:
                    score += 10
            else:
                score += 5
                penalties.append("weak rhythm")
        except:
            score += 10
        
        # === 2. MUSICAL STRUCTURE / REPETITION (0-15 points) ===
        # Continuous based on self-similarity
        try:
            hop_length = max(512, len(segment) // 200)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sample_rate, hop_length=hop_length)
            
            mid = chroma.shape[1] // 2
            if mid > 10:
                first_half = np.mean(chroma[:, :mid], axis=1)
                second_half = np.mean(chroma[:, mid:], axis=1)
                
                similarity = np.dot(first_half, second_half) / (
                    np.linalg.norm(first_half) * np.linalg.norm(second_half) + 1e-6
                )
                metrics['self_sim'] = similarity
                
                # Continuous: 0.6 sim = 0 pts, 0.95 sim = 15 pts
                struct_score = max(0, min(15, 15 * (similarity - 0.6) / 0.35))
                score += struct_score
                if similarity > 0.92:
                    bonuses.append("motif")
            else:
                score += 8
        except:
            score += 8
        
        # === 3. PEAK MOMENTS / CLIMAX DETECTION (0-20 points) ===
        hop = self.sample_rate // 4
        frame_energies = []
        for i in range(0, len(segment) - hop, hop):
            frame = segment[i:i+hop]
            frame_energies.append(np.sqrt(np.mean(frame**2)))
        
        if len(frame_energies) > 8:
            frame_energies = np.array(frame_energies)
            energy_mean = np.mean(frame_energies)
            energy_max = np.max(frame_energies)
            energy_std = np.std(frame_energies)
            
            # Peak ratio: how much does the loudest moment stand out?
            peak_ratio = energy_max / (energy_mean + 1e-6)
            metrics['peak_ratio'] = peak_ratio
            
            # Energy variance (normalized) - how much does energy vary?
            energy_cv = energy_std / (energy_mean + 1e-6)
            metrics['energy_cv'] = energy_cv
            
            # Count significant peaks
            peaks = []
            for i in range(1, len(frame_energies) - 1):
                if (frame_energies[i] > frame_energies[i-1] and 
                    frame_energies[i] > frame_energies[i+1] and
                    frame_energies[i] > energy_mean * 1.3):
                    peaks.append(i)
            metrics['num_peaks'] = len(peaks)
            
            # Continuous scoring: peak_ratio 1.0 = 0 pts, 2.0+ = 15 pts
            # Plus bonus for having 1-4 distinct peaks
            peak_score = max(0, min(15, 15 * (peak_ratio - 1.0)))
            if 1 <= len(peaks) <= 4:
                peak_score += 5
                bonuses.append("climax")
            elif len(peaks) > 4:
                peak_score += 2  # Too many peaks = chaotic
            score += peak_score
        else:
            score += 10
        
        # === 4. TENSION/RELEASE PATTERN (0-15 points) ===
        tension_score = 0
        if len(frame_energies) > 8:
            q_len = len(frame_energies) // 4
            if q_len > 1:
                q1 = np.mean(frame_energies[:q_len])
                q2 = np.mean(frame_energies[q_len:2*q_len])
                q3 = np.mean(frame_energies[2*q_len:3*q_len])
                q4 = np.mean(frame_energies[3*q_len:])
                
                # Compute arc magnitude (how much does energy move?)
                arc_magnitude = max(abs(q2-q1), abs(q3-q2), abs(q4-q3)) / (energy_mean + 1e-6)
                metrics['arc_mag'] = arc_magnitude
                
                # Build-up: q1 < q2 < q3
                is_buildup = q1 < q2 * 0.92 and q2 < q3 * 0.92
                # Release: high to low
                is_release = q1 > q2 * 1.08 and q2 > q3 * 1.08
                # Arc: build then release
                is_arc = q2 > q1 * 1.1 and q3 > q4 * 1.1
                # Middle climax
                middle_peak = max(q2, q3) > max(q1, q4) * 1.25
                
                # Continuous scoring based on arc magnitude
                tension_score = min(10, arc_magnitude * 30)
                
                # Bonus for clear pattern
                if is_arc or middle_peak:
                    tension_score += 5
                    bonuses.append("arc")
                elif is_buildup:
                    tension_score += 4
                    bonuses.append("build")
                elif is_release:
                    tension_score += 3
                    bonuses.append("release")
                
                score += min(15, tension_score)
            else:
                score += 5
        else:
            score += 5
        
        # === 5. HARMONIC INTEREST (0-15 points) ===
        try:
            hop_length = max(512, len(segment) // 100)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sample_rate, hop_length=hop_length)
            
            # Harmonic movement (chord changes)
            chroma_diff = np.diff(chroma, axis=1)
            harmonic_movement = np.mean(np.abs(chroma_diff))
            metrics['harm_move'] = harmonic_movement
            
            # Chromatic variety
            chroma_variance = np.mean(np.var(chroma, axis=1))
            metrics['harm_var'] = chroma_variance
            
            # Combined and scaled
            # Good values: movement > 0.02, variance > 0.02
            harm_score = min(8, harmonic_movement * 200) + min(7, chroma_variance * 150)
            score += harm_score
            if harmonic_movement > 0.025 and chroma_variance > 0.025:
                bonuses.append("rich harmony")
            elif harmonic_movement > 0.015:
                bonuses.append("harmony")
        except:
            score += 8
        
        # === 6. SPECTRAL INTEREST (0-10 points) ===
        try:
            spec_cent = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0]
            spec_contrast = librosa.feature.spectral_contrast(y=segment, sr=self.sample_rate)
            
            centroid_cv = np.std(spec_cent) / (np.mean(spec_cent) + 1e-6)
            contrast_mean = np.mean(spec_contrast)
            metrics['timbre_cv'] = centroid_cv
            metrics['contrast'] = contrast_mean
            
            # Continuous: centroid_cv 0.1-0.4, contrast 10-25
            timbre_score = min(5, centroid_cv * 15) + min(5, (contrast_mean - 10) / 3)
            score += max(0, timbre_score)
            if centroid_cv > 0.35 and contrast_mean > 22:
                bonuses.append("timbre")
        except:
            score += 5
        
        # === 7. ENERGY LEVEL (0-5 points) ===
        rms = np.sqrt(np.mean(segment**2))
        metrics['rms'] = rms
        # Continuous: 0.01 = 0 pts, 0.08 = 5 pts
        energy_score = max(0, min(5, 5 * (rms - 0.01) / 0.07))
        score += energy_score
        if rms < 0.015:
            penalties.append("quiet")
        
        # === 8. TUNING/NOODLING DETECTION (penalty up to -30 points) ===
        # Detect sections that sound like guitar tuning or random noodling
        # NOTE: Can't rely on timing being sloppy - some players tune in rhythm!
        tuning_penalty = 0
        try:
            # Tuning characteristics:
            # 1. Single notes (not chords) - narrow harmonic content
            # 2. Same pitches repeated (checking tuning)
            # 3. Limited pitch range (not melodic playing)
            # 4. Sparse polyphony (one note at a time)
            
            hop_length = 512
            
            # Check harmonic complexity - tuning = simple, single notes
            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sample_rate, hop_length=hop_length)
            
            # Count how many pitch classes are active at once (on average)
            # Tuning = mostly 1-2 notes, playing = 3+ (chords, harmonics)
            chroma_binary = (chroma > 0.3).astype(float)  # Threshold for "active" pitch
            avg_active_pitches = np.mean(np.sum(chroma_binary, axis=0))
            metrics['active_pitches'] = avg_active_pitches
            
            if avg_active_pitches < 2.0:
                tuning_penalty += 15
                penalties.append("single notes")
            elif avg_active_pitches < 2.5:
                tuning_penalty += 8
            
            # Check pitch variety - tuning repeats same notes, playing has variety
            # Use the dominant pitch in each frame
            dominant_pitches = np.argmax(chroma, axis=0)
            unique_pitches = len(np.unique(dominant_pitches))
            pitch_variety = unique_pitches / 12.0  # Normalize by total pitch classes
            metrics['pitch_variety'] = pitch_variety
            
            if pitch_variety < 0.4:  # Less than 5 different notes used
                tuning_penalty += 12
                penalties.append("repetitive pitch")
            elif pitch_variety < 0.5:
                tuning_penalty += 6
            
            # Check for "tuning pattern" - same note repeated with gaps
            # Look at pitch stability over time
            pitch_changes = np.sum(np.abs(np.diff(dominant_pitches)) > 0)
            pitch_change_rate = pitch_changes / (len(dominant_pitches) + 1)
            metrics['pitch_change_rate'] = pitch_change_rate
            
            # Very low change rate = holding/repeating same note (tuning)
            # But riffs can also be repetitive - only penalize VERY static
            if pitch_change_rate < 0.05:  # Almost no pitch movement at all
                tuning_penalty += 12
                penalties.append("static pitch")
            elif pitch_change_rate < 0.08:
                tuning_penalty += 6
            
            # Check spectral bandwidth - tuning is narrow, playing is full
            spec_bw = librosa.feature.spectral_bandwidth(y=segment, sr=self.sample_rate)[0]
            avg_bandwidth = np.mean(spec_bw)
            metrics['bandwidth'] = avg_bandwidth
            
            # Low bandwidth = thin sound (single notes)
            if avg_bandwidth < 1500:
                tuning_penalty += 8
                penalties.append("thin sound")
            elif avg_bandwidth < 2000:
                tuning_penalty += 4
            
        except Exception as e:
            pass
        
        # Apply tuning penalty (cap at -35)
        tuning_penalty = min(35, tuning_penalty)
        score -= tuning_penalty
        if tuning_penalty > 20:
            penalties.append("tuning?")
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        # Debug output for tuning - show key metrics
        metric_str = " ".join([f"{k}={v:.2f}" for k,v in list(metrics.items())[:6]])
        print(f"      {section.start_time:.0f}s-{section.end_time:.0f}s: {score:.1f} pts | {metric_str} | +{bonuses} -{penalties}")
        
        return score
    
    def _find_repeated_loops(
        self, 
        audio_data: np.ndarray, 
        sections: List[Section],
        similarity_threshold: float = 0.80
    ) -> List[List[Section]]:
        """
        Find sections that are actually repeated (same riff played multiple times).
        Returns groups of similar sections.
        """
        if not sections:
            return []
        
        # Compute fingerprint for each section
        fingerprints = []
        for section in sections:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            if len(segment) < self.sample_rate:  # Skip very short sections
                fingerprints.append(None)
                continue
            
            # Use chroma (harmonic content) + MFCC (timbre) for matching
            hop_length = max(512, len(segment) // 100)
            
            try:
                chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sample_rate, hop_length=hop_length)
                mfcc = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
                
                # Create compact fingerprint
                chroma_mean = np.mean(chroma, axis=1)
                mfcc_mean = np.mean(mfcc, axis=1)
                fingerprint = np.concatenate([chroma_mean, mfcc_mean])
                fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-6)
                fingerprints.append(fingerprint)
            except:
                fingerprints.append(None)
        
        # Group similar sections
        groups: List[List[Section]] = []
        used = set()
        
        for i, section in enumerate(sections):
            if i in used or fingerprints[i] is None:
                continue
            
            group = [section]
            used.add(i)
            
            for j, other in enumerate(sections):
                if j in used or j <= i or fingerprints[j] is None:
                    continue
                
                # Compute similarity
                sim = np.dot(fingerprints[i], fingerprints[j])
                
                if sim > similarity_threshold:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        # Sort by group size (most repeated first)
        groups.sort(key=lambda g: len(g), reverse=True)
        
        return groups
    
    def _pick_best_section(
        self, 
        audio_data: np.ndarray, 
        candidates: List[Section]
    ) -> Section:
        """Pick the best quality section from candidates."""
        if not candidates:
            raise ValueError("No candidates provided")
        
        if len(candidates) == 1:
            return candidates[0]
        
        best_score = -1
        best_section = candidates[0]
        
        for section in candidates:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            if len(segment) < self.sample_rate * 0.5:
                continue
            
            # Score based on: consistent loudness, no clipping, decent length
            rms = librosa.feature.rms(y=segment)[0]
            consistency = 1.0 / (np.std(rms) + 0.01)
            
            # Penalize clipping
            clipping = np.sum(np.abs(segment) > 0.99) / len(segment)
            
            # Prefer longer sections (more material)
            length_bonus = min(section.duration / 60, 1.0)  # Cap at 60s
            
            score = consistency * (1 - clipping * 10) * (0.5 + length_bonus * 0.5)
            
            if score > best_score:
                best_score = score
                best_section = section
        
        return best_section

    def create_doom_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        target_duration: Optional[float] = None,
        style_profile: StyleProfile = None
    ) -> Arrangement:
        """
        Create a stoner/doom arrangement using phrase-level selection.
        Optimized for musical coherence with:
        - Locality bias (prefer nearby phrases)
        - Harmonic/timbral similarity matching
        - Section continuity
        - Smart phrase grouping
        - Learned transition patterns (when style_profile provided)
        
        Doom structure: long, heavy, hypnotic repetition with slow dynamics.
        Starts quiet/ambient, builds into heavy riffs.
        
        Args:
            audio_data: The full audio data
            sections: Detected sections with phrases
            target_duration: Target duration for the arrangement
            style_profile: Optional learned style profile for transition scoring
        """
        if not sections:
            return Arrangement(sections=[], total_duration=0, structure=[])
        
        # Get all phrases for phrase-level arrangement
        all_phrases = self._get_all_phrases(sections)
        
        if not all_phrases:
            return Arrangement(sections=[], total_duration=0, structure=[])
        
        # Precompute features for all phrases (harmonic, timbral, energy)
        print("   Computing phrase features for coherence analysis...")
        phrase_features = self._precompute_phrase_features(audio_data, all_phrases)
        
        # Use sections as natural musical idea groups (they were detected by spectral change)
        # Each section represents a distinct musical idea
        phrase_groups = []
        for section in sections:
            if section.phrases:
                group = [(p, section) for p in section.phrases]
                phrase_groups.append(group)
        
        # Sort groups by average energy (quietest first for intro material)
        phrase_groups.sort(key=lambda g: np.mean([p.energy for p, _ in g]))
        
        print(f"   Found {len(phrase_groups)} distinct musical ideas")
        
        # Doom songs are longer - aim for 4-6 minutes
        total_available = sum(p.duration for p, _ in all_phrases)
        if target_duration is None:
            target_duration = min(360, max(180, total_available * 0.5))
        
        # Categorize phrase groups by average energy
        quiet_groups = []
        mid_groups = []
        heavy_groups = []
        
        all_group_energies = [np.mean([p.energy for p, _ in g]) for g in phrase_groups]
        if all_group_energies:
            median_group_energy = np.median(all_group_energies)
            q25 = np.percentile(all_group_energies, 25)
            q75 = np.percentile(all_group_energies, 75)
        else:
            median_group_energy, q25, q75 = 0.5, 0.25, 0.75
        
        for group in phrase_groups:
            avg_energy = np.mean([p.energy for p, _ in group])
            if avg_energy < q25:
                quiet_groups.append(group)
            elif avg_energy < q75:
                mid_groups.append(group)
            else:
                heavy_groups.append(group)
        
        # Ensure we have material for each category
        if not quiet_groups and phrase_groups:
            quiet_groups = [phrase_groups[0]]
        if not heavy_groups and phrase_groups:
            heavy_groups = [phrase_groups[-1]]
        if not mid_groups:
            mid_groups = heavy_groups[:1] if heavy_groups else quiet_groups[:1]
        
        # Doom structure: pick 2-3 main musical ideas and develop them
        # Select primary heavy riff group (the main "song")
        primary_heavy = heavy_groups[0] if heavy_groups else mid_groups[0]
        # Select secondary heavy riff (variation)
        secondary_heavy = heavy_groups[1] if len(heavy_groups) > 1 else primary_heavy
        # Select quiet material for intro/breakdown
        quiet_material = quiet_groups[0] if quiet_groups else mid_groups[0]
        # Select build material
        build_material = mid_groups[0] if mid_groups else primary_heavy
        
        # Calculate phrase counts based on learned section timing (if available)
        def get_phrase_count(section_label: str, default: int, phrase_duration: float = 26.0) -> int:
            """Calculate number of phrases based on learned section duration."""
            if style_profile and hasattr(style_profile, 'section_timing') and style_profile.section_timing:
                timing = style_profile.section_timing.get(section_label)
                if timing:
                    target_dur = timing.get("mean", default * phrase_duration)
                    # Calculate phrases needed for target duration
                    return max(1, round(target_dur / phrase_duration))
            return default
        
        # Estimate average phrase duration from available material
        avg_phrase_dur = 26.0  # Default ~26s for 8 bars at 73 BPM
        if all_phrases:
            avg_phrase_dur = np.mean([p.duration for p, _ in all_phrases])
        
        # Structure: use coherent groups with learned timing
        structure_plan = [
            ("intro", quiet_material, get_phrase_count("intro", 2, avg_phrase_dur)),
            ("build", build_material, get_phrase_count("buildup", 2, avg_phrase_dur)),
            ("riff", primary_heavy, get_phrase_count("chorus", 4, avg_phrase_dur)),
            ("riff", secondary_heavy, get_phrase_count("drop", 3, avg_phrase_dur)),
            ("breakdown", quiet_material, get_phrase_count("breakdown", 2, avg_phrase_dur)),
            ("riff", primary_heavy, get_phrase_count("chorus", 3, avg_phrase_dur)),
            ("outro", quiet_material, get_phrase_count("outro", 2, avg_phrase_dur)),
        ]
        
        if style_profile and hasattr(style_profile, 'section_timing'):
            print(f"   Using learned section timing from style")
        
        arrangement_sections = []
        current_time = 0.0
        previous_phrase = None
        previous_section = None  # Track for style-aware transition scoring
        used_phrases = set()
        
        for phase_name, phrase_group, num_phrases in structure_plan:
            if current_time >= target_duration:
                break
            
            phrases_added = 0
            
            # Sort phrases in group by their original time (preserve locality)
            sorted_group = sorted(phrase_group, key=lambda x: x[0].start_time)
            
            while phrases_added < num_phrases and current_time < target_duration:
                selected_phrase = None
                selected_section = None
                
                if previous_phrase:
                    # Score candidates from this group by transition quality
                    scored = []
                    for phrase, section in sorted_group:
                        phrase_id = (phrase.start_time, phrase.end_time)
                        # Prefer unused, but allow reuse for doom repetition
                        reuse_penalty = 0.1 if phrase_id in used_phrases else 0.0
                        
                        flow_score = self._score_phrase_transition(
                            audio_data, previous_phrase, phrase,
                            features=phrase_features,
                            locality_weight=0.35,  # Strong locality preference
                            style_profile=style_profile,
                            from_section=previous_section,
                            to_section=section
                        ) - reuse_penalty
                        
                        scored.append((phrase, section, flow_score))
                    
                    if scored:
                        scored.sort(key=lambda x: x[2], reverse=True)
                        selected_phrase, selected_section, _ = scored[0]
                else:
                    # First phrase - prefer earliest in original (natural start)
                    for phrase, section in sorted_group:
                        phrase_id = (phrase.start_time, phrase.end_time)
                        if phrase_id not in used_phrases:
                            selected_phrase = phrase
                            selected_section = section
                            break
                    
                    if not selected_phrase:
                        selected_phrase, selected_section = sorted_group[0]
                
                if not selected_phrase:
                    break
                
                # Mark as used
                phrase_id = (selected_phrase.start_time, selected_phrase.end_time)
                used_phrases.add(phrase_id)
                
                # Add to arrangement
                phrase_duration = selected_phrase.duration
                
                arrangement_sections.append((
                    Section(
                        start_time=selected_phrase.start_time,
                        end_time=selected_phrase.end_time,
                        label=phase_name,
                        energy=selected_phrase.energy,
                        features=selected_section.features
                    ),
                    current_time,
                    current_time + phrase_duration
                ))
                
                current_time += phrase_duration
                previous_phrase = selected_phrase
                previous_section = selected_section  # Track for style-aware scoring
                phrases_added += 1
        
        return Arrangement(
            sections=arrangement_sections,
            total_duration=current_time,
            structure=[s[0].label for s in arrangement_sections]
        )
    
    def _find_phrase_boundaries(
        self,
        audio_data: np.ndarray,
        section: Section,
        num_boundaries: int = 4
    ) -> List[float]:
        """
        Find natural phrase boundaries within a section.
        Good cut points should be:
        - Low energy moments (natural pauses)
        - Rhythmically stable (not during a fill or solo run)
        - Near beat boundaries
        
        Returns list of times (in seconds) that are good cut points.
        """
        start_sample = int(section.start_time * self.sample_rate)
        end_sample = int(section.end_time * self.sample_rate)
        section_audio = audio_data[start_sample:end_sample]
        
        if len(section_audio) < self.sample_rate:
            return [section.start_time, section.end_time]
        
        hop_length = 2048
        frame_length = 4096
        
        # Calculate energy
        rms = librosa.feature.rms(y=section_audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate spectral flux (how much the spectrum is changing)
        # High flux = melodic movement, solos, fills
        # Low flux = steady rhythmic playing, good for cuts
        spec = np.abs(librosa.stft(section_audio, hop_length=hop_length))
        spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
        # Pad to match rms length
        if len(spectral_flux) < len(rms):
            spectral_flux = np.pad(spectral_flux, (0, len(rms) - len(spectral_flux)))
        elif len(spectral_flux) > len(rms):
            spectral_flux = spectral_flux[:len(rms)]
        
        # Normalize flux
        flux_max = np.max(spectral_flux) if np.max(spectral_flux) > 0 else 1
        spectral_flux_norm = spectral_flux / flux_max
        
        # Combined score: low energy AND low spectral flux = good cut point
        # High flux penalty means we avoid cutting during solos/fills
        cut_score = rms + spectral_flux_norm * 0.5  # Lower is better
        
        # Find local minima in cut_score
        from scipy.signal import find_peaks
        
        inverted_score = -cut_score + np.max(cut_score)
        
        # Require at least 3 seconds between boundaries for musical phrases
        min_distance = int(3.0 * self.sample_rate / hop_length)
        peaks, _ = find_peaks(inverted_score, distance=min_distance, prominence=0.01)
        
        if len(peaks) == 0:
            return [section.start_time, section.end_time]
        
        # Convert to times and score each boundary
        boundary_candidates = []
        for peak in peaks:
            time_offset = librosa.frames_to_time(peak, sr=self.sample_rate, hop_length=hop_length)
            absolute_time = section.start_time + time_offset
            
            if section.start_time + 1.0 < absolute_time < section.end_time - 1.0:
                # Score: prefer low energy AND low flux (stable, quiet moments)
                score = cut_score[peak] if peak < len(cut_score) else 1.0
                boundary_candidates.append((absolute_time, score))
        
        # Sort by score (lower = better cut point)
        boundary_candidates.sort(key=lambda x: x[1])
        
        # Take the best boundaries
        best_boundaries = [section.start_time]
        best_boundaries.extend([t for t, _ in boundary_candidates[:num_boundaries]])
        best_boundaries.append(section.end_time)
        best_boundaries.sort()
        
        return best_boundaries
    
    def _find_best_entry_point(
        self,
        audio_data: np.ndarray,
        section: Section
    ) -> float:
        """Find the best place to enter a section (ideally on a downbeat after a quiet moment)."""
        boundaries = self._find_phrase_boundaries(audio_data, section)
        
        # The second boundary (first internal one) is often a good entry after intro
        if len(boundaries) > 2:
            return boundaries[1]
        return section.start_time
    
    def _find_best_exit_point(
        self,
        audio_data: np.ndarray,
        section: Section
    ) -> float:
        """Find the best place to exit a section (ideally at a phrase ending)."""
        boundaries = self._find_phrase_boundaries(audio_data, section)
        
        # The second-to-last boundary is often a good exit before the section "resolves"
        if len(boundaries) > 2:
            return boundaries[-2]
        return section.end_time
    
    def _analyze_transitions(
        self, 
        audio_data: np.ndarray, 
        sections: List[Section]
    ) -> dict:
        """Analyze section boundaries for smooth transitions."""
        transitions = {}
        analysis_samples = int(1.0 * self.sample_rate)  # 1 second
        
        for section in sections:
            start_sample = int(section.start_time * self.sample_rate)
            end_sample = int(section.end_time * self.sample_rate)
            
            start_seg = audio_data[start_sample:start_sample + analysis_samples]
            end_seg = audio_data[max(start_sample, end_sample - analysis_samples):end_sample]
            
            if len(start_seg) < 1000 or len(end_seg) < 1000:
                continue
            
            # Find best entry/exit points for this section
            best_entry = self._find_best_entry_point(audio_data, section)
            best_exit = self._find_best_exit_point(audio_data, section)
            
            transitions[id(section)] = {
                "start_energy": float(np.sqrt(np.mean(start_seg**2))),
                "end_energy": float(np.sqrt(np.mean(end_seg**2))),
                "start_chroma": self._get_section_chroma(start_seg),
                "end_chroma": self._get_section_chroma(end_seg),
                "best_entry": best_entry,
                "best_exit": best_exit,
            }
        
        return transitions
        
        return transitions
    
    def _get_section_chroma(self, audio_segment: np.ndarray) -> np.ndarray:
        """Get chroma vector for harmonic analysis."""
        if len(audio_segment) < 512:
            return np.zeros(12)
        chroma = librosa.feature.chroma_cqt(y=audio_segment, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        norm = np.linalg.norm(chroma_avg)
        return chroma_avg / norm if norm > 0 else chroma_avg
    
    def _score_section_flow(
        self,
        audio_data: np.ndarray,
        from_section: Section,
        to_section: Section,
        transitions: dict
    ) -> float:
        """Score how well two sections flow together."""
        from_id = id(from_section)
        to_id = id(to_section)
        
        if from_id not in transitions or to_id not in transitions:
            return 0.5
        
        from_t = transitions[from_id]
        to_t = transitions[to_id]
        
        score = 0.0
        
        # Energy continuity (30%)
        energy_diff = abs(from_t["end_energy"] - to_t["start_energy"])
        max_e = max(from_t["end_energy"], to_t["start_energy"], 0.01)
        score += (1.0 - min(energy_diff / max_e, 1.0)) * 0.30
        
        # Harmonic compatibility (40%)
        chroma_sim = np.dot(from_t["end_chroma"], to_t["start_chroma"])
        score += max(0, chroma_sim) * 0.40
        
        # Not same section (20%)
        if from_section.start_time != to_section.start_time:
            score += 0.20
        
        # Prefer natural progressions (10%)
        # Quieter after loud is OK, loud after quiet is OK (dynamics)
        score += 0.10
        
        return score

    def render_arrangement(
        self,
        audio_data: np.ndarray,
        arrangement: Arrangement,
        crossfade_duration: float = 0.05,  # Short crossfade just for click prevention
        snap_to_beats: bool = True
    ) -> np.ndarray:
        """
        Render an arrangement to audio with clean cuts at BAR boundaries.
        Detects loop length from rhythm track and aligns cuts to bar starts.
        
        Args:
            audio_data: Original audio data.
            arrangement: Arrangement plan.
            crossfade_duration: Duration of micro-fade for click prevention (default 50ms).
            snap_to_beats: Whether to snap cuts to beat/bar boundaries.
            
        Returns:
            Rendered audio.
        """
        if not arrangement.sections:
            return audio_data
        
        # Check if sections are in chronological order
        # If not, they've been rearranged and we should use hard cuts (crossfades between
        # unrelated audio sections sound weird)
        sections_rearranged = False
        prev_end = -1
        for section, _, _ in arrangement.sections:
            if section.start_time < prev_end - 0.1:  # Allow tiny overlap tolerance
                sections_rearranged = True
                break
            prev_end = section.end_time
        
        if sections_rearranged:
            print(f"   Sections rearranged - using hard cuts (crossfades would blend unrelated audio)")
        
        # Detect tempo, beats, and BAR structure
        bar_times = None
        beat_times = None
        bar_length = None
        
        if snap_to_beats:
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
                beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
                
                if len(beat_times) > 8:
                    # Estimate beats per bar (usually 4, sometimes 3 or 6)
                    beat_duration = np.median(np.diff(beat_times))
                    
                    # Try to detect loop length by finding repeating patterns
                    loop_length = self._detect_loop_length(audio_data, beat_duration, tempo)
                    
                    if loop_length:
                        bar_length = loop_length
                        print(f"   Detected loop/bar length: {bar_length:.2f}s ({bar_length/beat_duration:.1f} beats)")
                    else:
                        # Default to 4 beats per bar
                        bar_length = beat_duration * 4
                        print(f"   Using 4/4 bar length: {bar_length:.2f}s")
                    
                    # Generate bar boundary times
                    bar_times = self._generate_bar_times(beat_times, bar_length)
            except Exception as e:
                print(f"   Beat detection failed: {e}")
                beat_times = None
        
        # Process each section to find good cut points at BAR boundaries
        processed_sections = []
        for i, (section, new_start, new_end) in enumerate(arrangement.sections):
            start_time = section.start_time
            end_time = section.end_time
            
            # Snap to BAR boundaries (not just beats)
            if bar_times is not None and len(bar_times) > 0:
                # For section START: snap to nearest bar start
                start_time = self._snap_to_bar(start_time, bar_times, prefer='after')
                # For section END: snap to nearest bar END (which is next bar's start)
                end_time = self._snap_to_bar(end_time, bar_times, prefer='before')
            elif beat_times is not None and len(beat_times) > 0:
                # Fallback to beat snapping
                start_time = self._snap_to_beat(start_time, beat_times)
                end_time = self._snap_to_beat(end_time, beat_times)
            
            # Convert to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Fine-tune to nearest zero crossing for click-free cuts
            start_sample = self._find_zero_crossing(audio_data, start_sample, search_range=int(0.005 * self.sample_rate))
            end_sample = self._find_zero_crossing(audio_data, end_sample, search_range=int(0.005 * self.sample_rate))
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample > start_sample + self.sample_rate:  # At least 1 second
                processed_sections.append((start_sample, end_sample, section.label))
        
        if not processed_sections:
            return audio_data
        
        # Micro-fade for click prevention
        micro_fade = int(0.02 * self.sample_rate)  # 20ms for hard cuts
        
        # If sections were rearranged, use hard cuts (crossfades between unrelated audio sound weird)
        if sections_rearranged:
            print(f"   Using hard cuts with 20ms micro-fades")
            
            # Simple concatenation with micro-fades at each cut point
            output_parts = []
            
            for i, (start_sample, end_sample, label) in enumerate(processed_sections):
                section_audio = audio_data[start_sample:end_sample].copy().astype(np.float32)
                
                # Apply micro-fades at edges
                if len(section_audio) > micro_fade * 2:
                    # Fade in
                    fade_in = np.linspace(0, 1, micro_fade).astype(np.float32)
                    section_audio[:micro_fade] *= fade_in
                    # Fade out
                    fade_out = np.linspace(1, 0, micro_fade).astype(np.float32)
                    section_audio[-micro_fade:] *= fade_out
                
                output_parts.append(section_audio)
            
            # Concatenate all parts
            output = np.concatenate(output_parts)
            
            # Normalize if needed
            max_val = np.max(np.abs(output))
            if max_val > 0.95:
                output = output * 0.9 / max_val
            
            return output
        
        # For chronological sections, use BAR-aligned crossfade (1 full bar) for natural musical transitions
        crossfade_duration_sec = bar_length if bar_length else 1.5  # 1 bar, fallback to 1.5s
        crossfade_samples = int(crossfade_duration_sec * self.sample_rate)
        crossfade_samples = max(int(0.5 * self.sample_rate), crossfade_samples)  # At least 0.5s
        crossfade_samples = min(int(2.0 * self.sample_rate), crossfade_samples)  # Max 2 seconds
        
        print(f"   Using {crossfade_duration_sec:.2f}s bar-aligned crossfade")
        
        # Build output with bar-aligned crossfades between sections
        output_parts = []
        
        for i, (start_sample, end_sample, label) in enumerate(processed_sections):
            section_audio = audio_data[start_sample:end_sample].copy().astype(np.float32)
            
            if len(section_audio) < crossfade_samples * 2:
                output_parts.append(section_audio)
                continue
            
            if i == 0:
                # First section: micro fade in at very start
                fade_in = np.linspace(0, 1, micro_fade).astype(np.float32)
                section_audio[:micro_fade] *= fade_in
            
            if i == len(processed_sections) - 1:
                # Last section: micro fade out at very end
                fade_out = np.linspace(1, 0, micro_fade).astype(np.float32)
                section_audio[-micro_fade:] *= fade_out
            
            output_parts.append(section_audio)
        
        # Now combine with bar-aligned crossfades between sections
        if len(output_parts) == 0:
            return audio_data
        elif len(output_parts) == 1:
            return output_parts[0]
        
        # Calculate total length accounting for overlaps
        total_length = sum(len(p) for p in output_parts) - crossfade_samples * (len(output_parts) - 1)
        output = np.zeros(total_length, dtype=np.float32)
        
        pos = 0
        for i, part in enumerate(output_parts):
            if i == 0:
                # First section: copy fully, position for crossfade at end
                output[pos:pos + len(part)] = part
                pos += len(part) - crossfade_samples
            else:
                # Create equal-power crossfade over 1 bar
                fade_out = np.cos(np.linspace(0, np.pi/2, crossfade_samples)).astype(np.float32)
                fade_in = np.sin(np.linspace(0, np.pi/2, crossfade_samples)).astype(np.float32)
                
                # Apply crossfade in overlap region (1 bar)
                overlap_start = pos
                overlap_end = pos + crossfade_samples
                
                # Fade out the previous section's tail
                output[overlap_start:overlap_end] *= fade_out
                # Fade in and add the new section's head
                output[overlap_start:overlap_end] += part[:crossfade_samples] * fade_in
                
                # Copy the rest of the new section
                remaining = part[crossfade_samples:]
                output[overlap_end:overlap_end + len(remaining)] = remaining
                
                if i < len(output_parts) - 1:
                    pos = overlap_end + len(remaining) - crossfade_samples
                else:
                    pos = overlap_end + len(remaining)
        
        # Trim to actual content
        output = output[:pos]
        
        # Normalize if needed
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * 0.9 / max_val
        
        return output
    
    def _detect_loop_length(self, audio_data: np.ndarray, beat_duration: float, tempo: float) -> Optional[float]:
        """
        Detect the loop/riff length by finding repeating patterns in the rhythm track.
        Assumes 4/4 time signature. Returns the bar length in seconds.
        """
        # For 4/4 time, one bar = 4 beats
        bar_length = beat_duration * 4
        
        try:
            # Verify the 4-beat bar length by checking autocorrelation
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate, hop_length=hop_length)
            
            bar_samples = int(bar_length * self.sample_rate / hop_length)
            
            if bar_samples < 10 or bar_samples * 2 > len(onset_env):
                return bar_length  # Just use 4 beats
            
            # Check if the rhythm repeats every 4 beats (one bar)
            # Also check 8 beats (two bars) which is common for riffs
            for multiplier in [1, 2]:  # 1 bar or 2 bars
                test_samples = bar_samples * multiplier
                correlations = []
                
                for offset in range(0, min(len(onset_env) - test_samples * 2, test_samples * 8), test_samples):
                    seg1 = onset_env[offset:offset + test_samples]
                    seg2 = onset_env[offset + test_samples:offset + test_samples * 2]
                    
                    if len(seg1) == len(seg2) and len(seg1) > 0:
                        corr = np.corrcoef(seg1, seg2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations and np.mean(correlations) > 0.6:
                    # Found a strong repeating pattern
                    return bar_length * multiplier
            
            # Default to single bar (4 beats)
            return bar_length
            
        except Exception as e:
            return bar_length  # Default to 4 beats per bar
    
    def _generate_bar_times(self, beat_times: np.ndarray, bar_length: float) -> np.ndarray:
        """Generate bar boundary times from beat times and bar length."""
        if len(beat_times) == 0:
            return np.array([])
        
        # Start from first beat
        first_beat = beat_times[0]
        last_beat = beat_times[-1]
        
        # Generate bar times
        bar_times = []
        t = first_beat
        while t <= last_beat + bar_length:
            bar_times.append(t)
            t += bar_length
        
        return np.array(bar_times)
    
    def _snap_to_bar(self, time: float, bar_times: np.ndarray, prefer: str = 'nearest') -> float:
        """
        Snap a time to the nearest bar boundary.
        
        Args:
            time: Time to snap
            bar_times: Array of bar boundary times
            prefer: 'nearest', 'before', or 'after'
        """
        if len(bar_times) == 0:
            return time
        
        idx = np.argmin(np.abs(bar_times - time))
        
        if prefer == 'before':
            # Find the bar boundary AT or BEFORE the time
            candidates = bar_times[bar_times <= time + 0.1]  # Small tolerance
            if len(candidates) > 0:
                return candidates[-1]
        elif prefer == 'after':
            # Find the bar boundary AT or AFTER the time
            candidates = bar_times[bar_times >= time - 0.1]  # Small tolerance
            if len(candidates) > 0:
                return candidates[0]
        
        # Default: nearest
        return bar_times[idx]
    
    def _snap_to_beat(self, time: float, beat_times: np.ndarray) -> float:
        """Snap a time to the nearest beat."""
        if len(beat_times) == 0:
            return time
        
        idx = np.argmin(np.abs(beat_times - time))
        
        # Only snap if within 0.2 seconds of a beat
        if abs(beat_times[idx] - time) < 0.2:
            return beat_times[idx]
        return time
    
    def _find_zero_crossing(self, audio: np.ndarray, sample: int, search_range: int = 1000) -> int:
        """Find the nearest zero crossing to the given sample position."""
        start = max(0, sample - search_range)
        end = min(len(audio), sample + search_range)
        
        if start >= end:
            return sample
        
        segment = audio[start:end]
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        
        if len(zero_crossings) == 0:
            # No zero crossings, find lowest amplitude point
            min_idx = np.argmin(np.abs(segment))
            return start + min_idx
        
        # Find closest zero crossing to original position
        relative_pos = sample - start
        closest_idx = zero_crossings[np.argmin(np.abs(zero_crossings - relative_pos))]
        
        return start + closest_idx
    
    def _create_fade(self, length: int, direction: str) -> np.ndarray:
        """Create a smooth fade curve (equal power)."""
        if direction == 'in':
            # Equal power fade in
            t = np.linspace(0, np.pi / 2, length)
            return np.sin(t).astype(np.float32)
        else:
            # Equal power fade out
            t = np.linspace(0, np.pi / 2, length)
            return np.cos(t).astype(np.float32)
    
    def auto_arrange(
        self,
        audio_data: np.ndarray,
        template: str = "medium",
        target_duration: Optional[float] = None,
        style_profile: StyleProfile = None,
        allow_rearrange: bool = False,
        min_section_duration: Optional[float] = None,
        max_section_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, Arrangement, List[Section]]:
        """
        Automatically analyze and arrange audio into a song structure.
        
        Args:
            audio_data: Raw audio data.
            template: Arrangement length - "short" (~3min), "medium" (~5min), "long" (~8min), "full" (all material).
            target_duration: Target duration in seconds (overrides template).
            style_profile: Optional learned style profile for transition scoring.
            allow_rearrange: If True, allow reordering sections for better musical flow.
            min_section_duration: Minimum section duration in seconds.
            max_section_duration: Maximum section duration in seconds.
            
        Returns:
            Tuple of (arranged_audio, arrangement, detected_sections).
        """
        # Analyze sections with custom duration constraints
        sections = self.analyze_sections(
            audio_data, 
            min_section_duration=min_section_duration,
            max_section_duration=max_section_duration
        )
        
        # Create arrangement
        arrangement = self.create_arrangement(
            audio_data, sections, template, target_duration, style_profile,
            allow_rearrange=allow_rearrange
        )
        
        # Use short micro-fades for click prevention (not blending crossfades)
        crossfade_dur = 0.03  # 30ms - just to prevent clicks
        
        # Render
        arranged_audio = self.render_arrangement(audio_data, arrangement, crossfade_duration=crossfade_dur)
        
        return arranged_audio, arrangement, sections
    
    def _detect_best_template(self, sections: List[Section]) -> str:
        """Auto-detect the best template based on section characteristics."""
        labels = [s.label for s in sections]
        
        # Check for EDM-style elements
        if "drop" in labels or "buildup" in labels:
            return "edm"
        
        # Check for bridge (suggests more complex structure)
        if "bridge" in labels:
            return "pop"
        
        # Count choruses
        chorus_count = labels.count("chorus")
        if chorus_count >= 3:
            return "pop"
        elif chorus_count >= 2:
            return "simple"
        
        return "minimal"
    
    def get_arrangement_summary(
        self, 
        arrangement: Arrangement, 
        original_sections: List[Section]
    ) -> str:
        """Get a human-readable summary of the arrangement."""
        lines = [
            "🎵 Song Arrangement Summary",
            "=" * 50,
            "",
            f"Original sections detected: {len(original_sections)}",
            f"Arranged structure: {' → '.join(arrangement.structure)}",
            f"Total duration: {arrangement.total_duration:.1f}s ({arrangement.total_duration/60:.1f} min)",
            "",
            "Section Details:",
            "-" * 30
        ]
        
        for section, new_start, new_end in arrangement.sections:
            lines.append(
                f"  [{section.label.upper():^10}] "
                f"{new_start:.1f}s - {new_end:.1f}s "
                f"(from original {section.start_time:.1f}s-{section.end_time:.1f}s)"
            )
        
        return "\n".join(lines)

"""AI-powered song arranger that creates coherent song structures."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa


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
        
        # Common song structure templates
        self.templates = {
            "pop": ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
            "edm": ["intro", "buildup", "drop", "breakdown", "buildup", "drop", "outro"],
            "rock": ["intro", "verse", "chorus", "verse", "chorus", "solo", "chorus", "outro"],
            "simple": ["intro", "verse", "chorus", "verse", "chorus", "outro"],
            "minimal": ["intro", "main", "main", "outro"],
            "doom": ["intro", "verse", "verse", "chorus", "verse", "chorus", "breakdown", "verse", "outro"],
            "stoner": ["intro", "verse", "chorus", "breakdown", "verse", "chorus", "chorus", "outro"],
            "content": None,  # Will use actual detected sections
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
    
    def analyze_sections(self, audio_data: np.ndarray) -> List[Section]:
        """
        Analyze audio and identify distinct sections with characteristics.
        
        Args:
            audio_data: Raw audio data.
            
        Returns:
            List of detected sections with labels and phrases.
            
        Returns:
            List of detected sections with labels.
        """
        duration = len(audio_data) / self.sample_rate
        
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
        # Use fewer segments for cleaner structure (aim for ~20-30 second sections)
        num_segments = max(4, min(12, int(duration / 25)))
        
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
        
        # Merge very short segments (less than 8 seconds)
        bound_times = self._merge_short_segments(bound_times, min_duration=8.0)
        
        # Detect beat grid for phrase alignment
        tempo, beat_times, bar_times = self.detect_beat_grid(audio_data)
        
        # Snap section boundaries to nearest bar
        snapped_bounds = []
        for t in bound_times:
            snapped = self.snap_to_bar(t, bar_times)
            # Don't snap if it would create a very short section
            if len(snapped_bounds) == 0 or snapped - snapped_bounds[-1] >= 6.0:
                snapped_bounds.append(snapped)
            elif t - snapped_bounds[-1] >= 6.0:
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
        target_duration: Optional[float] = None
    ) -> Arrangement:
        """
        Create a song arrangement from detected sections.
        
        Args:
            audio_data: Raw audio data.
            sections: Detected sections.
            template: Song structure template to follow ("content" uses detected sections).
            target_duration: Target duration in seconds (None = auto).
            
        Returns:
            Arrangement plan.
        """
        # "content" mode: use the best sections from the actual recording
        if template == "content" or self.templates.get(template) is None:
            return self._create_content_based_arrangement(
                audio_data, sections, target_duration
            )
        
        # Doom/stoner specific arrangement
        if template in ["doom", "stoner"]:
            return self.create_doom_arrangement(audio_data, sections, target_duration)
        
        structure = self.templates.get(template, self.templates["simple"])
        
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
        locality_weight: float = 0.3
    ) -> float:
        """
        Score how well two phrases flow together.
        Optimized for musical coherence with locality bias.
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
            
            # 3. HARMONIC CONTINUITY (20%) - similar key/chords
            chroma_sim = np.dot(f_from["chroma"], f_to["chroma"])
            score += max(0, chroma_sim) * 0.20
            
            # 4. TIMBRAL CONTINUITY (15%) - similar instrument/tone
            timbre_sim = np.dot(f_from["timbre"], f_to["timbre"])
            score += max(0, timbre_sim) * 0.15
            
            # 5. ENERGY CONTINUITY (10%) - smooth energy transitions
            energy_diff = abs(f_from["energy"] - f_to["energy"])
            max_energy = max(f_from["energy"], f_to["energy"], 0.01)
            energy_score = 1.0 - min(energy_diff / max_energy, 1.0)
            score += energy_score * 0.10
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
    
    def create_doom_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        target_duration: Optional[float] = None
    ) -> Arrangement:
        """
        Create a stoner/doom arrangement using phrase-level selection.
        Optimized for musical coherence with:
        - Locality bias (prefer nearby phrases)
        - Harmonic/timbral similarity matching
        - Section continuity
        - Smart phrase grouping
        
        Doom structure: long, heavy, hypnotic repetition with slow dynamics.
        Starts quiet/ambient, builds into heavy riffs.
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
        
        # Structure: use coherent groups, not random phrases
        structure_plan = [
            ("intro", quiet_material, 2),           # 2 phrases from quiet group
            ("build", build_material, 2),           # 2 phrases building up
            ("riff", primary_heavy, 4),             # 4 phrases of main riff
            ("riff", secondary_heavy, 3),           # 3 phrases of variation
            ("breakdown", quiet_material, 2),       # Breathing room
            ("riff", primary_heavy, 3),             # Return to main riff
            ("outro", quiet_material, 2),           # Quiet ending
        ]
        
        arrangement_sections = []
        current_time = 0.0
        previous_phrase = None
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
                            locality_weight=0.35  # Strong locality preference
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
        crossfade_duration: float = 0.5,
        snap_to_beats: bool = True
    ) -> np.ndarray:
        """
        Render an arrangement to audio with smooth, musical transitions.
        Uses phrase-aware cutting for natural transitions.
        
        Args:
            audio_data: Original audio data.
            arrangement: Arrangement plan.
            crossfade_duration: Duration of crossfades between sections (default 0.5s).
            snap_to_beats: Whether to snap cuts to beat boundaries.
            
        Returns:
            Rendered audio.
        """
        if not arrangement.sections:
            return audio_data
        
        # Detect beats for the whole track for snapping
        beat_times = None
        if snap_to_beats:
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
                beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            except:
                beat_times = None
        
        # Pre-analyze all sections for phrase boundaries
        transitions = self._analyze_transitions(audio_data, [s for s, _, _ in arrangement.sections])
        
        # Process each section to find good cut points
        processed_sections = []
        for i, (section, new_start, new_end) in enumerate(arrangement.sections):
            section_id = id(section)
            
            # Use phrase-aware entry/exit points if available
            if section_id in transitions:
                trans = transitions[section_id]
                # For first section, use best entry point
                if i == 0:
                    start_time = trans.get("best_entry", section.start_time)
                else:
                    start_time = trans.get("best_entry", section.start_time)
                
                # For last section, use best exit point
                if i == len(arrangement.sections) - 1:
                    end_time = trans.get("best_exit", section.end_time)
                else:
                    end_time = trans.get("best_exit", section.end_time)
            else:
                start_time = section.start_time
                end_time = section.end_time
            
            # Snap to nearest beat if available
            if beat_times is not None and len(beat_times) > 0:
                start_time = self._snap_to_beat(start_time, beat_times)
                end_time = self._snap_to_beat(end_time, beat_times)
            
            # Convert to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Fine-tune to nearest zero crossing for click-free cuts
            start_sample = self._find_zero_crossing(audio_data, start_sample, search_range=int(0.01 * self.sample_rate))
            end_sample = self._find_zero_crossing(audio_data, end_sample, search_range=int(0.01 * self.sample_rate))
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample > start_sample + self.sample_rate:  # At least 1 second
                processed_sections.append((start_sample, end_sample, section.label))
        
        if not processed_sections:
            return audio_data
        
        # Build output with proper crossfades
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        
        # Calculate output length
        total_samples = sum(end - start for start, end, _ in processed_sections)
        # Subtract overlaps from crossfades (except first section)
        total_samples -= crossfade_samples * (len(processed_sections) - 1)
        total_samples = max(total_samples, self.sample_rate)  # At least 1 second
        
        output = np.zeros(total_samples, dtype=np.float32)
        output_pos = 0
        
        for i, (start_sample, end_sample, label) in enumerate(processed_sections):
            section_audio = audio_data[start_sample:end_sample].copy().astype(np.float32)
            section_length = len(section_audio)
            
            if section_length < crossfade_samples * 2:
                # Section too short for crossfade, just add it
                end_pos = min(output_pos + section_length, len(output))
                output[output_pos:end_pos] = section_audio[:end_pos - output_pos]
                output_pos = end_pos
                continue
            
            if i == 0:
                # First section: fade in at start, prepare for crossfade at end
                fade_in = self._create_fade(crossfade_samples, 'in')
                section_audio[:crossfade_samples] *= fade_in
                
                # Add full section
                end_pos = min(output_pos + section_length, len(output))
                output[output_pos:end_pos] = section_audio[:end_pos - output_pos]
                output_pos = end_pos - crossfade_samples  # Back up for overlap
                
            elif i == len(processed_sections) - 1:
                # Last section: crossfade in, fade out at end
                fade_in = self._create_fade(crossfade_samples, 'in')
                fade_out = self._create_fade(crossfade_samples, 'out')
                
                section_audio[:crossfade_samples] *= fade_in
                section_audio[-crossfade_samples:] *= fade_out
                
                # Overlap-add for crossfade
                end_pos = min(output_pos + section_length, len(output))
                add_length = min(section_length, end_pos - output_pos)
                output[output_pos:output_pos + add_length] += section_audio[:add_length]
                output_pos = output_pos + add_length
                
            else:
                # Middle section: crossfade in and out
                fade_in = self._create_fade(crossfade_samples, 'in')
                fade_out = self._create_fade(crossfade_samples, 'out')
                
                section_audio[:crossfade_samples] *= fade_in
                section_audio[-crossfade_samples:] *= fade_out
                
                # Overlap-add
                end_pos = min(output_pos + section_length, len(output))
                add_length = min(section_length, end_pos - output_pos)
                output[output_pos:output_pos + add_length] += section_audio[:add_length]
                output_pos = output_pos + add_length - crossfade_samples  # Back up for next overlap
        
        # Trim to actual content
        output = output[:output_pos + crossfade_samples]
        
        # Normalize to prevent clipping from overlap-add
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * 0.9 / max_val
        
        return output
    
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
        template: str = "auto",
        target_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, Arrangement, List[Section]]:
        """
        Automatically analyze and arrange audio into a song structure.
        
        Args:
            audio_data: Raw audio data.
            template: Structure template ("auto", "pop", "edm", "rock", "simple", "minimal").
            target_duration: Target duration in seconds.
            
        Returns:
            Tuple of (arranged_audio, arrangement, detected_sections).
        """
        # Analyze sections
        sections = self.analyze_sections(audio_data)
        
        # Auto-detect best template
        if template == "auto":
            template = self._detect_best_template(sections)
        
        # Create arrangement
        arrangement = self.create_arrangement(
            audio_data, sections, template, target_duration
        )
        
        # Use longer crossfades for doom/stoner (more hypnotic)
        crossfade_dur = 1.5 if template in ("doom", "stoner") else 0.5
        
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

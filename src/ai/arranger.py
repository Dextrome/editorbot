"""AI-powered song arranger that creates coherent song structures."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa


@dataclass
class Section:
    """Represents a detected section of audio."""
    start_time: float
    end_time: float
    label: str  # "intro", "verse", "chorus", "bridge", "outro", "buildup", "drop"
    energy: float  # Average energy level
    features: Dict[str, float]  # Characteristic features
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"Section({self.label}, {self.start_time:.1f}s-{self.end_time:.1f}s, energy={self.energy:.2f})"


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
    
    def analyze_sections(self, audio_data: np.ndarray) -> List[Section]:
        """
        Analyze audio and identify distinct sections with characteristics.
        
        Args:
            audio_data: Raw audio data.
            
        Returns:
            List of detected sections with labels.
        """
        duration = len(audio_data) / self.sample_rate
        
        # Extract features for segmentation
        hop_length = 512
        
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
            
            sections.append(Section(
                start_time=start_time,
                end_time=end_time,
                label=label,
                energy=energy,
                features=section_features
            ))
        
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
    
    def create_doom_arrangement(
        self,
        audio_data: np.ndarray,
        sections: List[Section],
        target_duration: Optional[float] = None
    ) -> Arrangement:
        """
        Create a stoner/doom arrangement emphasizing heavy repetition and slow build.
        
        Doom structure: long, heavy, hypnotic repetition with slow dynamics.
        Starts quiet/ambient, builds into heavy riffs.
        """
        if not sections:
            return Arrangement(sections=[], total_duration=0, structure=[])
        
        # Doom songs are longer - aim for 4-6 minutes
        total_available = sum(s.duration for s in sections)
        if target_duration is None:
            target_duration = min(360, max(200, total_available * 0.7))
        
        # Score each section for doom appropriateness
        scored = []
        for s in sections:
            # Doom score: prefer high energy, long duration, low variance (droning)
            doom_score = (
                s.energy * 2.0 +  # Heavy is good
                s.duration / 60.0 +  # Longer is better
                (1.0 / (1.0 + s.features.get("energy_variance", 0) * 5)) +  # Consistent/droning
                s.features.get("harmonicity", 0.5) * 0.5  # Some harmonic content
            )
            scored.append((s, doom_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize by energy - be more selective about what counts as "quiet"
        all_energies = [s.energy for s in sections]
        median_energy = np.median(all_energies)
        energy_std = np.std(all_energies)
        
        # Quiet = below median, Heavy = above median
        quiet = [s for s, _ in scored if s.energy < median_energy]
        heavy = [s for s, _ in scored if s.energy >= median_energy]
        
        # Sort quiet by energy (quietest first for intro)
        quiet.sort(key=lambda s: s.energy)
        # Sort heavy by doom score (best riffs first)
        heavy.sort(key=lambda s: next((score for sec, score in scored if sec == s), 0), reverse=True)
        
        # If we don't have quiet sections, use the beginning of the recording
        # (often has count-in or ambient room tone)
        if not quiet:
            # Create a synthetic "intro" from the very start
            first_section = min(sections, key=lambda s: s.start_time)
            quiet = [first_section]
        
        if not heavy:
            heavy = [s for s, _ in scored]
        
        # Doom structure: 
        # 1. Quiet/ambient intro (ease in)
        # 2. Build section (transition)
        # 3. Heavy riffs (the meat)
        # 4. Breakdown (breathing room)
        # 5. More heavy riffs
        # 6. Quiet outro (fade/drone out)
        
        arrangement_sections = []
        current_time = 0.0
        
        structure_plan = [
            ("intro", quiet, 0.08, True),        # Quiet intro - use quietest part
            ("build", heavy, 0.10, False),       # Transition into heavy (use start of a riff)
            ("riff", heavy, 0.22, False),        # Main heavy riff
            ("riff", heavy, 0.18, False),        # Second riff (doom repetition)
            ("breakdown", quiet, 0.10, True),    # Slow breakdown
            ("riff", heavy, 0.18, False),        # Back to heavy
            ("outro", quiet, 0.08, True),        # Quiet outro/drone
        ]
        
        used = set()
        
        for phase_name, candidates, ratio, prefer_quiet in structure_plan:
            if current_time >= target_duration:
                break
            
            phase_duration = target_duration * ratio
            
            # Find section
            selected = None
            
            # For intro/outro, specifically look for the quietest available
            if prefer_quiet and candidates:
                for c in sorted(candidates, key=lambda s: s.energy):
                    if id(c) not in used:
                        selected = c
                        used.add(id(c))
                        break
            else:
                for c in candidates:
                    if id(c) not in used:
                        selected = c
                        used.add(id(c))
                        break
            
            # If all used, reuse (doom is repetitive!)
            if not selected and candidates:
                selected = candidates[0]
            
            if selected:
                use_duration = min(phase_duration, selected.duration)
                use_duration = max(use_duration, selected.duration * 0.7)  # Use good chunk of section
                
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
    
    def render_arrangement(
        self,
        audio_data: np.ndarray,
        arrangement: Arrangement,
        crossfade_duration: float = 0.5,
        snap_to_beats: bool = True
    ) -> np.ndarray:
        """
        Render an arrangement to audio with smooth, musical transitions.
        
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
        
        # Process each section to find good cut points
        processed_sections = []
        for section, new_start, new_end in arrangement.sections:
            start_time = section.start_time
            end_time = section.end_time
            
            # Snap to nearest beat if available
            if beat_times is not None and len(beat_times) > 0:
                start_time = self._snap_to_beat(start_time, beat_times)
                end_time = self._snap_to_beat(end_time, beat_times)
            
            # Find good cut points (zero crossings or low energy)
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Adjust start to nearest zero crossing
            start_sample = self._find_zero_crossing(audio_data, start_sample, search_range=int(0.05 * self.sample_rate))
            
            # Adjust end to nearest zero crossing
            end_sample = self._find_zero_crossing(audio_data, end_sample, search_range=int(0.05 * self.sample_rate))
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample > start_sample:
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
        
        # Render
        arranged_audio = self.render_arrangement(audio_data, arrangement)
        
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

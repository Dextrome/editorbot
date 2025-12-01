"""
LoopRemix - Core analysis and remixing engine.

Analyzes loop-based jam recordings and creates coherent remixes.
"""
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

# Add shared folder to path for demucs_wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoopInfo:
    """Detected loop/bar information."""
    tempo: float              # BPM
    beat_duration: float      # seconds per beat
    bar_duration: float       # seconds per bar (4 beats)
    loop_bars: int            # bars per loop (typically 4, 8, or 16)
    loop_duration: float      # seconds per loop
    beat_times: np.ndarray    # timestamps of all beats
    bar_times: np.ndarray     # timestamps of bar downbeats


@dataclass
class StemData:
    """Separated audio stems from Demucs."""
    drums: Optional[np.ndarray] = None
    bass: Optional[np.ndarray] = None
    vocals: Optional[np.ndarray] = None
    other: Optional[np.ndarray] = None
    sample_rate: int = 44100
    
    @property
    def lead(self) -> Optional[np.ndarray]:
        """Vocals + other = the 'lead' or melodic content."""
        if self.vocals is not None and self.other is not None:
            return self.vocals + self.other
        return self.vocals or self.other
    
    @property
    def backing(self) -> Optional[np.ndarray]:
        """Drums + bass = the 'backing' or rhythm section."""
        if self.drums is not None and self.bass is not None:
            return self.drums + self.bass
        return self.drums or self.bass


@dataclass  
class Phrase:
    """A musical phrase extracted from the recording."""
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int
    bars: int                 # number of bars in this phrase
    energy: float             # average energy level
    quality_score: float      # overall quality (0-1)
    is_loop_aligned: bool     # starts on loop boundary
    lead_energy: float = 0.0  # energy in lead (vocals+other) stem
    backing_energy: float = 0.0  # energy in backing (drums+bass) stem
    start_energy: float = 0.0  # energy at the start of the phrase (for transition matching)
    end_energy: float = 0.0    # energy at the end of the phrase (for transition matching)
    spectral_centroid: float = 0.0  # brightness measure for progression tracking
    # Harmonic/melodic features from lead stem
    avg_chroma: Optional[np.ndarray] = None  # 12-element pitch class distribution
    start_chroma: Optional[np.ndarray] = None  # chroma at phrase start (for transitions)
    end_chroma: Optional[np.ndarray] = None  # chroma at phrase end (for transitions)
    dominant_pitch_class: int = 0  # most prominent pitch class (0-11, C=0)
    # Rhythmic texture features
    onset_density: float = 0.0  # onsets per second (how busy/active the phrase is)
    spectral_flux: float = 0.0  # spectral change rate (sustained vs staccato)
    end_onset_density: float = 0.0  # onset density at phrase end (for transition matching)


def separate_stems(audio_path: str, device: str = 'cuda') -> Optional[StemData]:
    """
    Separate audio into stems using Demucs.
    
    Args:
        audio_path: Path to audio file
        device: 'cuda' or 'cpu'
        
    Returns:
        StemData with separated stems, or None if separation failed
    """
    try:
        from demucs_wrapper import DemucsSeparator
        
        logger.info("Separating stems with Demucs...")
        separator = DemucsSeparator(model="htdemucs")
        stems = separator.separate(audio_path, device=device)
        
        sr = stems.get('_sr', 44100)
        stem_data = StemData(
            drums=stems.get('drums'),
            bass=stems.get('bass'),
            vocals=stems.get('vocals'),
            other=stems.get('other'),
            sample_rate=sr
        )
        
        logger.info("  Stem separation complete")
        return stem_data
        
    except ImportError as e:
        logger.warning(f"Demucs not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Stem separation failed: {e}")
        return None


class LoopAnalyzer:
    """Analyzes loop-based recordings to detect structure."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def analyze(self, audio: np.ndarray) -> LoopInfo:
        """
        Analyze audio to detect tempo, beats, bars, and loop structure.
        
        Args:
            audio: Audio data (mono or stereo)
            
        Returns:
            LoopInfo with detected structure
        """
        # Convert to mono for analysis
        if audio.ndim == 2:
            mono = librosa.to_mono(audio.T)
        else:
            mono = audio
        
        logger.info("Analyzing tempo and beat structure...")
        
        # Detect tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=self.sample_rate)
        
        # Handle tempo as array (newer librosa)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        beat_duration = 60.0 / tempo
        bar_duration = beat_duration * 4  # Assuming 4/4 time
        
        # Create bar times (every 4 beats)
        bar_times = beat_times[::4] if len(beat_times) >= 4 else beat_times
        
        # Detect loop length by analyzing repetition
        loop_bars = self._detect_loop_length(mono, bar_duration)
        loop_duration = bar_duration * loop_bars
        
        logger.info(f"  Tempo: {tempo:.1f} BPM")
        logger.info(f"  Bar duration: {bar_duration:.2f}s")
        logger.info(f"  Loop: {loop_bars} bars ({loop_duration:.2f}s)")
        logger.info(f"  Total beats: {len(beat_times)}, bars: {len(bar_times)}")
        
        return LoopInfo(
            tempo=tempo,
            beat_duration=beat_duration,
            bar_duration=bar_duration,
            loop_bars=loop_bars,
            loop_duration=loop_duration,
            beat_times=beat_times,
            bar_times=bar_times
        )
    
    def _detect_loop_length(self, mono: np.ndarray, bar_duration: float) -> int:
        """
        Detect the loop length in bars by finding repeating patterns.
        Common loop lengths: 4, 8, 16 bars.
        """
        # Compute onset strength for rhythm analysis
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=mono, sr=self.sample_rate, hop_length=hop_length)
        
        # Test common loop lengths: 4, 8, 16 bars
        best_loop = 4
        best_correlation = 0
        
        for loop_bars in [4, 8, 16]:
            loop_samples = int(bar_duration * loop_bars * self.sample_rate / hop_length)
            
            if loop_samples * 2 > len(onset_env):
                continue
            
            # Compute autocorrelation at this loop length
            correlations = []
            for offset in range(0, min(len(onset_env) - loop_samples * 2, loop_samples * 4), loop_samples):
                seg1 = onset_env[offset:offset + loop_samples]
                seg2 = onset_env[offset + loop_samples:offset + loop_samples * 2]
                
                if len(seg1) == len(seg2) and len(seg1) > 0:
                    corr = np.corrcoef(seg1, seg2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                avg_corr = np.mean(correlations)
                if avg_corr > best_correlation:
                    best_correlation = avg_corr
                    best_loop = loop_bars
        
        return best_loop


class PhraseDetector:
    """Detects musical phrases in the recording."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def detect_phrases(
        self, 
        audio: np.ndarray, 
        loop_info: LoopInfo,
        min_bars: int = 4,
        max_bars: int = 16,
        stems: Optional[StemData] = None
    ) -> List[Phrase]:
        """
        Detect musical phrases based on energy changes and loop alignment.
        
        When stems are available, uses the 'lead' (vocals+other) stem for phrase
        detection, which is more meaningful for loop jams where the backing stays constant.
        
        Args:
            audio: Audio data
            loop_info: Detected loop structure
            min_bars: Minimum phrase length in bars
            max_bars: Maximum phrase length in bars
            stems: Optional separated stems for better analysis
            
        Returns:
            List of detected phrases
        """
        # Convert to mono for analysis
        if audio.ndim == 2:
            mono = librosa.to_mono(audio.T)
        else:
            mono = audio
        
        # If we have stems, use lead (vocals+other) for phrase detection
        # This is better for loop jams where backing track stays constant
        lead_mono = None
        backing_mono = None
        if stems is not None:
            lead = stems.lead
            backing = stems.backing
            if lead is not None:
                if lead.ndim == 2:
                    lead_mono = librosa.to_mono(lead.T)
                else:
                    lead_mono = lead
                logger.info("  Using lead stem (vocals+other) for phrase detection")
            if backing is not None:
                if backing.ndim == 2:
                    backing_mono = librosa.to_mono(backing.T)
                else:
                    backing_mono = backing
        
        # Use lead stem for energy analysis if available, otherwise full mix
        analysis_audio = lead_mono if lead_mono is not None else mono
        
        logger.info("Detecting phrase boundaries...")
        
        # Compute frame-level energy
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=analysis_audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sample_rate, hop_length=hop_length)
        
        # Detect significant energy changes (potential phrase boundaries)
        rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')
        rms_diff = np.abs(np.diff(rms_smooth))
        
        # Find peaks in energy change (phrase boundaries)
        threshold = np.percentile(rms_diff, 85)
        boundary_frames = np.where(rms_diff > threshold)[0]
        boundary_times = rms_times[boundary_frames]
        
        # Snap boundaries to bar times
        bar_times = loop_info.bar_times
        snapped_boundaries = []
        
        for bt in boundary_times:
            # Find closest bar
            idx = np.argmin(np.abs(bar_times - bt))
            snapped = bar_times[idx]
            if snapped not in snapped_boundaries:
                snapped_boundaries.append(snapped)
        
        # Add start and end
        snapped_boundaries = sorted(set([0.0] + snapped_boundaries + [len(audio) / self.sample_rate]))
        
        # Create phrases from boundaries
        phrases = []
        min_duration = min_bars * loop_info.bar_duration
        max_duration = max_bars * loop_info.bar_duration
        
        i = 0
        while i < len(snapped_boundaries) - 1:
            start = snapped_boundaries[i]
            
            # Find end that gives us a good phrase length
            best_end = None
            for j in range(i + 1, len(snapped_boundaries)):
                end = snapped_boundaries[j]
                duration = end - start
                
                if duration >= min_duration:
                    if duration <= max_duration:
                        best_end = j
                        break
                    elif best_end is None:
                        # Take what we can get
                        best_end = j
                        break
            
            if best_end is None:
                best_end = len(snapped_boundaries) - 1
            
            end = snapped_boundaries[best_end]
            duration = end - start
            
            if duration >= min_duration * 0.5:  # Allow slightly shorter phrases
                start_sample = int(start * self.sample_rate)
                end_sample = int(end * self.sample_rate)
                
                # Calculate phrase properties
                phrase_audio = audio[start_sample:end_sample]
                if phrase_audio.ndim == 2:
                    phrase_mono = librosa.to_mono(phrase_audio.T)
                else:
                    phrase_mono = phrase_audio
                
                energy = np.sqrt(np.mean(phrase_mono ** 2))
                bars = round(duration / loop_info.bar_duration)
                
                # Calculate lead and backing energy if stems available
                lead_energy = 0.0
                backing_energy = 0.0
                if lead_mono is not None and end_sample <= len(lead_mono):
                    phrase_lead = lead_mono[start_sample:end_sample]
                    lead_energy = np.sqrt(np.mean(phrase_lead ** 2))
                if backing_mono is not None and end_sample <= len(backing_mono):
                    phrase_backing = backing_mono[start_sample:end_sample]
                    backing_energy = np.sqrt(np.mean(phrase_backing ** 2))
                
                # Check if aligned to loop boundary
                loop_position = (start % loop_info.loop_duration) / loop_info.loop_duration
                is_loop_aligned = loop_position < 0.1 or loop_position > 0.9
                
                # Calculate edge energies for transition matching
                # Use ~0.5 seconds at start and end for edge energy calculation
                edge_samples = int(0.5 * self.sample_rate)
                edge_samples = min(edge_samples, len(phrase_mono) // 4)  # Don't use more than 25% of phrase
                
                if edge_samples > 0:
                    start_edge = phrase_mono[:edge_samples]
                    end_edge = phrase_mono[-edge_samples:]
                    start_energy = np.sqrt(np.mean(start_edge ** 2))
                    end_energy = np.sqrt(np.mean(end_edge ** 2))
                else:
                    start_energy = energy
                    end_energy = energy
                
                # Calculate spectral centroid (brightness) for musical progression tracking
                spectral_cent = librosa.feature.spectral_centroid(y=phrase_mono, sr=self.sample_rate)[0]
                avg_centroid = np.mean(spectral_cent) if len(spectral_cent) > 0 else 1000.0
                
                # Extract chroma features from lead stem for harmonic analysis
                # Use lead_mono if available, otherwise phrase_mono
                chroma_audio = lead_mono[start_sample:end_sample] if lead_mono is not None and end_sample <= len(lead_mono) else phrase_mono
                avg_chroma, start_chroma, end_chroma, dominant_pitch = self._extract_chroma_features(
                    chroma_audio, edge_samples
                )
                
                # Extract rhythmic texture features (onset density, spectral flux)
                # Use lead stem for cleaner onset detection
                texture_audio = lead_mono[start_sample:end_sample] if lead_mono is not None and end_sample <= len(lead_mono) else phrase_mono
                onset_density, spectral_flux, end_onset_density = self._extract_rhythmic_texture(
                    texture_audio, edge_samples, duration
                )
                
                # Quality score based on energy consistency and alignment
                # When stems available, also factor in lead activity (more lead = more interesting)
                energy_var = np.var(librosa.feature.rms(y=phrase_mono)[0])
                quality = 1.0 - min(1.0, energy_var * 10)  # Lower variance = higher quality
                if is_loop_aligned:
                    quality += 0.1
                # Boost quality for sections with strong lead content
                if lead_energy > 0 and backing_energy > 0:
                    lead_ratio = lead_energy / (lead_energy + backing_energy)
                    quality += lead_ratio * 0.2  # Up to +0.2 for lead-heavy sections
                quality = min(1.0, quality)
                
                phrases.append(Phrase(
                    start_time=start,
                    end_time=end,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    bars=bars,
                    energy=energy,
                    quality_score=quality,
                    is_loop_aligned=is_loop_aligned,
                    lead_energy=lead_energy,
                    backing_energy=backing_energy,
                    start_energy=start_energy,
                    end_energy=end_energy,
                    spectral_centroid=avg_centroid,
                    avg_chroma=avg_chroma,
                    start_chroma=start_chroma,
                    end_chroma=end_chroma,
                    dominant_pitch_class=dominant_pitch,
                    onset_density=onset_density,
                    spectral_flux=spectral_flux,
                    end_onset_density=end_onset_density
                ))
            
            i = best_end
        
        logger.info(f"  Found {len(phrases)} phrases")
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, p in enumerate(phrases[:5]):  # Show first 5
            lead_info = f", lead={p.lead_energy:.3f}" if p.lead_energy > 0 else ""
            pitch_info = f", key~{pitch_names[p.dominant_pitch_class]}" if p.avg_chroma is not None else ""
            density_info = f", density={p.onset_density:.1f}/s" if p.onset_density > 0 else ""
            # Normalize brightness to 0-100% for readability
            bright_pct = min(100, max(0, p.spectral_centroid / 50))  # ~5000 Hz = 100%
            bright_info = f", bright={bright_pct:.0f}%"
            logger.info(f"    [{i}] {p.start_time:.1f}s - {p.end_time:.1f}s ({p.bars} bars, quality={p.quality_score:.2f}{lead_info}{pitch_info}{density_info}{bright_info})")
        if len(phrases) > 5:
            logger.info(f"    ... and {len(phrases) - 5} more")
        
        return phrases
    
    def _extract_chroma_features(
        self, 
        audio: np.ndarray, 
        edge_samples: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Extract chroma (pitch class) features from audio.
        
        Returns:
            avg_chroma: 12-element array of average pitch class energy
            start_chroma: chroma at phrase start
            end_chroma: chroma at phrase end  
            dominant_pitch_class: most prominent pitch (0-11)
        """
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(
                y=audio, 
                sr=self.sample_rate,
                hop_length=512,
                n_chroma=12
            )
            
            if chroma.size == 0:
                return None, None, None, 0
            
            # Average chroma across time
            avg_chroma = np.mean(chroma, axis=1)
            avg_chroma = avg_chroma / (np.sum(avg_chroma) + 1e-8)  # Normalize
            
            # Edge chroma for transition matching
            # Use ~0.5s at start/end (convert to frames)
            edge_frames = max(1, edge_samples // 512)
            
            start_chroma = np.mean(chroma[:, :edge_frames], axis=1)
            start_chroma = start_chroma / (np.sum(start_chroma) + 1e-8)
            
            end_chroma = np.mean(chroma[:, -edge_frames:], axis=1)
            end_chroma = end_chroma / (np.sum(end_chroma) + 1e-8)
            
            # Find dominant pitch class
            dominant_pitch = int(np.argmax(avg_chroma))
            
            return avg_chroma, start_chroma, end_chroma, dominant_pitch
            
        except Exception as e:
            logger.debug(f"Chroma extraction failed: {e}")
            return None, None, None, 0
    
    def _extract_rhythmic_texture(
        self,
        audio: np.ndarray,
        edge_samples: int,
        duration: float
    ) -> Tuple[float, float, float]:
        """
        Extract rhythmic texture features from audio.
        
        Returns:
            onset_density: onsets per second (how busy the phrase is)
            spectral_flux: average spectral change (sustained vs staccato)
            end_onset_density: onset density at phrase end
        """
        try:
            # Detect onsets (note attacks)
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, 
                sr=self.sample_rate,
                units='time'
            )
            
            # Overall onset density (onsets per second)
            onset_density = len(onsets) / duration if duration > 0 else 0.0
            
            # Spectral flux (how much the spectrum changes over time)
            # High flux = staccato/busy, low flux = sustained notes
            spectral_flux = np.mean(onset_env) if len(onset_env) > 0 else 0.0
            
            # End onset density (last ~1 second)
            # This helps match phrase endings to beginnings
            end_time = duration
            start_time = max(0, duration - 1.0)
            end_onsets = [o for o in onsets if start_time <= o <= end_time]
            end_onset_density = len(end_onsets) / (end_time - start_time) if (end_time - start_time) > 0 else onset_density
            
            return onset_density, spectral_flux, end_onset_density
            
        except Exception as e:
            logger.debug(f"Rhythmic texture extraction failed: {e}")
            return 0.0, 0.0, 0.0


class SongBuilder:
    """Builds a remix from detected phrases."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def build_remix(
        self,
        audio: np.ndarray,
        phrases: List[Phrase],
        loop_info: LoopInfo,
        target_duration: float = 300.0,
        crossfade_bars: float = 0.25
    ) -> np.ndarray:
        """
        Build a remix by selecting and arranging the best phrases.
        
        Args:
            audio: Original audio
            phrases: Detected phrases
            loop_info: Loop structure info
            target_duration: Target output duration in seconds
            crossfade_bars: Crossfade duration in bars (default 0.25 = ~1 beat)
            
        Returns:
            Remixed audio
        """
        logger.info(f"Building remix (target: {target_duration:.0f}s)...")
        
        if not phrases:
            logger.warning("No phrases to remix!")
            return audio
        
        # Sort phrases by quality
        sorted_phrases = sorted(phrases, key=lambda p: p.quality_score, reverse=True)
        
        # Select phrases to reach target duration
        selected = []
        current_duration = 0.0
        
        # Start with a good intro (lower energy, high quality)
        intro_candidates = [p for p in sorted_phrases if p.energy < np.median([x.energy for x in phrases])]
        if intro_candidates:
            intro = max(intro_candidates, key=lambda p: p.quality_score)
            selected.append(intro)
            current_duration += intro.end_time - intro.start_time
        
        # Add phrases until we reach target duration
        for phrase in sorted_phrases:
            if phrase in selected:
                continue
            
            phrase_duration = phrase.end_time - phrase.start_time
            if current_duration + phrase_duration > target_duration * 1.2:
                continue
            
            selected.append(phrase)
            current_duration += phrase_duration
            
            if current_duration >= target_duration:
                break
        
        # Reorder selected phrases for musical progression
        # Uses lookahead to evaluate sequences of 4-5 phrases for musical coherence
        if len(selected) >= 2:
            selected = self._order_for_musical_progression(selected)
            logger.info(f"  Reordered phrases for musical progression")
        
        logger.info(f"  Selected {len(selected)} phrases ({current_duration:.1f}s)")
        
        # Log transition energy matches
        if len(selected) >= 2:
            total_mismatch = 0
            for i in range(len(selected) - 1):
                end_e = selected[i].end_energy
                start_e = selected[i + 1].start_energy
                mismatch = abs(end_e - start_e)
                total_mismatch += mismatch
            avg_mismatch = total_mismatch / (len(selected) - 1)
            logger.info(f"  Average transition energy mismatch: {avg_mismatch:.4f}")
        
        # Build the output audio with crossfades
        crossfade_samples = int(crossfade_bars * loop_info.bar_duration * self.sample_rate)
        
        # Ensure stereo output
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        
        output_parts = []
        
        for i, phrase in enumerate(selected):
            phrase_audio = audio[phrase.start_sample:phrase.end_sample].copy()
            
            # Ensure stereo
            if phrase_audio.ndim == 1:
                phrase_audio = np.stack([phrase_audio, phrase_audio], axis=-1)
            
            if i == 0:
                # First phrase: just add it
                output_parts.append(phrase_audio)
            else:
                # Apply crossfade with previous
                prev_audio = output_parts[-1]
                
                # Determine crossfade length (limited by available audio)
                cf_len = min(crossfade_samples, len(prev_audio) // 4, len(phrase_audio) // 4)
                
                if cf_len > 100:
                    # Create crossfade with equal-power (cosine) curves
                    # This prevents volume dips during transitions
                    t = np.linspace(0, np.pi / 2, cf_len)
                    fade_out = np.cos(t)[:, None]  # cos(0)=1 -> cos(π/2)=0
                    fade_in = np.sin(t)[:, None]   # sin(0)=0 -> sin(π/2)=1
                    
                    # Crossfade region - equal power maintains perceived volume
                    crossfade = prev_audio[-cf_len:] * fade_out + phrase_audio[:cf_len] * fade_in
                    
                    # Trim previous and prepend crossfade to current
                    output_parts[-1] = prev_audio[:-cf_len]
                    output_parts.append(crossfade)
                    output_parts.append(phrase_audio[cf_len:])
                else:
                    output_parts.append(phrase_audio)
        
        # Concatenate all parts
        output = np.concatenate(output_parts, axis=0)
        
        # Apply fade in/out
        fade_samples = int(0.1 * self.sample_rate)  # 100ms fades
        if len(output) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)[:, None]
            fade_out = np.linspace(1, 0, fade_samples)[:, None]
            output[:fade_samples] *= fade_in
            output[-fade_samples:] *= fade_out
        
        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0.95:
            output = output * (0.95 / peak)
        
        logger.info(f"  Output duration: {len(output) / self.sample_rate:.1f}s")
        
        return output
    
    def _order_for_musical_progression(self, phrases: List[Phrase], lookahead: int = 4) -> List[Phrase]:
        """
        Reorder phrases to create a coherent musical progression.
        
        Uses lookahead to evaluate whether a sequence of phrases creates
        a sensible musical arc (build-up, peak, release patterns).
        
        Args:
            phrases: List of phrases to reorder
            lookahead: How many phrases ahead to consider (default 4)
            
        Returns:
            Reordered list of phrases
        """
        if len(phrases) <= 2:
            return phrases
        
        # Keep first phrase as intro (already selected for low energy/high quality)
        ordered = [phrases[0]]
        remaining = phrases[1:].copy()
        
        # Calculate energy and brightness ranges for normalization
        all_energies = [p.energy for p in phrases]
        all_centroids = [p.spectral_centroid for p in phrases]
        energy_range = max(all_energies) - min(all_energies) + 0.001
        centroid_range = max(all_centroids) - min(all_centroids) + 1.0
        
        def normalize_energy(e):
            return (e - min(all_energies)) / energy_range
        
        def normalize_centroid(c):
            return (c - min(all_centroids)) / centroid_range
        
        def score_sequence(sequence: List[Phrase]) -> float:
            """
            Score a sequence of phrases for musical coherence.
            
            Good progressions have:
            - Smooth energy transitions (small jumps)
            - Energy arcs (building or releasing, not random)
            - Brightness changes that match energy (brighter = more intense)
            - Harmonic compatibility (chroma similarity at transitions)
            - Sensible key/pitch progressions
            - Quality maintained throughout
            """
            if len(sequence) < 2:
                return 0.0
            
            score = 0.0
            
            # 1. Transition smoothness - energy matching at boundaries
            transition_penalty = 0.0
            for i in range(len(sequence) - 1):
                end_e = sequence[i].end_energy
                start_e = sequence[i + 1].start_energy
                avg_e = (end_e + start_e) / 2 + 0.001
                transition_penalty += abs(end_e - start_e) / avg_e
            transition_penalty /= (len(sequence) - 1)
            score -= transition_penalty * 0.3  # Penalize harsh transitions
            
            # 2. Harmonic transition smoothness - chroma matching at boundaries
            harmonic_score = 0.0
            harmonic_count = 0
            for i in range(len(sequence) - 1):
                curr_phrase = sequence[i]
                next_phrase = sequence[i + 1]
                
                if curr_phrase.end_chroma is not None and next_phrase.start_chroma is not None:
                    # Cosine similarity between end chroma of current and start chroma of next
                    chroma_sim = chroma_similarity(curr_phrase.end_chroma, next_phrase.start_chroma)
                    harmonic_score += chroma_sim
                    harmonic_count += 1
                    
            if harmonic_count > 0:
                harmonic_score /= harmonic_count
                score += harmonic_score * 0.25  # Reward harmonic continuity
            
            # 3. Key/pitch progression - favor musically related keys
            key_score = 0.0
            key_count = 0
            for i in range(len(sequence) - 1):
                curr_key = sequence[i].dominant_pitch_class
                next_key = sequence[i + 1].dominant_pitch_class
                
                # Calculate interval between keys
                interval = (next_key - curr_key) % 12
                
                # Reward musically related key movements:
                # - Same key (0): very good
                # - Perfect fifth (7) or fourth (5): classic progression
                # - Minor/major third (3, 4): smooth movement
                # - Whole step (2): common modulation
                key_bonus = {
                    0: 1.0,   # Same key
                    7: 0.9,   # Perfect 5th up (e.g., C->G)
                    5: 0.9,   # Perfect 4th up / 5th down (e.g., C->F)
                    4: 0.7,   # Major 3rd
                    3: 0.7,   # Minor 3rd
                    2: 0.6,   # Whole step
                    10: 0.6,  # Whole step down
                    9: 0.7,   # Minor 3rd down
                    8: 0.7,   # Major 3rd down
                }.get(interval, 0.3)  # Other intervals less favorable
                
                key_score += key_bonus
                key_count += 1
            
            if key_count > 0:
                key_score /= key_count
                score += key_score * 0.2  # Reward good key progressions
            
            # 4. Energy arc coherence - reward consistent direction
            energy_changes = []
            for i in range(len(sequence) - 1):
                e1 = normalize_energy(sequence[i].energy)
                e2 = normalize_energy(sequence[i + 1].energy)
                energy_changes.append(e2 - e1)
            
            # Reward if energy moves consistently in one direction (building or falling)
            if len(energy_changes) >= 2:
                # Check for consistent direction
                signs = [1 if c > 0.05 else (-1 if c < -0.05 else 0) for c in energy_changes]
                consistent = sum(1 for i in range(len(signs)-1) if signs[i] == signs[i+1] and signs[i] != 0)
                score += consistent * 0.1  # Reward consistent movement
            
            # 5. Peak/climax handling - high energy phrases should be surrounded by buildups
            for i, phrase in enumerate(sequence):
                norm_e = normalize_energy(phrase.energy)
                if norm_e > 0.7:  # High energy phrase (potential climax)
                    # Check if preceded by lower energy (buildup)
                    if i > 0:
                        prev_e = normalize_energy(sequence[i-1].energy)
                        if prev_e < norm_e:
                            score += 0.15  # Good buildup to peak
                    # Check if followed by lower energy (release)
                    if i < len(sequence) - 1:
                        next_e = normalize_energy(sequence[i+1].energy)
                        if next_e < norm_e:
                            score += 0.1  # Good release from peak
            
            # 6. Brightness-energy correlation - brighter phrases should be higher energy
            energy_list = [normalize_energy(p.energy) for p in sequence]
            bright_list = [normalize_centroid(p.spectral_centroid) for p in sequence]
            if len(energy_list) >= 2:
                correlation = np.corrcoef(energy_list, bright_list)[0, 1]
                if not np.isnan(correlation):
                    score += correlation * 0.1  # Reward brightness-energy correlation
            
            # 7. Quality bonus
            avg_quality = sum(p.quality_score for p in sequence) / len(sequence)
            score += avg_quality * 0.15
            
            # 8. Rhythmic texture continuity - penalize jarring changes in activity level
            # e.g., going from fast solo (high density) to held chord (low density)
            texture_penalty = 0.0
            texture_count = 0
            for i in range(len(sequence) - 1):
                curr_density = sequence[i].end_onset_density
                next_density = sequence[i + 1].onset_density
                
                if curr_density > 0 and next_density > 0:
                    # Calculate relative difference in onset density
                    # Large jumps (e.g., 8 onsets/s -> 1 onset/s) are penalized
                    ratio = max(curr_density, next_density) / (min(curr_density, next_density) + 0.1)
                    if ratio > 3.0:  # More than 3x difference is jarring
                        texture_penalty += (ratio - 3.0) * 0.1
                    texture_count += 1
            
            if texture_count > 0:
                texture_penalty /= texture_count
                score -= texture_penalty  # Penalize jarring texture changes
            
            # 9. Brightness continuity - penalize jarring brightness jumps at transitions
            # e.g., bright intense lead -> muddy noodling is jarring
            brightness_penalty = 0.0
            brightness_count = 0
            for i in range(len(sequence) - 1):
                curr_bright = normalize_centroid(sequence[i].spectral_centroid)
                next_bright = normalize_centroid(sequence[i + 1].spectral_centroid)
                
                # Penalize large brightness differences between adjacent phrases
                brightness_diff = abs(curr_bright - next_bright)
                if brightness_diff > 0.3:  # More than 30% brightness jump
                    # Quadratic penalty for extreme jumps
                    brightness_penalty += (brightness_diff - 0.3) ** 2 * 2.0
                brightness_count += 1
            
            if brightness_count > 0:
                brightness_penalty /= brightness_count
                score -= brightness_penalty  # Penalize jarring brightness changes
            
            return score
        
        def chroma_similarity(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
            """Compute cosine similarity between two chroma vectors."""
            dot = np.dot(chroma1, chroma2)
            norm1 = np.linalg.norm(chroma1)
            norm2 = np.linalg.norm(chroma2)
            if norm1 > 0 and norm2 > 0:
                return dot / (norm1 * norm2)
            return 0.0
        
        def evaluate_with_lookahead(current_sequence: List[Phrase], candidates: List[Phrase]) -> int:
            """
            Evaluate each candidate by looking ahead several phrases.
            Returns index of best candidate.
            """
            if not candidates:
                return 0
            
            best_idx = 0
            best_score = float('-inf')
            
            for i, candidate in enumerate(candidates):
                # Create hypothetical sequence with this candidate
                test_seq = current_sequence[-3:] + [candidate]  # Last 3 + candidate
                
                # If we have more candidates, try to project further
                if len(candidates) > 1:
                    other_candidates = [c for j, c in enumerate(candidates) if j != i]
                    # Greedily pick next few phrases for lookahead evaluation
                    lookahead_seq = test_seq.copy()
                    temp_remaining = other_candidates.copy()
                    
                    for _ in range(min(lookahead - 1, len(temp_remaining))):
                        if not temp_remaining:
                            break
                        # Pick next phrase by simple energy matching
                        last_end_e = lookahead_seq[-1].end_energy
                        best_next = min(temp_remaining, key=lambda p: abs(p.start_energy - last_end_e))
                        lookahead_seq.append(best_next)
                        temp_remaining.remove(best_next)
                    
                    test_seq = lookahead_seq
                
                # Score the projected sequence
                score = score_sequence(test_seq)
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            return best_idx
        
        # Build sequence using lookahead
        while remaining:
            best_idx = evaluate_with_lookahead(ordered, remaining)
            ordered.append(remaining.pop(best_idx))
        
        return ordered


def remix_loop_jam(
    input_path: str,
    output_path: str,
    target_duration: float = 300.0,
    min_phrase_bars: int = 4,
    max_phrase_bars: int = 16,
    use_demucs: bool = True,
    demucs_device: str = 'cuda'
) -> None:
    """
    Main function to remix a loop jam recording.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        target_duration: Target output duration in seconds
        min_phrase_bars: Minimum phrase length in bars
        max_phrase_bars: Maximum phrase length in bars
        use_demucs: Whether to use Demucs for stem separation
        demucs_device: Device for Demucs ('cuda' or 'cpu')
    """
    logger.info(f"Loading: {input_path}")
    audio, sr = sf.read(input_path)
    
    # Ensure float32
    audio = audio.astype(np.float32)
    
    duration = len(audio) / sr
    logger.info(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"  Sample rate: {sr}")
    
    # Optionally separate stems
    stems = None
    if use_demucs:
        stems = separate_stems(input_path, device=demucs_device)
        # Resample stems to match audio if needed
        if stems is not None and stems.sample_rate != sr:
            logger.info(f"  Resampling stems from {stems.sample_rate} to {sr}")
            for attr in ['drums', 'bass', 'vocals', 'other']:
                stem = getattr(stems, attr)
                if stem is not None:
                    if stem.ndim == 1:
                        resampled = librosa.resample(stem, orig_sr=stems.sample_rate, target_sr=sr)
                    else:
                        resampled = np.stack([librosa.resample(stem[:, ch], orig_sr=stems.sample_rate, target_sr=sr) 
                                            for ch in range(stem.shape[1])], axis=-1)
                    setattr(stems, attr, resampled)
            stems.sample_rate = sr
    
    # Analyze
    analyzer = LoopAnalyzer(sample_rate=sr)
    loop_info = analyzer.analyze(audio)
    
    # Detect phrases (using stems if available)
    detector = PhraseDetector(sample_rate=sr)
    phrases = detector.detect_phrases(audio, loop_info, min_phrase_bars, max_phrase_bars, stems=stems)
    
    # Build remix
    builder = SongBuilder(sample_rate=sr)
    remix = builder.build_remix(audio, phrases, loop_info, target_duration)
    
    # Save
    logger.info(f"Saving: {output_path}")
    sf.write(output_path, remix, sr)
    
    logger.info("Done!")

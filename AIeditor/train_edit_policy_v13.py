"""
Edit Policy V13 - Guitar-Focused Beat-Level Editing with Stem Separation

Based on V12's beat-level editing with phrase context, but uses Demucs stem separation
to focus on the "other" stem (guitar/lead instruments) for more coherent melodic phrases.

Key innovations over V12:
1. STEM SEPARATION: Uses Demucs to isolate guitar/lead from drums/bass/vocals
2. GUITAR-FOCUSED FEATURES: Extract spectral features from "other" stem
3. MELODIC PHRASE AWARENESS: Model learns guitar phrase boundaries, not drum hits
4. PHRASE CONTEXT: Model sees surrounding bars for musical phrase awareness  
5. DOWNBEAT PREFERENCE: Cuts prefer landing on bar boundaries (beat 1)

Architecture:
- Demucs stem separation (htdemucs model)
- Beat-level feature extraction from "other" stem
- Phrase context window (60 beats bidirectional LSTM)
- Position encoding: beat-in-bar, bar-in-phrase, position-in-song
- Same imitation learning + trajectory LSTM from V12

This should produce more musical edits that respect melodic phrase structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import random
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib
import warnings
from dataclasses import dataclass
import sys

# Add shared module path for Demucs wrapper
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from demucs_wrapper import DemucsSeparator

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_WORKERS = max(1, os.cpu_count() - 2)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()  # Mixed precision on GPU

# Feature cache directory - V13 uses separate cache for stem-based features
CACHE_DIR = Path("./feature_cache/v13")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Stem cache directory for Demucs-separated stems
STEM_CACHE_DIR = Path("./feature_cache/v13_stems")
STEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STEM SEPARATION (V13 ADDITION)
# =============================================================================

class StemSeparator:
    """
    Manages Demucs stem separation with caching.
    
    For guitar-focused editing, we combine vocals + other stems.
    This isolates the melodic content (guitar, keys, vocals) from
    drums and bass, allowing the model to learn phrase boundaries
    based on melodic content rather than rhythm section.
    """
    
    def __init__(self, sr: int = 22050, cache_dir: Path = STEM_CACHE_DIR):
        self.sr = sr
        self.cache_dir = cache_dir
        self._separator = None  # Lazy initialization
        
    @property
    def separator(self):
        """Lazy load Demucs separator (heavy model)."""
        if self._separator is None:
            logger.info("Loading Demucs separator (htdemucs)...")
            self._separator = DemucsSeparator(model="htdemucs")
            logger.info("Demucs loaded.")
        return self._separator
    
    def _get_cache_path(self, audio_path: Path) -> Path:
        """Get cache path for separated stems."""
        # Create hash from file path and modification time
        stat = audio_path.stat()
        cache_key = f"{audio_path.name}_{stat.st_mtime}_{stat.st_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{audio_path.stem}_{cache_hash}_stems.npz"
    
    def separate_stems(self, audio: np.ndarray, audio_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs.
        
        Returns dict with keys: 'drums', 'bass', 'other', 'vocals'
        Each stem is a numpy array at self.sr sample rate (mono).
        """
        # Check cache if path provided
        if audio_path is not None:
            cache_path = self._get_cache_path(audio_path)
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    logger.info(f"Loaded cached stems for {audio_path.name}")
                    return {k: cached[k] for k in cached.files}
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
        
        # Write audio to temp file (Demucs requires file input)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, self.sr)
        
        try:
            # Perform separation - resample output to our sr
            stems_raw = self.separator.separate(tmp_path, resample_to=self.sr)
            
            # Convert to mono if stereo and ensure consistent format
            stems = {}
            for key in ['drums', 'bass', 'other', 'vocals']:
                if key in stems_raw:
                    stem = stems_raw[key]
                    # Convert to mono if stereo
                    if stem.ndim == 2:
                        stem = stem.mean(axis=1)
                    stems[key] = stem.astype(np.float32)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # Cache results if path provided
        if audio_path is not None:
            try:
                np.savez_compressed(cache_path, **stems)
                logger.info(f"Cached stems for {audio_path.name}")
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
        
        return stems
    
    def get_guitar_focused_audio(self, audio: np.ndarray, audio_path: Optional[Path] = None) -> np.ndarray:
        """
        Get the guitar/melodic-focused audio by combining vocals + other stems.
        
        This removes drums and bass, leaving melodic content (guitar, keys, vocals).
        For instrumental music, this is mostly the "other" stem (guitar/lead).
        """
        stems = self.separate_stems(audio, audio_path)
        
        # Combine vocals + other for melodic content
        # This gives us: lead guitar, keys, synths, vocal melodies
        # We exclude: drums, bass (rhythm section)
        melodic = stems['vocals'] + stems['other']
        
        return melodic
    
    def get_all_stems_combined(self, audio: np.ndarray, audio_path: Optional[Path] = None,
                                include_drums: bool = False, include_bass: bool = False) -> np.ndarray:
        """
        Get customizable stem combination.
        
        By default returns vocals + other (melodic content).
        Can optionally include drums and/or bass.
        """
        stems = self.separate_stems(audio, audio_path)
        
        result = stems['vocals'] + stems['other']
        if include_drums:
            result = result + stems['drums']
        if include_bass:
            result = result + stems['bass']
        
        return result


# =============================================================================
# BEAT DETECTION & ANALYSIS
# =============================================================================

@dataclass
class BeatInfo:
    """Information about detected beats in a track."""
    beat_times: np.ndarray      # Time of each beat in seconds
    downbeat_mask: np.ndarray   # True for downbeats (beat 1 of bar)
    tempo: float                # Estimated BPM
    time_signature: int         # Beats per bar (usually 4)
    bar_indices: np.ndarray     # Which bar each beat belongs to
    beat_in_bar: np.ndarray     # Position within bar (0, 1, 2, 3 for 4/4)


def detect_beats(audio: np.ndarray, sr: int = 22050) -> BeatInfo:
    """
    Detect beats and estimate bar structure.
    Uses librosa's beat tracker with downbeat detection.
    """
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    if len(beat_times) < 4:
        # Too few beats, create artificial ones
        duration = len(audio) / sr
        tempo_val = 120.0
        beat_times = np.arange(0, duration, 60.0 / tempo_val)
        tempo = tempo_val
    
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    
    # Estimate time signature (assume 4/4 for now)
    time_sig = 4
    
    # Detect downbeats using onset strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
    n_beats = len(beat_times)
    downbeat_mask = np.zeros(n_beats, dtype=bool)
    beat_in_bar = np.zeros(n_beats, dtype=int)
    bar_indices = np.zeros(n_beats, dtype=int)
    
    # Find best phase for downbeats by checking onset strength
    best_phase = 0
    best_strength = 0
    for phase in range(time_sig):
        phase_beats = np.arange(phase, n_beats, time_sig)
        if len(phase_beats) > 0:
            phase_frames = beat_frames[phase_beats[phase_beats < len(beat_frames)]]
            if len(phase_frames) > 0:
                valid_frames = phase_frames[phase_frames < len(onset_env)]
                if len(valid_frames) > 0:
                    strength = np.mean(onset_env[valid_frames])
                    if strength > best_strength:
                        best_strength = strength
                        best_phase = phase
    
    # Assign bar structure
    current_bar = 0
    for i in range(n_beats):
        beat_pos = (i - best_phase) % time_sig
        if beat_pos < 0:
            beat_pos += time_sig
        
        beat_in_bar[i] = beat_pos
        downbeat_mask[i] = (beat_pos == 0)
        bar_indices[i] = current_bar
        
        if beat_pos == time_sig - 1:
            current_bar += 1
    
    return BeatInfo(
        beat_times=beat_times,
        downbeat_mask=downbeat_mask,
        tempo=tempo_val,
        time_signature=time_sig,
        bar_indices=bar_indices,
        beat_in_bar=beat_in_bar
    )


def get_beat_audio(audio: np.ndarray, sr: int, beat_times: np.ndarray, 
                   beat_idx: int) -> np.ndarray:
    """Extract audio for a single beat."""
    n_beats = len(beat_times)
    
    start_time = beat_times[beat_idx]
    if beat_idx < n_beats - 1:
        end_time = beat_times[beat_idx + 1]
    else:
        # Estimate based on tempo
        if n_beats > 1:
            avg_beat_dur = np.mean(np.diff(beat_times))
        else:
            avg_beat_dur = 0.5
        end_time = start_time + avg_beat_dur
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample]


# =============================================================================
# FAST FEATURE EXTRACTOR (optimized, no tempo - it's slow)
# =============================================================================

def extract_features_fast(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extract features from audio segment - FAST version.
    Removed tempo detection (slow) and optimized librosa calls.
    """
    features = []
    
    # Adaptive n_fft based on audio length (fixes warning for short segments)
    n_fft = min(2048, len(audio))
    if n_fft < 512:
        # Audio too short, pad it
        audio = np.pad(audio, (0, 512 - len(audio)))
        n_fft = 512
    hop_length = min(512, n_fft // 4)
    
    # Pre-compute STFT once (reused by multiple features)
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # MFCCs from pre-computed spectrogram (26 features)
    mel_spec = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))
    
    # Spectral features from pre-computed STFT (4 features)
    spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(S=S)
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    
    features.append(float(spec_cent.mean()))
    features.append(float(spec_bw.mean()))
    features.append(float(spec_flat.mean()))
    features.append(float(spec_rolloff.mean()))
    
    # RMS energy (2 features) - compute from audio directly to avoid frame_length issues
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    features.append(float(rms.mean()))
    features.append(float(rms.std()) if rms.size > 1 else 0.0)
    
    # Zero crossing rate (1 feature)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
    features.append(float(zcr.mean()))
    
    # Chroma from STFT (12 features)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    features.extend(chroma.mean(axis=1).tolist())
    
    # Spectral contrast (7 features) - need at least 7 frequency bands
    if S.shape[0] >= 7:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
        features.extend(contrast.mean(axis=1).tolist())
    else:
        features.extend([0.0] * 7)
    
    # Onset strength (2 features)
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr)
    features.append(float(onset_env.mean()))
    features.append(float(onset_env.std()) if len(onset_env) > 1 else 0.0)
    
    # Spectral flux (2 features)
    if S.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        features.append(float(flux.mean()))
        features.append(float(flux.std()) if len(flux) > 1 else 0.0)
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def _extract_segment_features_worker(args):
    """Worker function for parallel feature extraction."""
    segment, sr = args
    try:
        return extract_features_fast(segment, sr)
    except Exception as e:
        return None


def extract_features_parallel(segments: List[np.ndarray], sr: int = 22050, 
                             max_workers: int = None) -> np.ndarray:
    """Extract features from multiple segments in parallel."""
    if max_workers is None:
        max_workers = NUM_WORKERS
    
    # For small batches, don't bother with parallelism
    if len(segments) < 10:
        return np.array([extract_features_fast(seg, sr) for seg in segments])
    
    args_list = [(seg, sr) for seg in segments]
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_segment_features_worker, args) 
                   for args in args_list]
        
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                # Fallback to zeros if extraction failed
                results.append(np.zeros(56, dtype=np.float32))
    
    return np.array(results)


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def get_file_hash(path: Path) -> str:
    """Get a hash of file path and modification time for caching."""
    stat = path.stat()
    key = f"{path.absolute()}_{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_cached_features(path: Path) -> Optional[np.ndarray]:
    """Load cached features if available."""
    cache_file = CACHE_DIR / f"{get_file_hash(path)}.npy"
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except:
            pass
    return None


def save_cached_features(path: Path, features: np.ndarray):
    """Save features to cache."""
    cache_file = CACHE_DIR / f"{get_file_hash(path)}.npy"
    try:
        np.save(cache_file, features)
    except:
        pass


# =============================================================================
# AUDIO LOADING (optimized)
# =============================================================================

def load_audio_fast(path: Path, sr: int = 22050) -> np.ndarray:
    """Load and normalize audio - optimized."""
    # Use soundfile for wav (faster), librosa for mp3
    try:
        if path.suffix.lower() == '.wav':
            audio, file_sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        else:
            audio, _ = librosa.load(str(path), sr=sr, mono=True)
    except:
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
    
    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    
    return audio.astype(np.float32)


# =============================================================================
# ALIGNMENT: Find which raw segments were kept in the edit
# =============================================================================

class FastAudioAligner:
    """Align raw input to edited output - OPTIMIZED."""
    
    def __init__(self, sr: int = 22050, segment_duration: float = 3.0, 
                 hop_duration: float = 1.5):
        self.sr = sr
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.segment_samples = int(segment_duration * sr)
        self.hop_samples = int(hop_duration * sr)
    
    def extract_segments(self, audio: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract overlapping segments using numpy slicing."""
        n_segments = (len(audio) - self.segment_samples) // self.hop_samples + 1
        
        if n_segments <= 0:
            return [], []
        
        segments = []
        positions = []
        
        for i in range(n_segments):
            start = i * self.hop_samples
            end = start + self.segment_samples
            segments.append(audio[start:end])
            positions.append((start, end))
        
        return segments, positions
    
    def align(self, raw_audio: np.ndarray, edit_audio: np.ndarray,
              raw_path: Path = None, edit_path: Path = None,
              verbose: bool = False) -> Dict:
        """
        Align raw input to edited output using feature similarity.
        Uses caching for faster repeated runs.
        """
        # Extract segments
        raw_segments, raw_positions = self.extract_segments(raw_audio)
        edit_segments, edit_positions = self.extract_segments(edit_audio)
        
        if verbose:
            print(f"  Raw segments: {len(raw_segments)}, Edit segments: {len(edit_segments)}")
        
        # Try to load cached features
        raw_features = None
        edit_features = None
        
        if raw_path:
            raw_features = load_cached_features(raw_path)
        if edit_path:
            edit_features = load_cached_features(edit_path)
        
        # Extract features (parallel) if not cached
        if raw_features is None or len(raw_features) != len(raw_segments):
            if verbose:
                print(f"  Extracting raw features (parallel)...")
            raw_features = extract_features_parallel(raw_segments, self.sr)
            if raw_path:
                save_cached_features(raw_path, raw_features)
        else:
            if verbose:
                print(f"  Using cached raw features")
        
        if edit_features is None or len(edit_features) != len(edit_segments):
            if verbose:
                print(f"  Extracting edit features (parallel)...")
            edit_features = extract_features_parallel(edit_segments, self.sr)
            if edit_path:
                save_cached_features(edit_path, edit_features)
        else:
            if verbose:
                print(f"  Using cached edit features")
        
        # Expected keep ratio
        expected_keep_ratio = len(edit_audio) / len(raw_audio)
        expected_n_keep = int(len(raw_segments) * expected_keep_ratio)
        
        if verbose:
            print(f"  Expected keep ratio: {expected_keep_ratio:.1%} ({expected_n_keep} segments)")
        
        # Normalize for cosine similarity (vectorized)
        raw_norm = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-8)
        edit_norm = edit_features / (np.linalg.norm(edit_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix (batch operation)
        similarity_matrix = raw_norm @ edit_norm.T  # (n_raw, n_edit)
        
        # Best match score for each raw segment
        match_scores = similarity_matrix.max(axis=1).astype(np.float32)
        
        # Adaptive threshold based on expected keep count
        sorted_scores = np.sort(match_scores)[::-1]
        n_keep_target = max(1, int(expected_n_keep * 1.1))  # 10% margin
        
        if n_keep_target < len(sorted_scores):
            adaptive_threshold = sorted_scores[n_keep_target]
        else:
            adaptive_threshold = sorted_scores[-1] if len(sorted_scores) > 0 else 0.9
        
        # Require minimum absolute threshold
        final_threshold = max(adaptive_threshold, 0.92)
        
        keep_labels = (match_scores >= final_threshold).astype(np.float32)
        
        keep_ratio = keep_labels.mean()
        if verbose:
            print(f"  Threshold: {final_threshold:.3f}, Keep ratio: {keep_ratio:.1%} ({int(keep_labels.sum())}/{len(keep_labels)})")
        
        return {
            'raw_features': raw_features,
            'keep_labels': keep_labels,
            'match_scores': match_scores,
            'positions': raw_positions,
            'keep_ratio': keep_ratio,
            'expected_keep_ratio': expected_keep_ratio
        }


# =============================================================================
# BEAT-LEVEL ALIGNMENT (V12 ADDITION)
# =============================================================================

class BeatLevelAligner:
    """
    Align raw input to edited output at the beat level.
    
    Uses beat detection to segment audio into individual beats,
    then matches beats between raw and edit using chroma features.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract_beat_features(self, beat_audio: np.ndarray) -> np.ndarray:
        """Extract features from a single beat (handles short segments)."""
        return extract_features_fast(beat_audio, self.sr)
    
    def get_beats_audio(self, audio: np.ndarray, beat_info: BeatInfo) -> List[np.ndarray]:
        """Extract audio for each beat."""
        beats = []
        beat_samples = (beat_info.beat_times * self.sr).astype(int)
        
        for i in range(len(beat_samples) - 1):
            start = beat_samples[i]
            end = beat_samples[i + 1]
            if end > len(audio):
                end = len(audio)
            if start < len(audio):
                beats.append(audio[start:end])
        
        # Last beat to end of audio
        if len(beat_samples) > 0 and beat_samples[-1] < len(audio):
            beats.append(audio[beat_samples[-1]:])
        
        return beats
    
    def align_beats(self, raw_audio: np.ndarray, edit_audio: np.ndarray,
                   raw_beats: BeatInfo, edit_beats: BeatInfo,
                   verbose: bool = False) -> Dict:
        """
        Align raw beats to edit beats using feature similarity.
        
        Returns per-beat keep/cut labels and phrase-level context.
        """
        # Get beat audio
        raw_beat_audio = self.get_beats_audio(raw_audio, raw_beats)
        edit_beat_audio = self.get_beats_audio(edit_audio, edit_beats)
        
        n_raw = len(raw_beat_audio)
        n_edit = len(edit_beat_audio)
        
        if verbose:
            print(f"  Raw beats: {n_raw}, Edit beats: {n_edit}")
        
        if n_raw == 0 or n_edit == 0:
            # Fallback if beat detection failed
            return {
                'beat_features': np.zeros((1, 56), dtype=np.float32),
                'keep_labels': np.array([1.0]),
                'beat_positions': [(0, len(raw_audio))],
                'beat_info': raw_beats,
                'n_beats': 1
            }
        
        # Extract features for all beats
        raw_features = np.array([self.extract_beat_features(b) for b in raw_beat_audio])
        edit_features = np.array([self.extract_beat_features(b) for b in edit_beat_audio])
        
        # Expected keep ratio
        expected_keep_ratio = len(edit_audio) / len(raw_audio)
        expected_n_keep = int(n_raw * expected_keep_ratio)
        
        if verbose:
            print(f"  Expected keep ratio: {expected_keep_ratio:.1%} ({expected_n_keep} beats)")
        
        # Normalize for cosine similarity
        raw_norm = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-8)
        edit_norm = edit_features / (np.linalg.norm(edit_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = raw_norm @ edit_norm.T  # (n_raw, n_edit)
        
        # Best match score for each raw beat
        match_scores = similarity_matrix.max(axis=1).astype(np.float32)
        
        # Adaptive threshold
        sorted_scores = np.sort(match_scores)[::-1]
        n_keep_target = max(1, int(expected_n_keep * 1.1))
        
        if n_keep_target < len(sorted_scores):
            adaptive_threshold = sorted_scores[n_keep_target]
        else:
            adaptive_threshold = sorted_scores[-1] if len(sorted_scores) > 0 else 0.9
        
        final_threshold = max(adaptive_threshold, 0.88)  # Lower threshold for beats
        keep_labels = (match_scores >= final_threshold).astype(np.float32)
        
        # Beat positions in samples
        beat_samples = (raw_beats.beat_times * self.sr).astype(int)
        beat_positions = []
        for i in range(len(beat_samples) - 1):
            beat_positions.append((beat_samples[i], beat_samples[i + 1]))
        if len(beat_samples) > 0:
            beat_positions.append((beat_samples[-1], len(raw_audio)))
        
        keep_ratio = keep_labels.mean()
        if verbose:
            print(f"  Threshold: {final_threshold:.3f}, Keep ratio: {keep_ratio:.1%} ({int(keep_labels.sum())}/{len(keep_labels)})")
        
        return {
            'beat_features': raw_features,
            'keep_labels': keep_labels,
            'match_scores': match_scores,
            'beat_positions': beat_positions[:n_raw],
            'beat_info': raw_beats,
            'n_beats': n_raw,
            'keep_ratio': keep_ratio,
            'expected_keep_ratio': expected_keep_ratio
        }
    
    def align_beats_with_stems(self, raw_melodic: np.ndarray, edit_melodic: np.ndarray,
                               raw_beats: BeatInfo, edit_beats: BeatInfo,
                               verbose: bool = False) -> Dict:
        """
        V13: Align raw beats to edit beats using STEM-SEPARATED melodic audio.
        
        Same as align_beats, but audio is already stem-separated (vocals + other).
        Beat times come from full mix, but features are extracted from melodic content.
        This focuses alignment on guitar/lead phrases rather than drums.
        
        Args:
            raw_melodic: Melodic content (vocals + other) from raw audio
            edit_melodic: Melodic content (vocals + other) from edit audio  
            raw_beats: Beat info detected from FULL raw mix
            edit_beats: Beat info detected from FULL edit mix
        """
        # Get beat audio from melodic content using beat times from full mix
        raw_beat_audio = self.get_beats_audio(raw_melodic, raw_beats)
        edit_beat_audio = self.get_beats_audio(edit_melodic, edit_beats)
        
        n_raw = len(raw_beat_audio)
        n_edit = len(edit_beat_audio)
        
        if verbose:
            print(f"  Raw beats: {n_raw}, Edit beats: {n_edit} (melodic features)")
        
        if n_raw == 0 or n_edit == 0:
            # Fallback if beat detection failed
            return {
                'beat_features': np.zeros((1, 56), dtype=np.float32),
                'keep_labels': np.array([1.0]),
                'beat_positions': [(0, len(raw_melodic))],
                'beat_info': raw_beats,
                'n_beats': 1
            }
        
        # Extract features from MELODIC content (guitar/lead focused)
        # Use parallel extraction for larger beat counts
        if n_raw > 20:
            raw_features = extract_features_parallel(raw_beat_audio, self.sr)
        else:
            raw_features = np.array([self.extract_beat_features(b) for b in raw_beat_audio])
        
        if n_edit > 20:
            edit_features = extract_features_parallel(edit_beat_audio, self.sr)
        else:
            edit_features = np.array([self.extract_beat_features(b) for b in edit_beat_audio])
        
        # Expected keep ratio
        expected_keep_ratio = len(edit_melodic) / len(raw_melodic)
        expected_n_keep = int(n_raw * expected_keep_ratio)
        
        if verbose:
            print(f"  Expected keep ratio: {expected_keep_ratio:.1%} ({expected_n_keep} beats)")
        
        # Normalize for cosine similarity
        raw_norm = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-8)
        edit_norm = edit_features / (np.linalg.norm(edit_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = raw_norm @ edit_norm.T  # (n_raw, n_edit)
        
        # Best match score for each raw beat
        match_scores = similarity_matrix.max(axis=1).astype(np.float32)
        
        # Adaptive threshold
        sorted_scores = np.sort(match_scores)[::-1]
        n_keep_target = max(1, int(expected_n_keep * 1.1))
        
        if n_keep_target < len(sorted_scores):
            adaptive_threshold = sorted_scores[n_keep_target]
        else:
            adaptive_threshold = sorted_scores[-1] if len(sorted_scores) > 0 else 0.9
        
        final_threshold = max(adaptive_threshold, 0.88)  # Lower threshold for beats
        keep_labels = (match_scores >= final_threshold).astype(np.float32)
        
        # Beat positions in samples (from melodic audio)
        beat_samples = (raw_beats.beat_times * self.sr).astype(int)
        beat_positions = []
        for i in range(len(beat_samples) - 1):
            beat_positions.append((beat_samples[i], beat_samples[i + 1]))
        if len(beat_samples) > 0:
            beat_positions.append((beat_samples[-1], len(raw_melodic)))
        
        keep_ratio = keep_labels.mean()
        if verbose:
            print(f"  Threshold: {final_threshold:.3f}, Keep ratio: {keep_ratio:.1%} ({int(keep_labels.sum())}/{len(keep_labels)})")
        
        return {
            'beat_features': raw_features,  # Features from melodic content
            'keep_labels': keep_labels,
            'match_scores': match_scores,
            'beat_positions': beat_positions[:n_raw],
            'beat_info': raw_beats,
            'n_beats': n_raw,
            'keep_ratio': keep_ratio,
            'expected_keep_ratio': expected_keep_ratio
        }


# =============================================================================
# MODEL: Imitation Policy with Trajectory Context
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """LSTM that encodes the editing trajectory."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.feature_compress = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=65,  # 64 features + 1 action
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
    
    def forward(self, features: torch.Tensor, actions: torch.Tensor, 
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        if features.dim() == 2:
            features = features.unsqueeze(1)
            actions = actions.unsqueeze(1)
        
        compressed = self.feature_compress(features)
        actions_expanded = actions.unsqueeze(-1).float()
        lstm_input = torch.cat([compressed, actions_expanded], dim=-1)
        
        if hidden is None:
            output, hidden = self.lstm(lstm_input)
        else:
            output, hidden = self.lstm(lstm_input, hidden)
        
        return output[:, -1, :], hidden


# Default context: 60 beats ≈ 30 seconds at 120 BPM
DEFAULT_CONTEXT_BEATS = 60


class PhraseContextEncoder(nn.Module):
    """
    Bidirectional LSTM that provides phrase-level context for beat-level decisions.
    
    Looks at a window of beats (default 60 beats ≈ 30 seconds at 120 BPM) on each side
    to understand the musical phrase structure.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128, context_beats: int = DEFAULT_CONTEXT_BEATS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_beats = context_beats  # Beats on each side
        
        # Compress audio features
        self.feature_compress = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
        )
        
        # Position encoding: beat-in-bar (4), bar position, song position, is_downbeat
        self.position_dim = 4 + 1 + 1 + 1  # 7 total
        
        # Bidirectional LSTM for phrase context
        self.lstm = nn.LSTM(
            input_size=64 + self.position_dim,  # features + position encoding
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Output projection (combines forward and backward)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def encode_beat_position(self, beat_idx: int, beat_info: Optional['BeatInfo'],
                            n_total_beats: int, device: torch.device) -> torch.Tensor:
        """
        Encode the rhythmic position of a beat.
        
        Returns: tensor of shape (position_dim,)
        """
        # Default values
        beat_in_bar = [0.25, 0.25, 0.25, 0.25]  # One-hot for beat 1-4
        bar_position = beat_idx / max(n_total_beats, 1)
        song_position = beat_idx / max(n_total_beats, 1)
        is_downbeat = 0.0
        
        if beat_info is not None and beat_idx < len(beat_info.beat_in_bar):
            # Beat position in bar (1-4 for 4/4 time)
            bib = beat_info.beat_in_bar[beat_idx]
            beat_in_bar = [1.0 if i + 1 == bib else 0.0 for i in range(4)]
            
            # Is this a downbeat?
            is_downbeat = 1.0 if beat_info.downbeat_mask[beat_idx] else 0.0
            
            # Bar position within song
            if beat_info.bar_indices is not None and beat_idx < len(beat_info.bar_indices):
                bar_idx = beat_info.bar_indices[beat_idx]
                max_bar = beat_info.bar_indices.max() if len(beat_info.bar_indices) > 0 else 1
                bar_position = bar_idx / max(max_bar, 1)
        
        position = beat_in_bar + [bar_position, song_position, is_downbeat]
        return torch.tensor(position, dtype=torch.float32, device=device)
    
    def forward(self, all_features: torch.Tensor, center_idx: int,
                beat_info: Optional['BeatInfo'] = None) -> torch.Tensor:
        """
        Get phrase context for a specific beat.
        
        Args:
            all_features: (n_beats, feature_dim) all beat features in the track
            center_idx: index of the beat we want context for
            beat_info: optional BeatInfo with rhythmic annotations
        
        Returns:
            context: (hidden_dim,) phrase context embedding
        """
        n_beats = all_features.shape[0]
        device = all_features.device
        
        # Context window indices
        start_idx = max(0, center_idx - self.context_beats)
        end_idx = min(n_beats, center_idx + self.context_beats + 1)
        
        # Extract context window features
        context_features = all_features[start_idx:end_idx]  # (window_size, feature_dim)
        window_size = context_features.shape[0]
        
        # Compress features
        compressed = self.feature_compress(context_features)  # (window_size, 64)
        
        # Add position encoding
        positions = []
        for i in range(start_idx, end_idx):
            pos = self.encode_beat_position(i, beat_info, n_beats, device)
            positions.append(pos)
        positions = torch.stack(positions, dim=0)  # (window_size, position_dim)
        
        # Combine features and positions
        lstm_input = torch.cat([compressed, positions], dim=-1)  # (window_size, 64 + position_dim)
        lstm_input = lstm_input.unsqueeze(0)  # (1, window_size, ...)
        
        # Run bidirectional LSTM
        output, _ = self.lstm(lstm_input)  # (1, window_size, hidden_dim * 2)
        
        # Get the output at the center position (relative to window)
        center_in_window = center_idx - start_idx
        center_output = output[0, center_in_window, :]  # (hidden_dim * 2,)
        
        # Project to final output
        context = self.output_proj(center_output)  # (hidden_dim,)
        
        return context
    
    def forward_batch(self, all_features: torch.Tensor, 
                     beat_info: Optional['BeatInfo'] = None) -> torch.Tensor:
        """
        Get phrase context for all beats at once (more efficient).
        
        Args:
            all_features: (n_beats, feature_dim) all beat features
            beat_info: optional BeatInfo
        
        Returns:
            contexts: (n_beats, hidden_dim) context for each beat
        """
        n_beats = all_features.shape[0]
        device = all_features.device
        
        # Pad features for context at edges
        pad_size = self.context_beats
        # Repeat edge beats for padding
        pad_start = all_features[0:1].repeat(pad_size, 1)
        pad_end = all_features[-1:].repeat(pad_size, 1)
        padded_features = torch.cat([pad_start, all_features, pad_end], dim=0)
        
        # Compress all features
        compressed = self.feature_compress(padded_features)  # (n_beats + 2*pad, 64)
        
        # Create position encodings for all beats
        positions = []
        for i in range(-pad_size, n_beats + pad_size):
            actual_i = max(0, min(n_beats - 1, i))  # Clamp to valid range for beat_info
            pos = self.encode_beat_position(actual_i, beat_info, n_beats, device)
            positions.append(pos)
        positions = torch.stack(positions, dim=0)  # (n_beats + 2*pad, position_dim)
        
        # Combine
        lstm_input = torch.cat([compressed, positions], dim=-1)
        lstm_input = lstm_input.unsqueeze(0)  # (1, n_beats + 2*pad, ...)
        
        # Run bidirectional LSTM on full sequence
        output, _ = self.lstm(lstm_input)  # (1, n_beats + 2*pad, hidden_dim * 2)
        
        # Extract context for original beats (skip padding)
        output = output[0, pad_size:pad_size + n_beats, :]  # (n_beats, hidden_dim * 2)
        
        # Project
        contexts = self.output_proj(output)  # (n_beats, hidden_dim)
        
        return contexts


class ImitationPolicy(nn.Module):
    """Policy network that learns from human editing decisions."""
    
    def __init__(self, feature_dim: int, style_dim: int = 64, hidden_dim: int = 256,
                 dropout: float = 0.4):
        super().__init__()
        self.feature_dim = feature_dim
        self.style_dim = style_dim
        
        self.trajectory_encoder = TrajectoryEncoder(feature_dim, hidden_dim=128)
        self.style_proj = nn.Linear(feature_dim, style_dim)
        
        # Input: features + trajectory + position + style
        policy_input_dim = feature_dim + 128 + 3 + style_dim
        
        # Increased dropout for better generalization
        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),  # Less dropout near output
            nn.Linear(64, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_style_embedding(self, features: torch.Tensor) -> torch.Tensor:
        return self.style_proj(features)
    
    def forward(self, features: torch.Tensor, trajectory_hidden: Optional[Tuple],
                position_info: torch.Tensor, style_embedding: torch.Tensor,
                prev_features: Optional[torch.Tensor] = None,
                prev_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        
        batch_size = features.shape[0]
        
        if prev_features is not None and prev_action is not None:
            _, trajectory_hidden = self.trajectory_encoder(
                prev_features, prev_action, trajectory_hidden
            )
        
        if trajectory_hidden is not None:
            trajectory_emb = trajectory_hidden[0][-1]
        else:
            trajectory_emb = torch.zeros(batch_size, 128, device=features.device)
        
        policy_input = torch.cat([
            features, trajectory_emb, position_info, style_embedding
        ], dim=-1)
        
        keep_logit = self.policy(policy_input)
        value = self.value_head(policy_input)
        
        return keep_logit, value, trajectory_hidden


class BeatLevelPolicy(nn.Module):
    """
    V12 Beat-level policy that makes decisions per-beat using phrase context.
    
    Key differences from V11:
    - Works at beat level instead of 3s segments
    - Uses bidirectional phrase context (60 beats ≈ 30s on each side at 120 BPM)
    - Includes rhythmic position encoding (beat-in-bar, downbeat)
    """
    
    def __init__(self, feature_dim: int, style_dim: int = 64, hidden_dim: int = 256,
                 context_beats: int = DEFAULT_CONTEXT_BEATS, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.style_dim = style_dim
        self.context_beats = context_beats
        
        # Phrase context encoder (bidirectional LSTM)
        self.phrase_encoder = PhraseContextEncoder(
            feature_dim=feature_dim,
            hidden_dim=128,
            context_beats=context_beats
        )
        
        # Style projection
        self.style_proj = nn.Linear(feature_dim, style_dim)
        
        # Position encoding dimension (rhythmic info)
        # beat-in-bar (4) + bar_position + song_position + is_downbeat + beats_since_keep
        self.position_dim = 8
        
        # Policy input: beat_features + phrase_context + position + style
        policy_input_dim = feature_dim + 128 + self.position_dim + style_dim
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )
        
        # Value head for potential RL fine-tuning
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_style_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Compute style embedding from reference features."""
        return self.style_proj(features)
    
    def encode_position(self, beat_idx: int, beat_info: Optional['BeatInfo'],
                       n_total: int, beats_since_keep: int,
                       device: torch.device) -> torch.Tensor:
        """Encode rhythmic position for a beat."""
        # Default values
        beat_in_bar = [0.25, 0.25, 0.25, 0.25]  # Uniform distribution
        bar_position = beat_idx / max(n_total, 1)
        song_position = beat_idx / max(n_total, 1)
        is_downbeat = 0.0
        
        if beat_info is not None and beat_idx < len(beat_info.beat_in_bar):
            # Beat position in bar (1-4 for 4/4 time)
            bib = beat_info.beat_in_bar[beat_idx]
            beat_in_bar = [1.0 if i + 1 == bib else 0.0 for i in range(4)]
            
            # Is this a downbeat?
            is_downbeat = 1.0 if beat_info.downbeat_mask[beat_idx] else 0.0
            
            # Bar position within song
            if beat_info.bar_indices is not None and beat_idx < len(beat_info.bar_indices):
                bar_idx = beat_info.bar_indices[beat_idx]
                max_bar = beat_info.bar_indices.max() if len(beat_info.bar_indices) > 0 else 1
                bar_position = bar_idx / max(max_bar, 1)
        
        # Normalize beats_since_keep
        beats_since_keep_norm = min(beats_since_keep, 16) / 16.0
        
        position = beat_in_bar + [bar_position, song_position, is_downbeat, beats_since_keep_norm]
        return torch.tensor(position, dtype=torch.float32, device=device)
    
    def forward_single(self, all_features: torch.Tensor, beat_idx: int,
                      style_embedding: torch.Tensor,
                      beat_info: Optional['BeatInfo'] = None,
                      beats_since_keep: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single beat with full phrase context.
        
        Args:
            all_features: (n_beats, feature_dim) all beat features
            beat_idx: which beat to classify
            style_embedding: (style_dim,) reference style
            beat_info: optional rhythmic info
            beats_since_keep: for sequential decision making
        
        Returns:
            logit: keep/cut logit
            value: value estimate
        """
        device = all_features.device
        n_beats = all_features.shape[0]
        
        # Get current beat features
        beat_features = all_features[beat_idx]  # (feature_dim,)
        
        # Get phrase context
        phrase_context = self.phrase_encoder(all_features, beat_idx, beat_info)  # (128,)
        
        # Encode position
        position = self.encode_position(beat_idx, beat_info, n_beats, beats_since_keep, device)
        
        # Combine inputs
        policy_input = torch.cat([
            beat_features,
            phrase_context,
            position,
            style_embedding
        ], dim=-1).unsqueeze(0)  # (1, policy_input_dim)
        
        # Get prediction
        logit = self.policy(policy_input)
        value = self.value_head(policy_input)
        
        return logit.squeeze(), value.squeeze()
    
    def forward_batch(self, all_features: torch.Tensor, style_embedding: torch.Tensor,
                     beat_info: Optional['BeatInfo'] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient forward pass for all beats at once.
        
        Args:
            all_features: (n_beats, feature_dim) all beat features
            style_embedding: (style_dim,) reference style
            beat_info: optional rhythmic info
        
        Returns:
            logits: (n_beats,) keep/cut logits
            values: (n_beats,) value estimates
        """
        device = all_features.device
        n_beats = all_features.shape[0]
        
        # Get phrase context for all beats at once
        phrase_contexts = self.phrase_encoder.forward_batch(all_features, beat_info)  # (n_beats, 128)
        
        # Encode positions for all beats
        positions = []
        for i in range(n_beats):
            pos = self.encode_position(i, beat_info, n_beats, 0, device)  # beats_since_keep=0 for batch mode
            positions.append(pos)
        positions = torch.stack(positions, dim=0)  # (n_beats, position_dim)
        
        # Expand style embedding
        style_expanded = style_embedding.unsqueeze(0).expand(n_beats, -1)  # (n_beats, style_dim)
        
        # Combine inputs
        policy_input = torch.cat([
            all_features,
            phrase_contexts,
            positions,
            style_expanded
        ], dim=-1)  # (n_beats, policy_input_dim)
        
        # Get predictions
        logits = self.policy(policy_input).squeeze(-1)  # (n_beats,)
        values = self.value_head(policy_input).squeeze(-1)  # (n_beats,)
        
        return logits, values


# =============================================================================
# DATASET
# =============================================================================

class AlignedEditDataset(Dataset):
    """Dataset of aligned raw→edit segments with keep/cut labels."""
    
    def __init__(self, alignments: List[Dict], reference_features: np.ndarray,
                 scaler: StandardScaler):
        self.scaler = scaler
        
        ref_scaled = scaler.transform(reference_features)
        self.reference_centroid = torch.tensor(ref_scaled.mean(axis=0), dtype=torch.float32)
        
        self.samples = []
        
        for align_idx, alignment in enumerate(alignments):
            features = alignment['raw_features']
            labels = alignment['keep_labels']
            n_segments = len(features)
            
            features_scaled = scaler.transform(features)
            
            for i in range(n_segments):
                self.samples.append({
                    'features': torch.tensor(features_scaled[i], dtype=torch.float32),
                    'label': torch.tensor(labels[i], dtype=torch.float32),
                    'position': i / n_segments,
                    'track_idx': align_idx,
                    'segment_idx': i,
                    'n_segments': n_segments
                })
        
        labels_all = [s['label'].item() for s in self.samples]
        keep_ratio = sum(labels_all) / len(labels_all) if labels_all else 0
        logger.info(f"Dataset: {len(self.samples)} segments, keep ratio: {keep_ratio:.1%}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SequenceDataset(Dataset):
    """Dataset that returns full sequences for trajectory training."""
    
    def __init__(self, alignments: List[Dict], reference_features: np.ndarray,
                 scaler: StandardScaler):
        self.scaler = scaler
        
        ref_scaled = scaler.transform(reference_features)
        self.reference_centroid = torch.tensor(ref_scaled.mean(axis=0), dtype=torch.float32)
        
        self.sequences = []
        for alignment in alignments:
            features = alignment['raw_features']
            labels = alignment['keep_labels']
            features_scaled = scaler.transform(features)
            
            self.sequences.append({
                'features': torch.tensor(features_scaled, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.float32)
            })
        
        logger.info(f"SequenceDataset: {len(self.sequences)} tracks")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class BeatLevelDataset(Dataset):
    """
    V12 Dataset: Full tracks with beat-level labels and rhythmic info.
    
    Returns entire tracks for phrase-context training.
    """
    
    def __init__(self, alignments: List[Dict], reference_features: np.ndarray,
                 scaler: StandardScaler):
        self.scaler = scaler
        
        ref_scaled = scaler.transform(reference_features)
        self.reference_centroid = torch.tensor(ref_scaled.mean(axis=0), dtype=torch.float32)
        
        self.tracks = []
        total_beats = 0
        total_kept = 0
        
        for align_idx, alignment in enumerate(alignments):
            features = alignment['beat_features']
            labels = alignment['keep_labels']
            beat_info = alignment.get('beat_info')  # May be None
            
            features_scaled = scaler.transform(features)
            
            self.tracks.append({
                'features': torch.tensor(features_scaled, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.float32),
                'beat_info': beat_info,
                'n_beats': len(features)
            })
            
            total_beats += len(labels)
            total_kept += labels.sum()
        
        keep_ratio = total_kept / total_beats if total_beats > 0 else 0
        logger.info(f"BeatLevelDataset: {len(self.tracks)} tracks, {total_beats} beats, keep ratio: {keep_ratio:.1%}")
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        return self.tracks[idx]


# =============================================================================
# V12 TRAINER: Beat-Level Training
# =============================================================================

class V13Trainer:
    """
    Train beat-level policy from aligned human edits with STEM SEPARATION.
    
    V13 extracts features from Demucs-separated stems (vocals + other)
    to focus on melodic/guitar content rather than drums/bass.
    """
    
    def __init__(self, 
                 input_dir: str = "./training_data/input",
                 output_dir: str = "./training_data/desired_output",
                 reference_dir: str = "./training_data/reference",
                 model_dir: str = "./models"):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.sr = 22050
        
        # V13: Add stem separator for guitar-focused features
        self.stem_separator = StemSeparator(sr=self.sr)
        
        self.beat_aligner = BeatLevelAligner(sr=self.sr)
        self.segment_aligner = FastAudioAligner(sr=self.sr)  # For reference
        
        self.scaler = None
        self.model = None
        self.feature_dim = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching input/output pairs."""
        pairs = []
        input_files = list(self.input_dir.glob("*_raw.*"))
        
        for raw_file in input_files:
            base = raw_file.stem.replace("_raw", "")
            
            for ext in ['.wav', '.mp3']:
                edit_path = self.output_dir / f"{base}_edit{ext}"
                if edit_path.exists():
                    pairs.append((raw_file, edit_path))
                    break
        
        logger.info(f"Found {len(pairs)} input/output pairs")
        return pairs
    
    def find_reference_files(self) -> List[Path]:
        """Find all reference audio files."""
        refs = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            refs.extend(self.reference_dir.rglob(ext))
        logger.info(f"Found {len(refs)} reference files")
        return refs
    
    def prepare_training_data(self) -> Dict:
        """
        Prepare training data with beat-level alignment and STEM SEPARATION.
        
        V13: Extracts features from vocals+other stems (melodic content)
        instead of the full mix, to focus on guitar/lead phrase boundaries.
        
        Optimized with:
        - Pre-caching of all stems (Demucs is GPU-bound, runs sequentially but caches)
        - Parallel beat feature extraction after stems are cached
        """
        pairs = self.find_pairs()
        reference_files = self.find_reference_files()
        
        if len(pairs) == 0:
            raise ValueError("No input/output pairs found!")
        
        print("\n" + "=" * 70)
        print("V13: STEM-SEPARATED BEAT-LEVEL ALIGNMENT")
        print("=" * 70)
        print("Using Demucs to extract vocals + other stems (melodic content)")
        print("=" * 70)
        
        # =================================================================
        # PHASE 1: Pre-cache all stems (Demucs is GPU-bound, sequential)
        # This ensures all stems are cached before parallel processing
        # =================================================================
        print("\n--- PHASE 1: Pre-caching stems with Demucs ---")
        all_audio_paths = []
        for raw_path, edit_path in pairs:
            all_audio_paths.extend([raw_path, edit_path])
        
        ref_files = reference_files[:50]  # Limit references
        all_audio_paths.extend(ref_files)
        
        # Check which files need stem separation
        uncached = []
        for path in all_audio_paths:
            cache_path = self.stem_separator._get_cache_path(path)
            if not cache_path.exists():
                uncached.append(path)
        
        if uncached:
            print(f"  {len(uncached)} files need stem separation (out of {len(all_audio_paths)})")
            for path in tqdm(uncached, desc="Separating stems"):
                try:
                    audio = load_audio_fast(path, self.sr)
                    self.stem_separator.separate_stems(audio, path)
                except Exception as e:
                    logger.warning(f"  Stem separation failed for {path.name}: {e}")
        else:
            print(f"  All {len(all_audio_paths)} files already cached!")
        
        # =================================================================
        # PHASE 2: Sequential beat detection and alignment (memory-safe)
        # Process one pair at a time to avoid memory issues
        # =================================================================
        print("\n--- PHASE 2: Beat detection and alignment ---")
        
        import gc
        
        alignments = []
        all_features = []
        
        for raw_path, edit_path in tqdm(pairs, desc="Processing pairs"):
            try:
                # Load audio
                raw_audio = load_audio_fast(raw_path, self.sr)
                edit_audio = load_audio_fast(edit_path, self.sr)
                
                # Get cached melodic stems (should be instant from cache)
                raw_melodic = self.stem_separator.get_guitar_focused_audio(raw_audio, raw_path)
                edit_melodic = self.stem_separator.get_guitar_focused_audio(edit_audio, edit_path)
                
                # Detect beats from full mix
                raw_beats = detect_beats(raw_audio, self.sr)
                edit_beats = detect_beats(edit_audio, self.sr)
                
                # Align using melodic content
                alignment = self.beat_aligner.align_beats_with_stems(
                    raw_melodic, edit_melodic, raw_beats, edit_beats, verbose=False
                )
                
                alignments.append(alignment)
                all_features.append(alignment['beat_features'])
                print(f"  {raw_path.name}: {len(raw_beats.beat_times)} beats @ {raw_beats.tempo:.0f} BPM")
                
                # Free memory
                del raw_audio, edit_audio, raw_melodic, edit_melodic
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to process {raw_path.name}: {e}")
        
        # =================================================================
        # PHASE 3: Sequential reference feature extraction (memory-safe)
        # =================================================================
        print("\n--- PHASE 3: Reference feature extraction ---")
        
        reference_features = []
        for ref_path in tqdm(ref_files, desc="Processing references"):
            try:
                audio = load_audio_fast(ref_path, self.sr)
                melodic = self.stem_separator.get_guitar_focused_audio(audio, ref_path)
                beats = detect_beats(audio, self.sr)
                beat_audio = self.beat_aligner.get_beats_audio(melodic, beats)[:20]
                
                if beat_audio:
                    feats = np.array([extract_features_fast(b, self.sr) for b in beat_audio])
                    reference_features.extend(feats)
                
                # Free memory
                del audio, melodic, beat_audio
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed: {ref_path.name}: {e}")
        
        reference_features = np.array(reference_features) if reference_features else np.zeros((1, 56))
        print(f"Reference features: {len(reference_features)} beats")
        
        # Fit scaler
        all_features_flat = np.vstack(all_features)
        all_features_combined = np.vstack([all_features_flat, reference_features])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_features_combined)
        
        self.feature_dim = all_features_flat.shape[1]
        print(f"Feature dimension: {self.feature_dim}")
        
        # Stats
        total_beats = sum(len(a['beat_features']) for a in alignments)
        total_kept = sum(a['keep_labels'].sum() for a in alignments)
        
        print(f"\nTotal beats: {total_beats}")
        print(f"Total kept: {int(total_kept)} ({total_kept/total_beats:.1%})")
        
        # Save reference centroid - V13
        ref_scaled = self.scaler.transform(reference_features)
        ref_centroid = ref_scaled.mean(axis=0)
        np.save(self.model_dir / "reference_centroid_v13.npy", ref_centroid)
        
        return {
            'alignments': alignments,
            'reference_features': reference_features
        }
    
    def train(self, epochs: int = 120, batch_tracks: int = 4, lr: float = 1e-3,
               resume: bool = True) -> None:
        """
        Train V12 beat-level policy.
        
        Uses full-track training with phrase context.
        
        Args:
            epochs: Number of training epochs
            batch_tracks: Tracks per batch (not used currently, trains per-track)
            lr: Learning rate
            resume: If True, resume from last best checkpoint if available
        """
        data = self.prepare_training_data()
        
        print("\n" + "=" * 70)
        print(f"TRAINING V13 STEM-SEPARATED BEAT-LEVEL POLICY (device={DEVICE})")
        print("=" * 70)
        
        dataset = BeatLevelDataset(
            data['alignments'], data['reference_features'], self.scaler
        )
        
        # Compute class weights
        all_labels = []
        for track in dataset.tracks:
            all_labels.extend(track['labels'].tolist())
        n_keep = sum(all_labels)
        n_cut = len(all_labels) - n_keep
        weight_keep = np.sqrt(len(all_labels) / (2 * n_keep)) if n_keep > 0 else 1.0
        weight_cut = np.sqrt(len(all_labels) / (2 * n_cut)) if n_cut > 0 else 1.0
        print(f"Class weights: keep={weight_keep:.2f}, cut={weight_cut:.2f}")
        
        # Initialize model
        self.model = BeatLevelPolicy(
            feature_dim=self.feature_dim,
            style_dim=64,
            hidden_dim=256,
            context_beats=DEFAULT_CONTEXT_BEATS  # 60 beats ≈ 30s at 120 BPM
        ).to(DEVICE)
        
        # Try to resume from checkpoint
        start_epoch = 0
        best_loss = float('inf')
        best_acc = 0
        
        if resume:
            checkpoint_path = self.model_dir / "policy_v13_best.pt"
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                    if 'best_loss' in checkpoint:
                        best_loss = checkpoint['best_loss']
                    if 'best_acc' in checkpoint:
                        best_acc = checkpoint['best_acc']
                    print(f"Resumed from checkpoint: epoch {start_epoch}, best_loss={best_loss:.4f}, best_acc={best_acc:.1%}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}, starting fresh")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        remaining_epochs = epochs - start_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, remaining_epochs)
        grad_scaler = GradScaler(enabled=USE_AMP)
        
        ref_centroid = dataset.reference_centroid.to(DEVICE)
        
        label_smoothing = 0.1
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_beats = 0
            
            # Shuffle tracks
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)
            
            for track_idx in indices:
                track = dataset[track_idx]
                features = track['features'].to(DEVICE)
                labels = track['labels'].to(DEVICE)
                beat_info = track['beat_info']
                n_beats = len(features)
                
                optimizer.zero_grad()
                
                with autocast(enabled=USE_AMP):
                    # Get style embedding
                    style_emb = self.model.compute_style_embedding(ref_centroid)
                    
                    # Forward pass for all beats
                    logits, values = self.model.forward_batch(features, style_emb, beat_info)
                    
                    # Label smoothing
                    labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                    
                    # Weighted loss
                    weights = torch.where(labels == 1, weight_keep, weight_cut)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, labels_smooth, weight=weights
                    )
                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                total_loss += loss.item() * n_beats
                
                # Accuracy on original labels
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    total_correct += (preds == labels).sum().item()
                total_beats += n_beats
            
            scheduler.step()
            
            avg_loss = total_loss / total_beats
            accuracy = total_correct / total_beats
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1%}")
            
            # Save checkpoint on best epoch
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = accuracy
                self.save_model("policy_v13_best.pt", epoch=epoch, best_loss=best_loss, best_acc=best_acc)
                print(f"  -> New best! Saved checkpoint.")
            
            # Also save periodic checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_model(f"policy_v13_epoch{epoch+1}.pt", epoch=epoch, best_loss=best_loss, best_acc=best_acc)
                print(f"  -> Periodic checkpoint saved (epoch {epoch+1})")
        
        print(f"\nBest: Loss={best_loss:.4f}, Acc={best_acc:.1%}")
        self.save_model("policy_v13_final.pt", epoch=epochs-1, best_loss=best_loss, best_acc=best_acc)
    
    def train_with_trajectory(self, data: Dict, epochs: int = 60, lr: float = 5e-4,
                               resume: bool = True) -> None:
        """
        Fine-tune with sequential trajectory context using Truncated BPTT.
        
        This trains the model to consider beats_since_keep and make 
        sequential decisions that maintain musical flow.
        
        Args:
            data: Training data dict from prepare_training_data()
            epochs: Number of trajectory training epochs
            lr: Learning rate
            resume: If True, resume from last trajectory checkpoint if available
        """
        print("\n" + "=" * 70)
        print("V13 TRAINING WITH TRAJECTORY CONTEXT (TBPTT)")
        print("=" * 70)
        
        if self.model is None:
            raise ValueError("Run train() first!")
        
        seq_dataset = BeatLevelDataset(
            data['alignments'], data['reference_features'], self.scaler
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        ref_centroid = seq_dataset.reference_centroid.to(DEVICE)
        
        # Try to resume from trajectory checkpoint
        start_epoch = 0
        best_loss = float('inf')
        best_acc = 0
        
        if resume:
            checkpoint_path = self.model_dir / "policy_v13_trajectory_best.pt"
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                    if 'best_loss' in checkpoint:
                        best_loss = checkpoint['best_loss']
                    if 'best_acc' in checkpoint:
                        best_acc = checkpoint['best_acc']
                    print(f"Resumed trajectory training from epoch {start_epoch}, best_loss={best_loss:.4f}")
                except Exception as e:
                    logger.warning(f"Could not load trajectory checkpoint: {e}")
        
        window_size = 32  # TBPTT window size (beats)
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_beats = 0
            
            # Shuffle tracks
            indices = list(range(len(seq_dataset)))
            np.random.shuffle(indices)
            
            for track_idx in indices:
                track = seq_dataset[track_idx]
                features = track['features'].to(DEVICE)
                labels = track['labels'].to(DEVICE)
                beat_info = track['beat_info']
                n_beats = len(features)
                
                if n_beats < 10:
                    continue
                
                # Track beats_since_keep across windows
                beats_since_keep = 0
                
                # Process in TBPTT windows
                for win_start in range(0, n_beats, window_size):
                    win_end = min(win_start + window_size, n_beats)
                    window_len = win_end - win_start
                    
                    optimizer.zero_grad()
                    
                    # CRITICAL: Clone features and recompute style_emb for each window
                    # to break graph connections between windows (fixes backward() twice error)
                    window_features = features.clone()
                    style_emb = self.model.compute_style_embedding(ref_centroid)
                    
                    # Accumulate losses for this window
                    window_losses = []
                    window_correct = 0
                    
                    for i in range(win_start, win_end):
                        # Forward with sequential context
                        logit, _ = self.model.forward_single(
                            window_features, i, style_emb, beat_info, beats_since_keep
                        )
                        
                        # Loss
                        loss = F.binary_cross_entropy_with_logits(
                            logit, labels[i]
                        )
                        window_losses.append(loss)
                        
                        # Track for accuracy
                        with torch.no_grad():
                            pred = (torch.sigmoid(logit) > 0.5).float().item()
                            actual = labels[i].item()
                            window_correct += int(pred == actual)
                            
                            # Update beats_since_keep based on ground truth
                            if actual > 0.5:
                                beats_since_keep = 0
                            else:
                                beats_since_keep += 1
                    
                    # Single backward for window
                    window_loss = sum(window_losses) / len(window_losses)
                    window_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += window_loss.item() * window_len
                    total_correct += window_correct
                    total_beats += window_len
            
            avg_loss = total_loss / max(total_beats, 1)
            accuracy = total_correct / max(total_beats, 1)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1%}")
            
            # Save checkpoint on best epoch
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = accuracy
                self.save_model("policy_v13_trajectory_best.pt", epoch=epoch, best_loss=best_loss, best_acc=best_acc)
                print(f"  -> New best! Saved trajectory checkpoint.")
            
            # Also save periodic checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_model(f"policy_v13_trajectory_epoch{epoch+1}.pt", epoch=epoch, best_loss=best_loss, best_acc=best_acc)
                print(f"  -> Periodic checkpoint saved (epoch {epoch+1})")
        
        self.save_model("policy_v13_trajectory_final.pt", epoch=epochs-1, best_loss=best_loss, best_acc=best_acc)
        
        print("\n" + "=" * 70)
        print("V13 TRAJECTORY TRAINING COMPLETE")
        print("=" * 70)
    
    def save_model(self, filename: str, epoch: int = 0, best_loss: float = float('inf'),
                   best_acc: float = 0.0):
        """Save model and config with checkpoint info for resuming."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
            'context_beats': DEFAULT_CONTEXT_BEATS,
            'epoch': epoch,
            'best_loss': best_loss,
            'best_acc': best_acc,
        }, self.model_dir / filename)
        
        with open(self.model_dir / "scaler_v13.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        np.save(self.model_dir / "feature_dim_v13.npy", self.feature_dim)


# =============================================================================
# TRAINER (V11 - kept for reference)
# =============================================================================

class V11Trainer:
    """Train imitation policy from aligned human edits - OPTIMIZED."""
    
    def __init__(self, 
                 input_dir: str = "./training_data/input",
                 output_dir: str = "./training_data/desired_output",
                 reference_dir: str = "./training_data/reference",
                 model_dir: str = "./models",
                 segment_duration: float = 3.0,
                 hop_duration: float = 1.5):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sr = 22050
        
        self.aligner = FastAudioAligner(
            sr=self.sr,
            segment_duration=segment_duration,
            hop_duration=hop_duration
        )
        
        self.scaler = None
        self.model = None
        self.feature_dim = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching input/output pairs."""
        pairs = []
        input_files = list(self.input_dir.glob("*_raw.*"))
        
        for raw_file in input_files:
            base = raw_file.stem.replace("_raw", "")
            
            for ext in ['.wav', '.mp3']:
                edit_path = self.output_dir / f"{base}_edit{ext}"
                if edit_path.exists():
                    pairs.append((raw_file, edit_path))
                    break
        
        logger.info(f"Found {len(pairs)} input/output pairs")
        return pairs
    
    def find_reference_files(self) -> List[Path]:
        """Find all reference audio files."""
        refs = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            refs.extend(self.reference_dir.rglob(ext))
        logger.info(f"Found {len(refs)} reference files")
        return refs
    
    def prepare_training_data(self) -> Dict:
        """Prepare training data with parallel processing."""
        pairs = self.find_pairs()
        reference_files = self.find_reference_files()
        
        if len(pairs) == 0:
            raise ValueError("No input/output pairs found!")
        
        print("\n" + "=" * 70)
        print("V11: ALIGNING INPUT/OUTPUT PAIRS (OPTIMIZED)")
        print("=" * 70)
        
        alignments = []
        all_features = []
        
        for raw_path, edit_path in tqdm(pairs, desc="Aligning pairs"):
            print(f"\n{raw_path.name}")
            
            raw_audio = load_audio_fast(raw_path, self.sr)
            edit_audio = load_audio_fast(edit_path, self.sr)
            
            print(f"  Raw: {len(raw_audio)/self.sr:.1f}s, Edit: {len(edit_audio)/self.sr:.1f}s")
            
            alignment = self.aligner.align(
                raw_audio, edit_audio, 
                raw_path=raw_path, edit_path=edit_path,
                verbose=True
            )
            alignments.append(alignment)
            all_features.append(alignment['raw_features'])
        
        # Extract reference features (parallel)
        print("\n" + "=" * 70)
        print("EXTRACTING REFERENCE FEATURES (PARALLEL)")
        print("=" * 70)
        
        reference_features = []
        
        # Process references in batches
        ref_files = reference_files[:50]  # Limit
        
        for ref_path in tqdm(ref_files, desc="Processing references"):
            # Check cache first
            cached = load_cached_features(ref_path)
            if cached is not None:
                reference_features.extend(cached[:20])  # Limit per file
                continue
            
            try:
                audio = load_audio_fast(ref_path, self.sr)
                segments, _ = self.aligner.extract_segments(audio)
                
                if segments:
                    feats = extract_features_parallel(segments[:20], self.sr)
                    save_cached_features(ref_path, feats)
                    reference_features.extend(feats)
            except Exception as e:
                logger.warning(f"Failed: {ref_path.name}: {e}")
        
        reference_features = np.array(reference_features) if reference_features else np.zeros((1, 56))
        print(f"Reference features: {len(reference_features)} segments")
        
        # Fit scaler
        all_features_flat = np.vstack(all_features)
        all_features_combined = np.vstack([all_features_flat, reference_features])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_features_combined)
        
        self.feature_dim = all_features_flat.shape[1]
        print(f"Feature dimension: {self.feature_dim}")
        
        # Stats
        total_segments = sum(len(a['raw_features']) for a in alignments)
        total_kept = sum(a['keep_labels'].sum() for a in alignments)
        
        print(f"\nTotal segments: {total_segments}")
        print(f"Total kept: {int(total_kept)} ({total_kept/total_segments:.1%})")
        
        # Save reference centroid
        ref_scaled = self.scaler.transform(reference_features)
        ref_centroid = ref_scaled.mean(axis=0)
        np.save(self.model_dir / "reference_centroid_v11.npy", ref_centroid)
        
        return {
            'alignments': alignments,
            'reference_features': reference_features
        }
    
    def train_imitation(self, data: Dict, epochs: int = 100, 
                       batch_size: int = 128, lr: float = 1e-3) -> None:
        """Train with mixed precision for speed."""
        
        print("\n" + "=" * 70)
        print(f"TRAINING IMITATION POLICY (device={DEVICE}, AMP={USE_AMP})")
        print("=" * 70)
        
        dataset = AlignedEditDataset(
            data['alignments'], data['reference_features'], self.scaler
        )
        
        # Class weights - more balanced
        labels = [s['label'].item() for s in dataset.samples]
        n_keep = sum(labels)
        n_cut = len(labels) - n_keep
        # Use sqrt for smoother weighting
        weight_keep = np.sqrt(len(labels) / (2 * n_keep)) if n_keep > 0 else 1.0
        weight_cut = np.sqrt(len(labels) / (2 * n_cut)) if n_cut > 0 else 1.0
        print(f"Class weights: keep={weight_keep:.2f}, cut={weight_cut:.2f}")
        
        # Label smoothing
        label_smoothing = 0.1
        print(f"Label smoothing: {label_smoothing}")
        
        self.model = ImitationPolicy(
            feature_dim=self.feature_dim, style_dim=64, hidden_dim=256
        ).to(DEVICE)
        
        # Stronger weight decay for regularization
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = GradScaler(enabled=USE_AMP)
        
        ref_centroid = dataset.reference_centroid.to(DEVICE)
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        best_loss = float('inf')
        best_acc = 0
        
        # Data augmentation: add noise to features
        feature_noise_std = 0.15
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                features = batch['features'].to(DEVICE)
                labels_batch = batch['label'].to(DEVICE)
                positions = batch['position'].to(DEVICE).float()
                
                # Data augmentation: add Gaussian noise to features
                if self.model.training:
                    noise = torch.randn_like(features) * feature_noise_std
                    features = features + noise
                
                # Apply label smoothing
                labels_smooth = labels_batch * (1 - label_smoothing) + 0.5 * label_smoothing
                
                bs = features.shape[0]
                
                position_info = torch.stack([
                    positions,
                    torch.full((bs,), 0.35, device=DEVICE, dtype=torch.float32),
                    torch.zeros(bs, device=DEVICE, dtype=torch.float32)
                ], dim=1)
                
                style_emb = self.model.compute_style_embedding(ref_centroid.unsqueeze(0))
                style_emb = style_emb.expand(bs, -1)
                
                optimizer.zero_grad()
                
                with autocast(enabled=USE_AMP):
                    logits, _, _ = self.model(features, None, position_info, style_emb)
                    weights = torch.where(labels_batch == 1, weight_keep, weight_cut)
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), labels_smooth, weight=weights
                    )
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item() * bs
                # Accuracy on original labels (not smoothed)
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                correct += (preds == labels_batch).sum().item()
                total += bs
            
            scheduler.step()
            
            avg_loss = total_loss / total
            accuracy = correct / total
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1%}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = accuracy
                self.save_model("policy_v11_best.pt")
        
        print(f"\nBest: Loss={best_loss:.4f}, Acc={best_acc:.1%}")
        self.save_model("policy_v11_final.pt")
    
    def train_with_trajectory(self, data: Dict, epochs: int = 50, lr: float = 5e-4) -> None:
        """
        Fine-tune with trajectory context using Truncated BPTT.
        
        The key fix for "backward through graph twice" is to:
        1. Detach hidden states at the START of each window (before forward passes)
        2. Use retain_graph=False (default) since we fully detach between windows
        """
        print("\n" + "=" * 70)
        print("TRAINING WITH TRAJECTORY CONTEXT (TBPTT)")
        print("=" * 70)
        
        if self.model is None:
            raise ValueError("Run train_imitation first!")
        
        seq_dataset = SequenceDataset(
            data['alignments'], data['reference_features'], self.scaler
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        ref_centroid = seq_dataset.reference_centroid.to(DEVICE)
        
        best_loss = float('inf')
        window_size = 16  # TBPTT window size
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for seq in seq_dataset.sequences:
                features = seq['features'].to(DEVICE)
                labels = seq['labels'].to(DEVICE)
                n_segments = len(features)
                
                # Track trajectory hidden state VALUES across windows (no graph)
                carry_hidden_values = None
                
                # Process in TBPTT windows
                for win_start in range(0, n_segments, window_size):
                    win_end = min(win_start + window_size, n_segments)
                    window_len = win_end - win_start
                    
                    # Reset gradients for this window
                    optimizer.zero_grad()
                    
                    # CRITICAL: Recompute style_emb for each window to break graph
                    style_emb_base = self.model.compute_style_embedding(ref_centroid.unsqueeze(0))
                    
                    # Create fresh hidden state with VALUES from previous window
                    trajectory_hidden = None
                    if carry_hidden_values is not None:
                        trajectory_hidden = (
                            carry_hidden_values[0].clone(),  # Pure values, no graph
                            carry_hidden_values[1].clone()
                        )
                    
                    prev_features = None
                    prev_action = None
                    
                    # Accumulate losses for this window (single backward at end)
                    window_losses = []
                    window_correct = 0
                    
                    for i in range(win_start, win_end):
                        # Clone features to ensure no graph connection to previous windows
                        curr_features = features[i:i+1].clone()
                        curr_label = labels[i:i+1].clone()
                        
                        rel_pos = i / n_segments
                        kept_ratio = 0.35
                        segments_since_keep = 0
                        
                        position_info = torch.tensor([
                            [rel_pos, kept_ratio, segments_since_keep / 10]
                        ], device=DEVICE, dtype=torch.float32)
                        
                        # Forward pass - trajectory_hidden accumulates within window
                        logits, _, trajectory_hidden = self.model(
                            curr_features, trajectory_hidden, position_info, style_emb_base,
                            prev_features, prev_action
                        )
                        
                        # Compute loss for this step (accumulate, don't backward yet)
                        step_loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(-1), curr_label
                        )
                        window_losses.append(step_loss)
                        
                        # Track accuracy
                        with torch.no_grad():
                            pred = (torch.sigmoid(logits) > 0.5).float().item()
                            actual = curr_label.item()
                            window_correct += int(pred == actual)
                        
                        # Prev state for next step - detach to limit graph size
                        prev_features = curr_features.detach()
                        prev_action = curr_label.detach()
                    
                    # Single backward pass for entire window
                    window_loss = sum(window_losses) / len(window_losses)
                    window_loss.backward()
                    
                    # Save hidden state VALUES for next window (use .data for pure values)
                    if trajectory_hidden is not None:
                        carry_hidden_values = (
                            trajectory_hidden[0].data.clone(),
                            trajectory_hidden[1].data.clone()
                        )
                    else:
                        carry_hidden_values = None
                    
                    # Gradient clipping and optimizer step (once per window)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += window_loss.item() * window_len
                    total_correct += window_correct
                    total_samples += window_len
            
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1%}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("policy_v11_trajectory_best.pt")
        
        self.save_model("policy_v11_trajectory_final.pt")
    
    def save_model(self, filename: str):
        """Save model and config."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
        }, self.model_dir / filename)
        
        with open(self.model_dir / "scaler_v11.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        np.save(self.model_dir / "feature_dim_v11.npy", self.feature_dim)
    
    def train(self, include_trajectory: bool = True):
        """Full training pipeline."""
        data = self.prepare_training_data()
        self.train_imitation(data, epochs=100, batch_size=128)
        
        if include_trajectory:
            # Fine-tune with trajectory context for sequential editing awareness
            self.train_with_trajectory(data, epochs=50)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)


# =============================================================================
# INFERENCE: V11 Editor
# =============================================================================

class V11Editor:
    """Apply V11 policy to edit a track."""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        self.segment_duration = 3.0
        self.hop_duration = 1.5
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model."""
        self.feature_dim = int(np.load(self.model_dir / "feature_dim_v11.npy"))
        
        with open(self.model_dir / "scaler_v11.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.model = ImitationPolicy(
            feature_dim=self.feature_dim, style_dim=64, hidden_dim=256
        ).to(DEVICE)
        
        for model_file in ["policy_v11_trajectory_best.pt", "policy_v11_best.pt"]:
            path = self.model_dir / model_file
            if path.exists():
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded: {model_file}")
                break
        
        self.model.eval()
        
        ref_path = self.model_dir / "reference_centroid_v11.npy"
        if ref_path.exists():
            self.reference_centroid = np.load(ref_path)
        else:
            self.reference_centroid = np.zeros(self.feature_dim)
    
    def extract_segments(self, audio: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract segments."""
        segment_samples = int(self.segment_duration * self.sr)
        hop_samples = int(self.hop_duration * self.sr)
        
        segments = []
        positions = []
        
        start = 0
        while start + segment_samples <= len(audio):
            segments.append(audio[start:start + segment_samples])
            positions.append((start, start + segment_samples))
            start += hop_samples
        
        return segments, positions
    
    def process_track(self, input_path: str, output_path: str,
                     keep_ratio: float = 0.35,
                     use_trajectory: bool = True) -> Dict:
        """Process a track using V11 policy."""
        
        audio = load_audio_fast(Path(input_path), self.sr)
        duration = len(audio) / self.sr
        
        segments, positions = self.extract_segments(audio)
        n_segments = len(segments)
        
        if n_segments == 0:
            raise ValueError("Audio too short")
        
        # Extract features (parallel for speed)
        features = extract_features_parallel(segments, self.sr)
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        
        # Style embedding
        ref_tensor = torch.tensor(
            self.scaler.transform(self.reference_centroid.reshape(1, -1))[0],
            dtype=torch.float32
        ).to(DEVICE)
        
        scores = []
        
        with torch.no_grad():
            style_emb = self.model.compute_style_embedding(ref_tensor.unsqueeze(0))
            
            if use_trajectory:
                trajectory_hidden = None
                prev_features = None
                prev_action = None
                kept_so_far = 0
                segments_since_keep = 0
                
                for i in range(n_segments):
                    curr_features = features_tensor[i:i+1]
                    
                    rel_pos = i / n_segments
                    kept_ratio_so_far = kept_so_far / (i + 1) if i > 0 else keep_ratio
                    
                    position_info = torch.tensor([
                        [rel_pos, kept_ratio_so_far, min(segments_since_keep, 10) / 10]
                    ], device=DEVICE)
                    
                    logits, _, trajectory_hidden = self.model(
                        curr_features, trajectory_hidden, position_info, style_emb,
                        prev_features, prev_action
                    )
                    
                    score = torch.sigmoid(logits).item()
                    scores.append(score)
                    
                    pred_action = 1.0 if score > 0.5 else 0.0
                    if pred_action > 0.5:
                        kept_so_far += 1
                        segments_since_keep = 0
                    else:
                        segments_since_keep += 1
                    
                    prev_features = curr_features
                    prev_action = torch.tensor([pred_action], device=DEVICE)
            else:
                for i in range(n_segments):
                    position_info = torch.tensor([
                        [i / n_segments, keep_ratio, 0]
                    ], device=DEVICE)
                    
                    logits, _, _ = self.model(
                        features_tensor[i:i+1], None, position_info, style_emb
                    )
                    scores.append(torch.sigmoid(logits).item())
        
        scores = np.array(scores)
        
        # Select top segments
        n_keep = max(1, int(n_segments * keep_ratio))
        threshold = np.sort(scores)[::-1][min(n_keep - 1, len(scores) - 1)]
        keep_mask = scores >= threshold
        
        # Merge consecutive
        kept_regions = []
        in_region = False
        region_start = 0
        
        for i, keep in enumerate(keep_mask):
            if keep and not in_region:
                region_start = positions[i][0]
                in_region = True
            elif not keep and in_region:
                kept_regions.append((region_start, positions[i-1][1]))
                in_region = False
        
        if in_region:
            kept_regions.append((region_start, positions[-1][1]))
        
        # Merge close regions
        merged_regions = []
        min_gap = int(0.5 * self.sr)
        
        for start, end in kept_regions:
            if merged_regions and start - merged_regions[-1][1] < min_gap:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        
        # Build output
        output_segments = [audio[s:e] for s, e in merged_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        
        sf.write(output_path, output_audio, self.sr)
        
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_segments': n_segments,
            'n_regions': len(merged_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'score_stats': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
        }


# =============================================================================
# V12 EDITOR: Beat-Level Inference
# =============================================================================

class V13Editor:
    """
    V13 Beat-level editor with phrase context and STEM SEPARATION.
    
    Makes keep/cut decisions at the beat level using:
    - DEMUCS STEM SEPARATION: Extracts vocals + other for melodic focus
    - Bidirectional phrase context (looks at surrounding beats)
    - Rhythmic position encoding (beat-in-bar, downbeat detection)
    - Reference style matching
    
    V13 extracts features from the melodic content (vocals + other stems)
    instead of the full mix, helping identify guitar phrase boundaries.
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        
        # V13: Add stem separator
        self.stem_separator = StemSeparator(sr=self.sr)
        
        self._load_model()
    
    def _load_model(self):
        """Load trained V13 model."""
        self.feature_dim = int(np.load(self.model_dir / "feature_dim_v13.npy"))
        
        with open(self.model_dir / "scaler_v13.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load context_beats from checkpoint if available
        context_beats = DEFAULT_CONTEXT_BEATS
        checkpoint_loaded = False
        
        for model_file in ["policy_v13_trajectory_best.pt", "policy_v13_best.pt", "policy_v13_final.pt"]:
            path = self.model_dir / model_file
            if path.exists():
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                if 'context_beats' in checkpoint:
                    context_beats = checkpoint['context_beats']
                checkpoint_loaded = True
                break
        
        self.model = BeatLevelPolicy(
            feature_dim=self.feature_dim,
            style_dim=64,
            hidden_dim=256,
            context_beats=context_beats
        ).to(DEVICE)
        
        # Load weights
        if checkpoint_loaded:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded: {model_file} (context_beats={context_beats})")
        
        self.model.eval()
        
        ref_path = self.model_dir / "reference_centroid_v13.npy"
        if ref_path.exists():
            self.reference_centroid = np.load(ref_path)
        else:
            self.reference_centroid = np.zeros(self.feature_dim)
    
    def process_track(self, input_path: str, output_path: str,
                     keep_ratio: float = 0.35) -> Dict:
        """
        Process a track using V13 beat-level policy with stem separation.
        
        V13 uses Demucs to extract melodic content (vocals + other) for feature
        extraction, while using full mix for beat detection and output.
        """
        input_path_obj = Path(input_path)
        audio = load_audio_fast(input_path_obj, self.sr)
        duration = len(audio) / self.sr
        
        # V13: Separate stems and get melodic content for features
        logger.info("Separating stems with Demucs...")
        melodic_audio = self.stem_separator.get_guitar_focused_audio(audio, input_path_obj)
        
        # Detect beats from FULL MIX (better transient detection)
        logger.info("Detecting beats from full mix...")
        beat_info = detect_beats(audio, self.sr)
        n_beats = len(beat_info.beat_times) - 1  # -1 because we use intervals
        
        if n_beats <= 0:
            logger.warning("Could not detect beats, falling back to segment-based")
            raise ValueError("Beat detection failed")
        
        logger.info(f"Detected {n_beats} beats @ {beat_info.tempo:.1f} BPM")
        
        # Extract beat audio from MELODIC content for features
        beat_samples = (beat_info.beat_times * self.sr).astype(int)
        melodic_beat_list = []  # For features
        beat_positions = []      # For cutting from full audio
        
        for i in range(n_beats):
            start = beat_samples[i]
            end = beat_samples[i + 1] if i + 1 < len(beat_samples) else len(audio)
            if start < len(melodic_audio) and end > start:
                # Features from melodic content
                melodic_beat_list.append(melodic_audio[start:min(end, len(melodic_audio))])
                beat_positions.append((start, end))
        
        # V13: Extract features from MELODIC content (guitar-focused)
        logger.info("Extracting features from melodic content (vocals + other)...")
        features = np.array([extract_features_fast(b, self.sr) for b in melodic_beat_list])
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        
        # Style embedding
        ref_scaled = self.scaler.transform(self.reference_centroid.reshape(1, -1))
        ref_tensor = torch.tensor(ref_scaled[0], dtype=torch.float32).to(DEVICE)
        
        # Get scores using SEQUENTIAL forward (important for beats_since_keep)
        # This gives much better score differentiation than batch mode
        with torch.no_grad():
            style_emb = self.model.compute_style_embedding(ref_tensor)
            
            scores = []
            beats_since_keep = 0
            
            for i in range(len(features_tensor)):
                logit, _ = self.model.forward_single(
                    features_tensor, i, style_emb, beat_info, beats_since_keep
                )
                score = torch.sigmoid(logit).item()
                scores.append(score)
                
                # Update beats_since_keep based on running prediction
                if score > 0.5:
                    beats_since_keep = 0
                else:
                    beats_since_keep += 1
            
            scores = np.array(scores)
        
        # Select beats to keep
        n_keep = max(1, int(n_beats * keep_ratio))
        threshold = np.sort(scores)[::-1][min(n_keep - 1, len(scores) - 1)]
        keep_mask = scores >= threshold
        
        # Merge consecutive kept beats into regions
        # Also ensure we keep complete musical phrases (4-beat groups when possible)
        kept_regions = []
        in_region = False
        region_start = 0
        
        for i, keep in enumerate(keep_mask):
            if i < len(beat_positions):
                if keep and not in_region:
                    region_start = beat_positions[i][0]
                    in_region = True
                elif not keep and in_region:
                    kept_regions.append((region_start, beat_positions[i-1][1]))
                    in_region = False
        
        if in_region and len(beat_positions) > 0:
            kept_regions.append((region_start, beat_positions[-1][1]))
        
        # Merge regions that are close together (less than 2 beats apart)
        avg_beat_duration = (beat_info.beat_times[1] - beat_info.beat_times[0]) if len(beat_info.beat_times) > 1 else 0.5
        min_gap = int(2 * avg_beat_duration * self.sr)
        
        merged_regions = []
        for start, end in kept_regions:
            if merged_regions and start - merged_regions[-1][1] < min_gap:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        
        # Build output
        output_segments = [audio[s:e] for s, e in merged_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        
        sf.write(output_path, output_audio, self.sr)
        
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_beats': n_beats,
            'tempo': beat_info.tempo,
            'n_regions': len(merged_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'score_stats': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
        }


# =============================================================================
# HYBRID V12 EDITOR: V9 base + V12 beat-level refinement
# =============================================================================

class HybridV12Editor:
    """
    Hybrid editor that combines:
    - V9's quality scoring (generalizes to unseen tracks via reference contrastive learning)
    - V12's beat-level phrase context (60 beats bidirectional + rhythmic position encoding)
    
    Final score = v9_weight * V9_score + v12_weight * V12_adjustment
    
    V9 provides the foundation (what sounds good in general)
    V12 provides beat-level refinement with musical phrase awareness
    """
    
    def __init__(self, model_dir: str = "./models", v9_weight: float = 0.5, v12_weight: float = 0.5):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        
        self.v9_weight = v9_weight
        self.v12_weight = v12_weight
        
        self._load_v9_model()
        self._load_v12_model()
        
        logger.info(f"Hybrid V12 Editor loaded (V9 weight={v9_weight}, V12 weight={v12_weight})")
    
    def _load_v9_model(self):
        """Load V9 quality model."""
        from train_edit_policy_v9 import DualHeadModel
        
        self.v9_feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
        self.v9_reference_centroid = np.load(self.model_dir / "reference_centroid_v9.npy")
        self.v9_similarity_weight = float(np.load(self.model_dir / "similarity_weight_v9.npy"))
        
        self.v9_model = DualHeadModel(
            base_feature_dim=self.v9_feature_dim,
            embedding_dim=len(self.v9_reference_centroid)
        ).to(DEVICE)
        
        checkpoint = torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
        self.v9_model.load_state_dict(checkpoint)
        self.v9_model.eval()
        
        self.v9_ref_centroid_t = torch.FloatTensor(self.v9_reference_centroid).to(DEVICE)
        
        # V9 feature extractor
        from train_edit_policy_v10_simple import SegmentFeatureExtractor
        self.v9_extractor = SegmentFeatureExtractor(self.sr)
        
        logger.info("Loaded V9 model")
    
    def _load_v12_model(self):
        """Load V12 beat-level model."""
        try:
            self.v12_feature_dim = int(np.load(self.model_dir / "feature_dim_v12.npy"))
            
            with open(self.model_dir / "scaler_v12.pkl", 'rb') as f:
                self.v12_scaler = pickle.load(f)
            
            # Load context_beats from checkpoint
            context_beats = DEFAULT_CONTEXT_BEATS
            for model_file in ["policy_v12_trajectory_best.pt", "policy_v12_best.pt", "policy_v12_final.pt"]:
                path = self.model_dir / model_file
                if path.exists():
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                    if 'context_beats' in checkpoint:
                        context_beats = checkpoint['context_beats']
                    
                    self.v12_model = BeatLevelPolicy(
                        feature_dim=self.v12_feature_dim,
                        style_dim=64,
                        hidden_dim=256,
                        context_beats=context_beats
                    ).to(DEVICE)
                    
                    self.v12_model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded V12 model: {model_file}")
                    break
            
            self.v12_model.eval()
            
            ref_path = self.model_dir / "reference_centroid_v12.npy"
            if ref_path.exists():
                self.v12_reference_centroid = np.load(ref_path)
            else:
                self.v12_reference_centroid = np.zeros(self.v12_feature_dim)
            
            self.v12_loaded = True
        except Exception as e:
            logger.warning(f"Could not load V12 model: {e}, using V9 only")
            self.v12_loaded = False
    
    def compute_v9_scores_for_beats(self, beat_audio_list: List[np.ndarray], 
                                    beat_positions: List[Tuple[int, int]],
                                    audio: np.ndarray) -> np.ndarray:
        """
        Compute V9 scores for beats by mapping beats to 3s segments.
        
        V9 works on 3s segments, so we map each beat to the segment it falls within.
        """
        # Create V9 segments
        segment_duration = 3.0
        hop_duration = 1.5
        segment_samples = int(segment_duration * self.sr)
        hop_samples = int(hop_duration * self.sr)
        
        # Extract V9 segments
        segments = []
        segment_positions = []
        start = 0
        while start + segment_samples <= len(audio):
            segments.append(audio[start:start + segment_samples])
            segment_positions.append((start, start + segment_samples))
            start += hop_samples
        
        if not segments:
            return np.full(len(beat_audio_list), 0.5)
        
        # Extract V9 features for segments
        features = []
        for seg in segments:
            feat = self.v9_extractor.extract(seg)
            features.append(feat)
        features = np.array(features)
        
        # Create context windows (V9 uses 3-segment windows)
        from train_edit_policy_v10_simple import create_context_windows
        windowed = create_context_windows(features)
        windowed_tensor = torch.FloatTensor(windowed).to(DEVICE)
        
        with torch.no_grad():
            quality_logits, style_emb = self.v9_model(windowed_tensor)
            quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
            
            # Reference similarity
            ref_sim = torch.mm(style_emb, self.v9_ref_centroid_t.unsqueeze(1)).squeeze()
            ref_sim = ((ref_sim + 1) / 2).cpu().numpy()
        
        # Combined V9 score per segment
        segment_scores = quality + self.v9_similarity_weight * ref_sim
        
        # Map segment scores to beats
        beat_scores = []
        for beat_start, beat_end in beat_positions:
            beat_mid = (beat_start + beat_end) // 2
            
            # Find which segment this beat falls in
            best_seg = 0
            best_overlap = 0
            for seg_idx, (seg_start, seg_end) in enumerate(segment_positions):
                # Check overlap
                overlap_start = max(beat_start, seg_start)
                overlap_end = min(beat_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_seg = seg_idx
            
            beat_scores.append(segment_scores[best_seg])
        
        return np.array(beat_scores)
    
    def compute_v12_scores(self, features_tensor: torch.Tensor, beat_info: 'BeatInfo') -> np.ndarray:
        """Compute V12 beat-level scores with sequential context."""
        if not self.v12_loaded:
            return np.zeros(len(features_tensor))
        
        ref_scaled = self.v12_scaler.transform(self.v12_reference_centroid.reshape(1, -1))
        ref_tensor = torch.tensor(ref_scaled[0], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            style_emb = self.v12_model.compute_style_embedding(ref_tensor)
            
            scores = []
            beats_since_keep = 0
            
            for i in range(len(features_tensor)):
                logit, _ = self.v12_model.forward_single(
                    features_tensor, i, style_emb, beat_info, beats_since_keep
                )
                score = torch.sigmoid(logit).item()
                scores.append(score)
                
                # Update beats_since_keep
                if score > 0.5:
                    beats_since_keep = 0
                else:
                    beats_since_keep += 1
        
        return np.array(scores)
    
    def process_track(self, input_path: str, output_path: str,
                     keep_ratio: float = 0.35) -> Dict:
        """Process a track using hybrid V9+V12 beat-level scoring."""
        
        audio = load_audio_fast(Path(input_path), self.sr)
        duration = len(audio) / self.sr
        
        # Detect beats
        logger.info("Detecting beats...")
        beat_info = detect_beats(audio, self.sr)
        n_beats = len(beat_info.beat_times) - 1
        
        if n_beats <= 0:
            raise ValueError("Beat detection failed")
        
        logger.info(f"Detected {n_beats} beats @ {beat_info.tempo:.1f} BPM")
        
        # Extract beat audio
        beat_samples = (beat_info.beat_times * self.sr).astype(int)
        beat_audio_list = []
        beat_positions = []
        
        for i in range(n_beats):
            start = beat_samples[i]
            end = beat_samples[i + 1] if i + 1 < len(beat_samples) else len(audio)
            if start < len(audio) and end > start:
                beat_audio_list.append(audio[start:end])
                beat_positions.append((start, end))
        
        # Extract V12 features for beats
        features = np.array([extract_features_fast(b, self.sr) for b in beat_audio_list])
        features_scaled = self.v12_scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        
        # Get V9 scores (mapped to beats)
        logger.info("Computing V9 quality scores...")
        v9_scores = self.compute_v9_scores_for_beats(beat_audio_list, beat_positions, audio)
        
        # Normalize V9 to [0, 1]
        v9_min, v9_max = v9_scores.min(), v9_scores.max()
        if v9_max > v9_min:
            v9_normalized = (v9_scores - v9_min) / (v9_max - v9_min)
        else:
            v9_normalized = np.full(len(v9_scores), 0.5)
        
        # Get V12 scores with sequential context
        logger.info("Computing V12 beat-level scores...")
        v12_scores = self.compute_v12_scores(features_tensor, beat_info)
        
        # Combine scores
        hybrid_scores = self.v9_weight * v9_normalized + self.v12_weight * v12_scores
        
        # Select beats to keep
        n_keep = max(1, int(n_beats * keep_ratio))
        threshold = np.sort(hybrid_scores)[::-1][min(n_keep - 1, len(hybrid_scores) - 1)]
        keep_mask = hybrid_scores >= threshold
        
        # Merge consecutive kept beats
        kept_regions = []
        in_region = False
        region_start = 0
        
        for i, keep in enumerate(keep_mask):
            if i < len(beat_positions):
                if keep and not in_region:
                    region_start = beat_positions[i][0]
                    in_region = True
                elif not keep and in_region:
                    kept_regions.append((region_start, beat_positions[i-1][1]))
                    in_region = False
        
        if in_region and len(beat_positions) > 0:
            kept_regions.append((region_start, beat_positions[-1][1]))
        
        # Merge close regions
        avg_beat_duration = (beat_info.beat_times[1] - beat_info.beat_times[0]) if len(beat_info.beat_times) > 1 else 0.5
        min_gap = int(2 * avg_beat_duration * self.sr)
        
        merged_regions = []
        for start, end in kept_regions:
            if merged_regions and start - merged_regions[-1][1] < min_gap:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        
        # Build output
        output_segments = [audio[s:e] for s, e in merged_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        
        sf.write(output_path, output_audio, self.sr)
        
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_beats': n_beats,
            'tempo': beat_info.tempo,
            'n_regions': len(merged_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'v9_score_stats': {
                'min': float(v9_normalized.min()),
                'max': float(v9_normalized.max()),
                'mean': float(v9_normalized.mean()),
                'std': float(v9_normalized.std())
            },
            'v12_score_stats': {
                'min': float(v12_scores.min()),
                'max': float(v12_scores.max()),
                'mean': float(v12_scores.mean()),
                'std': float(v12_scores.std())
            },
            'hybrid_score_stats': {
                'min': float(hybrid_scores.min()),
                'max': float(hybrid_scores.max()),
                'mean': float(hybrid_scores.mean()),
                'std': float(hybrid_scores.std())
            }
        }


# =============================================================================
# HYBRID V13 EDITOR: V9 base + V13 stem-separated beat-level
# =============================================================================

class HybridV13Editor:
    """
    Hybrid editor combining V9 quality + V13 stem-separated beat-level.
    """
    
    def __init__(self, model_dir: str = "./models", v9_weight: float = 0.5, v13_weight: float = 0.5):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        self.v9_weight = v9_weight
        self.v13_weight = v13_weight
        self.stem_separator = StemSeparator(sr=self.sr)
        self._load_v9_model()
        self._load_v13_model()
        logger.info(f"Hybrid V13 Editor loaded (V9={v9_weight}, V13={v13_weight})")
    
    def _load_v9_model(self):
        from train_edit_policy_v9 import DualHeadModel
        self.v9_feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
        self.v9_reference_centroid = np.load(self.model_dir / "reference_centroid_v9.npy")
        self.v9_similarity_weight = float(np.load(self.model_dir / "similarity_weight_v9.npy"))
        self.v9_model = DualHeadModel(self.v9_feature_dim, len(self.v9_reference_centroid)).to(DEVICE)
        checkpoint = torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
        self.v9_model.load_state_dict(checkpoint)
        self.v9_model.eval()
        self.v9_ref_centroid_t = torch.FloatTensor(self.v9_reference_centroid).to(DEVICE)
        from train_edit_policy_v10_simple import SegmentFeatureExtractor
        self.v9_extractor = SegmentFeatureExtractor(self.sr)
        logger.info("Loaded V9 model")
    
    def _load_v13_model(self):
        try:
            self.v13_feature_dim = int(np.load(self.model_dir / "feature_dim_v13.npy"))
            with open(self.model_dir / "scaler_v13.pkl", 'rb') as f:
                self.v13_scaler = pickle.load(f)
            context_beats = DEFAULT_CONTEXT_BEATS
            for model_file in ["policy_v13_trajectory_best.pt", "policy_v13_best.pt", "policy_v13_final.pt"]:
                path = self.model_dir / model_file
                if path.exists():
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                    if 'context_beats' in checkpoint:
                        context_beats = checkpoint['context_beats']
                    self.v13_model = BeatLevelPolicy(self.v13_feature_dim, 64, 256, context_beats).to(DEVICE)
                    self.v13_model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded V13 model: {model_file}")
                    break
            self.v13_model.eval()
            ref_path = self.model_dir / "reference_centroid_v13.npy"
            self.v13_reference_centroid = np.load(ref_path) if ref_path.exists() else np.zeros(self.v13_feature_dim)
            self.v13_loaded = True
        except Exception as e:
            logger.warning(f"Could not load V13: {e}")
            self.v13_loaded = False
    
    def compute_v9_scores_for_beats(self, beat_positions, audio):
        segment_samples = int(3.0 * self.sr)
        hop_samples = int(1.5 * self.sr)
        segments, segment_positions = [], []
        start = 0
        while start + segment_samples <= len(audio):
            segments.append(audio[start:start + segment_samples])
            segment_positions.append((start, start + segment_samples))
            start += hop_samples
        if not segments:
            return np.full(len(beat_positions), 0.5)
        features = np.array([self.v9_extractor.extract(seg) for seg in segments])
        from train_edit_policy_v10_simple import create_context_windows
        windowed = create_context_windows(features)
        windowed_tensor = torch.FloatTensor(windowed).to(DEVICE)
        with torch.no_grad():
            quality_logits, style_emb = self.v9_model(windowed_tensor)
            quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
            ref_sim = torch.mm(style_emb, self.v9_ref_centroid_t.unsqueeze(1)).squeeze()
            ref_sim = ((ref_sim + 1) / 2).cpu().numpy()
        segment_scores = quality + self.v9_similarity_weight * ref_sim
        beat_scores = []
        for beat_start, beat_end in beat_positions:
            best_seg, best_overlap = 0, 0
            for seg_idx, (seg_start, seg_end) in enumerate(segment_positions):
                overlap = max(0, min(beat_end, seg_end) - max(beat_start, seg_start))
                if overlap > best_overlap:
                    best_overlap, best_seg = overlap, seg_idx
            beat_scores.append(segment_scores[best_seg])
        return np.array(beat_scores)
    
    def compute_v13_scores(self, features_tensor, beat_info):
        if not self.v13_loaded:
            return np.zeros(len(features_tensor))
        ref_scaled = self.v13_scaler.transform(self.v13_reference_centroid.reshape(1, -1))
        ref_tensor = torch.tensor(ref_scaled[0], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            style_emb = self.v13_model.compute_style_embedding(ref_tensor)
            scores, beats_since_keep = [], 0
            for i in range(len(features_tensor)):
                logit, _ = self.v13_model.forward_single(features_tensor, i, style_emb, beat_info, beats_since_keep)
                score = torch.sigmoid(logit).item()
                scores.append(score)
                beats_since_keep = 0 if score > 0.5 else beats_since_keep + 1
        return np.array(scores)
    
    def process_track(self, input_path, output_path, keep_ratio=0.35):
        input_path_obj = Path(input_path)
        audio = load_audio_fast(input_path_obj, self.sr)
        duration = len(audio) / self.sr
        logger.info("Separating stems...")
        melodic_audio = self.stem_separator.get_guitar_focused_audio(audio, input_path_obj)
        logger.info("Detecting beats...")
        beat_info = detect_beats(audio, self.sr)
        n_beats = len(beat_info.beat_times) - 1
        if n_beats <= 0:
            raise ValueError("Beat detection failed")
        logger.info(f"Detected {n_beats} beats @ {beat_info.tempo:.1f} BPM")
        beat_samples = (beat_info.beat_times * self.sr).astype(int)
        melodic_beat_list, beat_positions = [], []
        for i in range(n_beats):
            start, end = beat_samples[i], beat_samples[i+1] if i+1 < len(beat_samples) else len(audio)
            if start < len(melodic_audio) and end > start:
                melodic_beat_list.append(melodic_audio[start:min(end, len(melodic_audio))])
                beat_positions.append((start, end))
        logger.info("Extracting features...")
        features = np.array([extract_features_fast(b, self.sr) for b in melodic_beat_list])
        features_scaled = self.v13_scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        logger.info("Computing V9 scores...")
        v9_scores = self.compute_v9_scores_for_beats(beat_positions, audio)
        v9_min, v9_max = v9_scores.min(), v9_scores.max()
        v9_normalized = (v9_scores - v9_min) / (v9_max - v9_min) if v9_max > v9_min else np.full(len(v9_scores), 0.5)
        logger.info("Computing V13 scores...")
        v13_scores = self.compute_v13_scores(features_tensor, beat_info)
        hybrid_scores = self.v9_weight * v9_normalized + self.v13_weight * v13_scores
        n_keep = max(1, int(n_beats * keep_ratio))
        threshold = np.sort(hybrid_scores)[::-1][min(n_keep - 1, len(hybrid_scores) - 1)]
        keep_mask = hybrid_scores >= threshold
        kept_regions, in_region, region_start = [], False, 0
        for i, keep in enumerate(keep_mask):
            if i < len(beat_positions):
                if keep and not in_region:
                    region_start, in_region = beat_positions[i][0], True
                elif not keep and in_region:
                    kept_regions.append((region_start, beat_positions[i-1][1]))
                    in_region = False
        if in_region and beat_positions:
            kept_regions.append((region_start, beat_positions[-1][1]))
        avg_beat = (beat_info.beat_times[1] - beat_info.beat_times[0]) if len(beat_info.beat_times) > 1 else 0.5
        min_gap = int(2 * avg_beat * self.sr)
        merged_regions = []
        for start, end in kept_regions:
            if merged_regions and start - merged_regions[-1][1] < min_gap:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        output_segments = [audio[s:e] for s, e in merged_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        sf.write(output_path, output_audio, self.sr)
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_beats': n_beats,
            'tempo': beat_info.tempo,
            'n_regions': len(merged_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'v9_score_stats': {'min': float(v9_normalized.min()), 'max': float(v9_normalized.max()), 'mean': float(v9_normalized.mean())},
            'v13_score_stats': {'min': float(v13_scores.min()), 'max': float(v13_scores.max()), 'mean': float(v13_scores.mean())},
            'hybrid_score_stats': {'min': float(hybrid_scores.min()), 'max': float(hybrid_scores.max()), 'mean': float(hybrid_scores.mean())}
        }


# =============================================================================
# HYBRID V11 EDITOR: V9 base + V11 refinement
# =============================================================================

class HybridV11Editor:
    """
    Hybrid editor that combines:
    - V9's quality scoring (generalizes to unseen tracks via reference contrastive learning)
    - V11's alignment-learned patterns (user-specific preferences from training pairs)
    
    Final score = v9_weight * V9_score + v11_weight * V11_adjustment
    
    V9 provides the foundation (what sounds good in general)
    V11 provides refinement (what this specific user prefers to keep/cut)
    """
    
    def __init__(self, model_dir: str = "./models", v9_weight: float = 0.55, v11_weight: float = 0.45):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        self.segment_duration = 3.0  # V9 uses 3s segments
        self.hop_duration = 1.5
        
        self.v9_weight = v9_weight
        self.v11_weight = v11_weight
        
        self._load_v9_model()
        self._load_v11_model()
        
        logger.info(f"Hybrid V11 Editor loaded (V9 weight={v9_weight}, V11 weight={v11_weight})")
    
    def _load_v9_model(self):
        """Load V9 quality model."""
        from train_edit_policy_v9 import DualHeadModel
        
        self.v9_feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
        self.v9_reference_centroid = np.load(self.model_dir / "reference_centroid_v9.npy")
        self.v9_similarity_weight = float(np.load(self.model_dir / "similarity_weight_v9.npy"))
        
        self.v9_model = DualHeadModel(
            base_feature_dim=self.v9_feature_dim,
            embedding_dim=len(self.v9_reference_centroid)
        ).to(DEVICE)
        
        checkpoint = torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
        self.v9_model.load_state_dict(checkpoint)
        self.v9_model.eval()
        
        self.v9_ref_centroid_t = torch.FloatTensor(self.v9_reference_centroid).to(DEVICE)
        
        # V9 feature extractor
        from train_edit_policy_v10_simple import SegmentFeatureExtractor
        self.v9_extractor = SegmentFeatureExtractor(self.sr)
        
        logger.info("Loaded V9 model")
    
    def _load_v11_model(self):
        """Load V11 imitation model."""
        try:
            self.v11_feature_dim = int(np.load(self.model_dir / "feature_dim_v11.npy"))
            
            with open(self.model_dir / "scaler_v11.pkl", 'rb') as f:
                self.v11_scaler = pickle.load(f)
            
            self.v11_model = ImitationPolicy(
                feature_dim=self.v11_feature_dim, style_dim=64, hidden_dim=256
            ).to(DEVICE)
            
            for model_file in ["policy_v11_best.pt", "policy_v11_final.pt"]:
                path = self.model_dir / model_file
                if path.exists():
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                    self.v11_model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded V11 model: {model_file}")
                    break
            
            self.v11_model.eval()
            
            ref_path = self.model_dir / "reference_centroid_v11.npy"
            if ref_path.exists():
                self.v11_reference_centroid = np.load(ref_path)
            else:
                self.v11_reference_centroid = np.zeros(self.v11_feature_dim)
                
            self.v11_loaded = True
        except Exception as e:
            logger.warning(f"Could not load V11 model: {e}, using V9 only")
            self.v11_loaded = False
    
    def extract_segments(self, audio: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract overlapping segments."""
        segment_samples = int(self.segment_duration * self.sr)
        hop_samples = int(self.hop_duration * self.sr)
        
        segments = []
        positions = []
        
        start = 0
        while start + segment_samples <= len(audio):
            segments.append(audio[start:start + segment_samples])
            positions.append((start, start + segment_samples))
            start += hop_samples
        
        return segments, positions
    
    def compute_v9_scores(self, segments: List[np.ndarray]) -> np.ndarray:
        """Compute V9 quality scores."""
        # Extract V9 features
        features = []
        for seg in segments:
            feat = self.v9_extractor.extract(seg)
            features.append(feat)
        features = np.array(features)
        
        # Create context windows (V9 uses 3-segment windows)
        from train_edit_policy_v10_simple import create_context_windows
        windowed = create_context_windows(features)
        windowed_tensor = torch.FloatTensor(windowed).to(DEVICE)
        
        with torch.no_grad():
            quality_logits, style_emb = self.v9_model(windowed_tensor)
            quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
            
            # Reference similarity
            ref_sim = torch.mm(style_emb, self.v9_ref_centroid_t.unsqueeze(1)).squeeze()
            ref_sim = ((ref_sim + 1) / 2).cpu().numpy()  # Map to [0, 1]
        
        # Combined V9 score
        combined = quality + self.v9_similarity_weight * ref_sim
        
        return combined
    
    def compute_v11_adjustments(self, segments: List[np.ndarray], n_total: int) -> np.ndarray:
        """Compute V11 imitation-based adjustments."""
        if not self.v11_loaded:
            return np.zeros(len(segments))
        
        # Extract V11 features (uses shorter 1.5s segments internally, but we adapt)
        # For simplicity, extract features from the 3s segments using V11's approach
        features = []
        for seg in segments:
            # Use middle 1.5s portion to match V11's segment duration
            mid_start = len(seg) // 4  # Start at 25% 
            mid_end = mid_start + int(1.5 * self.sr)  # Take 1.5s
            subseg = seg[mid_start:mid_end]
            feat = extract_features_fast(subseg, self.sr)
            features.append(feat)
        features = np.array(features)
        
        # Scale features
        features_scaled = self.v11_scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        
        # Position info
        n = len(segments)
        positions = torch.tensor([i / n_total for i in range(n)]).to(DEVICE)
        position_info = torch.stack([
            positions,
            torch.full((n,), 0.35, device=DEVICE, dtype=torch.float32),
            torch.zeros(n, device=DEVICE, dtype=torch.float32)
        ], dim=1)
        
        # Style embedding
        ref_scaled = self.v11_scaler.transform(self.v11_reference_centroid.reshape(1, -1))
        ref_tensor = torch.tensor(ref_scaled.mean(axis=0), dtype=torch.float32).to(DEVICE)
        style_emb = self.v11_model.compute_style_embedding(ref_tensor.unsqueeze(0))
        style_emb = style_emb.expand(n, -1)
        
        # Get V11 scores
        with torch.no_grad():
            logits, _, _ = self.v11_model(features_tensor, None, position_info, style_emb)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Convert to adjustments: V11's opinion on keep vs cut
        # Center around 0: positive = V11 wants to keep, negative = V11 wants to cut
        adjustments = probs - 0.5  # Range: [-0.5, 0.5]
        
        return adjustments
    
    def score_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Compute hybrid scores for segments.
        
        Returns scores where higher = more likely to keep.
        """
        n = len(segments)
        
        # Get V9 base scores
        v9_scores = self.compute_v9_scores(segments)
        
        # Normalize V9 scores to [0, 1] range
        v9_min, v9_max = v9_scores.min(), v9_scores.max()
        if v9_max > v9_min:
            v9_normalized = (v9_scores - v9_min) / (v9_max - v9_min)
        else:
            v9_normalized = np.full(n, 0.5)
        
        # Get V11 adjustments
        v11_adjustments = self.compute_v11_adjustments(segments, n)
        
        # Combine: V9 base + V11 adjustment
        # V11 adjustments are in [-0.5, 0.5] range
        hybrid_scores = self.v9_weight * v9_normalized + self.v11_weight * (v11_adjustments + 0.5)
        
        return hybrid_scores
    
    def extract_tempo(self, segment: np.ndarray) -> float:
        """Estimate tempo of a segment using onset detection."""
        try:
            onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)[0]
            return float(tempo)
        except:
            return 120.0  # Default tempo
    
    def extract_chroma(self, segment: np.ndarray) -> np.ndarray:
        """Extract chroma features for key/harmony analysis."""
        try:
            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sr)
            return chroma.mean(axis=1)  # Average over time
        except:
            return np.zeros(12)
    
    def compute_transition_scores(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Compute transition compatibility scores between consecutive segments.
        
        Returns matrix where transition_scores[i, j] is how well segment j
        follows segment i (higher = better transition).
        """
        n = len(segments)
        
        # Extract features for all segments
        tempos = np.array([self.extract_tempo(seg) for seg in segments])
        chromas = np.array([self.extract_chroma(seg) for seg in segments])
        
        # Compute pairwise transition scores
        transition_scores = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    transition_scores[i, j] = 1.0
                    continue
                
                # Tempo compatibility: penalize large tempo changes
                tempo_ratio = min(tempos[i], tempos[j]) / max(tempos[i], tempos[j] + 1e-6)
                tempo_score = tempo_ratio ** 2  # Squared to penalize differences more
                
                # Key/chroma compatibility: cosine similarity
                chroma_i = chromas[i]
                chroma_j = chromas[j]
                norm_i = np.linalg.norm(chroma_i) + 1e-6
                norm_j = np.linalg.norm(chroma_j) + 1e-6
                chroma_sim = np.dot(chroma_i, chroma_j) / (norm_i * norm_j)
                chroma_score = (chroma_sim + 1) / 2  # Map from [-1,1] to [0,1]
                
                # Combined transition score
                transition_scores[i, j] = 0.5 * tempo_score + 0.5 * chroma_score
        
        return transition_scores
    
    def select_segments_with_transitions(self, scores: np.ndarray, 
                                         transition_scores: np.ndarray,
                                         target_keep: int,
                                         transition_weight: float = 0.3) -> np.ndarray:
        """
        Select segments using dynamic programming to optimize both
        segment quality and transition smoothness.
        
        Uses a greedy approach with lookahead to find good sequences.
        """
        n = len(scores)
        
        # Normalize quality scores to [0, 1]
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones(n) * 0.5
        
        # Greedy selection with transition awareness
        selected = []
        available = set(range(n))
        
        # Start with the highest scoring segment
        first = np.argmax(norm_scores)
        selected.append(first)
        available.remove(first)
        
        while len(selected) < target_keep and available:
            last_selected = selected[-1]
            
            best_next = None
            best_combined = -np.inf
            
            for candidate in available:
                # Quality score
                quality = norm_scores[candidate]
                
                # Transition score from last selected segment
                # Consider both forward and temporal proximity
                trans = transition_scores[last_selected, candidate]
                
                # Bonus for segments that are temporally close (maintains structure)
                temporal_dist = abs(candidate - last_selected)
                temporal_bonus = 1.0 / (1.0 + temporal_dist * 0.1)
                
                # Combined score
                combined = (1 - transition_weight) * quality + transition_weight * (0.7 * trans + 0.3 * temporal_bonus)
                
                if combined > best_combined:
                    best_combined = combined
                    best_next = candidate
            
            if best_next is not None:
                selected.append(best_next)
                available.remove(best_next)
            else:
                break
        
        # Create keep mask
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[selected] = True
        
        return keep_mask
    
    def process_track(self, input_path: str, output_path: str,
                     keep_ratio: float = 0.35,
                     use_transitions: bool = True,
                     transition_weight: float = 0.3) -> Dict:
        """Process a track using hybrid V9+V11 scoring with transition awareness."""
        
        audio = load_audio_fast(Path(input_path), self.sr)
        duration = len(audio) / self.sr
        
        segments, positions = self.extract_segments(audio)
        n_segments = len(segments)
        
        if n_segments == 0:
            raise ValueError("Audio too short")
        
        logger.info(f"Processing {n_segments} segments...")
        
        # Get hybrid scores
        scores = self.score_segments(segments)
        
        n_keep = max(1, int(n_segments * keep_ratio))
        
        if use_transitions and n_segments > 10:
            # Compute transition scores
            logger.info("Computing transition scores for tempo/key continuity...")
            transition_scores = self.compute_transition_scores(segments)
            
            # Select with transition awareness
            keep_mask = self.select_segments_with_transitions(
                scores, transition_scores, n_keep, transition_weight
            )
        else:
            # Simple threshold selection
            threshold = np.sort(scores)[::-1][min(n_keep - 1, len(scores) - 1)]
            keep_mask = scores >= threshold
        
        # Sort kept segments by original position for output
        kept_indices = np.where(keep_mask)[0]
        kept_indices = np.sort(kept_indices)  # Maintain temporal order
        
        # Merge consecutive/close kept regions
        kept_regions = []
        if len(kept_indices) > 0:
            region_start = positions[kept_indices[0]][0]
            region_end = positions[kept_indices[0]][1]
            
            for idx in kept_indices[1:]:
                seg_start, seg_end = positions[idx]
                # If this segment is close to the current region, extend it
                if seg_start - region_end < int(0.5 * self.sr):
                    region_end = seg_end
                else:
                    kept_regions.append((region_start, region_end))
                    region_start = seg_start
                    region_end = seg_end
            
            kept_regions.append((region_start, region_end))
        
        # Build output
        output_segments = [audio[s:e] for s, e in kept_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        
        sf.write(output_path, output_audio, self.sr)
        
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_segments': n_segments,
            'n_regions': len(kept_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'score_stats': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="V13 Stem-Separated Beat-Level Editor Training & Testing")
    parser.add_argument("--mode", choices=["train", "train_trajectory", "test"],
                       default="train", help="Mode: train, train_trajectory, test")
    parser.add_argument("--input", type=str, help="Input file for test mode")
    parser.add_argument("--output", type=str, help="Output file for test mode")
    parser.add_argument("--keep-ratio", type=float, default=0.35, help="Keep ratio (default: 0.35)")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--trajectory-epochs", type=int, default=60, help="Trajectory training epochs")
    parser.add_argument("--model-dir", type=str, default="./models", help="Model directory")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # V13 beat-level training with stem separation
        print("=" * 70)
        print("V13 STEM-SEPARATED TRAINING")
        print("Uses Demucs to extract vocals + other (melodic content)")
        print("=" * 70)
        
        trainer = V13Trainer(
            input_dir="./training_data/input",
            output_dir="./training_data/desired_output",
            reference_dir="./training_data/reference",
            model_dir=args.model_dir
        )
        trainer.train(epochs=args.epochs)
        
    elif args.mode == "train_trajectory":
        # Trajectory training (TBPTT) for sequential context
        print("=" * 70)
        print("V13 TRAJECTORY TRAINING (TBPTT) with Stem Separation")
        print("=" * 70)
        
        trainer = V13Trainer(
            input_dir="./training_data/input",
            output_dir="./training_data/desired_output",
            reference_dir="./training_data/reference",
            model_dir=args.model_dir
        )
        
        # First prepare data to initialize feature_dim and scaler
        training_data = trainer.prepare_training_data()
        
        # Initialize model (prepare_training_data sets feature_dim)
        trainer.model = BeatLevelPolicy(
            feature_dim=trainer.feature_dim,
            style_dim=64,
            hidden_dim=256,
            context_beats=DEFAULT_CONTEXT_BEATS
        ).to(DEVICE)
        print(f"Initialized model with feature_dim={trainer.feature_dim}")
        
        # Load best weights if available
        for model_file in ["policy_v13_best.pt", "policy_v13_final.pt"]:
            path = Path(args.model_dir) / model_file
            if path.exists():
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded existing V13 model weights: {model_file}")
                break
        
        trainer.train_with_trajectory(training_data, epochs=args.trajectory_epochs)
        
    elif args.mode == "test":
        # Test V13 editor (with stem separation)
        if not args.input:
            parser.error("--input required for test mode")
        
        output = args.output or args.input.replace(".wav", "_v13_out.wav")
        
        editor = V13Editor(args.model_dir)
        result = editor.process_track(args.input, output, keep_ratio=args.keep_ratio)
        
        print("\n" + "=" * 70)
        print("V13 STEM-SEPARATED EDIT RESULT")
        print("=" * 70)
        print(f"Input:  {result['input_duration']:.1f}s")
        print(f"Output: {result['output_duration']:.1f}s ({result['keep_ratio_actual']*100:.1f}% kept)")
        print(f"Beats:  {result['n_beats']} @ {result['tempo']:.1f} BPM")
        print(f"Regions: {result['n_regions']}")
        if 'score_stats' in result:
            stats = result['score_stats']
            print(f"\nScores: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                  f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print(f"\nOutput saved to: {output}")

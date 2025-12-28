"""Shared audio processing utilities.

Provides common audio loading, mel spectrogram extraction, beat detection,
and other audio analysis functions used by both rl_editor and super_editor.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import librosa
import soundfile as sf

from .audio_config import AudioConfig, get_default_audio_config

logger = logging.getLogger(__name__)


# =============================================================================
# In-memory audio caching
# =============================================================================

_audio_cache: Dict[Tuple[str, int, bool], Tuple[np.ndarray, int]] = {}
_audio_cache_enabled = True
_audio_cache_max_size_gb = 8.0
_audio_cache_current_size = 0


def set_audio_cache_enabled(enabled: bool) -> None:
    """Enable or disable in-memory audio caching."""
    global _audio_cache_enabled
    _audio_cache_enabled = enabled


def clear_audio_cache() -> None:
    """Clear the in-memory audio cache."""
    global _audio_cache, _audio_cache_current_size
    _audio_cache.clear()
    _audio_cache_current_size = 0
    logger.info("Audio cache cleared")


def get_audio_cache_stats() -> Dict[str, Any]:
    """Get audio cache statistics."""
    return {
        "enabled": _audio_cache_enabled,
        "n_cached": len(_audio_cache),
        "size_mb": _audio_cache_current_size / (1024 * 1024),
        "max_size_gb": _audio_cache_max_size_gb,
    }


# =============================================================================
# Audio I/O
# =============================================================================

def load_audio(
    filepath: str,
    sr: int = 22050,
    mono: bool = True,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load audio file with optional in-memory caching.

    Args:
        filepath: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        use_cache: Whether to use in-memory cache
        verbose: Log loading info

    Returns:
        Tuple of (audio array, sample rate)
    """
    global _audio_cache, _audio_cache_current_size

    cache_key = (str(filepath), sr, mono)

    # Check cache first
    if use_cache and _audio_cache_enabled and cache_key in _audio_cache:
        y, sr_loaded = _audio_cache[cache_key]
        if verbose:
            logger.debug(f"Cache hit for {filepath}")
        return y.copy(), sr_loaded

    try:
        y, sr_loaded = librosa.load(filepath, sr=sr, mono=mono)

        if verbose:
            logger.info(f"Loaded audio from {filepath}: shape={y.shape}, sr={sr_loaded}")

        # Add to cache if enabled and within size limit
        if use_cache and _audio_cache_enabled:
            audio_size = y.nbytes
            max_size_bytes = _audio_cache_max_size_gb * 1024 * 1024 * 1024

            if _audio_cache_current_size + audio_size < max_size_bytes:
                _audio_cache[cache_key] = (y.copy(), sr_loaded)
                _audio_cache_current_size += audio_size

        return y, sr_loaded
    except Exception as e:
        logger.error(f"Failed to load audio from {filepath}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    filepath: str,
    sr: int = 22050,
    subtype: str = "PCM_16",
    verbose: bool = False,
) -> None:
    """Save audio file.

    Args:
        audio: Audio array
        filepath: Path to save audio
        sr: Sample rate
        subtype: Audio subtype (PCM_16, FLOAT, etc.)
        verbose: Log save info
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filepath, audio, sr, subtype=subtype)
        if verbose:
            logger.info(f"Saved audio to {filepath}: shape={audio.shape}, sr={sr}")
    except Exception as e:
        logger.error(f"Failed to save audio to {filepath}: {e}")
        raise


# =============================================================================
# Mel Spectrogram
# =============================================================================

def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power_to_db: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Compute mel-spectrogram from audio.

    Args:
        y: Audio array
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length (defaults to n_fft)
        fmin: Minimum frequency
        fmax: Maximum frequency
        power_to_db: Convert to dB scale
        normalize: Normalize to [0, 1] range

    Returns:
        Mel spectrogram (n_mels, n_frames) or (T, n_mels) if transposed
    """
    if win_length is None:
        win_length = n_fft

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )

    if power_to_db:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if normalize:
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-8)

    return mel_spec


def compute_mel_spectrogram_from_config(
    y: np.ndarray,
    config: Optional[AudioConfig] = None,
    normalize: bool = True,
    transpose: bool = True,
) -> np.ndarray:
    """Compute mel spectrogram using AudioConfig.

    Args:
        y: Audio array
        config: AudioConfig (uses defaults if None)
        normalize: Normalize to [0, 1] range
        transpose: Return (T, n_mels) instead of (n_mels, T)

    Returns:
        Mel spectrogram
    """
    if config is None:
        config = get_default_audio_config()

    mel = compute_mel_spectrogram(
        y=y,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
        fmax=config.fmax,
        power_to_db=True,
        normalize=normalize,
    )

    if transpose:
        mel = mel.T  # (T, n_mels)

    return mel.astype(np.float32)


# =============================================================================
# Beat Detection
# =============================================================================

def detect_beats(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect beats in audio signal.

    Args:
        y: Audio array
        sr: Sample rate
        hop_length: Hop length for onset detection
        verbose: Log beat count

    Returns:
        Tuple of (beat_frames, beat_times)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    if verbose:
        logger.info(f"Detected {len(beat_frames)} beats at {tempo:.1f} BPM")

    return beat_frames, beat_times


def detect_beats_from_audio(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load audio and detect beats.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Tuple of (beat_frames, beat_times, tempo)
    """
    y, sr = load_audio(audio_path, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    return beat_frames, beat_times, float(tempo)


# =============================================================================
# Tempo and Energy
# =============================================================================

def estimate_tempo(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    verbose: bool = True,
) -> float:
    """Estimate tempo from audio signal.

    Args:
        y: Audio array
        sr: Sample rate
        hop_length: Hop length
        verbose: Log tempo

    Returns:
        Estimated tempo in BPM
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]

    if verbose:
        logger.info(f"Estimated tempo: {tempo:.1f} BPM")

    return float(tempo)


def get_energy_contour(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> np.ndarray:
    """Compute RMS energy contour over time.

    Args:
        y: Audio array
        sr: Sample rate
        hop_length: Hop length
        frame_length: Frame length for RMS

    Returns:
        Energy contour (n_frames,)
    """
    energy = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    return energy


def get_spectral_centroid(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute spectral centroid over time.

    Args:
        y: Audio array
        sr: Sample rate
        hop_length: Hop length
        n_fft: FFT size

    Returns:
        Spectral centroid (n_frames,)
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    return centroid


# =============================================================================
# MelExtractor class (for compatibility with super_editor)
# =============================================================================

class MelExtractor:
    """Extract mel spectrograms from audio files.

    Higher-level class that wraps the audio loading and mel extraction.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize extractor with audio config.

        Args:
            config: AudioConfig instance (uses defaults if None)
        """
        self.config = config or get_default_audio_config()

    def extract(self, audio_path: str, normalize: bool = True) -> np.ndarray:
        """Extract mel spectrogram from audio file.

        Args:
            audio_path: Path to audio file
            normalize: Normalize to [0, 1]

        Returns:
            mel: Mel spectrogram (T, n_mels)
        """
        y, sr = load_audio(audio_path, sr=self.config.sample_rate)
        return compute_mel_spectrogram_from_config(
            y, self.config, normalize=normalize, transpose=True
        )

    def extract_with_beats(
        self, audio_path: str, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mel spectrogram and detect beats.

        Args:
            audio_path: Path to audio file
            normalize: Normalize mel to [0, 1]

        Returns:
            mel: Mel spectrogram (T, n_mels)
            beat_times: Beat times in seconds (n_beats,)
        """
        y, sr = load_audio(audio_path, sr=self.config.sample_rate)

        mel = compute_mel_spectrogram_from_config(
            y, self.config, normalize=normalize, transpose=True
        )

        _, beat_times = detect_beats(
            y, sr=self.config.sample_rate, hop_length=self.config.hop_length
        )

        return mel, beat_times.astype(np.float32)

    def extract_with_features(
        self, audio_path: str, normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract mel spectrogram and additional features.

        Args:
            audio_path: Path to audio file
            normalize: Normalize mel to [0, 1]

        Returns:
            Dictionary with 'mel', 'beat_times', 'tempo', 'energy'
        """
        y, sr = load_audio(audio_path, sr=self.config.sample_rate)

        mel = compute_mel_spectrogram_from_config(
            y, self.config, normalize=normalize, transpose=True
        )

        beat_frames, beat_times, tempo = detect_beats_from_audio(
            audio_path, sr=self.config.sample_rate, hop_length=self.config.hop_length
        )

        energy = get_energy_contour(
            y, sr=self.config.sample_rate, hop_length=self.config.hop_length
        )

        return {
            'mel': mel,
            'beat_times': beat_times.astype(np.float32),
            'beat_frames': beat_frames,
            'tempo': tempo,
            'energy': energy.astype(np.float32),
        }

    def save(
        self,
        mel: np.ndarray,
        output_path: str,
        beat_times: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Save mel spectrogram to npz file.

        Args:
            mel: Mel spectrogram
            output_path: Path to save
            beat_times: Optional beat times
            **kwargs: Additional arrays to save
        """
        data = {'mel': mel}
        if beat_times is not None:
            data['beat_times'] = beat_times
        data.update(kwargs)
        np.savez_compressed(output_path, **data)


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass


def griffin_lim(
    mel: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
) -> np.ndarray:
    """Convert mel spectrogram to audio using Griffin-Lim.

    Args:
        mel: Mel spectrogram (T, n_mels) or (n_mels, T)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations

    Returns:
        Audio waveform
    """
    # Ensure (n_mels, T) format
    if mel.ndim == 2 and mel.shape[0] > mel.shape[1]:
        mel = mel.T

    # Invert mel to linear spectrogram
    mel_inv = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=n_fft)

    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(mel_inv, n_iter=n_iter, hop_length=hop_length)

    return audio

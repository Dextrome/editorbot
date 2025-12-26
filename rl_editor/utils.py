"""Utility functions for RL-based audio editor."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import librosa
import soundfile as sf


logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, name: str = "rl_editor") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory for log files
        name: Logger name

    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_path / f"{name}.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger_instance.addHandler(fh)
    logger_instance.addHandler(ch)

    return logger_instance


# In-memory audio cache for fast repeated access
_audio_cache: Dict[Tuple[str, int, bool], Tuple[np.ndarray, int]] = {}
_audio_cache_enabled = True
_audio_cache_max_size_gb = 8.0  # Max cache size in GB
_audio_cache_current_size = 0  # Current size in bytes


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


def get_audio_cache_stats() -> Dict[str, any]:
    """Get audio cache statistics."""
    return {
        "enabled": _audio_cache_enabled,
        "n_cached": len(_audio_cache),
        "size_mb": _audio_cache_current_size / (1024 * 1024),
        "max_size_gb": _audio_cache_max_size_gb,
    }


def load_audio(
    filepath: str, sr: int = 22050, mono: bool = True, verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """Load audio file with optional in-memory caching.

    Args:
        filepath: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        verbose: Log loading info (default False to reduce noise)

    Returns:
        Tuple of (audio array, sample rate)
    """
    global _audio_cache, _audio_cache_current_size
    
    cache_key = (filepath, sr, mono)
    
    # Check cache first
    if _audio_cache_enabled and cache_key in _audio_cache:
        y, sr_loaded = _audio_cache[cache_key]
        if verbose:
            logger.debug(f"Cache hit for {filepath}")
        return y.copy(), sr_loaded  # Return copy to prevent mutation
    
    try:
        y, sr_loaded = librosa.load(filepath, sr=sr, mono=mono)
        
        if verbose:
            logger.info(f"Loaded audio from {filepath}: shape={y.shape}, sr={sr_loaded}")
        
        # Add to cache if enabled and within size limit
        if _audio_cache_enabled:
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
    audio: np.ndarray, filepath: str, sr: int = 22050, subtype: str = "PCM_16",
    verbose: bool = False
) -> None:
    """Save audio file.

    Args:
        audio: Audio array
        filepath: Path to save audio
        sr: Sample rate
        subtype: Audio subtype (PCM_16, FLOAT, etc.)
        verbose: Log save info (default False to reduce noise)
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filepath, audio, sr, subtype=subtype)
        if verbose:
            logger.info(f"Saved audio to {filepath}: shape={audio.shape}, sr={sr}")
    except Exception as e:
        logger.error(f"Failed to save audio to {filepath}: {e}")
        raise


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Compute mel-spectrogram features.

    Args:
        y: Audio array
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel-spectrogram (n_mels, n_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def detect_beats(y: np.ndarray, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray]:
    """Detect beats in audio signal.

    Args:
        y: Audio array
        sr: Sample rate

    Returns:
        Tuple of (beat_frames, beat_times)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, units="frames"
    )[1]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    logger.info(f"Detected {len(beat_frames)} beats")
    return beat_frames, beat_times


def estimate_tempo(y: np.ndarray, sr: int = 22050) -> float:
    """Estimate tempo from audio signal.

    Args:
        y: Audio array
        sr: Sample rate

    Returns:
        Estimated tempo in BPM
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
    logger.info(f"Estimated tempo: {tempo:.1f} BPM")
    return float(tempo)


def get_energy_contour(
    y: np.ndarray, sr: int = 22050, hop_length: int = 512
) -> np.ndarray:
    """Compute energy contour over time.

    Args:
        y: Audio array
        sr: Sample rate
        hop_length: Hop length for frame-wise computation

    Returns:
        Energy contour (n_frames,)
    """
    # Frame-wise RMS energy
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    energy = librosa.feature.rms(S=librosa.power_to_db(S))[0]
    return energy


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

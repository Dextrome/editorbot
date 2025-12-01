"""Helper utility functions."""

import numpy as np


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string (MM:SS.mmm or HH:MM:SS.mmm).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def db_to_linear(db: float) -> float:
    """
    Convert decibels to linear amplitude.

    Args:
        db: Value in decibels.

    Returns:
        Linear amplitude value.
    """
    return 10 ** (db / 20)


def linear_to_db(linear: float, min_db: float = -100.0) -> float:
    """
    Convert linear amplitude to decibels.

    Args:
        linear: Linear amplitude value.
        min_db: Minimum dB value to return for zero/negative inputs.

    Returns:
        Value in decibels.
    """
    if linear <= 0:
        return min_db
    return 20 * np.log10(linear)


def samples_to_time(samples: int, sample_rate: int) -> float:
    """
    Convert sample count to time in seconds.

    Args:
        samples: Number of samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Time in seconds.
    """
    return samples / sample_rate


def time_to_samples(time_seconds: float, sample_rate: int) -> int:
    """
    Convert time in seconds to sample count.

    Args:
        time_seconds: Time in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Number of samples.
    """
    return int(time_seconds * sample_rate)


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to a target peak level.

    Args:
        audio: Audio data.
        target_db: Target peak level in dB.

    Returns:
        Normalized audio data.
    """
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio

    target_linear = db_to_linear(target_db)
    gain = target_linear / current_peak
    return audio * gain


def fade_in(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """
    Apply a fade-in to audio.

    Args:
        audio: Audio data.
        duration_samples: Fade duration in samples.

    Returns:
        Audio with fade-in applied.
    """
    fade_curve = np.linspace(0, 1, duration_samples)
    result = audio.copy()
    result[:duration_samples] *= fade_curve
    return result


def fade_out(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """
    Apply a fade-out to audio.

    Args:
        audio: Audio data.
        duration_samples: Fade duration in samples.

    Returns:
        Audio with fade-out applied.
    """
    fade_curve = np.linspace(1, 0, duration_samples)
    result = audio.copy()
    result[-duration_samples:] *= fade_curve
    return result


def crossfade(
    audio1: np.ndarray, audio2: np.ndarray, crossfade_samples: int
) -> np.ndarray:
    """
    Crossfade between two audio segments.

    Args:
        audio1: First audio segment.
        audio2: Second audio segment.
        crossfade_samples: Duration of crossfade in samples.

    Returns:
        Combined audio with crossfade.
    """
    # Fade out audio1
    fade_out_curve = np.linspace(1, 0, crossfade_samples)
    audio1_end = audio1[-crossfade_samples:] * fade_out_curve

    # Fade in audio2
    fade_in_curve = np.linspace(0, 1, crossfade_samples)
    audio2_start = audio2[:crossfade_samples] * fade_in_curve

    # Combine
    crossfaded = audio1_end + audio2_start

    return np.concatenate([audio1[:-crossfade_samples], crossfaded, audio2[crossfade_samples:]])

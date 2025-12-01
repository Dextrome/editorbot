"""
remixatron_blend.py - Utility for stem-based blending between beats for RemixatronAdapter.
"""
import numpy as np
from src.audio.ai_blend import AITransitionBlender
from typing import Optional

def soft_clip(arr: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft-tanh limiter to prevent harsh clipping."""
    peak = np.max(np.abs(arr))
    if peak > threshold:
        return threshold * np.tanh(arr / threshold)
    return arr

def blend_beats(
    prev_beat: np.ndarray,
    next_beat: np.ndarray,
    sample_rate: int = 44100,
    blend_duration: float = 0.2,
    prev_stems: dict = None,
    next_stems: dict = None,
    demucs_device: Optional[str] = None,
) -> np.ndarray:
    """
    Blend the end of prev_beat and start of next_beat using Demucs-based blending.
    """
    # Determine desired blend samples and ensure we only blend if both beats can provide the full blend window
    target_n_blend = int(sample_rate * blend_duration)
    def ensure_stereo(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return np.stack([arr, arr], axis=-1)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                return arr
            if arr.shape[0] == 2:
                return arr.T
            if arr.shape[1] == 1:
                return np.concatenate([arr, arr], axis=1)
            if arr.shape[0] == 1:
                mono = arr.flatten()
                return np.stack([mono, mono], axis=-1)
            if arr.shape[1] < arr.shape[0]:
                return arr[:, :2] if arr.shape[1] >= 2 else np.concatenate([arr, arr], axis=1)
            return arr.T[:, :2]
        return np.zeros((0, 2), dtype=arr.dtype)
    max_blend = min(len(prev_beat), len(next_beat))
    # If either beat is shorter than the desired blend window, fall back to simple concatenation
    if len(prev_beat) < target_n_blend or len(next_beat) < target_n_blend:
        return np.concatenate([ensure_stereo(prev_beat), ensure_stereo(next_beat)])
    n_blend = min(target_n_blend, max_blend)

    if len(prev_beat) < n_blend or len(next_beat) < n_blend:
        # Not enough samples, just concatenate
        return np.concatenate([ensure_stereo(prev_beat), ensure_stereo(next_beat)])
    blender = AITransitionBlender(sample_rate=sample_rate, demucs_device=demucs_device)
    # Use pre-separated stems for blending
    blend = blender.blend_sections(
        prev_beat[-n_blend:],
        next_beat[:n_blend],
        blend_duration=blend_duration,
        end_stems=prev_stems,
        start_stems=next_stems
    )
    # Remove overlap from prev_beat and next_beat, insert blend
    result = np.concatenate([
        ensure_stereo(prev_beat[:-n_blend]),
        ensure_stereo(blend),
        ensure_stereo(next_beat[n_blend:])
    ])
    # Apply soft clipping to the result to prevent harsh clipping
    return soft_clip(result, threshold=0.95)

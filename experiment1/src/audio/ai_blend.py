"""AI-based audio blending using Demucs for section transitions."""

import logging
import numpy as np
from .demucs_wrapper import DemucsSeparator

logger = logging.getLogger(__name__)

class AITransitionBlender:
    def __init__(self, sample_rate=44100, stem_method: str = 'demucs', demucs_device: str | None = None):
        self.sample_rate = sample_rate
        # Preferred stem method (currently only 'demucs' supported)
        self.stem_method = stem_method
        # Demucs device override (cpu/cuda) - passed when using the separator
        self.demucs_device = demucs_device
        self.demucs_separator = DemucsSeparator()

    def blend_sections(self, end_audio: np.ndarray, start_audio: np.ndarray, blend_duration: float = 1.0,
                      end_stems: dict = None, start_stems: dict = None) -> np.ndarray:
        """
        Blend the end of one section and the start of the next using Demucs stem separation and crossfade.
        Args:
            end_audio: Last N seconds of previous section.
            start_audio: First N seconds of next section.
            blend_duration: Duration in seconds to blend.
            end_stems: Pre-separated stems for end_audio (optional, avoids Demucs call)
            start_stems: Pre-separated stems for start_audio (optional, avoids Demucs call)
        Returns:
            Blended audio of length blend_duration.
        """
        # Determine number of samples to blend: can't be larger than either slice
        n_samples = int(self.sample_rate * blend_duration)
        # Ensure we don't exceed provided slice lengths
        n_samples = min(n_samples, len(end_audio), len(start_audio))
        end_audio = end_audio[-n_samples:]
        start_audio = start_audio[:n_samples]
        # If stems are not provided, raise error (no more on-the-fly separation)
        if end_stems is None or start_stems is None:
            raise ValueError("Stems must be provided for blending. Run Demucs once per section and pass stems here.")

        # Ensure all stems are stereo (n_samples, 2)

        def fix_stem_shape(arr, target_shape):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=-1)
            if arr.shape[0] < target_shape[0]:
                pad = np.zeros((target_shape[0] - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > target_shape[0]:
                arr = arr[:target_shape[0]]
            if arr.shape[1] != target_shape[1]:
                arr = arr[:, :target_shape[1]]
            return arr

        expected_shape = (n_samples, 2)
        for stems in (end_stems, start_stems):
            for key in stems:
                stems[key] = fix_stem_shape(stems[key], expected_shape)

        fade = np.linspace(0, 1, n_samples)[:, None]  # shape (n_samples, 1)
        # Combine all available stems (including vocals) for a full mix within the blend region
        stem_keys = set(list(end_stems.keys()) + list(start_stems.keys())) - {'_sr'}
        if not stem_keys:
            # Fallback to raw audio if no stems available
            acc_end = end_audio
            acc_start = start_audio
        else:
            # Sum all stems present (vocals included), then average by the number of stems
            acc_end = None
            acc_start = None
            for stem in stem_keys:
                se = end_stems.get(stem, np.zeros((n_samples, 2)))[-n_samples:]
                ss = start_stems.get(stem, np.zeros((n_samples, 2)))[:n_samples]
                if acc_end is None:
                    acc_end = se.copy().astype(np.float32)
                    acc_start = ss.copy().astype(np.float32)
                else:
                    acc_end += se
                    acc_start += ss
            acc_end = acc_end / float(max(1, len(stem_keys)))
            acc_start = acc_start / float(max(1, len(stem_keys)))
        # Match the RMS of both sides to reduce sudden level changes and reduce clipping risk.
        def rms(x):
            return np.sqrt(np.mean(x**2))
        rms_end = rms(acc_end)
        rms_start = rms(acc_start)
        if rms_end > 1e-9 and rms_start > 1e-9:
            rms_target = 0.5 * (rms_end + rms_start)
            acc_end *= rms_target / rms_end
            acc_start *= rms_target / rms_start

        # Perform the crossfade
        blended = (1 - fade) * acc_end + fade * acc_start

        # Apply soft clipping to prevent harsh distortion
        # Use tanh-based soft limiter for values exceeding threshold
        threshold = 0.95
        peak = np.max(np.abs(blended))
        if peak > threshold:
            # Soft clip using tanh for gentle limiting
            blended = threshold * np.tanh(blended / threshold)

        return blended

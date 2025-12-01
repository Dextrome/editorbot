"""Audio effects for enhancing and processing audio."""

from typing import Optional

import numpy as np
import librosa
from scipy import signal


class AudioEffects:
    """Collection of audio effects for processing recordings."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio effects processor.

        Args:
            sample_rate: Sample rate of the audio.
        """
        self.sample_rate = sample_rate

    def apply_eq(
        self,
        audio_data: np.ndarray,
        low_gain: float = 0.0,
        mid_gain: float = 0.0,
        high_gain: float = 0.0,
    ) -> np.ndarray:
        """
        Apply a 3-band equalizer to the audio.

        Args:
            audio_data: Input audio data.
            low_gain: Gain for low frequencies (dB).
            mid_gain: Gain for mid frequencies (dB).
            high_gain: Gain for high frequencies (dB).

        Returns:
            Processed audio data.
        """
        # Convert gains from dB to linear
        low_mult = 10 ** (low_gain / 20)
        mid_mult = 10 ** (mid_gain / 20)
        high_mult = 10 ** (high_gain / 20)

        # Design filters
        nyq = self.sample_rate / 2
        low_cutoff = 250 / nyq
        high_cutoff = 4000 / nyq

        # Low pass for bass
        b_low, a_low = signal.butter(4, low_cutoff, btype="low")
        # Band pass for mids
        b_mid, a_mid = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
        # High pass for treble
        b_high, a_high = signal.butter(4, high_cutoff, btype="high")

        # Apply filters. Use filtfilt for zero-phase when possible; fall back to lfilter for very short signals.
        try:
            low_band = signal.filtfilt(b_low, a_low, audio_data) * low_mult
            mid_band = signal.filtfilt(b_mid, a_mid, audio_data) * mid_mult
            high_band = signal.filtfilt(b_high, a_high, audio_data) * high_mult
        except ValueError:
            # Signal too short for filtfilt's pad length; use causal lfilter as a fallback
            low_band = signal.lfilter(b_low, a_low, audio_data) * low_mult
            mid_band = signal.lfilter(b_mid, a_mid, audio_data) * mid_mult
            high_band = signal.lfilter(b_high, a_high, audio_data) * high_mult

        return low_band + mid_band + high_band

    def apply_compression(
        self,
        audio_data: np.ndarray,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> np.ndarray:
        """
        Apply dynamic range compression.

        Args:
            audio_data: Input audio data.
            threshold: Threshold in dB.
            ratio: Compression ratio.
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.

        Returns:
            Compressed audio data.
        """
        # Convert to dB
        epsilon = 1e-10
        db = 20 * np.log10(np.abs(audio_data) + epsilon)

        # Calculate gain reduction
        over_threshold = db - threshold
        over_threshold = np.maximum(over_threshold, 0)
        gain_reduction_db = over_threshold * (1 - 1 / ratio)

        # Apply envelope follower (simplified)
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        envelope = np.zeros_like(gain_reduction_db)
        for i in range(1, len(gain_reduction_db)):
            if gain_reduction_db[i] > envelope[i - 1]:
                coeff = 1 - np.exp(-1 / attack_samples) if attack_samples > 0 else 1
            else:
                coeff = 1 - np.exp(-1 / release_samples) if release_samples > 0 else 1
            envelope[i] = envelope[i - 1] + coeff * (gain_reduction_db[i] - envelope[i - 1])

        # Apply gain reduction
        gain = 10 ** (-envelope / 20)
        return audio_data * gain

    def apply_reverb(
        self,
        audio_data: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
    ) -> np.ndarray:
        """
        Apply reverb effect using a simple algorithmic approach.

        Args:
            audio_data: Input audio data.
            room_size: Room size parameter (0-1).
            damping: Damping parameter (0-1).
            wet_level: Mix level for wet signal (0-1).

        Returns:
            Audio with reverb applied.
        """
        # Simple comb filter-based reverb
        delays = [int(d * room_size) for d in [1557, 1617, 1491, 1422]]
        gains = [0.7 * (1 - damping * 0.3) for _ in delays]

        reverb = np.zeros_like(audio_data)
        for delay, gain in zip(delays, gains):
            if delay > 0:
                # Pad only the time axis. For stereo (2D) audio, pad ((delay,0),(0,0)).
                if audio_data.ndim == 1:
                    padded = np.pad(audio_data, (delay, 0))
                    delayed = padded[:-delay]
                else:
                    padded = np.pad(audio_data, ((delay, 0), (0, 0)))
                    delayed = padded[:-delay, :]
            else:
                delayed = audio_data
            # Ensure delayed has same shape as reverb for broadcasting
            if delayed.shape != reverb.shape:
                # If shapes mismatch due to channels, try to convert delayed to stereo shape
                try:
                    if delayed.ndim == 1 and reverb.ndim == 2:
                        delayed = np.stack([delayed, delayed], axis=-1)
                    elif delayed.ndim == 2 and reverb.ndim == 1:
                        delayed = delayed[:, 0]
                except Exception:
                    # As a last resort, reshape to reverb shape with zeros
                    delayed = np.zeros_like(reverb)
            reverb += delayed * gain

        # Mix dry and wet
        dry_level = 1 - wet_level
        return audio_data * dry_level + reverb * wet_level

    def apply_noise_gate(
        self,
        audio_data: np.ndarray,
        threshold_db: float = -40.0,
        attack_ms: float = 1.0,
        release_ms: float = 100.0,
    ) -> np.ndarray:
        """
        Apply noise gate to reduce background noise.

        Args:
            audio_data: Input audio data.
            threshold_db: Gate threshold in dB.
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.

        Returns:
            Gated audio data.
        """
        # Calculate RMS envelope
        frame_length = int(0.01 * self.sample_rate)  # 10ms frames
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length)[0]
        
        # Interpolate RMS to match audio length
        rms_interp = np.interp(
            np.arange(len(audio_data)),
            np.linspace(0, len(audio_data), len(rms)),
            rms,
        )

        # Convert to dB and create gate
        epsilon = 1e-10
        rms_db = 20 * np.log10(rms_interp + epsilon)
        gate = (rms_db > threshold_db).astype(float)

        # Smooth the gate
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        smoothed_gate = np.zeros_like(gate)
        for i in range(1, len(gate)):
            if gate[i] > smoothed_gate[i - 1]:
                coeff = 1 - np.exp(-1 / attack_samples) if attack_samples > 0 else 1
            else:
                coeff = 1 - np.exp(-1 / release_samples) if release_samples > 0 else 1
            smoothed_gate[i] = smoothed_gate[i - 1] + coeff * (gate[i] - smoothed_gate[i - 1])

        return audio_data * smoothed_gate

    def remove_noise(
        self, audio_data: np.ndarray, noise_reduction_amount: float = 0.75
    ) -> np.ndarray:
        """
        Simple spectral noise reduction.

        Args:
            audio_data: Input audio data.
            noise_reduction_amount: Amount of noise reduction (0-1).

        Returns:
            Noise-reduced audio data.
        """
        # Compute STFT
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise floor from quietest frames
        frame_energy = np.sum(magnitude**2, axis=0)
        noise_frames = frame_energy < np.percentile(frame_energy, 10)
        noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)

        # Spectral subtraction
        magnitude_clean = magnitude - noise_reduction_amount * noise_profile
        magnitude_clean = np.maximum(magnitude_clean, 0)

        # Reconstruct signal
        stft_clean = magnitude_clean * np.exp(1j * phase)
        return librosa.istft(stft_clean)

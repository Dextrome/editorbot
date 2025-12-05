"""Data augmentation for audio editing training.

Provides augmentation transforms that preserve edit semantics:
- Pitch shifting (changes key, preserves structure)
- Time stretching (changes tempo, preserves beat structure)
- Noise injection (adds background, preserves content)
- Gain variation (volume changes)
- EQ filtering (tonal changes)

All transforms are designed to be applied to paired (raw, edited) data
while preserving the edit labels (KEEP/CUT/LOOP decisions).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for advanced augmentations
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available - some augmentations disabled")

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Master switch
    enabled: bool = True
    
    # Pitch shifting
    pitch_shift_enabled: bool = True
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)  # semitones
    pitch_shift_prob: float = 0.5
    
    # Time stretching  
    time_stretch_enabled: bool = True
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)  # rate multiplier
    time_stretch_prob: float = 0.5
    
    # Noise injection
    noise_enabled: bool = True
    noise_snr_range: Tuple[float, float] = (20.0, 40.0)  # dB
    noise_prob: float = 0.3
    noise_types: Tuple[str, ...] = ("white", "pink")
    
    # Gain variation
    gain_enabled: bool = True
    gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB
    gain_prob: float = 0.5
    
    # EQ filtering (simple high/low shelf)
    eq_enabled: bool = True
    eq_gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB
    eq_prob: float = 0.3
    
    # Probability of applying any augmentation (overall)
    augment_prob: float = 0.8
    
    # Maximum augmentations per sample
    max_augments: int = 3


class AudioAugmentor:
    """Apply augmentations to audio signals."""
    
    def __init__(
        self,
        sr: int = 22050,
        config: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize augmentor.
        
        Args:
            sr: Sample rate
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.sr = sr
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(seed)
        
        # Available augmentation functions
        self._augmentations = self._build_augmentation_list()
    
    def _build_augmentation_list(self) -> List[Tuple[str, callable, float]]:
        """Build list of (name, function, probability) tuples."""
        augments = []
        
        if self.config.pitch_shift_enabled and HAS_LIBROSA:
            augments.append(("pitch_shift", self._pitch_shift, self.config.pitch_shift_prob))
        
        if self.config.time_stretch_enabled and HAS_LIBROSA:
            augments.append(("time_stretch", self._time_stretch, self.config.time_stretch_prob))
        
        if self.config.noise_enabled:
            augments.append(("noise", self._add_noise, self.config.noise_prob))
        
        if self.config.gain_enabled:
            augments.append(("gain", self._apply_gain, self.config.gain_prob))
        
        if self.config.eq_enabled and HAS_SCIPY:
            augments.append(("eq", self._apply_eq, self.config.eq_prob))
        
        return augments
    
    def __call__(
        self,
        audio: np.ndarray,
        return_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Apply random augmentations to audio.
        
        Args:
            audio: Input audio signal
            return_info: If True, return augmentation info dict
            
        Returns:
            Augmented audio (and optionally info dict)
        """
        if not self.config.enabled:
            return (audio, {}) if return_info else audio
        
        # Check if we should augment at all
        if self.rng.random() > self.config.augment_prob:
            return (audio, {"augmented": False}) if return_info else audio
        
        # Select which augmentations to apply
        applied = []
        augmented = audio.copy()
        
        # Shuffle augmentations and apply up to max_augments
        indices = self.rng.permutation(len(self._augmentations))
        
        for idx in indices:
            if len(applied) >= self.config.max_augments:
                break
            
            name, func, prob = self._augmentations[idx]
            if self.rng.random() < prob:
                try:
                    augmented, info = func(augmented)
                    applied.append((name, info))
                except Exception as e:
                    logger.warning(f"Augmentation {name} failed: {e}")
        
        if return_info:
            return augmented, {"augmented": True, "applied": applied}
        return augmented
    
    def augment_pair(
        self,
        raw_audio: np.ndarray,
        edited_audio: np.ndarray,
        return_info: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict]]:
        """Apply the SAME augmentations to raw and edited audio.
        
        This ensures edit labels remain valid after augmentation.
        
        Args:
            raw_audio: Raw/input audio
            edited_audio: Edited/output audio
            return_info: If True, return augmentation info
            
        Returns:
            Tuple of (augmented_raw, augmented_edited) and optionally info
        """
        if not self.config.enabled:
            if return_info:
                return raw_audio, edited_audio, {"augmented": False}
            return raw_audio, edited_audio
        
        # Check if we should augment
        if self.rng.random() > self.config.augment_prob:
            if return_info:
                return raw_audio, edited_audio, {"augmented": False}
            return raw_audio, edited_audio
        
        # Store random state to apply same transforms
        state = self.rng.get_state()
        
        # Apply to raw
        applied_raw = []
        aug_raw = raw_audio.copy()
        
        indices = self.rng.permutation(len(self._augmentations))
        selected = []
        
        for idx in indices:
            if len(selected) >= self.config.max_augments:
                break
            name, func, prob = self._augmentations[idx]
            if self.rng.random() < prob:
                selected.append((name, func, self.rng.get_state()))
        
        # Apply selected augmentations to both with same params
        aug_raw = raw_audio.copy()
        aug_edited = edited_audio.copy()
        applied = []
        
        for name, func, aug_state in selected:
            try:
                # Apply to raw
                self.rng.set_state(aug_state)
                aug_raw, info_raw = func(aug_raw)
                
                # Apply same params to edited
                self.rng.set_state(aug_state)
                aug_edited, info_edited = func(aug_edited)
                
                applied.append((name, info_raw))
            except Exception as e:
                logger.warning(f"Pair augmentation {name} failed: {e}")
        
        if return_info:
            return aug_raw, aug_edited, {"augmented": True, "applied": applied}
        return aug_raw, aug_edited
    
    def _pitch_shift(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply pitch shifting."""
        low, high = self.config.pitch_shift_range
        n_steps = self.rng.uniform(low, high)
        
        # librosa pitch shift
        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=n_steps
        )
        
        return shifted, {"n_steps": n_steps}
    
    def _time_stretch(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply time stretching."""
        low, high = self.config.time_stretch_range
        rate = self.rng.uniform(low, high)
        
        # librosa time stretch
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        return stretched, {"rate": rate}
    
    def _add_noise(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Add background noise."""
        low, high = self.config.noise_snr_range
        snr_db = self.rng.uniform(low, high)
        noise_type = self.rng.choice(self.config.noise_types)
        
        # Generate noise
        if noise_type == "white":
            noise = self.rng.randn(len(audio))
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(audio))
        else:
            noise = self.rng.randn(len(audio))
        
        # Scale noise to achieve desired SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            noise = noise * scale
        
        noisy = audio + noise
        
        # Prevent clipping
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val
        
        return noisy, {"snr_db": snr_db, "noise_type": noise_type}
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink (1/f) noise."""
        # Simple approximation using filtered white noise
        white = self.rng.randn(length)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # 1/f amplitude scaling
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=length)
        
        return pink / np.std(pink)
    
    def _apply_gain(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply gain (volume) change."""
        low, high = self.config.gain_range
        gain_db = self.rng.uniform(low, high)
        
        gain_linear = 10 ** (gain_db / 20)
        gained = audio * gain_linear
        
        # Soft clip if needed
        max_val = np.max(np.abs(gained))
        if max_val > 1.0:
            gained = np.tanh(gained)  # Soft saturation
        
        return gained, {"gain_db": gain_db}
    
    def _apply_eq(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply simple EQ (high/low shelf)."""
        low, high = self.config.eq_gain_range
        
        # Random low shelf gain
        low_gain_db = self.rng.uniform(low, high)
        # Random high shelf gain
        high_gain_db = self.rng.uniform(low, high)
        
        # Design simple biquad filters
        # Low shelf at 200 Hz
        low_shelf = self._design_shelf_filter(200, low_gain_db, "low")
        # High shelf at 4000 Hz
        high_shelf = self._design_shelf_filter(4000, high_gain_db, "high")
        
        # Apply filters
        eq_audio = scipy.signal.lfilter(low_shelf[0], low_shelf[1], audio)
        eq_audio = scipy.signal.lfilter(high_shelf[0], high_shelf[1], eq_audio)
        
        return eq_audio.astype(np.float32), {
            "low_gain_db": low_gain_db,
            "high_gain_db": high_gain_db,
        }
    
    def _design_shelf_filter(
        self,
        freq: float,
        gain_db: float,
        shelf_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Design a shelf filter.
        
        Args:
            freq: Shelf frequency in Hz
            gain_db: Gain in dB
            shelf_type: "low" or "high"
            
        Returns:
            (b, a) filter coefficients
        """
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.707 - 1) + 2)
        
        cos_w0 = np.cos(w0)
        
        if shelf_type == "low":
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        else:  # high
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        
        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1, a1 / a0, a2 / a0])
        
        return b, a


class BatchAugmentor:
    """Augmentor optimized for batch processing with GPU support."""
    
    def __init__(
        self,
        sr: int = 22050,
        config: Optional[AugmentationConfig] = None,
        device: str = "cuda",
    ):
        """Initialize batch augmentor.
        
        Args:
            sr: Sample rate
            config: Augmentation configuration
            device: Device for GPU operations
        """
        self.sr = sr
        self.config = config or AugmentationConfig()
        self.device = device if HAS_TORCHAUDIO else "cpu"
        
        if HAS_TORCHAUDIO and device == "cuda" and torch.cuda.is_available():
            self._use_gpu = True
        else:
            self._use_gpu = False
            self.cpu_augmentor = AudioAugmentor(sr=sr, config=config)
    
    def augment_batch(
        self,
        batch: Union[np.ndarray, "torch.Tensor"],
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Augment a batch of audio.
        
        Args:
            batch: Batch of audio (batch_size, n_samples) or (batch_size, channels, n_samples)
            
        Returns:
            Augmented batch
        """
        if not self.config.enabled:
            return batch
        
        if self._use_gpu and isinstance(batch, torch.Tensor):
            return self._gpu_augment_batch(batch)
        else:
            return self._cpu_augment_batch(batch)
    
    def _gpu_augment_batch(self, batch: "torch.Tensor") -> "torch.Tensor":
        """GPU-accelerated batch augmentation using torchaudio."""
        batch = batch.to(self.device)
        
        # Pitch shift using torchaudio (if available)
        if self.config.pitch_shift_enabled:
            # Note: torchaudio.functional.pitch_shift requires specific setup
            # For now, fall back to CPU for pitch shift
            pass
        
        # Gain is easy on GPU
        if self.config.gain_enabled:
            low, high = self.config.gain_range
            gain_db = torch.empty(batch.shape[0], 1, device=self.device).uniform_(low, high)
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply with probability
            mask = torch.rand(batch.shape[0], 1, device=self.device) < self.config.gain_prob
            batch = torch.where(mask, batch * gain_linear, batch)
        
        # Noise on GPU
        if self.config.noise_enabled:
            low, high = self.config.noise_snr_range
            snr_db = torch.empty(batch.shape[0], 1, device=self.device).uniform_(low, high)
            
            noise = torch.randn_like(batch)
            signal_power = torch.mean(batch ** 2, dim=-1, keepdim=True)
            noise_power = torch.mean(noise ** 2, dim=-1, keepdim=True)
            
            scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10)) + 1e-10))
            noise = noise * scale
            
            mask = torch.rand(batch.shape[0], 1, device=self.device) < self.config.noise_prob
            batch = torch.where(mask, batch + noise, batch)
        
        # Clip
        batch = torch.clamp(batch, -1.0, 1.0)
        
        return batch
    
    def _cpu_augment_batch(
        self,
        batch: Union[np.ndarray, "torch.Tensor"],
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """CPU batch augmentation."""
        is_tensor = isinstance(batch, torch.Tensor) if HAS_TORCHAUDIO else False
        
        if is_tensor:
            batch_np = batch.cpu().numpy()
        else:
            batch_np = batch
        
        # Process each sample
        augmented = []
        for i in range(len(batch_np)):
            aug = self.cpu_augmentor(batch_np[i])
            augmented.append(aug)
        
        result = np.stack(augmented)
        
        if is_tensor:
            return torch.from_numpy(result).to(batch.device)
        return result


def get_default_augmentation_config() -> AugmentationConfig:
    """Get default augmentation config."""
    return AugmentationConfig()


def get_conservative_augmentation_config() -> AugmentationConfig:
    """Get conservative augmentation config (mild transforms)."""
    return AugmentationConfig(
        enabled=True,
        pitch_shift_enabled=True,
        pitch_shift_range=(-1.0, 1.0),
        pitch_shift_prob=0.3,
        time_stretch_enabled=True,
        time_stretch_range=(0.95, 1.05),
        time_stretch_prob=0.3,
        noise_enabled=True,
        noise_snr_range=(30.0, 50.0),
        noise_prob=0.2,
        gain_enabled=True,
        gain_range=(-3.0, 3.0),
        gain_prob=0.4,
        eq_enabled=False,
        augment_prob=0.6,
        max_augments=2,
    )


def get_aggressive_augmentation_config() -> AugmentationConfig:
    """Get aggressive augmentation config (strong transforms)."""
    return AugmentationConfig(
        enabled=True,
        pitch_shift_enabled=True,
        pitch_shift_range=(-4.0, 4.0),
        pitch_shift_prob=0.6,
        time_stretch_enabled=True,
        time_stretch_range=(0.85, 1.15),
        time_stretch_prob=0.6,
        noise_enabled=True,
        noise_snr_range=(15.0, 35.0),
        noise_prob=0.4,
        gain_enabled=True,
        gain_range=(-9.0, 9.0),
        gain_prob=0.6,
        eq_enabled=True,
        eq_gain_range=(-9.0, 9.0),
        eq_prob=0.4,
        augment_prob=0.9,
        max_augments=4,
    )

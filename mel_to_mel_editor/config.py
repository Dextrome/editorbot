"""Configuration for Mel-to-Mel Editor.

Direct audio transformation without labels - learns end-to-end.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Import shared audio config
from shared.audio_config import AudioConfig


@dataclass
class ModelConfig:
    """U-Net model configuration."""

    # Input/output
    n_mels: int = 128

    # Encoder channels (downsampling path)
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Bottleneck
    bottleneck_channels: int = 512

    # Decoder channels (upsampling path) - mirrors encoder
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])

    # Convolution settings
    kernel_size: int = 5

    # Residual learning - predict delta instead of full output
    use_residual: bool = True

    # Attention in bottleneck
    use_attention: bool = True
    attention_heads: int = 8

    # Dropout
    dropout: float = 0.1


@dataclass
class LossConfig:
    """Loss function configuration."""

    # L1 reconstruction loss
    l1_weight: float = 1.0

    # MSE loss
    mse_weight: float = 0.0

    # Multi-scale STFT loss (on reconstructed audio)
    stft_weight: float = 0.5
    stft_fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])

    # Perceptual loss (feature matching)
    perceptual_weight: float = 0.0

    # Preserve unchanged regions (where raw ≈ edited)
    preservation_weight: float = 0.5
    preservation_threshold: float = 0.1  # Regions where diff < threshold


@dataclass
class TrainConfig:
    """Training configuration."""

    # Audio
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Loss
    loss: LossConfig = field(default_factory=LossConfig)

    # Training
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 100
    gradient_clip: float = 1.0

    # LR schedule
    warmup_epochs: int = 5
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"

    # Sequence length
    max_seq_len: int = 1024  # ~23 seconds

    # Data augmentation
    use_augmentation: bool = True
    augment_noise: float = 0.01
    augment_time_stretch: bool = False

    # Data
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1

    # Checkpointing
    save_interval: int = 10

    # Device
    device: str = "cuda"
    use_mixed_precision: bool = True

    # Logging
    log_dir: str = "./logs/mel_to_mel"
    use_tensorboard: bool = True

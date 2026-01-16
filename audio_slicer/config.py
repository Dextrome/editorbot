"""Configuration for audio slicer."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Model architecture config."""
    n_mels: int = 128
    segment_frames: int = 128  # ~1.5 seconds per segment

    # Encoder
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    latent_dim: int = 128

    # Scoring head
    use_projection_head: bool = True
    projection_dim: int = 64

    # Regularization
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training config."""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Contrastive learning
    temperature: float = 0.1

    # Augmentation for negatives
    noise_std: float = 0.1
    time_mask_ratio: float = 0.2
    freq_mask_ratio: float = 0.2

    # Data
    val_split: float = 0.1
    num_workers: int = 0

    # Misc
    use_mixed_precision: bool = True
    save_every: int = 10

    # Segment extraction
    segments_per_track: int = 50  # Random segments per track
    overlap_ratio: float = 0.5


@dataclass
class SlicerConfig:
    """Full config."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

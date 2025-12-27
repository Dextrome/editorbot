"""Configuration dataclasses for Super Editor.

Contains all hyperparameters for both training phases.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    fmin: float = 0.0
    fmax: float = 8000.0


@dataclass
class Phase1Config:
    """Configuration for Phase 1: Supervised Reconstruction.

    Trains an autoencoder that takes (raw_mel, edit_labels) and produces edited_mel.
    """

    # Audio
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Model architecture
    encoder_dim: int = 512
    decoder_dim: int = 512
    n_encoder_layers: int = 6
    n_decoder_layers: int = 3
    n_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Edit label vocabulary size
    n_edit_labels: int = 8  # CUT, KEEP, LOOP, FADE_IN, FADE_OUT, EFFECT, TRANSITION, PAD

    # Input constraints
    max_seq_len: int = 2048  # ~47 seconds at 22050/512

    # Training
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_clip: float = 1.0
    epochs: int = 100

    # LR schedule
    warmup_steps: int = 1000
    lr_decay_steps: int = 100000

    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 0.0
    stft_weight: float = 1.0
    consistency_weight: float = 0.5
    perceptual_weight: float = 0.0  # Optional: feature-based loss

    # Multi-scale STFT params
    stft_fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    stft_hop_sizes: List[int] = field(default_factory=lambda: [128, 256, 512])
    stft_win_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])

    # Data
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1

    # Checkpointing
    save_interval: int = 1000
    log_interval: int = 100

    # Device
    device: str = "cuda"
    use_mixed_precision: bool = True


@dataclass
class Phase2Config:
    """Configuration for Phase 2: RL Edit Prediction.

    Trains an RL agent to predict edit_labels that maximize output quality
    when passed through the frozen Phase 1 reconstruction model.
    """

    # Audio (should match Phase 1)
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Edit predictor architecture
    predictor_dim: int = 256
    n_predictor_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Value network
    value_dim: int = 256
    n_value_layers: int = 3

    # Edit label vocabulary (must match Phase 1)
    n_edit_labels: int = 8

    # Input constraints (must match Phase 1)
    max_seq_len: int = 2048

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    entropy_coeff_decay: bool = True
    entropy_coeff_min: float = 0.001
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    batch_size: int = 32
    n_epochs_per_update: int = 4
    rollout_steps: int = 128
    total_epochs: int = 1000

    # Reward weights
    reconstruction_reward_weight: float = 10.0
    label_accuracy_reward_weight: float = 5.0
    duration_match_reward_weight: float = 3.0
    smoothness_penalty_weight: float = 2.0

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    save_interval: int = 100
    log_interval: int = 10

    # Device
    device: str = "cuda"
    use_mixed_precision: bool = True

    # Phase 1 model (frozen)
    reconstruction_model_path: str = ""


# Edit label definitions
class EditLabel:
    """Edit label vocabulary."""
    CUT = 0
    KEEP = 1
    LOOP = 2
    FADE_IN = 3
    FADE_OUT = 4
    EFFECT = 5
    TRANSITION = 6
    PAD = 7

    NAMES = ['CUT', 'KEEP', 'LOOP', 'FADE_IN', 'FADE_OUT', 'EFFECT', 'TRANSITION', 'PAD']

    @classmethod
    def to_name(cls, label: int) -> str:
        return cls.NAMES[label] if 0 <= label < len(cls.NAMES) else 'UNKNOWN'

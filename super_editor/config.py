"""Configuration dataclasses for Super Editor.

Contains all hyperparameters for both training phases.
AudioConfig and EditLabel are imported from the shared module.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Import from shared module for consistency across the project
from shared.audio_config import AudioConfig, EditLabel

# Re-export for backward compatibility
__all__ = ['AudioConfig', 'EditLabel', 'Phase1Config', 'Phase2Config']


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

    # Logging
    log_dir: str = "./logs"
    use_tensorboard: bool = True


@dataclass
class Phase2Config:
    """Configuration for Phase 2: RL Edit Prediction.

    Trains an RL agent to predict edit_labels that maximize output quality
    when passed through the frozen Phase 1 reconstruction model.

    Improvements from rl_editor:
    - Curriculum learning (start with shorter sequences)
    - Observation normalization (RunningMeanStd)
    - Better PPO config (target KL, entropy decay)
    - Auxiliary tasks (tempo, energy, phrase, mel reconstruction)
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

    # PPO hyperparameters (improved based on rl_editor)
    learning_rate: float = 1e-4  # Lower than before, with decay
    lr_decay: bool = True
    lr_decay_type: str = "cosine"  # Options: "cosine", "linear", "exponential"
    lr_min_ratio: float = 0.1  # Decay to 10% of initial LR
    lr_warmup_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02  # Early stopping on KL divergence (from rl_editor)
    entropy_coeff: float = 0.02
    entropy_coeff_decay: bool = True
    entropy_coeff_min: float = 0.005
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0  # Tighter gradient clipping

    # Training
    batch_size: int = 32
    n_epochs_per_update: int = 4
    rollout_steps: int = 128
    total_epochs: int = 1000
    gradient_accumulation_steps: int = 4  # From rl_editor

    # Reward weights (tuned for balanced learning signal)
    reconstruction_reward_weight: float = 1.0   # Scaled down to not dominate
    label_accuracy_reward_weight: float = 10.0  # Direct supervision signal
    duration_match_reward_weight: float = 3.0
    smoothness_penalty_weight: float = 1.0      # Less harsh

    # Curriculum learning (from rl_editor)
    use_curriculum: bool = True
    curriculum_initial_seq_len: int = 512  # Start with short sequences
    curriculum_final_seq_len: int = 2048  # End with full sequences
    curriculum_warmup_epochs: int = 100  # Epochs to linearly increase

    # Observation normalization (from rl_editor)
    use_observation_normalization: bool = True
    observation_clip_range: float = 10.0

    # Auxiliary tasks (from rl_editor)
    use_auxiliary_tasks: bool = True
    aux_tempo_weight: float = 0.1  # Predict tempo bin from encoded state
    aux_energy_weight: float = 0.1  # Predict energy bin from encoded state
    aux_phrase_weight: float = 0.1  # Predict phrase boundary from encoded state
    aux_mel_reconstruction_weight: float = 0.5  # Reconstruct mel from encoded state

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    save_interval: int = 100
    log_interval: int = 10

    # Device
    device: str = "cuda"
    use_mixed_precision: bool = True

    # Logging
    log_dir: str = "./logs"
    use_tensorboard: bool = True

    # Phase 1 model (frozen)
    reconstruction_model_path: str = ""

    def get_curriculum_seq_len(self, epoch: int) -> int:
        """Get sequence length for curriculum learning.

        Args:
            epoch: Current epoch

        Returns:
            Sequence length for this epoch
        """
        if not self.use_curriculum:
            return self.max_seq_len

        if epoch >= self.curriculum_warmup_epochs:
            return self.curriculum_final_seq_len

        # Linear interpolation
        progress = epoch / self.curriculum_warmup_epochs
        seq_len = int(
            self.curriculum_initial_seq_len +
            progress * (self.curriculum_final_seq_len - self.curriculum_initial_seq_len)
        )
        return seq_len

    def get_entropy_coeff(self, epoch: int, total_epochs: Optional[int] = None) -> float:
        """Get entropy coefficient with decay.

        Args:
            epoch: Current epoch
            total_epochs: Total training epochs (uses self.total_epochs if None)

        Returns:
            Entropy coefficient for this epoch
        """
        if not self.entropy_coeff_decay:
            return self.entropy_coeff

        if total_epochs is None:
            total_epochs = self.total_epochs

        progress = min(epoch / total_epochs, 1.0)
        # Exponential decay from entropy_coeff to entropy_coeff_min
        return self.entropy_coeff_min + (self.entropy_coeff - self.entropy_coeff_min) * (1 - progress)

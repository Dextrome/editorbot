"""Configuration module for RL-based audio editor.

Uses dataclass Config pattern for all hyperparameters.
Simplified for train_v2.py pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


# Stub for backwards compatibility with old checkpoints
@dataclass
class ActionSpaceConfig:
    """Legacy action space config - kept for checkpoint compatibility."""
    pass


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 22050
    hop_length: int = 512
    n_mels: int = 128
    n_fft: int = 2048


@dataclass
class StateConfig:
    """State representation configuration."""

    beat_context_size: int = 3  # beats before/after current
    use_stem_features: bool = True
    use_mel_spectrogram: bool = True
    use_beat_descriptors: bool = True
    use_global_features: bool = True


@dataclass
class RewardConfig:
    """Reward signal configuration."""

    # === MONTE CARLO MODE ===
    # When True: Zero step rewards, episode-only rewards, mean-baseline advantages
    # This forces true multi-step credit assignment
    use_monte_carlo: bool = True  # NEW: Pure Monte Carlo learning
    
    use_sparse_rewards: bool = True
    use_dense_rewards: bool = True
    use_learned_rewards: bool = False  # Disabled - needs real audio features to be useful
    use_trajectory_rewards: bool = True  # End-of-episode audio quality reward
    trajectory_reward_scale: float = 100.0  # Scale factor for trajectory rewards
    step_reward_scale: float = 0.05  # ZERO step rewards in Monte Carlo mode
    target_keep_ratio: float = 0.45  # Target ratio of beats to keep (~35%)
    target_max_duration_s: float = 600.0  # 10 minutes max output duration
    duration_penalty_weight: float = 0.1  # Penalty multiplier for exceeding target (reduced from 0.5)
    max_loop_ratio: float = 0.12  # Max fraction of beats that should be looped (12%, reduced from 15%)
    loop_penalty_weight: float = 0.5  # Penalty for exceeding max loop ratio
    loop_repetition_penalty: float = 0.2  # Penalty for looping beats near other looped beats
    loop_proximity_window: int = 8  # Beats within this window count as "nearby" loops
    tempo_consistency_weight: float = 1.0
    energy_flow_weight: float = 1.0
    phrase_completeness_weight: float = 0.8
    transition_quality_weight: float = 0.9
    keep_ratio_weight: float = 1.0  # Weight for hitting target keep ratio
    transition_smoothness_weight: float = 1.5  # Weight for smooth transitions (no clicks)
    
    # === PURE AUDIO QUALITY REWARDS ===
    # NO ground truth labels - model learns what sounds good
    spectral_continuity_weight: float = 1.2  # Penalize spectral discontinuities at edit points
    beat_alignment_quality_weight: float = 1.0  # Reward for clean beat-aligned cuts
    section_coherence_weight: float = 1.0  # Reward for keeping consecutive beats together
    flow_continuity_weight: float = 1.0  # Beat-to-beat transition flow
    ground_truth_weight: float = 0.0  # DISABLED - force learning from audio quality, not copying labels
    
    # Step rewards (only used if use_monte_carlo=False)
    step_phrase_boundary_bonus: float = 0.3  # Bonus for cutting at phrase boundaries
    step_coherence_bonus: float = 0.2  # Bonus for keeping beats adjacent to other kept beats
    step_energy_continuity_weight: float = 0.15  # Penalty for sudden energy jumps


@dataclass
class ModelConfig:
    """Neural network model configuration."""

    policy_hidden_dim: int = 512
    policy_n_layers: int = 5
    policy_dropout: float = 0.15
    
    # NATTEN Hybrid Encoder (local attention + global pooling)
    natten_kernel_size: int = 33  # Local neighborhood size (odd number)
    natten_n_heads: int = 8  # Number of attention heads
    natten_n_layers: int = 3  # Number of NATTEN layers
    natten_dilation: int = 2  # Dilation for sparse global context (1 = no dilation)
    
    # Value network
    value_hidden_dim: int = 384
    value_n_layers: int = 4
    value_dropout: float = 0.1


@dataclass
class PPOConfig:
    """PPO training configuration."""

    learning_rate: float = 2e-5  # Lower LR for stability with high entropy
    lr_decay: bool = False  # Enable learning rate decay
    lr_decay_type: str = "cosine"  # Options: "cosine", "linear", "exponential", "step"
    lr_min_ratio: float = 0.5  # Decay from 2e-5 to 1e-5
    lr_warmup_epochs: int = 10  # Warmup epochs before decay starts
    lr_decay_epochs: int = 500  # Total epochs for decay (for step/exponential)
    lr_step_factor: float = 0.5  # Factor for step decay
    lr_step_interval: int = 100  # Epochs between step decays
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2  # PPO clipping ratio - tighter for stability
    target_kl: float = 0.02  # Tighter KL target to prevent collapse
    entropy_coeff: float = 0.3  # High entropy for exploration (decay to 0.05)
    entropy_coeff_decay: bool = True  # Decay entropy over training
    entropy_coeff_min: float = 0.05  # Higher floor to maintain exploration
    value_loss_coeff: float = 0.5  # Standard value coeff (targets are normalized now)
    max_grad_norm: float = 0.5  # Standard grad clipping
    n_epochs: int = 6  # More PPO epochs for stable updates
    batch_size: int = 2048  # Larger batch for better GPU utilization (was 128)
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 2  # Reduced since batch is larger
    use_mixed_precision: bool = True  # Enabled with proper inf grad handling


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    device: str = "cuda"
    save_dir: str = "./models"
    log_dir: str = "./logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "rl-audio-editor"


@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration."""
    
    # Feature mode: "basic" (4 features), "enhanced" (60+ features), "full" (with stems)
    feature_mode: str = "full"
    use_stem_features: bool = True


@dataclass
class AugmentationCfg:
    """Data augmentation configuration.
    
    NOTE: pitch_shift and time_stretch are DISABLED by default because they
    change audio timing, which invalidates cached features and requires
    expensive re-extraction (~5-15s per file). Keep them disabled for fast
    training. Noise/gain/EQ are safe and fast.
    """
    
    # Master switch
    enabled: bool = True  # Enable fast augmentations (noise, gain, EQ)
    
    # Pitch shifting - DISABLED (too slow, requires feature re-extraction)
    pitch_shift_enabled: bool = False
    pitch_shift_min: float = -2.0  # semitones
    pitch_shift_max: float = 2.0
    pitch_shift_prob: float = 0.5
    
    # Time stretching - DISABLED (too slow, requires feature re-extraction)
    time_stretch_enabled: bool = False
    time_stretch_min: float = 0.9  # rate multiplier
    time_stretch_max: float = 1.1
    time_stretch_prob: float = 0.5
    
    # Noise injection
    noise_enabled: bool = True
    noise_snr_min: float = 20.0  # dB
    noise_snr_max: float = 40.0
    noise_prob: float = 0.3
    
    # Gain variation
    gain_enabled: bool = True
    gain_min: float = -6.0  # dB
    gain_max: float = 6.0
    gain_prob: float = 0.5
    
    # EQ filtering
    eq_enabled: bool = True
    eq_gain_min: float = -6.0  # dB
    eq_gain_max: float = 6.0
    eq_prob: float = 0.3
    
    # Overall probability and limits
    augment_prob: float = 0.65
    max_augments: int = 3


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Main training data directory
    # Expected structure:
    #   data_dir/
    #     input/           - Raw audio files (*_raw.wav)
    #     desired_output/  - Human-edited versions (*_edit.wav)
    #     reference/       - Additional finished tracks (optional, no pairs needed)
    #     test_input/      - Test files for inference (optional)
    data_dir: str = "./training_data"
    
    # Subdirectory names (relative to data_dir)
    raw_subdir: str = "input"
    edited_subdir: str = "desired_output"
    reference_subdir: str = "reference"
    test_subdir: str = "test_input"
    
    # Caching - centralized in rl_editor/cache/
    cache_features: bool = True
    cache_dir: str = "./rl_editor/cache"
    
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    
    # Feature extraction
    use_stems: bool = True  # Pre-cached - OK to enable
    
    # Train/val split (fraction of paired data to use for validation)
    val_split: float = 0.0  # 0 = use all for training (recommended with small datasets)


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    state: StateConfig = field(default_factory=StateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "audio": self.audio.__dict__,
            "state": self.state.__dict__,
            "reward": self.reward.__dict__,
            "model": self.model.__dict__,
            "ppo": self.ppo.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "augmentation": self.augmentation.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            audio=AudioConfig(**config_dict.get("audio", {})),
            state=StateConfig(**config_dict.get("state", {})),
            reward=RewardConfig(**config_dict.get("reward", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            ppo=PPOConfig(**config_dict.get("ppo", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            features=FeatureExtractionConfig(**config_dict.get("features", {})),
            augmentation=AugmentationCfg(**config_dict.get("augmentation", {})),
        )


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()

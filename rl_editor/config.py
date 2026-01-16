"""Configuration module for RL-based audio editor.

Uses dataclass Config pattern for all hyperparameters.
Used by train.py with factored action space.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# Import AudioConfig from shared module for backward compatibility
# New code should import from shared directly
from shared.audio_config import AudioConfig


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
    use_sparse_rewards: bool = True
    use_dense_rewards: bool = True
    use_learned_rewards: bool = False  # Disabled - needs real audio features to be useful
    target_keep_ratio: float = 0.45  # Target ratio of beats to keep (~45%)
    tempo_consistency_weight: float = 1.0
    energy_flow_weight: float = 1.0
    phrase_completeness_weight: float = 0.8
    transition_quality_weight: float = 0.9

    # === Reward stabilization (fixes for training plateau) ===
    # Duration reward clamping to reduce variance
    duration_reward_clip_min: float = -10.0  # Clamp extreme negative penalties
    duration_reward_clip_max: float = 10.0   # Clamp extreme positive rewards

    # Curriculum on keep ratio target (start lenient, tighten over time)
    use_keep_ratio_curriculum: bool = True
    target_keep_ratio_initial: float = 0.60  # Start with easier target
    target_keep_ratio_final: float = 0.45    # End with actual target
    keep_ratio_curriculum_epochs: int = 8000  # Epochs to anneal over (set high for resumed training)

    # Cut action bonus to overcome "keep everything" inertia
    cut_action_bonus: float = 0.3  # Small bonus per cut action to encourage cutting


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

    learning_rate: float = 5e-5  # Moderate LR for resumed training (was 1e-4)
    lr_decay: bool = True  # Enable learning rate decay by default
    lr_decay_type: str = "cosine"  # Options: "cosine", "linear", "exponential", "step"
    lr_min_ratio: float = 0.4  # Decay to 40% of initial LR (2e-5 minimum)
    lr_warmup_epochs: int = 10  # Warmup epochs before decay starts
    lr_decay_epochs: int = 500  # Total epochs for decay (for step/exponential)
    lr_step_factor: float = 0.5  # Factor for step decay
    lr_step_interval: int = 100  # Epochs between step decays
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2  # Standard PPO clipping - allows exploration
    target_kl: float = 0.015  # Tighter KL to prevent too-aggressive updates (was 0.02)
    # Boost entropy to escape local minimum (model stuck at 67% keep ratio)
    entropy_coeff: float = 0.05  # Higher to encourage exploration (was 0.02)
    entropy_coeff_decay: bool = True  # Decay entropy over training
    entropy_coeff_min: float = 0.02  # Keep some exploration even late (was 0.005)
    value_loss_coeff: float = 0.1  # Standard PPO balance
    # Clip returns (helps prevent FP16 overflow and huge losses)
    # Widen return clipping to effectively disable aggressive clipping during recovery
    return_clip_min: float = -1e6
    return_clip_max: float = 1e6
    # L2 weight decay for optimizer - disable for now to avoid training drift
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # Tighter gradient clipping to avoid large updates
    n_epochs: int = 8  # PPO epochs per update
    batch_size: int = 2048  # Large batch for better GPU utilization
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_mixed_precision: bool = True  # FP16 for faster training (safe with bug fixes)


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    device: str = "cuda"
    save_dir: str = "./models"
    log_dir: str = "./logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "rl-audio-editor"
    # Evaluation truncation: 0 = full track, >0 = truncate to this many beats for eval
    # Increase default to a large snippet (2500 beats) for near-full-track eval during HPO
    eval_max_beats: int = 2500
    # Debug mode: enable verbose diagnostics (entropy comparisons, state distribution logs)
    # Disable for faster training
    debug_diagnostics: bool = False
    # Use torch.compile() for faster forward passes (PyTorch 2.0+)
    # May have startup overhead but speeds up training significantly
    use_torch_compile: bool = False
    # Checkpoint save interval (epochs)
    checkpoint_interval: int = 50
    # TensorBoard logging interval (epochs) - reduce for faster training
    log_interval: int = 10
    # Curriculum learning parameters
    curriculum_initial_beats: int = 1000  # Starting max beats per sample
    curriculum_final_beats: int = 4000    # Ending max beats per sample
    curriculum_initial_short_prob: float = 1.0  # Start with 100% short samples
    curriculum_final_short_prob: float = 0.25   # End with 25% short samples
    curriculum_steps: int = 4000  # Epochs to anneal curriculum over


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
    
    # Caching - centralized cache folder
    cache_features: bool = True
    cache_dir: str = "F:/editorbot/cache"
    
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    
    # Feature extraction
    use_stems: bool = True  # Pre-cached - OK to enable
    
    # Train/val split (fraction of paired data to use for validation)
    val_split: float = 0.0  # 0 = use all for training (recommended with small datasets)
    # Include reference (extra finished tracks) when loading paired dataset
    include_reference: bool = False


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

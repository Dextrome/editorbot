"""Configuration for pointer network."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PointerNetworkConfig:
    """Configuration for the pointer network model."""

    # Audio parameters (must match training data)
    n_mels: int = 128
    hop_length: int = 256
    sr: int = 22050

    # Encoder architecture
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    encoder_kernel_size: int = 5
    encoder_stride: int = 2  # Downsample factor per layer

    # Transformer dimensions
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Chunk processing (for long sequences)
    chunk_size: int = 512  # Process this many frames at a time
    chunk_overlap: int = 64  # Overlap between chunks

    # Pointer output
    max_output_length: int = 65536  # Maximum output sequence length

    # V2 architecture features
    use_pre_norm: bool = True  # Pre-LayerNorm for stability
    use_stems: bool = False  # Multi-stem encoder
    n_stems: int = 4  # Number of stems (drums, bass, vocals, other)

    # V2 Full-Sequence Architecture (Delta Prediction)
    compression_ratio: float = 0.67  # Expected output/input ratio (edit is ~67% of raw)
    attn_window_size: int = 512  # Window size for position-aware cross-attention
    max_delta: int = 64  # Max small delta offset (larger = jump prediction)
    n_global_tokens: int = 64  # Number of global summary tokens
    global_token_stride: int = 1000  # Frames per global token

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-5  # Lower LR for stability
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip: float = 0.5  # Tighter clipping to prevent NaN

    @property
    def downsample_factor(self) -> int:
        """Total downsampling from encoder."""
        return self.encoder_stride ** len(self.encoder_channels)

    @property
    def frames_per_second(self) -> float:
        """Mel frames per second."""
        return self.sr / self.hop_length


@dataclass
class TrainConfig:
    """Training configuration."""

    model: PointerNetworkConfig = field(default_factory=PointerNetworkConfig)

    # Data paths
    cache_dir: str = "F:/editorbot/cache"
    pointer_dir: str = "F:/editorbot/training_data/pointer_sequences"
    save_dir: str = "F:/editorbot/models/pointer_network_full"

    # Training params
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500  # Linear warmup steps
    label_smoothing: float = 0.1  # Label smoothing for cross-entropy loss

    # Performance optimizations
    num_workers: int = 0  # DataLoader workers (0 = main process, faster when data preloaded)
    prefetch_factor: int = 2  # Batches to prefetch per worker
    use_amp: bool = True  # Automatic mixed precision
    use_bfloat16: bool = True  # Use bfloat16 instead of float16 (more stable)
    use_gradient_checkpoint: bool = False  # Gradient checkpointing (saves VRAM, slower)
    use_compile: bool = False  # torch.compile (requires Triton, not available on Windows)
    gradient_accumulation_steps: int = 1  # Accumulate gradients
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs
    scheduler_type: str = "warmup_cosine"  # "warmup_cosine", "onecycle", or "none"
    max_lr_mult: float = 3.0  # OneCycleLR max_lr = learning_rate * max_lr_mult

    # Logging
    log_every: int = 10
    save_every: int = 10
    eval_every: int = 5

    # Data filtering
    exclude_samples: List[str] = field(default_factory=list)  # Sample name patterns to exclude
    real_only_validation: bool = True  # Use only real (non-synthetic) samples for validation

    # Augmentation
    augmentation_enabled: bool = False  # Enable data augmentation during training
    augmentation_noise_prob: float = 0.5
    augmentation_noise_level_min: float = 0.05
    augmentation_noise_level_max: float = 0.15
    augmentation_spec_augment_prob: float = 0.5
    augmentation_freq_masks: int = 2
    augmentation_freq_mask_width: int = 20
    augmentation_time_masks: int = 2
    augmentation_time_mask_width: int = 50
    augmentation_gain_prob: float = 0.5
    augmentation_gain_scale_min: float = 0.7
    augmentation_gain_scale_max: float = 1.3
    augmentation_channel_dropout_prob: float = 0.3
    augmentation_channel_dropout_rate: float = 0.1
    augmentation_chunk_shuffle_prob: float = 0.3
    augmentation_chunk_size: int = 1000
    augmentation_crop_prob: float = 0.5
    augmentation_crop_min_len: int = 500
    augmentation_crop_max_len: int = 5000


def get_mini_config() -> PointerNetworkConfig:
    """Minimal config for proof-of-concept testing.

    ~3M params vs ~15M for full config. Use this to:
    - Validate the data pipeline works
    - Confirm loss decreases
    - Test overfitting on a few examples
    - Fast iteration on hyperparameters
    """
    return PointerNetworkConfig(
        # Keep audio params the same
        n_mels=128,
        hop_length=256,
        sr=22050,
        # Smaller encoder
        encoder_channels=[32, 64],
        encoder_kernel_size=5,
        encoder_stride=2,
        # Smaller transformer
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        # Same chunk settings
        chunk_size=512,
        chunk_overlap=64,
        max_output_length=65536,
        # V2 full-sequence settings (smaller for mini)
        compression_ratio=0.67,
        attn_window_size=256,  # Smaller window for mini model
        max_delta=32,
        n_global_tokens=32,
        global_token_stride=500,
        # Training - higher LR ok for smaller model
        batch_size=4,
        learning_rate=3e-4,
        warmup_steps=100,
        max_epochs=50,
        gradient_clip=1.0,
    )


def get_mini_train_config() -> TrainConfig:
    """Training config using mini model for proof-of-concept."""
    return TrainConfig(
        model=get_mini_config(),
        save_dir="F:/editorbot/models/pointer_network_mini",
        batch_size=4,
        epochs=50,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        use_amp=True,
        use_compile=False,  # Faster startup for quick experiments
        log_every=5,
        save_every=10,
        eval_every=5,
    )
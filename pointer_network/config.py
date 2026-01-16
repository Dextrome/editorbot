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

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip: float = 1.0

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
    cache_dir: str = "F:/editorbot/training_data/super_editor_cache"
    pointer_dir: str = "F:/editorbot/training_data/pointer_sequences"
    save_dir: str = "F:/editorbot/models/pointer_network"

    # Training params
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Logging
    log_every: int = 10
    save_every: int = 10
    eval_every: int = 5

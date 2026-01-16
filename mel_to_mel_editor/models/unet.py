"""U-Net model for mel-to-mel transformation.

Architecture:
    - Encoder: Downsampling conv blocks
    - Bottleneck: Self-attention + conv
    - Decoder: Upsampling conv blocks with skip connections
    - Residual: Output = input + predicted_delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..config import ModelConfig


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
        )

        # Residual connection if dimensions match
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class DownBlock(nn.Module):
    """Downsampling block: ConvBlock + downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, 1, dropout)
        self.downsample = nn.Conv1d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.conv(x)
        return self.downsample(h), h  # Return downsampled and skip connection


class UpBlock(nn.Module):
    """Upsampling block: upsample + concat skip + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, 2, 1)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, kernel_size, 1, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for the bottleneck."""

    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) for attention
        B, C, T = x.shape
        x = x.transpose(1, 2)  # (B, T, C)

        # Self-attention
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        x = residual + out
        return x.transpose(1, 2)  # (B, C, T)


class MelUNet(nn.Module):
    """U-Net for mel-to-mel transformation.

    Input: (B, T, n_mels) - raw mel spectrogram
    Output: (B, T, n_mels) - edited mel spectrogram
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_mels = config.n_mels

        # Input projection: (B, T, n_mels) -> (B, C, T)
        self.input_proj = nn.Conv1d(n_mels, config.encoder_channels[0], 1)

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        in_ch = config.encoder_channels[0]
        for out_ch in config.encoder_channels:
            self.encoders.append(DownBlock(in_ch, out_ch, config.kernel_size, config.dropout))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(
            in_ch, config.bottleneck_channels, config.kernel_size, 1, config.dropout
        )

        if config.use_attention:
            self.attention = SelfAttention(config.bottleneck_channels, config.attention_heads)
        else:
            self.attention = nn.Identity()

        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        in_ch = config.bottleneck_channels
        skip_channels = list(reversed(config.encoder_channels))

        for i, out_ch in enumerate(config.decoder_channels):
            skip_ch = skip_channels[i]
            self.decoders.append(UpBlock(in_ch, skip_ch, out_ch, config.kernel_size, config.dropout))
            in_ch = out_ch

        # Output projection: (B, C, T) -> (B, T, n_mels)
        self.output_proj = nn.Sequential(
            nn.Conv1d(config.decoder_channels[-1], n_mels, 1),
            nn.Tanh(),  # Output in [-1, 1], will be scaled
        )

        # Residual scaling (learnable)
        if config.use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw mel spectrogram (B, T, n_mels) in [0, 1]

        Returns:
            Edited mel spectrogram (B, T, n_mels) in [0, 1]
        """
        # Store input for residual
        raw_input = x

        # (B, T, n_mels) -> (B, n_mels, T)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.attention(x)

        # Decoder with skip connections
        skips = list(reversed(skips))
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])

        # Output projection
        x = self.output_proj(x)

        # (B, n_mels, T) -> (B, T, n_mels)
        x = x.transpose(1, 2)

        # Residual learning: output = input + scaled_delta
        if self.config.use_residual:
            # x is in [-1, 1], scale to small delta
            delta = x * self.residual_scale
            output = raw_input + delta
        else:
            # Direct output, scale from [-1, 1] to [0, 1]
            output = (x + 1) / 2

        # Clamp to valid range
        output = torch.clamp(output, 0, 1)

        return output

    @classmethod
    def from_checkpoint(cls, path: str, config: Optional[ModelConfig] = None) -> 'MelUNet':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        if config is None:
            config = checkpoint.get('model_config', ModelConfig())

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

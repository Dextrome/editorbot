"""Mel spectrogram encoder for pointer network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MelEncoder(nn.Module):
    """Encodes mel spectrograms into frame embeddings.

    Takes (batch, n_mels, time) and outputs (batch, time', d_model)
    where time' = time / downsample_factor.
    """

    def __init__(
        self,
        n_mels: int = 128,
        channels: list = [64, 128, 256],
        kernel_size: int = 5,
        stride: int = 2,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.d_model = d_model
        self.downsample_factor = stride ** len(channels)

        # Build convolutional layers
        layers = []
        in_ch = n_mels
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, stride, dropout))
            in_ch = out_ch

        self.conv_layers = nn.ModuleList(layers)

        # Project to d_model
        self.out_proj = nn.Linear(channels[-1], d_model)

        # Positional encoding will be added by the transformer

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) mel spectrogram

        Returns:
            embeddings: (batch, time', d_model) frame embeddings
        """
        x = mel  # (batch, n_mels, time)

        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)

        # x: (batch, channels[-1], time')
        x = x.transpose(1, 2)  # (batch, time', channels[-1])
        x = self.out_proj(x)  # (batch, time', d_model)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 65536, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerMelEncoder(nn.Module):
    """Full encoder: CNN + Transformer self-attention."""

    def __init__(
        self,
        n_mels: int = 128,
        channels: list = [64, 128, 256],
        kernel_size: int = 5,
        stride: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 65536,
    ):
        super().__init__()

        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.d_model = d_model
        self.downsample_factor = self.mel_encoder.downsample_factor

    def forward(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) mel spectrogram
            mask: optional padding mask

        Returns:
            embeddings: (batch, time', d_model)
        """
        # CNN encoding
        x = self.mel_encoder(mel)  # (batch, time', d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer self-attention
        x = self.transformer(x, src_key_padding_mask=mask)

        return x

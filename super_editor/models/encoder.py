"""Edit Encoder for Phase 1 reconstruction.

Takes raw mel spectrogram and edit labels, produces latent representation.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from ..config import Phase1Config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            (B, T, D) tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EditEncoder(nn.Module):
    """Encodes raw mel + edit labels into latent representation.

    Architecture:
        raw_mel -> Linear -> mel_feat
        edit_labels -> Embedding -> label_feat
        concat(mel_feat, label_feat) -> Linear -> combined
        combined -> PositionalEncoding -> TransformerEncoder -> latent
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Project mel spectrogram to encoder dimension
        self.mel_proj = nn.Sequential(
            nn.Linear(config.audio.n_mels, config.encoder_dim),
            nn.LayerNorm(config.encoder_dim),
            nn.GELU(),
        )

        # Edit label embedding
        self.edit_embed = nn.Embedding(config.n_edit_labels, config.encoder_dim // 4)

        # Combine mel + edit embeddings
        combined_dim = config.encoder_dim + config.encoder_dim // 4
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, config.encoder_dim),
            nn.LayerNorm(config.encoder_dim),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.encoder_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.encoder_dim),
        )

    def forward(
        self,
        mel: torch.Tensor,  # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T) int
        mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = valid
    ) -> torch.Tensor:
        """
        Args:
            mel: Raw mel spectrogram (B, T, n_mels)
            edit_labels: Edit labels per frame (B, T)
            mask: Valid frame mask (B, T), True = valid

        Returns:
            latent: Encoded representation (B, T, encoder_dim)
        """
        B, T, _ = mel.shape

        # Project mel
        mel_feat = self.mel_proj(mel)  # (B, T, encoder_dim)

        # Embed edit labels
        label_feat = self.edit_embed(edit_labels)  # (B, T, encoder_dim//4)

        # Combine
        combined = torch.cat([mel_feat, label_feat], dim=-1)  # (B, T, encoder_dim + encoder_dim//4)
        x = self.combine(combined)  # (B, T, encoder_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask (True = ignore for PyTorch transformer)
        if mask is not None:
            attn_mask = ~mask  # Invert: True -> ignore
        else:
            attn_mask = None

        # Transformer encoding
        latent = self.transformer(x, src_key_padding_mask=attn_mask)

        return latent


class ConvEditEncoder(nn.Module):
    """Alternative encoder using convolutions + transformer.

    Better for capturing local patterns before global attention.
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Convolutional front-end for mel
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.audio.n_mels, config.encoder_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.encoder_dim // 2, config.encoder_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.encoder_dim // 2, config.encoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Edit label embedding
        self.edit_embed = nn.Embedding(config.n_edit_labels, config.encoder_dim // 4)

        # Combine
        self.combine = nn.Linear(config.encoder_dim + config.encoder_dim // 4, config.encoder_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.encoder_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.encoder_dim),
        )

    def forward(
        self,
        mel: torch.Tensor,
        edit_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Conv expects (B, C, T)
        mel_conv = mel.transpose(1, 2)  # (B, n_mels, T)
        conv_feat = self.conv_layers(mel_conv)  # (B, encoder_dim, T)
        conv_feat = conv_feat.transpose(1, 2)  # (B, T, encoder_dim)

        # Edit labels
        label_feat = self.edit_embed(edit_labels)  # (B, T, encoder_dim//4)

        # Combine
        combined = torch.cat([conv_feat, label_feat], dim=-1)
        x = self.combine(combined)

        # Positional encoding + transformer
        x = self.pos_encoder(x)

        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        latent = self.transformer(x, src_key_padding_mask=attn_mask)
        return latent

"""Quality scoring model using contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import json
from pathlib import Path

from ..config import ModelConfig


class ConvBlock(nn.Module):
    """Conv block with normalization and activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class QualityScorer(nn.Module):
    """Learn quality embeddings via contrastive learning.

    Takes mel segments, encodes them to latent space.
    Similar (good) segments cluster together.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.n_mels, config.hidden_dims[0])

        # Encoder (conv over time)
        encoder_layers = []
        in_dim = config.hidden_dims[0]
        for out_dim in config.hidden_dims:
            encoder_layers.append(ConvBlock(in_dim, out_dim))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Global pooling + latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

        # Optional projection head for contrastive learning
        if config.use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.projection_dim),
            )
        else:
            self.projection_head = None

        # Quality scoring head - outputs logits (no sigmoid for autocast compatibility)
        self.quality_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel segment to latent vector.

        Args:
            mel: (B, T, n_mels) mel segment

        Returns:
            (B, latent_dim) latent vector
        """
        # Project input
        x = self.input_proj(mel)  # (B, T, hidden)

        # Conv expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.encoder(x)  # (B, hidden[-1], T')

        # Global average pooling
        x = x.mean(dim=-1)  # (B, hidden[-1])

        # Project to latent
        z = self.latent_proj(x)  # (B, latent_dim)

        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent to contrastive space."""
        if self.projection_head is not None:
            return F.normalize(self.projection_head(z), dim=-1)
        return F.normalize(z, dim=-1)

    def score_logits(self, mel: torch.Tensor) -> torch.Tensor:
        """Get raw logits for quality scoring (for training with BCE loss).

        Args:
            mel: (B, T, n_mels) mel segment

        Returns:
            (B,) raw logits
        """
        z = self.encode(mel)
        logits = self.quality_head(z).squeeze(-1)
        return logits

    def score(self, mel: torch.Tensor) -> torch.Tensor:
        """Score quality of mel segment.

        Args:
            mel: (B, T, n_mels) mel segment

        Returns:
            (B,) quality score in [0, 1]
        """
        logits = self.score_logits(mel)
        return torch.sigmoid(logits)

    def forward(
        self,
        mel: torch.Tensor,
        return_projection: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            mel: (B, T, n_mels) mel segment
            return_projection: Whether to return projection for contrastive loss

        Returns:
            Dict with 'embedding', 'score', and optionally 'projection'
        """
        z = self.encode(mel)
        score = self.quality_head(z).squeeze(-1)

        result = {
            'embedding': z,
            'score': score,
        }

        if return_projection:
            result['projection'] = self.project(z)

        return result

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'config': self.config.__dict__,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str) -> 'QualityScorer':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])

        return model

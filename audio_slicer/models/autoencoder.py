"""Quality Autoencoder - learns to reconstruct "good" audio.

Like FaceSwap: train only on edited/good audio, then use
reconstruction error to identify good vs bad segments in raw audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from pathlib import Path

from ..config import ModelConfig


class Encoder(nn.Module):
    """Compress mel segment to latent."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input: (B, T, n_mels)
        layers = []
        in_dim = config.n_mels

        for i, out_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Temporal compression with conv
        self.conv = nn.Sequential(
            nn.Conv1d(config.hidden_dims[-1], config.hidden_dims[-1], 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_dims[-1], config.hidden_dims[-1], 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_dims[-1], config.latent_dim, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_mels)
        Returns:
            (B, latent_dim, T') compressed latent
        """
        x = self.mlp(x)  # (B, T, hidden)
        x = x.transpose(1, 2)  # (B, hidden, T)
        z = self.conv(x)  # (B, latent_dim, T')
        return z


class Decoder(nn.Module):
    """Reconstruct mel from latent."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Temporal upsampling
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim, config.hidden_dims[-1], 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(config.hidden_dims[-1], config.hidden_dims[-1], 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(config.hidden_dims[-1], config.hidden_dims[-1], 4, stride=2, padding=1),
            nn.GELU(),
        )

        # Output projection
        layers = []
        in_dim = config.hidden_dims[-1]

        for out_dim in reversed(config.hidden_dims[:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, config.n_mels))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T') latent
            target_len: Target sequence length
        Returns:
            (B, T, n_mels) reconstructed mel
        """
        x = self.deconv(z)  # (B, hidden, T'')
        x = x.transpose(1, 2)  # (B, T'', hidden)

        # Interpolate to exact target length
        if x.size(1) != target_len:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
            x = x.transpose(1, 2)

        x = self.mlp(x)  # (B, T, n_mels)
        return x


class QualityAutoencoder(nn.Module):
    """Autoencoder trained on edited audio only.

    Low reconstruction error = segment matches "good" audio distribution.
    High reconstruction error = segment is different (probably bad).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel to latent."""
        return self.encoder(mel)

    def decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """Decode latent to mel."""
        return self.decoder(z, target_len)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: (B, T, n_mels) input mel segment

        Returns:
            Dict with 'reconstruction', 'latent', 'loss'
        """
        T = mel.size(1)

        # Encode
        z = self.encode(mel)  # (B, latent_dim, T')

        # Decode
        recon = self.decode(z, T)  # (B, T, n_mels)

        # Reconstruction loss
        loss = F.mse_loss(recon, mel)

        return {
            'reconstruction': recon,
            'latent': z,
            'loss': loss,
        }

    def reconstruction_error(self, mel: torch.Tensor) -> torch.Tensor:
        """Get per-sample reconstruction error (quality score inverse).

        Args:
            mel: (B, T, n_mels)

        Returns:
            (B,) reconstruction error per sample (lower = better quality)
        """
        with torch.no_grad():
            result = self.forward(mel)
            recon = result['reconstruction']
            # Per-sample MSE
            error = ((recon - mel) ** 2).mean(dim=(1, 2))
        return error

    def quality_score(self, mel: torch.Tensor) -> torch.Tensor:
        """Get quality score (higher = better).

        Converts reconstruction error to [0, 1] score.
        """
        error = self.reconstruction_error(mel)
        # Convert error to score: low error = high score
        # Use exponential mapping: score = exp(-error * scale)
        score = torch.exp(-error * 10)
        return score

    def save(self, path: str):
        """Save checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'config': self.config.__dict__,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str) -> 'QualityAutoencoder':
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])

        return model

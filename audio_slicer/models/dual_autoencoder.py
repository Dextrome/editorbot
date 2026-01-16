"""Dual Autoencoder - FaceSwap-style audio transformation.

Like FaceSwap:
- Shared encoder learns content features
- Decoder_raw learns to output raw audio style
- Decoder_edited learns to output edited audio style

At inference:
- Encode raw audio → Decode with edited decoder
- This transforms raw audio to "edited quality"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path

from ..config import ModelConfig


class ConvEncoder(nn.Module):
    """Shared encoder - learns content representation."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(config.n_mels, config.hidden_dims[0])

        # Downsampling conv blocks
        layers = []
        in_dim = config.hidden_dims[0]
        for out_dim in config.hidden_dims:
            layers.append(nn.Conv1d(in_dim, out_dim, 4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = out_dim

        # Final projection to latent
        layers.append(nn.Conv1d(in_dim, config.latent_dim, 3, padding=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_mels) mel spectrogram
        Returns:
            (B, latent_dim, T') compressed latent
        """
        x = self.input_proj(x)  # (B, T, hidden)
        x = x.transpose(1, 2)  # (B, hidden, T)
        z = self.conv(x)  # (B, latent_dim, T')
        return z


class ConvDecoder(nn.Module):
    """Decoder - learns to output specific style (raw or edited)."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Upsampling conv blocks (mirror encoder)
        layers = []
        in_dim = config.latent_dim

        for out_dim in reversed(config.hidden_dims):
            layers.append(nn.ConvTranspose1d(in_dim, out_dim, 4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = out_dim

        self.deconv = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.n_mels),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T') latent
            target_len: Target sequence length
        Returns:
            (B, T, n_mels) reconstructed mel
        """
        x = self.deconv(z)  # (B, hidden[0], T'')

        # Interpolate to exact length
        if x.size(2) != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        x = x.transpose(1, 2)  # (B, T, hidden[0])
        x = self.output_proj(x)  # (B, T, n_mels)
        return x


class DualAutoencoder(nn.Module):
    """FaceSwap-style dual autoencoder for audio.

    Shared encoder + two decoders (raw and edited).
    Train both simultaneously, then use cross-decoding at inference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = ConvEncoder(config)

        # Two decoders
        self.decoder_raw = ConvDecoder(config)
        self.decoder_edited = ConvDecoder(config)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel to latent."""
        return self.encoder(mel)

    def decode_raw(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """Decode to raw style."""
        return self.decoder_raw(z, target_len)

    def decode_edited(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """Decode to edited style."""
        return self.decoder_edited(z, target_len)

    def forward_raw(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for raw audio."""
        T = mel.size(1)
        z = self.encode(mel)
        recon = self.decode_raw(z, T)
        loss = F.mse_loss(recon, mel)
        return {'reconstruction': recon, 'latent': z, 'loss': loss}

    def forward_edited(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for edited audio."""
        T = mel.size(1)
        z = self.encode(mel)
        recon = self.decode_edited(z, T)
        loss = F.mse_loss(recon, mel)
        return {'reconstruction': recon, 'latent': z, 'loss': loss}

    def transform_raw_to_edited(self, mel: torch.Tensor) -> torch.Tensor:
        """Transform raw audio to edited style (the magic!)."""
        T = mel.size(1)
        z = self.encode(mel)
        transformed = self.decode_edited(z, T)
        return transformed

    def get_transformation_score(self, raw_mel: torch.Tensor) -> torch.Tensor:
        """Score how well raw transforms to edited style.

        Low error = raw segment is compatible with edited style (keep it)
        High error = raw segment doesn't fit edited pattern (cut it)

        Returns:
            (B,) score in [0, 1], higher = better
        """
        with torch.no_grad():
            # Transform raw → edited style
            transformed = self.transform_raw_to_edited(raw_mel)

            # Re-encode transformed and decode back to raw
            # If transformation is faithful, this should match original
            z_transformed = self.encode(transformed)
            T = raw_mel.size(1)
            back_to_raw = self.decode_raw(z_transformed, T)

            # Cycle consistency error
            cycle_error = ((back_to_raw - raw_mel) ** 2).mean(dim=(1, 2))

            # Also measure how "edited-like" the transformed version is
            # by checking its reconstruction error with edited decoder
            z_raw = self.encode(raw_mel)
            edited_recon = self.decode_edited(z_raw, T)
            z_edited_recon = self.encode(edited_recon)
            back_to_edited = self.decode_edited(z_edited_recon, T)
            edit_consistency = ((back_to_edited - edited_recon) ** 2).mean(dim=(1, 2))

            # Combined score (lower error = higher score)
            total_error = cycle_error + edit_consistency
            score = torch.exp(-total_error * 5)

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
    def from_checkpoint(cls, path: str) -> 'DualAutoencoder':
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])

        return model

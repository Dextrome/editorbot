"""Reconstruction Model combining Encoder and Decoder for Phase 1.

This is the main model that takes (raw_mel, edit_labels) and produces edited_mel.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ..config import Phase1Config
from .encoder import EditEncoder, ConvEditEncoder
from .decoder import MelDecoder, MultiScaleMelDecoder


class ReconstructionModel(nn.Module):
    """Full reconstruction model: (raw_mel, edit_labels) -> edited_mel.

    Architecture:
        1. EditEncoder: raw_mel + edit_labels -> latent
        2. Optional projection if encoder_dim != decoder_dim
        3. MelDecoder: latent + raw_mel -> edited_mel
    """

    def __init__(self, config: Phase1Config, use_conv_encoder: bool = False):
        super().__init__()
        self.config = config

        # Encoder
        if use_conv_encoder:
            self.encoder = ConvEditEncoder(config)
        else:
            self.encoder = EditEncoder(config)

        # Projection if dims differ
        if config.encoder_dim != config.decoder_dim:
            self.latent_proj = nn.Linear(config.encoder_dim, config.decoder_dim)
        else:
            self.latent_proj = nn.Identity()

        # Decoder
        self.decoder = MelDecoder(config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        raw_mel: torch.Tensor,      # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T) int
        mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Args:
            raw_mel: Original mel spectrogram (B, T, n_mels)
            edit_labels: Edit labels per frame (B, T)
            mask: Valid frame mask (B, T), True = valid

        Returns:
            pred_mel: Predicted edited mel (B, T, n_mels)
        """
        # Encode
        latent = self.encoder(raw_mel, edit_labels, mask)  # (B, T, encoder_dim)

        # Project if needed
        latent = self.latent_proj(latent)  # (B, T, decoder_dim)

        # Decode (pass edit_labels for label-aware gating)
        pred_mel = self.decoder(latent, raw_mel, edit_labels, mask)  # (B, T, n_mels)

        return pred_mel

    def encode(
        self,
        raw_mel: torch.Tensor,
        edit_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent representation (for analysis/debugging)."""
        latent = self.encoder(raw_mel, edit_labels, mask)
        return self.latent_proj(latent)

    def decode(
        self,
        latent: torch.Tensor,
        raw_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode from latent (for analysis/debugging)."""
        return self.decoder(latent, raw_mel, mask)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: Optional[Phase1Config] = None) -> 'ReconstructionModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if config is None:
            config = checkpoint.get('config', Phase1Config())

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        **kwargs,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint.update(kwargs)
        torch.save(checkpoint, path)


class MultiScaleReconstructionModel(nn.Module):
    """Reconstruction model with multi-scale outputs.

    Produces mel at multiple scales for coarse-to-fine training.
    """

    def __init__(self, config: Phase1Config, use_conv_encoder: bool = False):
        super().__init__()
        self.config = config

        # Encoder
        if use_conv_encoder:
            self.encoder = ConvEditEncoder(config)
        else:
            self.encoder = EditEncoder(config)

        # Projection
        if config.encoder_dim != config.decoder_dim:
            self.latent_proj = nn.Linear(config.encoder_dim, config.decoder_dim)
        else:
            self.latent_proj = nn.Identity()

        # Multi-scale decoder
        self.decoder = MultiScaleMelDecoder(config)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        raw_mel: torch.Tensor,
        edit_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_scales: bool = False,
    ):
        """
        Args:
            raw_mel: Original mel spectrogram (B, T, n_mels)
            edit_labels: Edit labels per frame (B, T)
            mask: Valid frame mask (B, T)
            return_all_scales: If True, return all scale outputs

        Returns:
            If return_all_scales:
                (pred_mel, pred_mel_half, pred_mel_quarter)
            Else:
                pred_mel
        """
        latent = self.encoder(raw_mel, edit_labels, mask)
        latent = self.latent_proj(latent)

        return self.decoder(latent, raw_mel, mask, return_all_scales=return_all_scales)

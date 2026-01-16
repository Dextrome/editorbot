"""Loss functions for mel-to-mel training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..config import LossConfig


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for audio quality.

    Computes spectral convergence and log magnitude loss at multiple FFT sizes.
    Applied to audio reconstructed from mel (requires vocoder during training,
    or can be approximated on mel directly).
    """

    def __init__(self, fft_sizes: list = [512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel (B, T, n_mels)
            target: Target mel (B, T, n_mels)

        Returns:
            Multi-scale loss
        """
        # Flatten mel to pseudo-waveform for STFT-like comparison
        # This is an approximation - real STFT would need audio
        pred_flat = pred.flatten(1)  # (B, T*n_mels)
        target_flat = target.flatten(1)

        total_loss = 0
        for fft_size in self.fft_sizes:
            hop = fft_size // 4

            # Compute STFT
            pred_stft = torch.stft(
                pred_flat, fft_size, hop, fft_size,
                window=torch.hann_window(fft_size, device=pred.device),
                return_complex=True
            )
            target_stft = torch.stft(
                target_flat, fft_size, hop, fft_size,
                window=torch.hann_window(fft_size, device=target.device),
                return_complex=True
            )

            # Magnitude
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Spectral convergence
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

            # Log magnitude loss
            log_loss = F.l1_loss(
                torch.log(pred_mag + 1e-8),
                torch.log(target_mag + 1e-8)
            )

            total_loss += sc_loss + log_loss

        return total_loss / len(self.fft_sizes)


class PreservationLoss(nn.Module):
    """Penalize changes in regions where raw ≈ edited.

    If the raw and target are similar, the model should preserve them.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel (B, T, n_mels)
            target: Target mel (B, T, n_mels)
            raw: Raw input mel (B, T, n_mels)

        Returns:
            Preservation loss
        """
        # Find regions where raw ≈ target (unchanged regions)
        diff = torch.abs(raw - target).mean(dim=-1)  # (B, T)
        preserve_mask = (diff < self.threshold).float().unsqueeze(-1)  # (B, T, 1)

        # In preserved regions, pred should match raw (not target, since they're same)
        preserve_loss = (torch.abs(pred - raw) * preserve_mask).sum() / (preserve_mask.sum() + 1e-8)

        return preserve_loss


class CombinedLoss(nn.Module):
    """Combined loss for mel-to-mel training."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        if config.stft_weight > 0:
            self.stft_loss = MultiScaleSTFTLoss(config.stft_fft_sizes)

        if config.preservation_weight > 0:
            self.preservation_loss = PreservationLoss(config.preservation_threshold)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Predicted mel (B, T, n_mels)
            target: Target mel (B, T, n_mels)
            raw: Raw input mel (B, T, n_mels) - for preservation loss

        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        total = 0

        # L1 loss
        if self.config.l1_weight > 0:
            l1 = F.l1_loss(pred, target)
            losses['l1'] = l1
            total += self.config.l1_weight * l1

        # MSE loss
        if self.config.mse_weight > 0:
            mse = F.mse_loss(pred, target)
            losses['mse'] = mse
            total += self.config.mse_weight * mse

        # Multi-scale STFT loss
        if self.config.stft_weight > 0:
            stft = self.stft_loss(pred, target)
            losses['stft'] = stft
            total += self.config.stft_weight * stft

        # Preservation loss
        if self.config.preservation_weight > 0 and raw is not None:
            preserve = self.preservation_loss(pred, target, raw)
            losses['preservation'] = preserve
            total += self.config.preservation_weight * preserve

        losses['total'] = total
        return losses

"""Reconstruction loss functions for mel spectrogram training.

Includes L1, MSE, and Multi-Scale STFT losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class L1MelLoss(nn.Module):
    """L1 loss on mel spectrogram."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,    # (B, T, n_mels)
        target: torch.Tensor,  # (B, T, n_mels)
        mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel spectrogram
            target: Target mel spectrogram
            mask: Valid frame mask, True = valid

        Returns:
            L1 loss value
        """
        loss = torch.abs(pred - target)

        if mask is not None:
            # Expand mask to mel dimension
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            loss = loss * mask_expanded

            if self.reduction == 'mean':
                # Average over valid positions
                return loss.sum() / (mask_expanded.sum() * pred.size(-1) + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class MSEMelLoss(nn.Module):
    """MSE loss on mel spectrogram."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = (pred - target) ** 2

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            loss = loss * mask_expanded

            if self.reduction == 'mean':
                return loss.sum() / (mask_expanded.sum() * pred.size(-1) + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class STFTLoss(nn.Module):
    """STFT-based loss for a single scale.

    Combines spectral convergence and log magnitude losses.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Register window buffer
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(
        self,
        pred: torch.Tensor,    # (B, T, n_mels) or (B, L) waveform
        target: torch.Tensor,  # Same shape as pred
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: Predicted signal
            target: Target signal

        Returns:
            (spectral_convergence_loss, log_magnitude_loss)
        """
        # Flatten if mel spectrogram input
        if pred.dim() == 3:
            pred = pred.reshape(pred.size(0), -1)
            target = target.reshape(target.size(0), -1)

        # STFT
        pred_stft = torch.stft(
            pred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # Magnitudes
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        # Spectral convergence: ||S_target - S_pred|| / ||S_target||
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

        # Log magnitude loss
        log_pred = torch.log(pred_mag + 1e-8)
        log_target = torch.log(target_mag + 1e-8)
        mag_loss = F.l1_loss(log_pred, log_target)

        return sc_loss, mag_loss


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss.

    Computes STFT loss at multiple resolutions to capture
    both fine-grained and coarse frequency information.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_sizes: List[int] = [512, 1024, 2048],
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
    ):
        super().__init__()
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=fft, hop_length=hop, win_length=win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted signal (B, T, n_mels) or (B, L)
            target: Target signal

        Returns:
            Combined multi-scale STFT loss
        """
        total_sc = 0.0
        total_mag = 0.0

        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(pred, target)
            total_sc += sc
            total_mag += mag

        n_scales = len(self.stft_losses)
        total_loss = (self.sc_weight * total_sc + self.mag_weight * total_mag) / n_scales

        return total_loss


class SpectralConvergenceLoss(nn.Module):
    """Standalone spectral convergence loss."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.reshape(pred.size(0), -1)
            target = target.reshape(target.size(0), -1)

        pred_stft = torch.stft(
            pred, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True
        )
        target_stft = torch.stft(
            target, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True
        )

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        return torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)


class LogMagnitudeLoss(nn.Module):
    """Standalone log magnitude STFT loss."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.reshape(pred.size(0), -1)
            target = target.reshape(target.size(0), -1)

        pred_stft = torch.stft(
            pred, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True
        )
        target_stft = torch.stft(
            target, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True
        )

        log_pred = torch.log(torch.abs(pred_stft) + 1e-8)
        log_target = torch.log(torch.abs(target_stft) + 1e-8)

        return F.l1_loss(log_pred, log_target)

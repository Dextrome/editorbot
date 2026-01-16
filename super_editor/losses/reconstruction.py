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

        # STFT - ensure window is on same device
        window = self.window.to(pred.device)
        pred_stft = torch.stft(
            pred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
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


class LabelConditionedLoss(nn.Module):
    """Loss that enforces label-dependent behavior.

    - CUT (0): Output should be silent (zero)
    - KEEP (1): Output should match input (pass-through)

    This forces the model to actually use the labels!
    """

    def __init__(self, cut_weight: float = 1.0, keep_weight: float = 1.0):
        super().__init__()
        self.cut_weight = cut_weight
        self.keep_weight = keep_weight

    def forward(
        self,
        pred: torch.Tensor,       # (B, T, n_mels)
        raw_mel: torch.Tensor,    # (B, T, n_mels) - original input
        edit_labels: torch.Tensor,  # (B, T) - 0=CUT, 1=KEEP
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cut_loss: L1 loss pushing CUT frames toward zero
            keep_loss: L1 loss keeping KEEP frames same as input
        """
        B, T, M = pred.shape

        # Create label masks
        cut_mask = (edit_labels == 0).unsqueeze(-1).float()  # (B, T, 1)
        keep_mask = (edit_labels == 1).unsqueeze(-1).float()  # (B, T, 1)

        if mask is not None:
            valid_mask = mask.unsqueeze(-1).float()
            cut_mask = cut_mask * valid_mask
            keep_mask = keep_mask * valid_mask

        # CUT loss: output should be zero for CUT frames
        cut_loss = (torch.abs(pred) * cut_mask).sum() / (cut_mask.sum() * M + 1e-8)

        # KEEP loss: output should match input for KEEP frames
        keep_diff = torch.abs(pred - raw_mel) * keep_mask
        keep_loss = keep_diff.sum() / (keep_mask.sum() * M + 1e-8)

        return self.cut_weight * cut_loss, self.keep_weight * keep_loss


class LabelContrastiveLoss(nn.Module):
    """Contrastive loss ensuring CUT and KEEP outputs are different.

    Pushes CUT frame outputs to be dissimilar from KEEP frame outputs.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred: torch.Tensor,       # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        Encourages CUT outputs to be at least `margin` different from KEEP outputs.
        """
        B, T, M = pred.shape

        # Compute mean output for CUT and KEEP frames per batch
        cut_mask = (edit_labels == 0).unsqueeze(-1).float()
        keep_mask = (edit_labels == 1).unsqueeze(-1).float()

        # Mean mel for CUT and KEEP frames
        cut_sum = (pred * cut_mask).sum(dim=1)  # (B, M)
        cut_count = cut_mask.sum(dim=1) + 1e-8  # (B, 1)
        cut_mean = cut_sum / cut_count

        keep_sum = (pred * keep_mask).sum(dim=1)
        keep_count = keep_mask.sum(dim=1) + 1e-8
        keep_mean = keep_sum / keep_count

        # Contrastive: difference should be at least margin
        diff = torch.abs(cut_mean - keep_mean).mean(dim=-1)  # (B,)
        loss = F.relu(self.margin - diff).mean()

        return loss

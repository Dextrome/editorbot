"""Loss functions for pointer network training.

Includes pointer prediction, length prediction, smoothness, and auxiliary losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PointerLoss(nn.Module):
    """Cross-entropy loss for pointer prediction with label smoothing."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,      # (B, T, vocab_size)
        targets: torch.Tensor,     # (B, T)
        mask: Optional[torch.Tensor] = None,  # (B, T) True = valid
    ) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape

        # Prepare targets
        targets = targets.clone()
        if mask is not None:
            targets[~mask] = self.ignore_index

        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss


class LengthLoss(nn.Module):
    """MSE loss for length prediction."""

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        pred_length: torch.Tensor,   # (B,) or (B, 1)
        true_length: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        pred_length = pred_length.squeeze()
        true_length = true_length.float()

        if self.normalize:
            # Normalize by max length for stability
            max_len = true_length.max().clamp(min=1)
            loss = F.mse_loss(pred_length / max_len, true_length / max_len)
        else:
            loss = F.mse_loss(pred_length, true_length)

        return loss


class StopLoss(nn.Module):
    """Binary cross-entropy for stop token prediction."""

    def __init__(self, pos_weight: float = 5.0):
        """
        Args:
            pos_weight: Weight for positive class (STOP) to handle imbalance.
                       Since STOP only appears once per sequence, it's rare.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        stop_logits: torch.Tensor,   # (B, T) or (B, T, 1)
        seq_lengths: torch.Tensor,   # (B,) actual sequence lengths
        max_len: int,
    ) -> torch.Tensor:
        batch_size = stop_logits.size(0)
        device = stop_logits.device

        if stop_logits.dim() == 3:
            stop_logits = stop_logits.squeeze(-1)

        # Create targets: 1 at sequence end, 0 elsewhere
        stop_targets = torch.zeros_like(stop_logits)
        for i in range(batch_size):
            end_pos = min(seq_lengths[i].item(), max_len - 1)
            stop_targets[i, int(end_pos)] = 1.0

        # Create mask for valid positions (up to seq_length + 1 for stop)
        mask = torch.zeros_like(stop_logits, dtype=torch.bool)
        for i in range(batch_size):
            valid_len = min(seq_lengths[i].item() + 1, max_len)
            mask[i, :int(valid_len)] = True

        # Apply pos_weight
        weight = torch.ones_like(stop_targets)
        weight[stop_targets == 1] = self.pos_weight

        loss = F.binary_cross_entropy_with_logits(
            stop_logits[mask],
            stop_targets[mask],
            weight=weight[mask],
        )
        return loss


class SmoothnessLoss(nn.Module):
    """Penalize non-smooth pointer sequences.

    Encourages consecutive pointers to be close together,
    which results in more natural-sounding edits.
    """

    def __init__(self, max_jump: int = 100, margin: float = 0.0):
        """
        Args:
            max_jump: Jumps larger than this are expected (cuts/loops) and not penalized.
            margin: Don't penalize jumps smaller than this.
        """
        super().__init__()
        self.max_jump = max_jump
        self.margin = margin

    def forward(
        self,
        pointers: torch.Tensor,  # (B, T) predicted pointer indices
        mask: Optional[torch.Tensor] = None,  # (B, T) True = valid
    ) -> torch.Tensor:
        # Compute differences between consecutive pointers
        diffs = torch.abs(pointers[:, 1:].float() - pointers[:, :-1].float())

        # Only penalize medium jumps (small jumps are fine, large jumps are intentional cuts)
        # Penalize if margin < diff < max_jump
        penalties = torch.clamp(diffs - self.margin, min=0)
        penalties = torch.where(
            diffs > self.max_jump,
            torch.zeros_like(penalties),  # Don't penalize large jumps
            penalties
        )

        if mask is not None:
            # Both positions must be valid
            valid = mask[:, 1:] & mask[:, :-1]
            if valid.sum() > 0:
                return penalties[valid].mean()
            return torch.tensor(0.0, device=pointers.device)

        return penalties.mean()


class MonotonicityLoss(nn.Module):
    """Encourage monotonically increasing pointers within segments.

    For sequential playback (no cuts/loops), pointers should generally increase.
    """

    def __init__(self, tolerance: int = 10):
        super().__init__()
        self.tolerance = tolerance

    def forward(
        self,
        pointers: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Expected: pointer[t+1] >= pointer[t] - tolerance
        expected_min = pointers[:, :-1] - self.tolerance
        violations = torch.clamp(expected_min - pointers[:, 1:].float(), min=0)

        if mask is not None:
            valid = mask[:, 1:] & mask[:, :-1]
            if valid.sum() > 0:
                return violations[valid].mean()
            return torch.tensor(0.0, device=pointers.device)

        return violations.mean()


class CombinedPointerLoss(nn.Module):
    """Combined loss for pointer network training."""

    def __init__(
        self,
        pointer_weight: float = 1.0,
        length_weight: float = 0.1,
        stop_weight: float = 0.5,
        smoothness_weight: float = 0.01,
        kl_weight: float = 0.01,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.pointer_loss = PointerLoss(label_smoothing=label_smoothing)
        self.length_loss = LengthLoss()
        self.stop_loss = StopLoss()
        self.smoothness_loss = SmoothnessLoss()

        self.pointer_weight = pointer_weight
        self.length_weight = length_weight
        self.stop_weight = stop_weight
        self.smoothness_weight = smoothness_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,         # (B, T) target pointer indices
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict with keys like 'logits', 'length_pred', 'stop_logits', 'kl_loss'
            targets: Target pointer indices
            mask: Valid position mask (True = valid)

        Returns:
            Dict of individual losses and total loss
        """
        losses = {}
        total = 0.0

        # Pointer prediction loss
        if 'logits' in outputs:
            ptr_loss = self.pointer_loss(outputs['logits'], targets, mask)
            losses['pointer_loss'] = ptr_loss
            total = total + self.pointer_weight * ptr_loss

        # Length prediction loss
        if 'length_pred' in outputs and mask is not None:
            true_lengths = mask.sum(dim=1)
            len_loss = self.length_loss(outputs['length_pred'], true_lengths)
            losses['length_loss'] = len_loss
            total = total + self.length_weight * len_loss

        # Stop prediction loss
        if 'stop_logits' in outputs and mask is not None:
            true_lengths = mask.sum(dim=1)
            stop_loss = self.stop_loss(
                outputs['stop_logits'],
                true_lengths,
                max_len=targets.size(1),
            )
            losses['stop_loss'] = stop_loss
            total = total + self.stop_weight * stop_loss

        # VAE KL loss
        if 'kl_loss' in outputs:
            losses['kl_loss'] = outputs['kl_loss']
            total = total + self.kl_weight * outputs['kl_loss']

        # Smoothness loss (on predicted pointers)
        if 'logits' in outputs and self.smoothness_weight > 0:
            pred_pointers = outputs['logits'].argmax(dim=-1)
            smooth_loss = self.smoothness_loss(pred_pointers, mask)
            losses['smoothness_loss'] = smooth_loss
            total = total + self.smoothness_weight * smooth_loss

        losses['total_loss'] = total
        return losses

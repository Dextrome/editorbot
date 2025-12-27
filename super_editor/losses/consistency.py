"""Edit consistency losses for Super Editor.

These losses ensure that regions marked as KEEP remain unchanged,
and that transitions between edit regions are smooth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..config import EditLabel


class EditConsistencyLoss(nn.Module):
    """Loss that penalizes changes in KEEP regions.

    When edit_label == KEEP (1), the prediction should match
    the raw input exactly. This helps the model learn to
    preserve unedited regions.
    """

    def __init__(self, keep_label: int = EditLabel.KEEP, reduction: str = 'mean'):
        super().__init__()
        self.keep_label = keep_label
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,         # (B, T, n_mels)
        raw: torch.Tensor,          # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T) int
        mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel spectrogram
            raw: Raw input mel spectrogram (should be preserved for KEEP)
            edit_labels: Edit labels per frame
            mask: Valid frame mask

        Returns:
            Consistency loss for KEEP regions
        """
        # Create KEEP mask: 1 where label == KEEP, 0 elsewhere
        keep_mask = (edit_labels == self.keep_label).float()  # (B, T)

        # Combine with validity mask if provided
        if mask is not None:
            keep_mask = keep_mask * mask.float()

        # Expand to mel dimension
        keep_mask = keep_mask.unsqueeze(-1)  # (B, T, 1)

        # L1 difference weighted by keep mask
        diff = torch.abs(pred - raw) * keep_mask

        if self.reduction == 'mean':
            # Average over KEEP positions only
            n_keep = keep_mask.sum() + 1e-8
            return diff.sum() / (n_keep * pred.size(-1))
        elif self.reduction == 'sum':
            return diff.sum()
        else:
            return diff


class SmoothnessLoss(nn.Module):
    """Loss that encourages smooth transitions in predictions.

    Penalizes large frame-to-frame changes, especially at edit boundaries.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,  # (B, T, n_mels)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel spectrogram
            mask: Valid frame mask

        Returns:
            Temporal smoothness loss
        """
        # Frame-to-frame difference
        diff = pred[:, 1:, :] - pred[:, :-1, :]  # (B, T-1, n_mels)
        loss = torch.abs(diff)

        if mask is not None:
            # Only compute smoothness for valid transitions
            # Both current and next frame must be valid
            valid_transitions = mask[:, 1:] & mask[:, :-1]  # (B, T-1)
            loss = loss * valid_transitions.unsqueeze(-1).float()

            if self.reduction == 'mean':
                n_valid = valid_transitions.sum() + 1e-8
                return loss.sum() / (n_valid * pred.size(-1))
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


class EditBoundaryLoss(nn.Module):
    """Loss that focuses on edit boundary regions.

    Extra supervision at frames where edit label changes,
    since these transitions are hardest to get right.
    """

    def __init__(self, boundary_width: int = 2, reduction: str = 'mean'):
        super().__init__()
        self.boundary_width = boundary_width
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,         # (B, T, n_mels)
        target: torch.Tensor,       # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel spectrogram
            target: Target mel spectrogram
            edit_labels: Edit labels per frame
            mask: Valid frame mask

        Returns:
            Boundary-focused loss
        """
        B, T, D = pred.shape

        # Find edit boundaries (where label changes)
        label_changes = (edit_labels[:, 1:] != edit_labels[:, :-1])  # (B, T-1)

        # Expand boundary regions
        boundary_mask = torch.zeros(B, T, device=pred.device)

        for w in range(-self.boundary_width, self.boundary_width + 1):
            if w == 0:
                continue
            shifted_idx = torch.arange(T-1, device=pred.device) + w + 1
            valid_idx = (shifted_idx >= 0) & (shifted_idx < T)
            if valid_idx.any():
                boundary_mask[:, shifted_idx[valid_idx]] += label_changes[:, valid_idx].float()

        # Also include the actual change positions
        boundary_mask[:, 1:] += label_changes.float()

        # Clamp to [0, 1]
        boundary_mask = boundary_mask.clamp(0, 1)

        # Combine with validity mask
        if mask is not None:
            boundary_mask = boundary_mask * mask.float()

        # Expand to mel dimension
        boundary_mask = boundary_mask.unsqueeze(-1)  # (B, T, 1)

        # Weighted L1 loss at boundaries
        diff = torch.abs(pred - target) * boundary_mask

        if self.reduction == 'mean':
            n_boundary = boundary_mask.sum() + 1e-8
            return diff.sum() / (n_boundary * D)
        elif self.reduction == 'sum':
            return diff.sum()
        else:
            return diff


class LabelSpecificLoss(nn.Module):
    """Apply different loss weights based on edit label type.

    Allows stronger supervision for harder edit types (e.g., EFFECT, TRANSITION).
    """

    def __init__(
        self,
        n_labels: int = 8,
        label_weights: Optional[dict] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.reduction = reduction

        # Default: equal weights
        if label_weights is None:
            label_weights = {i: 1.0 for i in range(n_labels)}

        # Convert to tensor
        weights = torch.zeros(n_labels)
        for label, weight in label_weights.items():
            weights[label] = weight
        self.register_buffer('label_weights', weights)

    def forward(
        self,
        pred: torch.Tensor,         # (B, T, n_mels)
        target: torch.Tensor,       # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted mel spectrogram
            target: Target mel spectrogram
            edit_labels: Edit labels per frame
            mask: Valid frame mask

        Returns:
            Label-weighted reconstruction loss
        """
        # Get weight for each position
        position_weights = self.label_weights[edit_labels]  # (B, T)

        if mask is not None:
            position_weights = position_weights * mask.float()

        # Expand to mel dimension
        position_weights = position_weights.unsqueeze(-1)  # (B, T, 1)

        # Weighted L1 loss
        diff = torch.abs(pred - target) * position_weights

        if self.reduction == 'mean':
            n_weighted = position_weights.sum() + 1e-8
            return diff.sum() / (n_weighted * pred.size(-1))
        elif self.reduction == 'sum':
            return diff.sum()
        else:
            return diff

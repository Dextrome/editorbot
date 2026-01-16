"""Combined loss for Phase 1 training."""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..config import Phase1Config
from .reconstruction import (
    L1MelLoss, MSEMelLoss, MultiScaleSTFTLoss,
    LabelConditionedLoss, LabelContrastiveLoss
)
from .consistency import EditConsistencyLoss, SmoothnessLoss


class Phase1Loss(nn.Module):
    """Combined loss for Phase 1 supervised reconstruction training.

    Combines:
        - L1 reconstruction loss
        - MSE reconstruction loss (optional)
        - Multi-scale STFT loss
        - Edit consistency loss (preserve KEEP regions)
        - Label-conditioned loss (CUT->zero, KEEP->input)
        - Label contrastive loss (CUT and KEEP should differ)
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Individual loss functions
        self.l1_loss = L1MelLoss(reduction="mean")
        self.mse_loss = MSEMelLoss(reduction="mean")
        self.stft_loss = MultiScaleSTFTLoss(
            fft_sizes=config.stft_fft_sizes,
            hop_sizes=config.stft_hop_sizes,
            win_sizes=config.stft_win_sizes,
        )
        self.consistency_loss = EditConsistencyLoss(keep_label=1)  # KEEP=1
        self.smoothness_loss = SmoothnessLoss(reduction="mean")

        # Label-enforcing losses (critical for making model use labels!)
        self.label_conditioned_loss = LabelConditionedLoss(cut_weight=1.0, keep_weight=1.0)
        self.label_contrastive_loss = LabelContrastiveLoss(margin=0.5)

        # Weights from config
        self.l1_weight = config.l1_weight
        self.mse_weight = config.mse_weight
        self.stft_weight = config.stft_weight
        self.consistency_weight = config.consistency_weight
        self.label_conditioned_weight = getattr(config, "label_conditioned_weight", 2.0)
        self.label_contrastive_weight = getattr(config, "label_contrastive_weight", 0.5)

    def forward(
        self,
        pred: torch.Tensor,         # (B, T, n_mels)
        target: torch.Tensor,       # (B, T, n_mels)
        raw: torch.Tensor,          # (B, T, n_mels) - original input
        edit_labels: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Predicted edited mel spectrogram
            target: Target edited mel spectrogram
            raw: Raw input mel spectrogram
            edit_labels: Edit labels per frame
            mask: Valid frame mask

        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}

        # L1 reconstruction loss
        if self.l1_weight > 0:
            losses["l1"] = self.l1_loss(pred, target, mask)
        else:
            losses["l1"] = torch.tensor(0.0, device=pred.device)

        # MSE reconstruction loss
        if self.mse_weight > 0:
            losses["mse"] = self.mse_loss(pred, target, mask)
        else:
            losses["mse"] = torch.tensor(0.0, device=pred.device)

        # Multi-scale STFT loss
        if self.stft_weight > 0:
            losses["stft"] = self.stft_loss(pred, target)
        else:
            losses["stft"] = torch.tensor(0.0, device=pred.device)

        # Edit consistency loss (preserve KEEP regions)
        if self.consistency_weight > 0:
            losses["consistency"] = self.consistency_loss(pred, raw, edit_labels, mask)
        else:
            losses["consistency"] = torch.tensor(0.0, device=pred.device)

        # Label-conditioned loss (CUT->zero, KEEP->input)
        if self.label_conditioned_weight > 0:
            cut_loss, keep_loss = self.label_conditioned_loss(pred, raw, edit_labels, mask)
            losses["cut_loss"] = cut_loss
            losses["keep_loss"] = keep_loss
        else:
            losses["cut_loss"] = torch.tensor(0.0, device=pred.device)
            losses["keep_loss"] = torch.tensor(0.0, device=pred.device)

        # Label contrastive loss (CUT and KEEP should differ)
        if self.label_contrastive_weight > 0:
            losses["contrastive"] = self.label_contrastive_loss(pred, edit_labels)
        else:
            losses["contrastive"] = torch.tensor(0.0, device=pred.device)

        # Compute total loss
        losses["total"] = (
            self.l1_weight * losses["l1"] +
            self.mse_weight * losses["mse"] +
            self.stft_weight * losses["stft"] +
            self.consistency_weight * losses["consistency"] +
            self.label_conditioned_weight * (losses["cut_loss"] + losses["keep_loss"]) +
            self.label_contrastive_weight * losses["contrastive"]
        )

        return losses

    def update_weights(
        self,
        l1_weight: Optional[float] = None,
        mse_weight: Optional[float] = None,
        stft_weight: Optional[float] = None,
        consistency_weight: Optional[float] = None,
        label_conditioned_weight: Optional[float] = None,
        label_contrastive_weight: Optional[float] = None,
    ):
        """Update loss weights during training (e.g., for curriculum)."""
        if l1_weight is not None:
            self.l1_weight = l1_weight
        if mse_weight is not None:
            self.mse_weight = mse_weight
        if stft_weight is not None:
            self.stft_weight = stft_weight
        if consistency_weight is not None:
            self.consistency_weight = consistency_weight
        if label_conditioned_weight is not None:
            self.label_conditioned_weight = label_conditioned_weight
        if label_contrastive_weight is not None:
            self.label_contrastive_weight = label_contrastive_weight


class MultiScalePhase1Loss(nn.Module):
    """Loss for multi-scale reconstruction model.

    Applies reconstruction loss at multiple scales.
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Base loss
        self.base_loss = Phase1Loss(config)

        # Scale weights (coarser = less weight)
        self.scale_weights = [1.0, 0.5, 0.25]  # full, half, quarter

    def forward(
        self,
        pred_full: torch.Tensor,
        pred_half: torch.Tensor,
        pred_quarter: torch.Tensor,
        target: torch.Tensor,
        raw: torch.Tensor,
        edit_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_full: Full resolution prediction
            pred_half: Half resolution prediction (upsampled)
            pred_quarter: Quarter resolution prediction (upsampled)
            target: Target mel spectrogram
            raw: Raw input mel spectrogram
            edit_labels: Edit labels
            mask: Valid frame mask

        Returns:
            Dictionary with combined losses
        """
        losses = {}

        # Full scale loss
        full_losses = self.base_loss(pred_full, target, raw, edit_labels, mask)
        for k, v in full_losses.items():
            losses[f"full_{k}"] = v

        # Half scale loss (simplified - just L1)
        losses["half_l1"] = self.base_loss.l1_loss(pred_half, target, mask)

        # Quarter scale loss
        losses["quarter_l1"] = self.base_loss.l1_loss(pred_quarter, target, mask)

        # Combine
        losses["total"] = (
            self.scale_weights[0] * full_losses["total"] +
            self.scale_weights[1] * losses["half_l1"] +
            self.scale_weights[2] * losses["quarter_l1"]
        )

        return losses

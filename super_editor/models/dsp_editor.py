"""DSP-based Editor - Applies hard-coded effects based on edit labels.

No neural network needed - just deterministic DSP operations.
This replaces the full ReconstructionModel for Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .decoder import (
    find_contiguous_segments,
    apply_fade_in,
    apply_fade_out,
    apply_loop,
    apply_effect,
    apply_transition,
)


class DSPEditor(nn.Module):
    """Applies deterministic DSP effects based on edit labels.

    Edit labels:
        0 = CUT (silence)
        1 = KEEP (passthrough)
        2 = LOOP (repeat previous audio)
        3 = FADE_IN (0→1 ramp)
        4 = FADE_OUT (1→0 ramp)
        5 = EFFECT (low-pass filter)
        6 = TRANSITION (volume dip)
        7 = PAD (silence, same as CUT)

    No training needed - all effects are hard-coded DSP operations.
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so PyTorch doesn't complain about empty module
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(
        self,
        raw_mel: torch.Tensor,      # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T) int
        mask: Optional[torch.Tensor] = None,  # (B, T) bool - ignored
    ) -> torch.Tensor:
        """Apply DSP effects based on edit labels.

        Args:
            raw_mel: Original mel spectrogram (B, T, n_mels)
            edit_labels: Edit labels per frame (B, T)
            mask: Ignored (kept for interface compatibility)

        Returns:
            edited_mel: Processed mel spectrogram (B, T, n_mels)
        """
        B, T, M = raw_mel.shape

        # Start with raw mel
        output = raw_mel.clone()

        # Vectorized operations for simple labels (faster than looping)
        # CUT (0) and PAD (7) -> silence
        cut_mask = ((edit_labels == 0) | (edit_labels == 7)).unsqueeze(-1)  # (B, T, 1)
        output = output * (~cut_mask).float()

        # KEEP (1) -> already set (no-op)

        # For complex operations (LOOP, FADE_IN, etc.) we still need per-batch processing
        # but only if those labels are present
        has_complex = (
            (edit_labels == 2).any() |  # LOOP
            (edit_labels == 3).any() |  # FADE_IN
            (edit_labels == 4).any() |  # FADE_OUT
            (edit_labels == 5).any() |  # EFFECT
            (edit_labels == 6).any()    # TRANSITION
        )

        if has_complex:
            for b in range(B):
                batch_labels = edit_labels[b]  # (T,)

                # Skip if this batch has no complex labels
                if not ((batch_labels >= 2) & (batch_labels <= 6)).any():
                    continue

                batch_output = output[b]  # (T, M)

                # LOOP (2) -> repeat previous audio segment
                loop_segments = find_contiguous_segments(batch_labels, 2)
                for start, end in loop_segments:
                    batch_output = apply_loop(batch_output, start, end)

                # FADE_IN (3) -> ramp 0→1
                fade_in_segments = find_contiguous_segments(batch_labels, 3)
                for start, end in fade_in_segments:
                    batch_output = apply_fade_in(batch_output, start, end)

                # FADE_OUT (4) -> ramp 1→0
                fade_out_segments = find_contiguous_segments(batch_labels, 4)
                for start, end in fade_out_segments:
                    batch_output = apply_fade_out(batch_output, start, end)

                # EFFECT (5) -> low-pass filter
                effect_segments = find_contiguous_segments(batch_labels, 5)
                for start, end in effect_segments:
                    batch_output = apply_effect(batch_output, start, end)

                # TRANSITION (6) -> volume dip
                transition_segments = find_contiguous_segments(batch_labels, 6)
                for start, end in transition_segments:
                    batch_output = apply_transition(batch_output, start, end)

                output[b] = batch_output

        # Ensure output stays in valid range
        output = torch.clamp(output, 0, 1)
        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str = None) -> 'DSPEditor':
        """Create DSPEditor (checkpoint ignored since no weights to load)."""
        return cls()

    def eval(self):
        """Override eval - always in eval mode (no training)."""
        return self

    def train(self, mode: bool = True):
        """Override train - always in eval mode (no training)."""
        return self

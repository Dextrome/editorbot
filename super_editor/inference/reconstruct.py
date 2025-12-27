"""Reconstruction inference using Phase 1 model."""

import torch
import numpy as np
from typing import Optional, Tuple, Union

from ..config import Phase1Config, EditLabel
from ..models import ReconstructionModel


@torch.no_grad()
def reconstruct_mel(
    model: ReconstructionModel,
    raw_mel: Union[np.ndarray, torch.Tensor],  # (T, n_mels) or (B, T, n_mels)
    edit_labels: Union[np.ndarray, torch.Tensor],  # (T,) or (B, T)
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Reconstruct edited mel spectrogram from raw mel and edit labels.

    Args:
        model: Trained ReconstructionModel
        raw_mel: Raw mel spectrogram
        edit_labels: Edit labels for each frame
        device: Device to use for inference

    Returns:
        pred_mel: Predicted edited mel spectrogram (T, n_mels) or (B, T, n_mels)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Convert to tensors if needed
    if isinstance(raw_mel, np.ndarray):
        raw_mel = torch.from_numpy(raw_mel).float()
    if isinstance(edit_labels, np.ndarray):
        edit_labels = torch.from_numpy(edit_labels).long()

    # Add batch dimension if needed
    single_sample = raw_mel.dim() == 2
    if single_sample:
        raw_mel = raw_mel.unsqueeze(0)
        edit_labels = edit_labels.unsqueeze(0)

    # Move to device
    raw_mel = raw_mel.to(device)
    edit_labels = edit_labels.to(device)

    # Forward pass
    pred_mel = model(raw_mel, edit_labels)

    # Remove batch dimension if needed
    if single_sample:
        pred_mel = pred_mel.squeeze(0)

    return pred_mel.cpu().numpy()


@torch.no_grad()
def reconstruct_with_mask(
    model: ReconstructionModel,
    raw_mel: torch.Tensor,      # (B, T, n_mels)
    edit_labels: torch.Tensor,  # (B, T)
    mask: torch.Tensor,         # (B, T)
) -> torch.Tensor:
    """Reconstruct with explicit padding mask."""
    model.eval()
    return model(raw_mel, edit_labels, mask)


def create_edit_labels_from_regions(
    length: int,
    regions: list,  # List of (start, end, label_name)
) -> np.ndarray:
    """Create edit labels array from region specifications.

    Args:
        length: Length of the sequence
        regions: List of (start_frame, end_frame, label_name) tuples
                 where label_name is one of: 'CUT', 'KEEP', 'LOOP', etc.

    Returns:
        labels: Edit labels array (length,)

    Example:
        labels = create_edit_labels_from_regions(100, [
            (0, 20, 'KEEP'),
            (20, 40, 'CUT'),
            (40, 60, 'LOOP'),
            (60, 100, 'KEEP'),
        ])
    """
    labels = np.ones(length, dtype=np.int64)  # Default: KEEP

    label_map = {
        'CUT': EditLabel.CUT,
        'KEEP': EditLabel.KEEP,
        'LOOP': EditLabel.LOOP,
        'FADE_IN': EditLabel.FADE_IN,
        'FADE_OUT': EditLabel.FADE_OUT,
        'EFFECT': EditLabel.EFFECT,
        'TRANSITION': EditLabel.TRANSITION,
        'PAD': EditLabel.PAD,
    }

    for start, end, label_name in regions:
        if label_name in label_map:
            labels[start:end] = label_map[label_name]
        else:
            print(f"Warning: Unknown label '{label_name}', skipping")

    return labels


def apply_edit_labels(
    raw_mel: np.ndarray,  # (T, n_mels)
    edit_labels: np.ndarray,  # (T,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply edit labels to raw mel (simple rule-based, for testing).

    This provides a rule-based approximation of what the neural network
    should learn. Useful for testing and comparison.

    Args:
        raw_mel: Raw mel spectrogram
        edit_labels: Edit labels

    Returns:
        edited_mel: Edited mel spectrogram
        output_indices: Mapping from output to input indices
    """
    T = len(raw_mel)
    output_frames = []
    output_indices = []

    i = 0
    while i < T:
        label = edit_labels[i]

        if label == EditLabel.CUT:
            # Skip this frame
            i += 1

        elif label == EditLabel.KEEP:
            # Keep this frame
            output_frames.append(raw_mel[i])
            output_indices.append(i)
            i += 1

        elif label == EditLabel.LOOP:
            # Repeat this frame (simplified: just duplicate)
            output_frames.append(raw_mel[i])
            output_indices.append(i)
            output_frames.append(raw_mel[i])
            output_indices.append(i)
            i += 1

        elif label == EditLabel.FADE_IN:
            # Apply fade in (linear ramp)
            # Find extent of fade region
            fade_start = i
            while i < T and edit_labels[i] == EditLabel.FADE_IN:
                i += 1
            fade_end = i

            for j in range(fade_start, fade_end):
                fade_factor = (j - fade_start) / max(1, fade_end - fade_start)
                output_frames.append(raw_mel[j] * fade_factor)
                output_indices.append(j)

        elif label == EditLabel.FADE_OUT:
            # Apply fade out
            fade_start = i
            while i < T and edit_labels[i] == EditLabel.FADE_OUT:
                i += 1
            fade_end = i

            for j in range(fade_start, fade_end):
                fade_factor = 1 - (j - fade_start) / max(1, fade_end - fade_start)
                output_frames.append(raw_mel[j] * fade_factor)
                output_indices.append(j)

        elif label == EditLabel.EFFECT:
            # Placeholder: just keep the frame
            output_frames.append(raw_mel[i])
            output_indices.append(i)
            i += 1

        elif label == EditLabel.TRANSITION:
            # Placeholder: just keep the frame
            output_frames.append(raw_mel[i])
            output_indices.append(i)
            i += 1

        else:  # PAD or unknown
            i += 1

    if len(output_frames) == 0:
        # Return empty array with correct shape
        return np.zeros((0, raw_mel.shape[1]), dtype=raw_mel.dtype), np.array([])

    edited_mel = np.stack(output_frames)
    return edited_mel, np.array(output_indices)

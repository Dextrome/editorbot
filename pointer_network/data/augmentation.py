"""
Data augmentation for pointer network training.

Two categories:
- Category A: Safe augmentations (no pointer adjustment needed)
- Category B: Pointer-aware augmentations (require pointer remapping)
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    enabled: bool = True

    # A1: Additive noise
    noise_enabled: bool = True
    noise_probability: float = 0.5
    noise_level_min: float = 0.05
    noise_level_max: float = 0.15

    # A2: SpecAugment (frequency and time masking)
    spec_augment_enabled: bool = True
    spec_augment_probability: float = 0.5
    freq_masks: int = 2
    freq_mask_width: int = 20
    time_masks: int = 2
    time_mask_width: int = 50

    # A3: Gain/amplitude scaling
    gain_enabled: bool = True
    gain_probability: float = 0.5
    gain_scale_min: float = 0.7
    gain_scale_max: float = 1.3

    # A4: Channel dropout
    channel_dropout_enabled: bool = True
    channel_dropout_probability: float = 0.3
    channel_dropout_rate: float = 0.1

    # B1: Chunk shuffling
    chunk_shuffle_enabled: bool = True
    chunk_shuffle_probability: float = 0.3
    chunk_size: int = 1000

    # B2: Crop augmentation
    crop_enabled: bool = True
    crop_probability: float = 0.5
    crop_min_len: int = 500
    crop_max_len: int = 5000

    # B3: Concatenation (handled at dataset level, not here)
    concatenation_enabled: bool = False
    concatenation_probability: float = 0.2


# =============================================================================
# Category A: Safe Augmentations (no pointer adjustment)
# =============================================================================

def add_noise(
    mel: torch.Tensor,
    noise_level_min: float = 0.05,
    noise_level_max: float = 0.15
) -> torch.Tensor:
    """Add Gaussian noise to mel spectrogram.

    Args:
        mel: (n_mels, time) mel spectrogram
        noise_level_min: minimum noise level (relative to mel std)
        noise_level_max: maximum noise level

    Returns:
        Augmented mel spectrogram
    """
    noise_level = random.uniform(noise_level_min, noise_level_max)
    noise = torch.randn_like(mel) * mel.std() * noise_level
    return mel + noise


def freq_mask(
    mel: torch.Tensor,
    num_masks: int = 2,
    max_width: int = 20
) -> Tuple[torch.Tensor, list]:
    """Zero out random frequency bands.

    Args:
        mel: (n_mels, time) mel spectrogram
        num_masks: number of frequency masks to apply
        max_width: maximum width of each mask

    Returns:
        Tuple of (masked mel, list of (start, width) tuples for reproducibility)
    """
    mel = mel.clone()
    n_mels = mel.shape[0]
    masks_applied = []

    for _ in range(num_masks):
        width = random.randint(1, max_width)
        start = random.randint(0, max(0, n_mels - width))
        mel[start:start + width, :] = 0
        masks_applied.append((start, width))

    return mel, masks_applied


def time_mask(
    mel: torch.Tensor,
    num_masks: int = 2,
    max_width: int = 50
) -> Tuple[torch.Tensor, list]:
    """Zero out random time segments.

    Args:
        mel: (n_mels, time) mel spectrogram
        num_masks: number of time masks to apply
        max_width: maximum width of each mask (frames)

    Returns:
        Tuple of (masked mel, list of (start, width) tuples for reproducibility)
    """
    mel = mel.clone()
    time_steps = mel.shape[1]
    masks_applied = []

    for _ in range(num_masks):
        width = random.randint(1, min(max_width, max(1, time_steps // 10)))
        start = random.randint(0, max(0, time_steps - width))
        mel[:, start:start + width] = 0
        masks_applied.append((start, width))

    return mel, masks_applied


def apply_freq_mask(mel: torch.Tensor, masks: list) -> torch.Tensor:
    """Apply pre-computed frequency masks to mel spectrogram."""
    mel = mel.clone()
    for start, width in masks:
        mel[start:start + width, :] = 0
    return mel


def apply_time_mask(mel: torch.Tensor, masks: list) -> torch.Tensor:
    """Apply pre-computed time masks to mel spectrogram."""
    mel = mel.clone()
    for start, width in masks:
        mel[:, start:start + width] = 0
    return mel


def scale_amplitude(
    mel: torch.Tensor,
    scale_min: float = 0.7,
    scale_max: float = 1.3
) -> Tuple[torch.Tensor, float]:
    """Scale mel spectrogram amplitude.

    Args:
        mel: (n_mels, time) mel spectrogram
        scale_min: minimum scaling factor
        scale_max: maximum scaling factor

    Returns:
        Tuple of (scaled mel, scale factor used)
    """
    scale = random.uniform(scale_min, scale_max)
    return mel * scale, scale


def channel_dropout(
    mel: torch.Tensor,
    dropout_prob: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly zero out entire mel frequency channels.

    Args:
        mel: (n_mels, time) mel spectrogram
        dropout_prob: probability of dropping each channel

    Returns:
        Tuple of (mel with dropped channels, dropout mask for reproducibility)
    """
    mel = mel.clone()
    mask = torch.rand(mel.shape[0]) > dropout_prob
    mel = mel * mask.unsqueeze(1).float()
    return mel, mask


def apply_channel_dropout(mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply pre-computed channel dropout mask."""
    mel = mel.clone()
    return mel * mask.unsqueeze(1).float()


# =============================================================================
# Category B: Pointer-Aware Augmentations
# =============================================================================

def chunk_shuffle(
    raw_mel: torch.Tensor,
    edit_mel: torch.Tensor,
    pointers: torch.Tensor,
    chunk_size: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle chunks of raw audio, adjust pointers accordingly.

    The edit_mel and pointers stay in original order, but pointers
    are remapped to point to new positions in shuffled raw_mel.

    Uses vectorized operations to avoid memory leaks from large Python dicts.

    Args:
        raw_mel: (n_mels, raw_time) raw mel spectrogram
        edit_mel: (n_mels, edit_time) edited mel spectrogram
        pointers: (edit_time,) indices into raw_mel
        chunk_size: size of chunks to shuffle

    Returns:
        (shuffled_raw_mel, edit_mel, remapped_pointers)
    """
    raw_time = raw_mel.shape[1]
    n_chunks = raw_time // chunk_size

    if n_chunks < 2:
        return raw_mel, edit_mel, pointers

    # Handle remainder by including it in the last chunk
    # This avoids variable-size chunks
    usable_frames = n_chunks * chunk_size
    remainder = raw_time - usable_frames

    # Create chunk permutation
    perm = torch.randperm(n_chunks)

    # Build index mapping tensor (vectorized, no Python loops over frames)
    # For each position in original, compute its new position
    old_indices = torch.arange(usable_frames, dtype=torch.long)
    chunk_ids = old_indices // chunk_size  # Which chunk each frame belongs to
    within_chunk = old_indices % chunk_size  # Position within chunk

    # Find where each chunk moves to in the new order
    # inverse_perm[i] tells us the new position of chunk i
    inverse_perm = torch.zeros_like(perm)
    inverse_perm[perm] = torch.arange(n_chunks)

    # New position = (new_chunk_position * chunk_size) + within_chunk_offset
    new_chunk_positions = inverse_perm[chunk_ids]
    new_indices = new_chunk_positions * chunk_size + within_chunk

    # Build full mapping tensor
    mapping = torch.zeros(raw_time, dtype=torch.long)
    mapping[:usable_frames] = new_indices
    # Remainder frames stay at the end (after all shuffled chunks)
    if remainder > 0:
        mapping[usable_frames:] = torch.arange(usable_frames, raw_time)

    # Shuffle the raw_mel using the permutation
    # Reshape to (n_mels, n_chunks, chunk_size), permute chunks, reshape back
    main_part = raw_mel[:, :usable_frames].reshape(raw_mel.shape[0], n_chunks, chunk_size)
    shuffled_main = main_part[:, perm, :].reshape(raw_mel.shape[0], usable_frames)

    if remainder > 0:
        shuffled_raw = torch.cat([shuffled_main, raw_mel[:, usable_frames:]], dim=1)
    else:
        shuffled_raw = shuffled_main

    # Remap pointers using vectorized indexing
    # Clamp pointers to valid range first
    clamped_pointers = pointers.clamp(0, raw_time - 1)
    new_pointers = mapping[clamped_pointers]

    return shuffled_raw, edit_mel, new_pointers


def random_crop(
    raw_mel: torch.Tensor,
    edit_mel: torch.Tensor,
    pointers: torch.Tensor,
    min_crop_len: int = 500,
    max_crop_len: int = 5000,
    context_pad: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly crop a window from the edit sequence.

    Raw mel is cropped to only include frames that are pointed to,
    plus some context. Pointers are adjusted accordingly.

    Args:
        raw_mel: (n_mels, raw_time)
        edit_mel: (n_mels, edit_time)
        pointers: (edit_time,)
        min_crop_len: minimum crop length
        max_crop_len: maximum crop length
        context_pad: extra frames to include around pointed region

    Returns:
        (cropped_raw_mel, cropped_edit_mel, adjusted_pointers)
    """
    edit_time = edit_mel.shape[1]

    # Clamp crop length to sequence length
    actual_max = min(max_crop_len, edit_time)
    if actual_max <= min_crop_len:
        return raw_mel, edit_mel, pointers

    # Determine crop length
    crop_len = random.randint(min_crop_len, actual_max)

    # Random start position in edit sequence
    max_start = edit_time - crop_len
    if max_start <= 0:
        return raw_mel, edit_mel, pointers

    edit_start = random.randint(0, max_start)
    edit_end = edit_start + crop_len

    # Crop edit mel and pointers
    cropped_edit = edit_mel[:, edit_start:edit_end]
    cropped_pointers = pointers[edit_start:edit_end].clone()

    # Find range of raw frames needed
    min_raw_ptr = cropped_pointers.min().item()
    max_raw_ptr = cropped_pointers.max().item()

    # Add context padding
    raw_start = max(0, min_raw_ptr - context_pad)
    raw_end = min(raw_mel.shape[1], max_raw_ptr + context_pad + 1)

    # Crop raw mel
    cropped_raw = raw_mel[:, raw_start:raw_end]

    # Adjust pointers to new raw indices
    adjusted_pointers = cropped_pointers - raw_start

    # Clamp to valid range
    adjusted_pointers = adjusted_pointers.clamp(0, cropped_raw.shape[1] - 1)

    return cropped_raw, cropped_edit, adjusted_pointers


# =============================================================================
# Augmentor Class
# =============================================================================

class Augmentor:
    """Applies random augmentations to mel spectrograms and pointers."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def __call__(
        self,
        raw_mel: torch.Tensor,
        edit_mel: torch.Tensor,
        pointers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentations to a sample.

        Args:
            raw_mel: (n_mels, raw_time) raw mel spectrogram
            edit_mel: (n_mels, edit_time) edited mel spectrogram
            pointers: (edit_time,) pointer indices

        Returns:
            (augmented_raw_mel, augmented_edit_mel, adjusted_pointers)
        """
        if not self.config.enabled:
            return raw_mel, edit_mel, pointers

        cfg = self.config

        # Make copies to avoid modifying originals
        raw_mel = raw_mel.clone()
        edit_mel = edit_mel.clone()
        pointers = pointers.clone()

        # === Category B: Pointer-aware augmentations (apply first) ===

        # B2: Crop augmentation
        if cfg.crop_enabled and random.random() < cfg.crop_probability:
            raw_mel, edit_mel, pointers = random_crop(
                raw_mel, edit_mel, pointers,
                min_crop_len=cfg.crop_min_len,
                max_crop_len=cfg.crop_max_len
            )

        # B1: Chunk shuffling
        if cfg.chunk_shuffle_enabled and random.random() < cfg.chunk_shuffle_probability:
            raw_mel, edit_mel, pointers = chunk_shuffle(
                raw_mel, edit_mel, pointers,
                chunk_size=cfg.chunk_size
            )

        # === Category A: Safe augmentations ===

        # A1: Additive noise (apply different noise to each)
        if cfg.noise_enabled and random.random() < cfg.noise_probability:
            raw_mel = add_noise(raw_mel, cfg.noise_level_min, cfg.noise_level_max)
            edit_mel = add_noise(edit_mel, cfg.noise_level_min, cfg.noise_level_max)

        # A2: SpecAugment (apply same masks to both)
        if cfg.spec_augment_enabled and random.random() < cfg.spec_augment_probability:
            # Frequency masking
            raw_mel, freq_masks_applied = freq_mask(
                raw_mel, cfg.freq_masks, cfg.freq_mask_width
            )
            edit_mel = apply_freq_mask(edit_mel, freq_masks_applied)

            # Time masking (apply independently since sequences have different lengths)
            raw_mel, _ = time_mask(raw_mel, cfg.time_masks, cfg.time_mask_width)
            edit_mel, _ = time_mask(edit_mel, cfg.time_masks, cfg.time_mask_width)

        # A3: Gain scaling (apply same scale to both)
        if cfg.gain_enabled and random.random() < cfg.gain_probability:
            raw_mel, scale = scale_amplitude(raw_mel, cfg.gain_scale_min, cfg.gain_scale_max)
            edit_mel = edit_mel * scale

        # A4: Channel dropout (apply same mask to both)
        if cfg.channel_dropout_enabled and random.random() < cfg.channel_dropout_probability:
            raw_mel, dropout_mask = channel_dropout(raw_mel, cfg.channel_dropout_rate)
            edit_mel = apply_channel_dropout(edit_mel, dropout_mask)

        return raw_mel, edit_mel, pointers

    def validate_pointers(
        self,
        raw_mel: torch.Tensor,
        pointers: torch.Tensor
    ) -> bool:
        """Check that all pointers are valid indices into raw_mel."""
        raw_len = raw_mel.shape[1]
        return (pointers >= 0).all() and (pointers < raw_len).all()

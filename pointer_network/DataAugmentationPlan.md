# Pointer Network Data Augmentation Plan

## Overview

This document outlines data augmentation strategies for the pointer network training pipeline. The goal is to increase effective dataset size and improve generalization without breaking pointer validity.

**Current state:** 45 training samples, ~15M parameter model
**Target:** 10-100x effective data increase through augmentation

## Implementation Status

✅ **IMPLEMENTED** - All augmentation functions are available in `pointer_network/data/augmentation.py`

To enable augmentation, set `augmentation_enabled: true` in your config JSON or TrainConfig.

Unit tests: `python -m pytest pointer_network/tests/test_augmentation.py -v`

---

## Augmentation Categories

### Category A: Safe Augmentations (No Pointer Adjustment)

These augmentations modify the mel spectrograms but don't change frame counts or alignment. Apply identically to both raw and edited mels.

### Category B: Pointer-Aware Augmentations

These augmentations change sequence structure and require corresponding pointer adjustments.

---

## Category A: Safe Augmentations

### A1. Additive Noise

**Purpose:** Improve robustness to recording quality variations

**Implementation:**
```python
def add_noise(mel: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to mel spectrogram.

    Args:
        mel: (n_mels, time) mel spectrogram
        noise_level: std of noise relative to mel std

    Returns:
        Augmented mel spectrogram
    """
    noise = torch.randn_like(mel) * mel.std() * noise_level
    return mel + noise
```

**Parameters:**
- `noise_level`: 0.05 - 0.15 (randomize within range)
- `probability`: 0.5 (apply to 50% of samples)

**Notes:**
- Apply SAME noise pattern to both raw_mel and edit_mel to maintain consistency
- Or apply DIFFERENT noise to simulate different recording conditions (more realistic)

---

### A2. SpecAugment (Frequency & Time Masking)

**Purpose:** Regularization, forces model to use broader context

**Implementation:**
```python
def freq_mask(mel: torch.Tensor, num_masks: int = 2, max_width: int = 20) -> torch.Tensor:
    """Zero out random frequency bands.

    Args:
        mel: (n_mels, time) mel spectrogram
        num_masks: number of frequency masks to apply
        max_width: maximum width of each mask

    Returns:
        Masked mel spectrogram
    """
    mel = mel.clone()
    n_mels = mel.shape[0]

    for _ in range(num_masks):
        width = random.randint(1, max_width)
        start = random.randint(0, n_mels - width)
        mel[start:start + width, :] = 0

    return mel


def time_mask(mel: torch.Tensor, num_masks: int = 2, max_width: int = 50) -> torch.Tensor:
    """Zero out random time segments.

    Args:
        mel: (n_mels, time) mel spectrogram
        num_masks: number of time masks to apply
        max_width: maximum width of each mask (frames)

    Returns:
        Masked mel spectrogram
    """
    mel = mel.clone()
    time_steps = mel.shape[1]

    for _ in range(num_masks):
        width = random.randint(1, min(max_width, time_steps // 10))
        start = random.randint(0, time_steps - width)
        mel[:, start:start + width] = 0

    return mel
```

**Parameters:**
- Frequency masks: 1-3 masks, width 10-30 mel bins
- Time masks: 1-3 masks, width 20-100 frames
- `probability`: 0.5

**Notes:**
- Apply IDENTICAL masks to both raw_mel and edit_mel
- Store mask indices and apply to both tensors
- Time masking requires care: masked frames in edit still point to (unmasked) raw frames - this is actually fine and teaches robustness

---

### A3. Gain/Amplitude Scaling

**Purpose:** Robustness to volume variations in recordings

**Implementation:**
```python
def scale_amplitude(mel: torch.Tensor, min_scale: float = 0.8, max_scale: float = 1.2) -> torch.Tensor:
    """Scale mel spectrogram amplitude.

    Args:
        mel: (n_mels, time) mel spectrogram
        min_scale: minimum scaling factor
        max_scale: maximum scaling factor

    Returns:
        Scaled mel spectrogram
    """
    scale = random.uniform(min_scale, max_scale)
    return mel * scale
```

**Parameters:**
- Scale range: 0.7 - 1.3
- `probability`: 0.5

**Notes:**
- Can apply SAME scale to both (simulate overall volume change)
- Or DIFFERENT scales (simulate gain staging differences between raw/edited)

---

### A4. Channel Dropout (Mel Band Dropout)

**Purpose:** Prevent over-reliance on specific frequency ranges

**Implementation:**
```python
def channel_dropout(mel: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
    """Randomly zero out entire mel frequency channels.

    Args:
        mel: (n_mels, time) mel spectrogram
        dropout_prob: probability of dropping each channel

    Returns:
        Mel with dropped channels
    """
    mel = mel.clone()
    mask = torch.rand(mel.shape[0]) > dropout_prob
    mel = mel * mask.unsqueeze(1)
    return mel
```

**Parameters:**
- `dropout_prob`: 0.05 - 0.15
- `probability`: 0.3

**Notes:**
- Apply SAME dropout mask to both raw_mel and edit_mel

---

## Category B: Pointer-Aware Augmentations

### B1. Chunk Shuffling

**Purpose:** Teach model to handle reordered recordings (common in real edits)

**Implementation:**
```python
def chunk_shuffle(
    raw_mel: torch.Tensor,
    edit_mel: torch.Tensor,
    pointers: torch.Tensor,
    chunk_size: int = 1000,
    shuffle_prob: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle chunks of raw audio, adjust pointers accordingly.

    The edit_mel and pointers stay in original order, but pointers
    are remapped to point to new positions in shuffled raw_mel.

    Args:
        raw_mel: (n_mels, raw_time) raw mel spectrogram
        edit_mel: (n_mels, edit_time) edited mel spectrogram
        pointers: (edit_time,) indices into raw_mel
        chunk_size: size of chunks to shuffle
        shuffle_prob: probability of shuffling

    Returns:
        (shuffled_raw_mel, edit_mel, remapped_pointers)
    """
    if random.random() > shuffle_prob:
        return raw_mel, edit_mel, pointers

    raw_time = raw_mel.shape[1]
    n_chunks = raw_time // chunk_size

    if n_chunks < 2:
        return raw_mel, edit_mel, pointers

    # Split into chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(raw_mel[:, start:end])

    # Handle remainder
    remainder_start = n_chunks * chunk_size
    if remainder_start < raw_time:
        chunks.append(raw_mel[:, remainder_start:])

    # Shuffle and track new positions
    chunk_indices = list(range(len(chunks)))
    random.shuffle(chunk_indices)

    # Build mapping: old_frame_idx -> new_frame_idx
    old_to_new = {}
    new_pos = 0
    for new_chunk_idx, old_chunk_idx in enumerate(chunk_indices):
        old_start = old_chunk_idx * chunk_size
        chunk_len = chunks[old_chunk_idx].shape[1]
        for offset in range(chunk_len):
            old_to_new[old_start + offset] = new_pos + offset
        new_pos += chunk_len

    # Reassemble raw_mel
    shuffled_raw = torch.cat([chunks[i] for i in chunk_indices], dim=1)

    # Remap pointers
    new_pointers = torch.tensor([old_to_new.get(p.item(), p.item()) for p in pointers])

    return shuffled_raw, edit_mel, new_pointers
```

**Parameters:**
- `chunk_size`: 500 - 2000 frames (~5-20 seconds at typical hop)
- `shuffle_prob`: 0.3

**Notes:**
- Only shuffle raw_mel, keep edit_mel order intact
- Pointers are remapped to point to new locations
- Simulates out-of-order recording sessions

---

### B2. Crop Augmentation

**Purpose:** Train on varied sequence lengths, reduce memory, add variety

**Implementation:**
```python
def random_crop(
    raw_mel: torch.Tensor,
    edit_mel: torch.Tensor,
    pointers: torch.Tensor,
    min_crop_len: int = 500,
    max_crop_len: int = 5000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly crop a window from the edit sequence.

    Raw mel is cropped to only include frames that are pointed to,
    plus some context. Pointers are adjusted accordingly.

    Args:
        raw_mel: (n_mels, raw_time)
        edit_mel: (n_mels, edit_time)
        pointers: (edit_time,)
        min_crop_len: minimum crop length
        max_crop_len: maximum crop length

    Returns:
        (cropped_raw_mel, cropped_edit_mel, adjusted_pointers)
    """
    edit_time = edit_mel.shape[1]

    # Determine crop length
    crop_len = random.randint(min_crop_len, min(max_crop_len, edit_time))

    # Random start position in edit sequence
    max_start = edit_time - crop_len
    if max_start <= 0:
        return raw_mel, edit_mel, pointers

    edit_start = random.randint(0, max_start)
    edit_end = edit_start + crop_len

    # Crop edit mel and pointers
    cropped_edit = edit_mel[:, edit_start:edit_end]
    cropped_pointers = pointers[edit_start:edit_end]

    # Find range of raw frames needed
    min_raw_ptr = cropped_pointers.min().item()
    max_raw_ptr = cropped_pointers.max().item()

    # Add context padding
    context_pad = 100
    raw_start = max(0, min_raw_ptr - context_pad)
    raw_end = min(raw_mel.shape[1], max_raw_ptr + context_pad + 1)

    # Crop raw mel
    cropped_raw = raw_mel[:, raw_start:raw_end]

    # Adjust pointers to new raw indices
    adjusted_pointers = cropped_pointers - raw_start

    # Clamp to valid range
    adjusted_pointers = adjusted_pointers.clamp(0, cropped_raw.shape[1] - 1)

    return cropped_raw, cropped_edit, adjusted_pointers
```

**Parameters:**
- `min_crop_len`: 500 frames
- `max_crop_len`: 5000 frames (or percentage of sequence)
- `probability`: 0.5

**Notes:**
- Reduces sequence length, good for memory
- Creates more training variety from same sample
- Must include enough raw context for pointers to remain valid

---

### B3. Sample Concatenation

**Purpose:** Create longer sequences, increase effective dataset size

**Implementation:**
```python
def concatenate_samples(
    sample1: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sample2: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate two samples into one longer sample.

    Args:
        sample1: (raw_mel1, edit_mel1, pointers1)
        sample2: (raw_mel2, edit_mel2, pointers2)

    Returns:
        (combined_raw, combined_edit, combined_pointers)
    """
    raw1, edit1, ptr1 = sample1
    raw2, edit2, ptr2 = sample2

    raw1_len = raw1.shape[1]

    # Concatenate raw mels
    combined_raw = torch.cat([raw1, raw2], dim=1)

    # Concatenate edit mels
    combined_edit = torch.cat([edit1, edit2], dim=1)

    # Adjust pointers for second sample (offset by raw1 length)
    ptr2_adjusted = ptr2 + raw1_len

    # Concatenate pointers
    combined_pointers = torch.cat([ptr1, ptr2_adjusted])

    return combined_raw, combined_edit, combined_pointers
```

**Parameters:**
- `probability`: 0.2 (apply to 20% of batches)
- May need to limit to avoid OOM on very long sequences

**Notes:**
- Requires access to multiple samples (implement in collate_fn or dataset)
- Doubles sequence length - watch memory usage
- Creates novel combinations the model hasn't seen
- Consider concatenating samples with similar characteristics (tempo, genre)

---

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Create `augmentation.py` module** in `pointer_network/data/`
   - Implement all individual augmentation functions
   - Create `AugmentationConfig` dataclass for parameters
   - Create `Augmentor` class that applies random augmentations

2. **Integrate with `PointerDataset`**
   - Add `augment: bool` parameter to dataset
   - Add `augmentation_config: AugmentationConfig` parameter
   - Apply augmentations in `__getitem__`

3. **Add config support**
   - Add augmentation settings to JSON configs
   - Allow per-augmentation probability tuning

### Phase 2: Implementation Order

Implement in order of complexity/risk:

1. **Additive noise** - simplest, low risk
2. **Gain scaling** - simple, low risk
3. **Channel dropout** - simple, low risk
4. **SpecAugment (freq/time mask)** - medium complexity
5. **Crop augmentation** - medium complexity, pointer adjustment
6. **Chunk shuffling** - higher complexity, pointer remapping
7. **Concatenation** - highest complexity, requires dataset changes

### Phase 3: Validation

1. **Unit tests** for each augmentation
   - Verify pointer validity after augmentation
   - Check tensor shapes are correct
   - Test edge cases (empty, very short sequences)

2. **Visual inspection**
   - Plot augmented vs original mels
   - Verify augmentations look reasonable

3. **Training validation**
   - Compare training curves with/without augmentation
   - Monitor for training instability
   - Check validation metrics

---

## Configuration Example

```json
{
    "augmentation": {
        "enabled": true,
        "noise": {
            "enabled": true,
            "probability": 0.5,
            "level_min": 0.05,
            "level_max": 0.15
        },
        "gain": {
            "enabled": true,
            "probability": 0.5,
            "scale_min": 0.7,
            "scale_max": 1.3
        },
        "channel_dropout": {
            "enabled": true,
            "probability": 0.3,
            "dropout_rate": 0.1
        },
        "spec_augment": {
            "enabled": true,
            "probability": 0.5,
            "freq_masks": 2,
            "freq_width": 20,
            "time_masks": 2,
            "time_width": 50
        },
        "crop": {
            "enabled": true,
            "probability": 0.5,
            "min_len": 500,
            "max_len": 5000
        },
        "chunk_shuffle": {
            "enabled": true,
            "probability": 0.3,
            "chunk_size": 1000
        },
        "concatenation": {
            "enabled": false,
            "probability": 0.2
        }
    }
}
```

---

## Expected Impact

| Augmentation | Effective Data Multiplier | Training Stability |
|--------------|---------------------------|-------------------|
| Noise | 2x | High |
| Gain | 1.5x | High |
| Channel dropout | 1.5x | High |
| SpecAugment | 2x | High |
| Crop | 3-5x | Medium |
| Chunk shuffle | 2-3x | Medium |
| Concatenation | 2x | Low (memory risk) |

**Combined estimate:** 10-20x effective dataset size increase

---

## Risks and Mitigations

1. **Training instability**
   - Mitigation: Start with low augmentation probabilities, increase gradually
   - Mitigation: Disable augmentation for first few epochs (warmup)

2. **Pointer validity corruption**
   - Mitigation: Unit tests with assertion checks
   - Mitigation: Validate pointers are in-bounds after each augmentation

3. **Memory issues (concatenation)**
   - Mitigation: Limit max concatenated length
   - Mitigation: Use gradient checkpointing

4. **Augmentation too strong**
   - Mitigation: Monitor validation metrics
   - Mitigation: Use augmentation probability scheduling (decrease over training)

---

## Files to Create/Modify

### New Files
- `pointer_network/data/augmentation.py` - Augmentation implementations
- `pointer_network/data/augmentation_config.py` - Config dataclasses
- `pointer_network/tests/test_augmentation.py` - Unit tests

### Modified Files
- `pointer_network/data/dataset.py` - Integrate augmentation
- `pointer_network/config.py` - Add augmentation config fields
- `pointer_network/configs/full.json` - Add augmentation settings
- `pointer_network/configs/mini.json` - Add augmentation settings

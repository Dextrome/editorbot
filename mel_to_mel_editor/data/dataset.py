"""Dataset for paired mel spectrograms."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List

from ..config import TrainConfig


class PairedMelDataset(Dataset):
    """Dataset of paired (raw, edited) mel spectrograms.

    Expects cache directory with:
        features/{pair_id}_raw.npz  -> {'mel': (T, n_mels)}
        features/{pair_id}_edit.npz -> {'mel': (T, n_mels)}
    """

    def __init__(
        self,
        cache_dir: str,
        config: TrainConfig,
        split: str = 'train',
    ):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Find all pairs
        self.pairs = self._find_pairs()

        # Train/val split
        n_val = max(1, int(len(self.pairs) * config.val_split))
        if split == 'train':
            self.pairs = self.pairs[n_val:]
        else:
            self.pairs = self.pairs[:n_val]

        print(f"PairedMelDataset ({split}): {len(self.pairs)} pairs")

    def _find_pairs(self) -> List[str]:
        """Find all raw/edit pairs in cache."""
        features_dir = self.cache_dir / 'features'
        if not features_dir.exists():
            raise ValueError(f"Features directory not found: {features_dir}")

        # Find all raw files
        raw_files = list(features_dir.glob('*_raw.npz'))

        pairs = []
        for raw_path in raw_files:
            pair_id = raw_path.stem.replace('_raw', '')
            edit_path = features_dir / f'{pair_id}_edit.npz'

            if edit_path.exists():
                pairs.append(pair_id)

        return sorted(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_id = self.pairs[idx]
        features_dir = self.cache_dir / 'features'

        # Load mels
        raw_data = np.load(features_dir / f'{pair_id}_raw.npz')
        edit_data = np.load(features_dir / f'{pair_id}_edit.npz')

        raw_mel = raw_data['mel']  # (T_raw, n_mels)
        edit_mel = edit_data['mel']  # (T_edit, n_mels)

        # Align lengths (use shorter one)
        min_len = min(len(raw_mel), len(edit_mel))
        raw_mel = raw_mel[:min_len]
        edit_mel = edit_mel[:min_len]

        # Random crop if too long
        if len(raw_mel) > self.max_seq_len:
            start = random.randint(0, len(raw_mel) - self.max_seq_len)
            raw_mel = raw_mel[start:start + self.max_seq_len]
            edit_mel = edit_mel[start:start + self.max_seq_len]

        # Data augmentation (training only)
        if self.split == 'train' and self.config.use_augmentation:
            # Add small noise
            if self.config.augment_noise > 0:
                noise = np.random.randn(*raw_mel.shape) * self.config.augment_noise
                raw_mel = np.clip(raw_mel + noise, 0, 1)

        # Convert to tensors
        raw_mel = torch.from_numpy(raw_mel).float()
        edit_mel = torch.from_numpy(edit_mel).float()

        return {
            'raw_mel': raw_mel,
            'edit_mel': edit_mel,
            'pair_id': pair_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max length
    max_len = max(b['raw_mel'].size(0) for b in batch)

    raw_mels = []
    edit_mels = []
    masks = []

    for b in batch:
        T = b['raw_mel'].size(0)
        pad_len = max_len - T

        if pad_len > 0:
            raw_mel = F.pad(b['raw_mel'], (0, 0, 0, pad_len))
            edit_mel = F.pad(b['edit_mel'], (0, 0, 0, pad_len))
            mask = torch.cat([torch.ones(T), torch.zeros(pad_len)]).bool()
        else:
            raw_mel = b['raw_mel']
            edit_mel = b['edit_mel']
            mask = torch.ones(T).bool()

        raw_mels.append(raw_mel)
        edit_mels.append(edit_mel)
        masks.append(mask)

    return {
        'raw_mel': torch.stack(raw_mels),
        'edit_mel': torch.stack(edit_mels),
        'mask': torch.stack(masks),
    }


# Need F for padding
import torch.nn.functional as F

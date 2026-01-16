"""Dataset for paired (raw, edited) mel spectrograms with edit labels."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from ..config import Phase1Config, EditLabel


class MelAugmentation:
    """Data augmentation for mel spectrograms.

    Applies augmentations consistently to both raw and edit mels.
    """

    def __init__(
        self,
        freq_mask_prob: float = 0.5,
        freq_mask_max: int = 20,
        time_mask_prob: float = 0.5,
        time_mask_max: int = 50,
        gain_prob: float = 0.5,
        gain_range: Tuple[float, float] = (0.8, 1.2),
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        time_stretch_prob: float = 0.3,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
    ):
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_max = freq_mask_max
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max
        self.gain_prob = gain_prob
        self.gain_range = gain_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range

    def freq_mask(self, mel: np.ndarray) -> np.ndarray:
        """Apply frequency masking (SpecAugment)."""
        if random.random() > self.freq_mask_prob:
            return mel

        n_mels = mel.shape[1]
        mask_width = random.randint(1, self.freq_mask_max)
        mask_start = random.randint(0, n_mels - mask_width)

        mel = mel.copy()
        mel[:, mask_start:mask_start + mask_width] = 0
        return mel

    def time_mask(self, mel: np.ndarray) -> np.ndarray:
        """Apply time masking (SpecAugment)."""
        if random.random() > self.time_mask_prob:
            return mel

        T = mel.shape[0]
        mask_width = random.randint(1, min(self.time_mask_max, T // 4))
        mask_start = random.randint(0, T - mask_width)

        mel = mel.copy()
        mel[mask_start:mask_start + mask_width, :] = 0
        return mel

    def apply_gain(self, mel: np.ndarray, gain: float) -> np.ndarray:
        """Apply gain scaling."""
        return mel * gain

    def add_noise(self, mel: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        if random.random() > self.noise_prob:
            return mel

        noise = np.random.normal(0, self.noise_std, mel.shape).astype(np.float32)
        return mel + noise

    def time_stretch(
        self,
        raw_mel: np.ndarray,
        edit_mel: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple time stretch via interpolation."""
        if random.random() > self.time_stretch_prob:
            return raw_mel, edit_mel, labels

        factor = random.uniform(*self.time_stretch_range)
        T_new = int(len(raw_mel) * factor)

        if T_new < 10:  # Too short
            return raw_mel, edit_mel, labels

        # Interpolate mels
        from scipy.ndimage import zoom
        raw_mel = zoom(raw_mel, (factor, 1), order=1)
        edit_mel = zoom(edit_mel, (factor, 1), order=1)

        # Stretch labels (nearest neighbor)
        labels = zoom(labels.astype(float), factor, order=0).astype(np.int64)

        return raw_mel, edit_mel, labels

    def __call__(
        self,
        raw_mel: np.ndarray,
        edit_mel: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply all augmentations."""
        # Time stretch (affects both equally)
        raw_mel, edit_mel, labels = self.time_stretch(raw_mel, edit_mel, labels)

        # Shared gain (same for both)
        if random.random() < self.gain_prob:
            gain = random.uniform(*self.gain_range)
            raw_mel = self.apply_gain(raw_mel, gain)
            edit_mel = self.apply_gain(edit_mel, gain)

        # SpecAugment on raw only (edit is target, keep clean)
        raw_mel = self.freq_mask(raw_mel)
        raw_mel = self.time_mask(raw_mel)

        # Noise on raw only
        raw_mel = self.add_noise(raw_mel)

        return raw_mel, edit_mel, labels


class PairedMelDataset(Dataset):
    """Dataset of paired (raw, edited) mel spectrograms with edit labels.

    Expected cache structure:
        cache_dir/
            features/
                {pair_id}_raw.npz  -> {'mel': (T, n_mels), 'beat_times': (n_beats,)}
                {pair_id}_edit.npz -> {'mel': (T', n_mels), 'beat_times': (n_beats',)}
            labels/
                {pair_id}_labels.npy -> (T,) int array of edit labels

    If edit labels file doesn't exist, you can generate them using
    EditLabelInferencer from preprocessing.py.
    """

    def __init__(
        self,
        cache_dir: str,
        config: Phase1Config,
        split: str = 'train',  # 'train' or 'val'
        max_samples: Optional[int] = None,
        use_augmentation: bool = True,
    ):
        """
        Args:
            cache_dir: Directory containing features/ and labels/ subdirs
            config: Phase1Config with audio settings
            split: 'train' or 'val'
            max_samples: Maximum number of samples to use (for debugging)
            use_augmentation: Whether to apply data augmentation (train only)
        """
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Data augmentation (only for training)
        self.augmentation = None
        if use_augmentation and split == 'train':
            self.augmentation = MelAugmentation()
            print("Data augmentation enabled for training")

        # Find all pair IDs
        self.pair_ids = self._find_pairs()

        # Split into train/val
        random.seed(42)
        shuffled = self.pair_ids.copy()
        random.shuffle(shuffled)

        n_val = int(len(shuffled) * config.val_split)
        if split == 'val':
            self.pair_ids = shuffled[:n_val]
        else:
            self.pair_ids = shuffled[n_val:]

        if max_samples is not None:
            self.pair_ids = self.pair_ids[:max_samples]

        print(f"PairedMelDataset ({split}): {len(self.pair_ids)} pairs")

    def _find_pairs(self) -> List[str]:
        """Find all valid pair IDs in the cache directory."""
        features_dir = self.cache_dir / 'features'

        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        # Find all raw mel files
        pair_ids = []
        for f in features_dir.glob('*_raw.npz'):
            pair_id = f.stem.replace('_raw', '')

            # Check that edit file also exists
            edit_file = features_dir / f'{pair_id}_edit.npz'
            if edit_file.exists():
                pair_ids.append(pair_id)

        return sorted(pair_ids)

    def _load_mel(self, path: Path) -> np.ndarray:
        """Load mel spectrogram from npz file."""
        data = np.load(path)
        mel = data['mel']  # (T, n_mels)
        return mel.astype(np.float32)

    def _load_labels(self, pair_id: str) -> Optional[np.ndarray]:
        """Load edit labels if available, else return None."""
        labels_file = self.cache_dir / 'labels' / f'{pair_id}_labels.npy'
        if labels_file.exists():
            return np.load(labels_file).astype(np.int64)
        return None

    def _align_sequences(
        self,
        raw_mel: np.ndarray,
        edit_mel: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align raw and edited mel sequences.

        The edited mel may be shorter (due to cuts) or longer (due to loops).
        We need to align them for supervised training.
        """
        T_raw = len(raw_mel)
        T_edit = len(edit_mel)

        # If no labels provided, create default KEEP labels
        if labels is None:
            labels = np.ones(T_raw, dtype=np.int64)  # All KEEP

        # Simple alignment: truncate to minimum length
        # More sophisticated: use DTW or label-based alignment
        T_min = min(T_raw, T_edit, len(labels))

        raw_mel = raw_mel[:T_min]
        edit_mel = edit_mel[:T_min]
        labels = labels[:T_min]

        return raw_mel, edit_mel, labels

    def _random_crop(
        self,
        raw_mel: np.ndarray,
        edit_mel: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random crop to max_seq_len if needed."""
        T = len(raw_mel)

        if T <= self.max_seq_len:
            return raw_mel, edit_mel, labels

        # Random start position
        start = random.randint(0, T - self.max_seq_len)
        end = start + self.max_seq_len

        return (
            raw_mel[start:end],
            edit_mel[start:end],
            labels[start:end],
        )

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_id = self.pair_ids[idx]

        # Load mel spectrograms
        raw_mel = self._load_mel(self.cache_dir / 'features' / f'{pair_id}_raw.npz')
        edit_mel = self._load_mel(self.cache_dir / 'features' / f'{pair_id}_edit.npz')

        # Load edit labels (if available)
        labels = self._load_labels(pair_id)

        # Align sequences
        raw_mel, edit_mel, labels = self._align_sequences(raw_mel, edit_mel, labels)

        # Apply data augmentation (training only)
        if self.augmentation is not None:
            raw_mel, edit_mel, labels = self.augmentation(raw_mel, edit_mel, labels)

        # Crop to max_seq_len
        if self.split == 'train':
            # Random crop for training
            raw_mel, edit_mel, labels = self._random_crop(raw_mel, edit_mel, labels)
        elif len(raw_mel) > self.max_seq_len:
            # Center crop for validation
            start = (len(raw_mel) - self.max_seq_len) // 2
            end = start + self.max_seq_len
            raw_mel = raw_mel[start:end]
            edit_mel = edit_mel[start:end]
            labels = labels[start:end]

        # Convert to tensors
        return {
            'pair_id': pair_id,
            'raw_mel': torch.from_numpy(raw_mel.copy()),      # (T, n_mels)
            'edit_mel': torch.from_numpy(edit_mel.copy()),    # (T, n_mels)
            'edit_labels': torch.from_numpy(labels.copy()),   # (T,)
            'length': len(raw_mel),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Pads sequences to the same length within a batch.
    """
    # Find max length in batch
    max_len = max(item['length'] for item in batch)
    n_mels = batch[0]['raw_mel'].size(-1)

    B = len(batch)

    # Initialize padded tensors
    raw_mel = torch.zeros(B, max_len, n_mels)
    edit_mel = torch.zeros(B, max_len, n_mels)
    edit_labels = torch.full((B, max_len), EditLabel.PAD, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    pair_ids = []

    # Fill in data
    for i, item in enumerate(batch):
        T = item['length']
        raw_mel[i, :T] = item['raw_mel']
        edit_mel[i, :T] = item['edit_mel']
        edit_labels[i, :T] = item['edit_labels']
        mask[i, :T] = True
        lengths[i] = T
        pair_ids.append(item['pair_id'])

    return {
        'pair_ids': pair_ids,
        'raw_mel': raw_mel,
        'edit_mel': edit_mel,
        'edit_labels': edit_labels,
        'mask': mask,
        'lengths': lengths,
    }


def create_dataloader(
    cache_dir: str,
    config: Phase1Config,
    split: str = 'train',
    shuffle: bool = True,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for the paired mel dataset."""
    dataset = PairedMelDataset(cache_dir, config, split=split)

    if num_workers is None:
        num_workers = config.num_workers

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=split == 'train',
    )

    return loader


class InMemoryDataset(Dataset):
    """Dataset that keeps all data in memory for faster training.

    Use this for small datasets that fit in RAM.
    """

    def __init__(
        self,
        cache_dir: str,
        config: Phase1Config,
        split: str = 'train',
    ):
        self.config = config
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Load all data into memory
        base_dataset = PairedMelDataset(cache_dir, config, split=split)

        self.data = []
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            self.data.append({
                'raw_mel': item['raw_mel'].numpy(),
                'edit_mel': item['edit_mel'].numpy(),
                'edit_labels': item['edit_labels'].numpy(),
                'pair_id': item['pair_id'],
            })

        print(f"InMemoryDataset ({split}): Loaded {len(self.data)} pairs into memory")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        raw_mel = item['raw_mel']
        edit_mel = item['edit_mel']
        labels = item['edit_labels']

        # Random crop for training
        if self.split == 'train' and len(raw_mel) > self.max_seq_len:
            start = random.randint(0, len(raw_mel) - self.max_seq_len)
            end = start + self.max_seq_len
            raw_mel = raw_mel[start:end]
            edit_mel = edit_mel[start:end]
            labels = labels[start:end]

        return {
            'pair_id': item['pair_id'],
            'raw_mel': torch.from_numpy(raw_mel.copy()),
            'edit_mel': torch.from_numpy(edit_mel.copy()),
            'edit_labels': torch.from_numpy(labels.copy()),
            'length': len(raw_mel),
        }

"""Dataset for audio segments - loads from both raw and edited audio."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional

from ..config import TrainConfig


class AudioSegmentDatasetFast(Dataset):
    """Pre-loads all data into memory for fast training."""

    def __init__(
        self,
        cache_dir: str,
        config: TrainConfig,
        split: str = 'train',
    ):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.split = split
        self.segment_frames = config.model.segment_frames

        # Find all pairs
        pairs = self._find_pairs()

        # Train/val split
        n_val = max(1, int(len(pairs) * config.val_split))
        if split == 'train':
            pairs = pairs[n_val:]
        else:
            pairs = pairs[:n_val]

        # Pre-load ALL data into memory
        print(f"Pre-loading {len(pairs)} pairs into memory...")
        self.raw_mels = []
        self.edit_mels = []

        features_dir = self.cache_dir / 'features'
        for pair_id in pairs:
            raw_data = np.load(features_dir / f'{pair_id}_raw.npz')
            edit_data = np.load(features_dir / f'{pair_id}_edit.npz')
            self.raw_mels.append(raw_data['mel'].astype(np.float32))
            self.edit_mels.append(edit_data['mel'].astype(np.float32))

        # Count total segments
        self.raw_total_frames = sum(len(m) for m in self.raw_mels)
        self.edit_total_frames = sum(len(m) for m in self.edit_mels)

        # Number of samples per epoch
        self.n_samples = config.segments_per_track * len(pairs)

        print(f"AudioSegmentDatasetFast ({split}):")
        print(f"  Pairs loaded: {len(pairs)}")
        print(f"  Raw frames: {self.raw_total_frames:,}")
        print(f"  Edit frames: {self.edit_total_frames:,}")
        print(f"  Samples per epoch: {self.n_samples}")

    def _find_pairs(self) -> List[str]:
        features_dir = self.cache_dir / 'features'
        raw_files = list(features_dir.glob('*_raw.npz'))
        pairs = []
        for raw_path in raw_files:
            pair_id = raw_path.stem.replace('_raw', '')
            edit_path = features_dir / f'{pair_id}_edit.npz'
            if edit_path.exists():
                pairs.append(pair_id)
        return sorted(pairs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # Random raw segment
        raw_idx = random.randint(0, len(self.raw_mels) - 1)
        raw_mel = self.raw_mels[raw_idx]
        if len(raw_mel) > self.segment_frames:
            start = random.randint(0, len(raw_mel) - self.segment_frames)
            raw_segment = raw_mel[start:start + self.segment_frames]
        else:
            raw_segment = raw_mel

        # Random edit segment
        edit_idx = random.randint(0, len(self.edit_mels) - 1)
        edit_mel = self.edit_mels[edit_idx]
        if len(edit_mel) > self.segment_frames:
            start = random.randint(0, len(edit_mel) - self.segment_frames)
            edit_segment = edit_mel[start:start + self.segment_frames]
        else:
            edit_segment = edit_mel

        return {
            'raw_mel': torch.from_numpy(raw_segment),
            'edit_mel': torch.from_numpy(edit_segment),
        }


class AudioSegmentDataset(Dataset):
    """Dataset of audio segments from raw and edited audio.

    For FaceSwap-style training:
    - Extract random segments from raw audio
    - Extract random segments from edited audio
    - Train encoder to learn shared representation
    - Train separate decoders for each style
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
        self.segment_frames = config.model.segment_frames
        self.segments_per_track = config.segments_per_track

        # Find all pairs
        self.pairs = self._find_pairs()

        # Train/val split
        n_val = max(1, int(len(self.pairs) * config.val_split))
        if split == 'train':
            self.pairs = self.pairs[n_val:]
        else:
            self.pairs = self.pairs[:n_val]

        # Pre-compute segment positions for each track
        self.raw_segments = []
        self.edited_segments = []
        self._precompute_segments()

        print(f"AudioSegmentDataset ({split}):")
        print(f"  Pairs: {len(self.pairs)}")
        print(f"  Raw segments: {len(self.raw_segments)}")
        print(f"  Edited segments: {len(self.edited_segments)}")

    def _find_pairs(self) -> List[str]:
        """Find all raw/edit pairs in cache."""
        features_dir = self.cache_dir / 'features'
        if not features_dir.exists():
            raise ValueError(f"Features directory not found: {features_dir}")

        raw_files = list(features_dir.glob('*_raw.npz'))
        pairs = []

        for raw_path in raw_files:
            pair_id = raw_path.stem.replace('_raw', '')
            edit_path = features_dir / f'{pair_id}_edit.npz'
            if edit_path.exists():
                pairs.append(pair_id)

        return sorted(pairs)

    def _precompute_segments(self):
        """Pre-compute random segment positions."""
        features_dir = self.cache_dir / 'features'

        for pair_id in self.pairs:
            # Load mel lengths
            raw_data = np.load(features_dir / f'{pair_id}_raw.npz')
            edit_data = np.load(features_dir / f'{pair_id}_edit.npz')

            raw_len = len(raw_data['mel'])
            edit_len = len(edit_data['mel'])

            # Generate random segment positions
            # Raw segments
            if raw_len > self.segment_frames:
                for _ in range(self.segments_per_track):
                    start = random.randint(0, raw_len - self.segment_frames)
                    self.raw_segments.append((pair_id, start, 'raw'))

            # Edited segments
            if edit_len > self.segment_frames:
                for _ in range(self.segments_per_track):
                    start = random.randint(0, edit_len - self.segment_frames)
                    self.edited_segments.append((pair_id, start, 'edit'))

    def __len__(self) -> int:
        # Return combined length
        return max(len(self.raw_segments), len(self.edited_segments))

    def _load_segment(self, pair_id: str, start: int, source: str) -> np.ndarray:
        """Load a segment from cache."""
        features_dir = self.cache_dir / 'features'
        suffix = '_raw.npz' if source == 'raw' else '_edit.npz'
        data = np.load(features_dir / f'{pair_id}{suffix}')
        mel = data['mel']
        segment = mel[start:start + self.segment_frames]
        return segment

    def __getitem__(self, idx: int) -> dict:
        # Get both raw and edited segment for this index
        raw_idx = idx % len(self.raw_segments)
        edit_idx = idx % len(self.edited_segments)

        raw_pair_id, raw_start, _ = self.raw_segments[raw_idx]
        edit_pair_id, edit_start, _ = self.edited_segments[edit_idx]

        raw_segment = self._load_segment(raw_pair_id, raw_start, 'raw')
        edit_segment = self._load_segment(edit_pair_id, edit_start, 'edit')

        # Convert to tensors
        raw_segment = torch.from_numpy(raw_segment).float()
        edit_segment = torch.from_numpy(edit_segment).float()

        return {
            'raw_mel': raw_segment,
            'edit_mel': edit_segment,
        }


class EditedOnlyDataset(Dataset):
    """Dataset of only edited audio segments.

    For single-autoencoder approach where we only learn "good" audio.
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
        self.segment_frames = config.model.segment_frames
        self.segments_per_track = config.segments_per_track * 2  # More since only edited

        # Find all edited files
        self.edited_files = self._find_edited()

        # Train/val split
        n_val = max(1, int(len(self.edited_files) * config.val_split))
        if split == 'train':
            self.edited_files = self.edited_files[n_val:]
        else:
            self.edited_files = self.edited_files[:n_val]

        # Pre-compute segments
        self.segments = []
        self._precompute_segments()

        print(f"EditedOnlyDataset ({split}): {len(self.segments)} segments from {len(self.edited_files)} files")

    def _find_edited(self) -> List[str]:
        """Find all edited files."""
        features_dir = self.cache_dir / 'features'
        return sorted([f.stem.replace('_edit', '') for f in features_dir.glob('*_edit.npz')])

    def _precompute_segments(self):
        """Pre-compute segment positions."""
        features_dir = self.cache_dir / 'features'

        for file_id in self.edited_files:
            data = np.load(features_dir / f'{file_id}_edit.npz')
            mel_len = len(data['mel'])

            if mel_len > self.segment_frames:
                for _ in range(self.segments_per_track):
                    start = random.randint(0, mel_len - self.segment_frames)
                    self.segments.append((file_id, start))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict:
        file_id, start = self.segments[idx]
        features_dir = self.cache_dir / 'features'

        data = np.load(features_dir / f'{file_id}_edit.npz')
        segment = data['mel'][start:start + self.segment_frames]

        return {
            'mel': torch.from_numpy(segment).float(),
        }

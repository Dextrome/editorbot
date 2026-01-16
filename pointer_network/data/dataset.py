"""Dataset for pointer network training."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
import random


class PointerDataset(Dataset):
    """Dataset that loads mel spectrograms and pointer sequences.

    Each sample contains:
    - raw_mel: mel spectrogram of raw audio
    - target_pointers: sequence of frame indices (edited -> raw mapping)
    """

    def __init__(
        self,
        cache_dir: str,
        pointer_dir: str,
        max_raw_frames: int = 65536,
        max_edit_frames: int = 65536,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 64,
    ):
        """
        Args:
            cache_dir: directory containing cached mel spectrograms
            pointer_dir: directory containing pointer sequences (.npy files)
            max_raw_frames: maximum raw audio frames to use
            max_edit_frames: maximum edited frames (output length)
            chunk_size: if set, split long sequences into chunks
            chunk_overlap: overlap between chunks
        """
        self.cache_dir = Path(cache_dir)
        self.pointer_dir = Path(pointer_dir)
        self.max_raw_frames = max_raw_frames
        self.max_edit_frames = max_edit_frames
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Find all pointer files
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples")

    def _find_samples(self) -> List[Dict]:
        """Find all valid training samples."""
        samples = []

        for pointer_file in self.pointer_dir.glob("*_pointers.npy"):
            base_name = pointer_file.stem.replace("_pointers", "")

            # Find corresponding info file
            info_file = self.pointer_dir / f"{base_name}_info.json"
            if not info_file.exists():
                continue

            # Load info to get paths
            with open(info_file) as f:
                info = json.load(f)

            # Find raw mel file in cache
            raw_path = Path(info['raw_path'])
            raw_mel_file = self.cache_dir / f"{raw_path.stem}_mel.npy"

            if not raw_mel_file.exists():
                # Try without _raw suffix
                raw_mel_file = self.cache_dir / f"{base_name}_raw_mel.npy"

            if not raw_mel_file.exists():
                print(f"Warning: mel file not found for {base_name}")
                continue

            samples.append({
                'name': base_name,
                'pointer_file': pointer_file,
                'info_file': info_file,
                'raw_mel_file': raw_mel_file,
                'raw_frames': info['raw_frames'],
                'edit_frames': info['edit_frames'],
            })

        return samples

    def __len__(self) -> int:
        if self.chunk_size is None:
            return len(self.samples)
        else:
            # Count total chunks across all samples
            total = 0
            for sample in self.samples:
                n_chunks = max(1, (sample['edit_frames'] - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap))
                total += n_chunks
            return total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.chunk_size is None:
            return self._get_full_sample(idx)
        else:
            return self._get_chunk(idx)

    def _get_full_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a full sample (may be truncated if too long)."""
        sample = self.samples[idx]

        # Load raw mel
        raw_mel = np.load(sample['raw_mel_file'])  # (n_mels, time) or (time, n_mels)
        if raw_mel.shape[0] != 128:  # Assume n_mels=128
            raw_mel = raw_mel.T  # Transpose if needed

        # Load pointers
        pointers = np.load(sample['pointer_file'])

        # Truncate if needed
        if raw_mel.shape[1] > self.max_raw_frames:
            raw_mel = raw_mel[:, :self.max_raw_frames]
            # Also need to filter pointers that point beyond the truncation
            valid_mask = pointers < self.max_raw_frames
            pointers = np.where(valid_mask, pointers, -1)

        if len(pointers) > self.max_edit_frames:
            pointers = pointers[:self.max_edit_frames]

        # Handle invalid pointers (-1)
        pointers = np.clip(pointers, 0, raw_mel.shape[1] - 1)

        return {
            'raw_mel': torch.from_numpy(raw_mel).float(),
            'target_pointers': torch.from_numpy(pointers).long(),
            'name': sample['name'],
        }

    def _get_chunk(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a chunk from a sample."""
        # Find which sample and chunk this index corresponds to
        chunk_idx = idx
        for sample in self.samples:
            n_chunks = max(1, (sample['edit_frames'] - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap))
            if chunk_idx < n_chunks:
                return self._get_chunk_from_sample(sample, chunk_idx)
            chunk_idx -= n_chunks

        # Fallback (shouldn't happen)
        return self._get_full_sample(0)

    def _get_chunk_from_sample(self, sample: Dict, chunk_idx: int) -> Dict[str, torch.Tensor]:
        """Get a specific chunk from a sample."""
        # Load raw mel
        raw_mel = np.load(sample['raw_mel_file'])
        if raw_mel.shape[0] != 128:
            raw_mel = raw_mel.T

        # Load pointers
        pointers = np.load(sample['pointer_file'])

        # Calculate chunk boundaries
        stride = self.chunk_size - self.chunk_overlap
        start = chunk_idx * stride
        end = min(start + self.chunk_size, len(pointers))

        # Get chunk of pointers
        chunk_pointers = pointers[start:end]

        # Find the range of raw frames needed for this chunk
        valid_pointers = chunk_pointers[chunk_pointers >= 0]
        if len(valid_pointers) > 0:
            min_ptr = max(0, valid_pointers.min() - 100)  # Add some context
            max_ptr = min(raw_mel.shape[1], valid_pointers.max() + 100)
        else:
            min_ptr, max_ptr = 0, min(raw_mel.shape[1], self.max_raw_frames)

        # Truncate raw mel to relevant range
        raw_mel_chunk = raw_mel[:, min_ptr:max_ptr]

        # Adjust pointers to be relative to chunk start
        chunk_pointers = chunk_pointers - min_ptr
        chunk_pointers = np.clip(chunk_pointers, 0, raw_mel_chunk.shape[1] - 1)

        return {
            'raw_mel': torch.from_numpy(raw_mel_chunk).float(),
            'target_pointers': torch.from_numpy(chunk_pointers).long(),
            'name': f"{sample['name']}_chunk{chunk_idx}",
            'raw_offset': min_ptr,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max lengths
    max_raw_len = max(item['raw_mel'].shape[1] for item in batch)
    max_tgt_len = max(len(item['target_pointers']) for item in batch)

    batch_size = len(batch)
    n_mels = batch[0]['raw_mel'].shape[0]

    # Initialize tensors
    raw_mel = torch.zeros(batch_size, n_mels, max_raw_len)
    target_pointers = torch.full((batch_size, max_tgt_len), -1, dtype=torch.long)
    raw_padding_mask = torch.ones(batch_size, max_raw_len, dtype=torch.bool)
    tgt_padding_mask = torch.ones(batch_size, max_tgt_len, dtype=torch.bool)

    names = []

    for i, item in enumerate(batch):
        raw_len = item['raw_mel'].shape[1]
        tgt_len = len(item['target_pointers'])

        raw_mel[i, :, :raw_len] = item['raw_mel']
        target_pointers[i, :tgt_len] = item['target_pointers']
        raw_padding_mask[i, :raw_len] = False
        tgt_padding_mask[i, :tgt_len] = False
        names.append(item['name'])

    return {
        'raw_mel': raw_mel,
        'target_pointers': target_pointers,
        'raw_padding_mask': raw_padding_mask,
        'tgt_padding_mask': tgt_padding_mask,
        'names': names,
    }


def create_dataloader(
    cache_dir: str,
    pointer_dir: str,
    batch_size: int = 4,
    chunk_size: Optional[int] = 512,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a dataloader for pointer network training."""
    dataset = PointerDataset(
        cache_dir=cache_dir,
        pointer_dir=pointer_dir,
        chunk_size=chunk_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

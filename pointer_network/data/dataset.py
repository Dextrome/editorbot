"""Dataset for pointer network training."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
import random
from functools import lru_cache


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
        use_mmap: bool = True,
        preload_pointers: bool = True,
    ):
        """
        Args:
            cache_dir: directory containing cached mel spectrograms
            pointer_dir: directory containing pointer sequences (.npy files)
            max_raw_frames: maximum raw audio frames to use
            max_edit_frames: maximum edited frames (output length)
            chunk_size: if set, split long sequences into chunks
            chunk_overlap: overlap between chunks
            use_mmap: use memory-mapped file loading (faster, lower memory)
            preload_pointers: preload all pointer sequences into memory
        """
        self.cache_dir = Path(cache_dir)
        self.pointer_dir = Path(pointer_dir)
        self.max_raw_frames = max_raw_frames
        self.max_edit_frames = max_edit_frames
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_mmap = use_mmap
        self.preload_pointers = preload_pointers

        # Find all pointer files
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples")

        # Preload pointer sequences (they're small)
        if self.preload_pointers:
            self._preloaded_pointers = {}
            for sample in self.samples:
                self._preloaded_pointers[sample['name']] = np.load(sample['pointer_file'])
            print(f"Preloaded {len(self._preloaded_pointers)} pointer sequences")

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

            # Find raw mel file in cache - support both .npy and .npz formats
            raw_path = Path(info['raw_path'])
            raw_mel_file = None

            # Try different possible cache file patterns
            possible_patterns = [
                # .npz format (super_editor_cache style): {stem}.npz with 'mel' key
                self.cache_dir / "features" / f"{raw_path.stem}.npz",
                self.cache_dir / f"{raw_path.stem}.npz",
                # .npy format: {stem}_mel.npy
                self.cache_dir / f"{raw_path.stem}_mel.npy",
                self.cache_dir / f"{base_name}_raw_mel.npy",
            ]

            for pattern in possible_patterns:
                if pattern.exists():
                    raw_mel_file = pattern
                    break

            if raw_mel_file is None:
                print(f"Warning: mel file not found for {base_name}")
                print(f"  Tried: {[str(p) for p in possible_patterns]}")
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

    def _load_mel(self, mel_file: Path) -> np.ndarray:
        """Load mel spectrogram with optional memory mapping.

        Supports both:
        - .npy files: direct mel array
        - .npz files: archive with 'mel' key (super_editor_cache format)
        """
        if mel_file.suffix == '.npz':
            # .npz format - load archive and extract 'mel' key
            # mmap not supported for .npz, but file is cached anyway
            data = np.load(mel_file)
            raw_mel = data['mel']
        else:
            # .npy format - direct array
            if self.use_mmap:
                raw_mel = np.load(mel_file, mmap_mode='r')
            else:
                raw_mel = np.load(mel_file)

        # Ensure shape is (n_mels, time) for consistency
        if raw_mel.shape[0] != 128:  # Assume n_mels=128
            # Data is (time, n_mels) - transpose to (n_mels, time)
            raw_mel = np.array(raw_mel.T)
        return raw_mel

    def _load_pointers(self, sample: Dict) -> np.ndarray:
        """Load pointers, using preloaded cache if available."""
        if self.preload_pointers:
            return self._preloaded_pointers[sample['name']].copy()
        return np.load(sample['pointer_file'])

    def _get_full_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a full sample (may be truncated if too long)."""
        sample = self.samples[idx]

        # Load raw mel (memory-mapped for efficiency)
        raw_mel = self._load_mel(sample['raw_mel_file'])

        # Load pointers (from preloaded cache or disk)
        pointers = self._load_pointers(sample)

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
        # Load raw mel (memory-mapped for efficiency)
        raw_mel = self._load_mel(sample['raw_mel_file'])

        # Load pointers (from preloaded cache or disk)
        pointers = self._load_pointers(sample)

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

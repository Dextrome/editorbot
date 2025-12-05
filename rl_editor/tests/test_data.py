"""Tests for data loading module."""

import pytest
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from rl_editor.config import Config
from rl_editor.data import AudioDataset, create_dataloader

class TestAudioDataset:
    """Test AudioDataset class."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a temporary data directory with dummy audio files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create a few dummy wav files
        sr = 22050
        for i in range(3):
            duration = 1.0 + i * 0.5  # Variable duration
            y = np.random.uniform(-1, 1, int(sr * duration))
            sf.write(data_dir / f"test_{i}.wav", y, sr)
            
        return data_dir

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    def test_dataset_init(self, data_dir):
        """Test dataset initialization."""
        config = Config()
        dataset = AudioDataset(str(data_dir), config)
        assert len(dataset) == 3

    def test_dataset_getitem(self, data_dir):
        """Test getting an item from the dataset."""
        config = Config()
        dataset = AudioDataset(str(data_dir), config)
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert "audio" in item
        assert "mel" in item
        assert "beats" in item
        assert "path" in item
        assert isinstance(item["audio"], torch.Tensor)
        assert isinstance(item["mel"], torch.Tensor)

    def test_dataset_caching(self, data_dir, cache_dir):
        """Test dataset caching mechanism."""
        config = Config()
        dataset = AudioDataset(str(data_dir), config, cache_dir=str(cache_dir))
        
        # First access (should process and save to cache)
        item1 = dataset[0]
        cache_file = list(cache_dir.rglob("*.pt"))[0]
        assert cache_file.exists()
        
        # Second access (should load from cache)
        # We can verify this by modifying the cache and seeing if it loads the modified version
        # But for now, just checking it runs without error is enough
        item2 = dataset[0]
        assert torch.equal(item1["audio"], item2["audio"])

    def test_dataloader(self, data_dir):
        """Test dataloader creation and iteration."""
        config = Config()
        dataset = AudioDataset(str(data_dir), config)
        dataloader = create_dataloader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert "audio" in batch[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

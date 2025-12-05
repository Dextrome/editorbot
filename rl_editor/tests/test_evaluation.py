"""Tests for evaluation module."""

import pytest
import numpy as np
import torch
from pathlib import Path
from rl_editor.config import Config
from rl_editor.agent import Agent
from rl_editor.evaluation import Evaluator
from rl_editor.data import AudioDataset
from rl_editor.state import AudioState

class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, length=5):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return dummy data compatible with Evaluator
        n_beats = 32
        return {
            "beat_times": torch.linspace(0, 10, n_beats),
            "mel": torch.randn(128, n_beats),
            "beat_features": torch.randn(n_beats, 3), # 3 features per beat
            "tempo": torch.tensor(120.0),
            "path": f"test_{idx}.wav"
        }

class TestEvaluator:
    """Test Evaluator class."""

    @pytest.fixture
    def config(self):
        conf = Config()
        conf.training.device = "cpu"
        return conf

    @pytest.fixture
    def agent(self, config):
        # Input dim should match StateRepresentation feature_dim
        # Based on previous error, it is 35 for the default config + mock data
        return Agent(config, input_dim=35, n_actions=10)

    def test_evaluator_init(self, config, agent, tmp_path):
        """Test evaluator initialization."""
        evaluator = Evaluator(config, agent, output_dir=str(tmp_path))
        assert evaluator.output_dir.exists()

    def test_evaluate_dataset(self, config, agent, tmp_path):
        """Test evaluating on a dataset."""
        evaluator = Evaluator(config, agent, output_dir=str(tmp_path))
        dataset = MockDataset(length=2)
        
        metrics = evaluator.evaluate_dataset(dataset, n_episodes=2)
        
        assert "mean_reward" in metrics
        assert "mean_length" in metrics
        assert len(list(tmp_path.glob("*.png"))) > 0 # Check if plots were generated

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

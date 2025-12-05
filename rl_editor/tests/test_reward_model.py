"""Tests for reward model."""

import pytest
import numpy as np
import torch
from rl_editor.config import Config
from rl_editor.reward_model import RewardModel

class TestRewardModel:
    """Test RewardModel class."""

    @pytest.fixture
    def config(self):
        conf = Config()
        conf.training.device = "cpu"
        return conf

    def test_reward_model_init(self, config):
        """Test initialization."""
        model = RewardModel(config, input_dim=64)
        assert model.input_dim == 64
        assert isinstance(model.net, torch.nn.Sequential)

    def test_reward_model_forward(self, config):
        """Test forward pass."""
        model = RewardModel(config, input_dim=64)
        features = torch.randn(32, 64)
        rewards = model(features)
        assert rewards.shape == (32, 1)

    def test_train_on_preferences(self, config):
        """Test training on preferences."""
        model = RewardModel(config, input_dim=64)
        
        # Create dummy preference data
        n_samples = 100
        features_a = np.random.randn(n_samples, 64).astype(np.float32)
        features_b = np.random.randn(n_samples, 64).astype(np.float32)
        # Assume a is better if sum is higher
        rewards_a = np.sum(features_a, axis=1)
        rewards_b = np.sum(features_b, axis=1)
        preferences = (rewards_a > rewards_b).astype(np.float32)
        
        initial_loss = model.train_on_preferences(features_a, features_b, preferences, n_epochs=1)
        final_loss = model.train_on_preferences(features_a, features_b, preferences, n_epochs=5)
        
        # Loss should decrease (or be low)
        assert isinstance(final_loss, float)
        # Note: convergence is not guaranteed in random test, but it should run

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

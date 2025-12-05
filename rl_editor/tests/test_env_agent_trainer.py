"""Unit tests for environment, agent, and trainer modules."""

import pytest
import numpy as np
import torch
from rl_editor.config import Config
from rl_editor.environment import AudioEditingEnv
from rl_editor.agent import Agent, PolicyNetwork, ValueNetwork
from rl_editor.trainer import PPOTrainer
from rl_editor.state import AudioState


class TestAgent:
    """Test agent networks."""

    def test_policy_network_init(self):
        """Test PolicyNetwork initialization."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        net = PolicyNetwork(config, input_dim=64, n_actions=128)
        assert net.input_dim == 64
        assert net.n_actions == 128

    def test_policy_network_forward(self):
        """Test PolicyNetwork forward pass."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        net = PolicyNetwork(config, input_dim=64, n_actions=128)

        state = torch.randn(32, 64)
        logits, probs = net(state)

        assert logits.shape == (32, 128)
        assert probs.shape == (32, 128)
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(32), atol=1e-5)

    def test_policy_network_with_mask(self):
        """Test PolicyNetwork with action mask."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        net = PolicyNetwork(config, input_dim=64, n_actions=128)

        state = torch.randn(2, 64)
        mask = torch.zeros(2, 128, dtype=torch.bool)
        mask[:, :10] = True  # Only first 10 actions valid

        logits, probs = net(state, mask)

        # Invalid actions should have ~zero probability
        assert torch.allclose(probs[:, 10:], torch.zeros(2, 118), atol=0.01)

    def test_value_network_init(self):
        """Test ValueNetwork initialization."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        net = ValueNetwork(config, input_dim=64)
        assert net.input_dim == 64

    def test_value_network_forward(self):
        """Test ValueNetwork forward pass."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        net = ValueNetwork(config, input_dim=64)

        state = torch.randn(32, 64)
        values = net(state)

        assert values.shape == (32,)

    def test_agent_init(self):
        """Test Agent initialization."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        agent = Agent(config, input_dim=64, n_actions=128)
        assert agent.device == torch.device("cpu")

    def test_agent_select_action(self):
        """Test agent action selection."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        agent = Agent(config, input_dim=64, n_actions=128)

        state = np.random.randn(64).astype(np.float32)
        action, log_prob = agent.select_action(torch.from_numpy(state))

        assert 0 <= action < 128
        assert isinstance(log_prob, float)

    def test_agent_compute_value(self):
        """Test agent value computation."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        agent = Agent(config, input_dim=64, n_actions=128)

        state = np.random.randn(64).astype(np.float32)
        value = agent.compute_value(torch.from_numpy(state))

        assert isinstance(value, float)

    def test_agent_save_load(self, tmp_path):
        """Test agent save/load functionality."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        agent1 = Agent(config, input_dim=64, n_actions=128)

        # Save
        checkpoint_path = tmp_path / "agent.pt"
        agent1.save(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Load
        agent2 = Agent(config, input_dim=64, n_actions=128)
        agent2.load(str(checkpoint_path))

        # Check weights are same
        for p1, p2 in zip(agent1.policy_net.parameters(), agent2.policy_net.parameters()):
            assert torch.allclose(p1, p2)


class TestEnvironment:
    """Test audio editing environment."""

    def create_test_audio_state(self) -> AudioState:
        """Create test audio state."""
        beat_times = np.linspace(0, 8, 32)  # 32 beats over 8 seconds
        beat_features = np.random.randn(32, 3)
        return AudioState(
            beat_index=0,
            beat_times=beat_times,
            beat_features=beat_features,
            tempo=120.0,
        )

    def test_env_init(self):
        """Test environment initialization."""
        config = Config()
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)
        assert env.config == config
        assert env.audio_state == audio_state

    def test_env_reset(self):
        """Test environment reset."""
        config = Config()
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)

        obs, info = env.reset()
        assert obs is not None
        assert obs.dtype == np.float32
        assert "step" in info
        assert "n_beats" in info
        assert info["n_beats"] == 32

    def test_env_step(self):
        """Test environment step."""
        config = Config()
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)

        obs, _ = env.reset()
        assert env.action_space is not None

        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == obs.shape
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_env_episode(self):
        """Test full episode."""
        config = Config()
        config.training.total_timesteps = 100  # Shorter for testing
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)
        env.set_max_steps(50)

        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        # Reward can be positive or negative depending on edit actions taken
        assert isinstance(episode_reward, (float, np.floating))

    def test_env_action_masking(self):
        """Test environment action masking."""
        config = Config()
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)

        obs, _ = env.reset()
        mask = env.action_space.get_action_mask(
            current_beat_index=0,
            remaining_duration=100.0,
            edited_beats=[],
            total_beats=32,
        )

        assert mask.shape[0] == env.action_space.n_discrete_actions
        assert np.any(mask)  # At least some actions should be valid

    def test_env_observation_dim(self):
        """Test observation dimension matches config."""
        config = Config()
        audio_state = self.create_test_audio_state()
        env = AudioEditingEnv(config, audio_state)

        obs, _ = env.reset()
        expected_dim = env.state_rep.feature_dim if env.state_rep else 0
        assert obs.shape[0] == expected_dim


class TestPPOTrainer:
    """Test PPO trainer."""

    def create_test_audio_state(self) -> AudioState:
        """Create test audio state."""
        beat_times = np.linspace(0, 8, 32)
        beat_features = np.random.randn(32, 3)
        return AudioState(
            beat_index=0,
            beat_times=beat_times,
            beat_features=beat_features,
            tempo=120.0,
        )

    def test_trainer_init(self):
        """Test trainer initialization."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        assert trainer.config == config
        assert trainer.device == torch.device("cpu")

    def test_trainer_initialize_env_and_agent(self):
        """Test initializing environment and agent."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        audio_state = self.create_test_audio_state()

        trainer.initialize_env_and_agent(audio_state)

        assert trainer.env is not None
        assert trainer.agent is not None
        assert trainer.policy_optimizer is not None
        assert trainer.value_optimizer is not None

    def test_trainer_collect_rollouts(self):
        """Test collecting rollouts."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        audio_state = self.create_test_audio_state()

        trainer.initialize_env_and_agent(audio_state)
        rollout_data = trainer.collect_rollouts(n_steps=10)

        assert "states" in rollout_data
        assert "actions" in rollout_data
        assert "rewards" in rollout_data
        assert "returns" in rollout_data
        assert len(rollout_data["states"]) > 0

    def test_trainer_update(self):
        """Test trainer update step."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        audio_state = self.create_test_audio_state()

        trainer.initialize_env_and_agent(audio_state)
        rollout_data = trainer.collect_rollouts(n_steps=16)
        metrics = trainer.update(rollout_data)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "episode_reward" in metrics

    def test_trainer_save_load_checkpoint(self, tmp_path):
        """Test checkpoint save/load."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        audio_state = self.create_test_audio_state()

        trainer.initialize_env_and_agent(audio_state)

        # Save
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Load
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.global_step == 0  # Should be same as saved

    def test_trainer_evaluate(self):
        """Test trainer evaluation."""
        config = Config()
        config.training.device = "cpu"  # Force CPU for tests
        trainer = PPOTrainer(config)
        audio_state = self.create_test_audio_state()

        trainer.initialize_env_and_agent(audio_state)
        metrics = trainer.evaluate(n_episodes=2)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "min_reward" in metrics
        assert "max_reward" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

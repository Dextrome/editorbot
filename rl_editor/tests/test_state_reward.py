"""Unit tests for state representation and reward modules."""

import pytest
import numpy as np
from rl_editor.config import Config
from rl_editor.state import (
    AudioState,
    EditHistory,
    StateRepresentation,
)
from rl_editor.reward import RewardCalculator, RewardComponents, compute_trajectory_return


class TestEditHistory:
    """Test edit history tracking."""

    def test_init(self):
        """Test EditHistory initialization."""
        history = EditHistory()
        assert history.kept_beats == set()
        assert history.cut_beats == set()
        assert history.total_duration_edited == 0.0

    def test_add_keep(self):
        """Test adding kept beats."""
        history = EditHistory()
        history.add_keep(5)
        assert 5 in history.kept_beats

        # Duplicate should not be added (sets handle this automatically)
        history.add_keep(5)
        assert len([b for b in history.kept_beats if b == 5]) == 1

    def test_add_cut(self):
        """Test adding cut beats."""
        history = EditHistory()
        history.add_cut(3)
        assert 3 in history.cut_beats

    def test_add_loop(self):
        """Test adding looped beats."""
        history = EditHistory()
        history.add_loop(2)
        assert 2 in history.looped_beats

    def test_add_crossfade(self):
        """Test adding crossfaded pairs."""
        history = EditHistory()
        history.add_crossfade(1, 3)
        assert (1, 3) in history.crossfaded_pairs

    def test_add_reorder(self):
        """Test adding reordered pairs."""
        history = EditHistory()
        history.add_reorder(4, 2)
        # Should be stored sorted
        assert (2, 4) in history.reordered_pairs

    def test_get_edited_beats(self):
        """Test getting all edited beats."""
        history = EditHistory()
        history.add_keep(0)
        history.add_cut(2)
        history.add_loop(5)
        history.add_crossfade(1, 3)

        edited = history.get_edited_beats()
        assert set(edited) == {0, 2, 5, 1, 3}
        assert edited == sorted(edited)


class TestStateRepresentation:
    """Test state representation construction."""

    def test_init(self):
        """Test StateRepresentation initialization."""
        config = Config()
        state_rep = StateRepresentation(config)
        assert state_rep.config == config
        assert state_rep.feature_dim > 0

    def test_feature_dim_calculation(self):
        """Test feature dimension calculation."""
        config = Config()
        state_rep = StateRepresentation(config)

        # With all features enabled
        dim = state_rep.feature_dim
        assert dim > 0

        # Should be deterministic
        assert state_rep.feature_dim == dim

    def test_beat_context_extraction(self):
        """Test beat context extraction."""
        config = Config()
        state_rep = StateRepresentation(config)

        beat_features = np.random.randn(32, 10)  # 32 beats, 10 features each
        context = state_rep.get_beat_context(16, beat_features, context_size=2)

        # Context size 2: 2 before + current + 2 after = 5 beats
        expected_dim = 5 * 10
        assert context.shape[0] == expected_dim

    def test_beat_context_padding(self):
        """Test beat context with boundary beats."""
        config = Config()
        state_rep = StateRepresentation(config)

        beat_features = np.ones((8, 3))
        # First beat context (should pad at start)
        context = state_rep.get_beat_context(0, beat_features, context_size=2)
        assert context.shape[0] == 5 * 3
        # First two positions should be zero (padding)
        assert np.allclose(context[:6], 0)

        # Last beat context (should pad at end)
        context = state_rep.get_beat_context(7, beat_features, context_size=2)
        assert context.shape[0] == 5 * 3
        # Last two positions should be zero (padding)
        assert np.allclose(context[-6:], 0)


class TestRewardCalculator:
    """Test reward calculation."""

    def test_init(self):
        """Test RewardCalculator initialization."""
        config = Config()
        calc = RewardCalculator(config)
        assert calc.config == config

    def test_sparse_reward_positive(self):
        """Test sparse reward with positive human rating."""
        config = Config()
        calc = RewardCalculator(config)

        components = calc.compute_sparse_reward(human_rating=8.0)
        assert components.sparse_reward is not None
        assert -1.0 <= components.sparse_reward <= 1.0
        assert components.sparse_reward > 0  # Rating 8 should be positive

    def test_sparse_reward_negative(self):
        """Test sparse reward with negative human rating."""
        config = Config()
        calc = RewardCalculator(config)

        components = calc.compute_sparse_reward(human_rating=2.0)
        assert components.sparse_reward is not None
        assert components.sparse_reward < 0  # Rating 2 should be negative

    def test_sparse_reward_neutral(self):
        """Test sparse reward with neutral rating."""
        config = Config()
        calc = RewardCalculator(config)

        components = calc.compute_sparse_reward(human_rating=5.0)
        assert components.sparse_reward == 0.0

    def test_sparse_reward_no_feedback(self):
        """Test sparse reward with no feedback."""
        config = Config()
        calc = RewardCalculator(config)

        components = calc.compute_sparse_reward(human_rating=-1)
        assert components.sparse_reward is None

    def test_dense_reward_components(self):
        """Test that dense reward has components."""
        config = Config()
        calc = RewardCalculator(config)

        # Create synthetic audio
        sr = 22050
        duration = 2.0
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
        edited_audio = original_audio.copy()
        beat_times = np.linspace(0, duration, 32)
        beat_frames = np.arange(len(beat_times))

        components = calc.compute_dense_reward(
            edited_audio, original_audio, beat_times, beat_frames, sr
        )

        assert 0.0 <= components.tempo_consistency <= 1.0
        assert 0.0 <= components.energy_flow <= 1.0
        assert 0.0 <= components.phrase_completeness <= 1.0
        assert 0.0 <= components.transition_quality <= 1.0
        assert 0.0 <= components.dense_reward <= 1.0

    def test_combined_reward(self):
        """Test combined reward computation."""
        config = Config()
        calc = RewardCalculator(config)

        sr = 22050
        duration = 2.0
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
        edited_audio = original_audio.copy()
        beat_times = np.linspace(0, duration, 32)
        beat_frames = np.arange(len(beat_times))

        components = calc.compute_combined_reward(
            edited_audio=edited_audio,
            original_audio=original_audio,
            beat_times=beat_times,
            beat_frames=beat_frames,
            human_rating=7.0,
            learned_reward=0.3,
            sr=sr,
        )

        assert components.sparse_reward is not None
        assert components.learned_reward is not None
        assert -1.0 <= components.total_reward <= 1.0


class TestTrajectoryReturn:
    """Test trajectory return computation."""

    def test_single_step(self):
        """Test trajectory with single step."""
        rewards = [1.0]
        returns, mean = compute_trajectory_return(rewards, normalize=False)
        assert len(returns) == 1
        assert returns[0] == 1.0

    def test_two_steps(self):
        """Test trajectory with two steps."""
        rewards = [1.0, 2.0]
        returns, mean = compute_trajectory_return(rewards, gamma=0.99, normalize=False)
        assert len(returns) == 2
        # Reverse: process 2.0 first (return=2.0), then 1.0 (return=1.0 + 0.99*2.0)
        assert returns[1] == 2.0
        assert abs(returns[0] - (1.0 + 0.99 * 2.0)) < 1e-6

    def test_normalization(self):
        """Test return normalization."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns_normalized, _ = compute_trajectory_return(rewards, normalize=True)
        returns = np.array(returns_normalized)

        # After normalization: mean should be ~0, std should be ~1
        assert abs(np.mean(returns)) < 1e-5
        assert abs(np.std(returns) - 1.0) < 0.1

    def test_discount_factor(self):
        """Test different discount factors."""
        rewards = [1.0, 1.0, 1.0]

        returns_undiscounted, _ = compute_trajectory_return(rewards, gamma=1.0, normalize=False)
        returns_discounted, _ = compute_trajectory_return(rewards, gamma=0.9, normalize=False)

        # Discounted returns should be smaller
        assert returns_undiscounted[0] > returns_discounted[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for action space module."""

import pytest
import numpy as np
from rl_editor.actions import (
    ActionType,
    Action,
    KeepAction,
    CutAction,
    LoopAction,
    CrossfadeAction,
    ReorderAction,
    ActionSpace,
)


class TestActionTypes:
    """Test individual action classes."""

    def test_keep_action(self):
        """Test KeepAction creation and properties."""
        action = KeepAction(beat_index=5)
        assert action.action_type == ActionType.KEEP
        assert action.beat_index == 5
        assert "KEEP" in str(action)

    def test_cut_action(self):
        """Test CutAction creation and properties."""
        action = CutAction(beat_index=3)
        assert action.action_type == ActionType.CUT
        assert action.beat_index == 3

    def test_loop_action(self):
        """Test LoopAction creation and properties."""
        action = LoopAction(beat_index=2, n_times=3)
        assert action.action_type == ActionType.LOOP
        assert action.beat_index == 2
        assert action.n_times == 3

    def test_crossfade_action(self):
        """Test CrossfadeAction creation and properties."""
        action = CrossfadeAction(beat_i=1, beat_j=2, duration_seconds=1.5)
        assert action.action_type == ActionType.CROSSFADE
        assert action.beat_index == 1
        assert action.beat_j == 2
        assert action.duration_seconds == 1.5

    def test_reorder_action(self):
        """Test ReorderAction creation and properties."""
        action = ReorderAction(section_a=3, section_b=1)
        assert action.action_type == ActionType.REORDER
        assert action.beat_index == 3
        assert action.section_b == 1


class TestActionSpace:
    """Test ActionSpace class."""

    def test_initialization(self):
        """Test ActionSpace initialization."""
        action_space = ActionSpace(n_beats=32)
        assert action_space.n_beats == 32
        assert action_space.n_keep == 32
        assert action_space.n_cut == 32
        assert action_space.n_loop == 32 * 4  # max_loop_times=4
        assert action_space.n_crossfade > 0
        assert action_space.n_reorder == 32 * 31  # n_beats * (n_beats-1)
        assert action_space.n_discrete_actions > 0

    def test_action_space_size_calculation(self):
        """Test action space size calculations for different beat counts."""
        for n_beats in [8, 16, 32, 64]:
            space = ActionSpace(n_beats=n_beats)
            expected_keep = n_beats
            expected_cut = n_beats
            expected_loop = n_beats * 4
            expected_reorder = n_beats * (n_beats - 1)
            
            assert space.n_keep == expected_keep
            assert space.n_cut == expected_cut
            assert space.n_loop == expected_loop
            assert space.n_reorder == expected_reorder

    def test_decode_keep_action(self):
        """Test decoding KEEP actions."""
        action_space = ActionSpace(n_beats=8)
        for i in range(action_space.n_keep):
            action = action_space.decode_action(i)
            assert isinstance(action, KeepAction)
            assert action.beat_index == i

    def test_decode_cut_action(self):
        """Test decoding CUT actions."""
        action_space = ActionSpace(n_beats=8)
        offset = action_space.n_keep
        for i in range(action_space.n_cut):
            action = action_space.decode_action(offset + i)
            assert isinstance(action, CutAction)
            assert action.beat_index == i

    def test_decode_loop_action(self):
        """Test decoding LOOP actions."""
        action_space = ActionSpace(n_beats=8, max_loop_times=4)
        offset = action_space.n_keep + action_space.n_cut
        
        # Test first loop action
        action = action_space.decode_action(offset)
        assert isinstance(action, LoopAction)
        assert action.beat_index == 0
        assert action.n_times == 2  # min loop times
        
        # Test last loop action for beat 0
        action = action_space.decode_action(offset + 3)
        assert action.beat_index == 0
        assert action.n_times == 5  # max loop times (2+4-1)

    def test_decode_crossfade_action(self):
        """Test decoding CROSSFADE actions."""
        action_space = ActionSpace(n_beats=8)
        offset = action_space.n_keep + action_space.n_cut + action_space.n_loop
        
        # First crossfade action should exist
        action = action_space.decode_action(offset)
        assert isinstance(action, CrossfadeAction)
        assert action.beat_index == 0
        assert action.beat_j > 0
        assert 0.1 <= action.duration_seconds <= 2.0

    def test_decode_reorder_action(self):
        """Test decoding REORDER actions."""
        action_space = ActionSpace(n_beats=8)
        offset = (
            action_space.n_keep
            + action_space.n_cut
            + action_space.n_loop
            + action_space.n_crossfade
        )
        
        action = action_space.decode_action(offset)
        assert isinstance(action, ReorderAction)
        assert action.beat_index == 0
        assert action.section_b > 0

    def test_decode_out_of_bounds(self):
        """Test decoding with out-of-bounds action index."""
        action_space = ActionSpace(n_beats=8)
        
        with pytest.raises(ValueError):
            action_space.decode_action(-1)
        
        with pytest.raises(ValueError):
            action_space.decode_action(action_space.n_discrete_actions)

    def test_encode_keep_action(self):
        """Test encoding KEEP actions."""
        action_space = ActionSpace(n_beats=8)
        
        for i in range(8):
            action = KeepAction(i)
            idx = action_space.encode_action(action)
            assert idx == i
            assert action_space.decode_action(idx).beat_index == i

    def test_encode_cut_action(self):
        """Test encoding CUT actions."""
        action_space = ActionSpace(n_beats=8)
        
        for i in range(8):
            action = CutAction(i)
            idx = action_space.encode_action(action)
            decoded = action_space.decode_action(idx)
            assert isinstance(decoded, CutAction)
            assert decoded.beat_index == i

    def test_encode_loop_action(self):
        """Test encoding LOOP actions."""
        action_space = ActionSpace(n_beats=8, max_loop_times=4)
        
        action = LoopAction(beat_index=2, n_times=3)
        idx = action_space.encode_action(action)
        decoded = action_space.decode_action(idx)
        assert isinstance(decoded, LoopAction)
        assert decoded.beat_index == 2
        assert decoded.n_times == 3

    def test_encode_crossfade_action(self):
        """Test encoding CROSSFADE actions."""
        action_space = ActionSpace(n_beats=8)
        
        action = CrossfadeAction(beat_i=1, beat_j=3, duration_seconds=1.0)
        idx = action_space.encode_action(action)
        decoded = action_space.decode_action(idx)
        assert isinstance(decoded, CrossfadeAction)
        assert decoded.beat_index == 1
        assert decoded.beat_j == 3
        # Duration may not be exact due to discretization
        assert abs(decoded.duration_seconds - 1.0) < 0.5

    def test_encode_reorder_action(self):
        """Test encoding REORDER actions."""
        action_space = ActionSpace(n_beats=8)
        
        action = ReorderAction(section_a=2, section_b=5)
        idx = action_space.encode_action(action)
        decoded = action_space.decode_action(idx)
        assert isinstance(decoded, ReorderAction)
        assert decoded.beat_index == 2
        assert decoded.section_b == 5

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip for all action types."""
        action_space = ActionSpace(n_beats=16)
        
        test_actions = [
            KeepAction(5),
            CutAction(3),
            LoopAction(7, 2),
            CrossfadeAction(1, 4, 1.5),
            ReorderAction(2, 8),
        ]
        
        for original_action in test_actions:
            encoded_idx = action_space.encode_action(original_action)
            decoded_action = action_space.decode_action(encoded_idx)
            
            assert type(decoded_action) == type(original_action)
            assert decoded_action.beat_index == original_action.beat_index

    def test_action_mask_all_valid(self):
        """Test action mask when all actions are valid."""
        action_space = ActionSpace(n_beats=8)
        mask = action_space.get_action_mask(
            current_beat_index=0,
            remaining_duration=100.0,
            edited_beats=[],
            total_beats=8,
        )
        
        assert mask.shape == (action_space.n_discrete_actions,)
        assert mask.dtype == bool
        # Most actions should be valid
        assert np.sum(mask) > 0

    def test_action_mask_excludes_edited_beats(self):
        """Test action mask excludes already edited beats."""
        action_space = ActionSpace(n_beats=8)
        edited_beats = [0, 2, 5]
        mask = action_space.get_action_mask(
            current_beat_index=3,
            remaining_duration=100.0,
            edited_beats=edited_beats,
            total_beats=8,
        )
        
        # KEEP and CUT actions for edited beats should be invalid
        for beat_idx in edited_beats:
            keep_idx = beat_idx
            cut_idx = action_space.n_keep + beat_idx
            assert not mask[keep_idx], f"KEEP action for beat {beat_idx} should be masked"
            assert not mask[cut_idx], f"CUT action for beat {beat_idx} should be masked"

    def test_sample_action_without_mask(self):
        """Test sampling random action without mask."""
        action_space = ActionSpace(n_beats=8)
        
        for _ in range(100):
            action_idx = action_space.sample()
            assert 0 <= action_idx < action_space.n_discrete_actions

    def test_sample_action_with_mask(self):
        """Test sampling random action with mask."""
        action_space = ActionSpace(n_beats=8)
        mask = np.zeros(action_space.n_discrete_actions, dtype=bool)
        # Only allow KEEP actions for beats 0 and 1
        mask[0] = True
        mask[1] = True
        
        for _ in range(100):
            action_idx = action_space.sample(mask)
            assert action_idx in [0, 1]

    def test_sample_with_empty_mask(self):
        """Test sampling with all-False mask (fallback behavior)."""
        action_space = ActionSpace(n_beats=8)
        mask = np.zeros(action_space.n_discrete_actions, dtype=bool)
        
        # Should still return a valid index
        action_idx = action_space.sample(mask)
        assert 0 <= action_idx < action_space.n_discrete_actions

    def test_crossfade_duration_discretization(self):
        """Test that crossfade durations are properly discretized."""
        action_space = ActionSpace(
            n_beats=8,
            max_crossfade_duration=2.0,
            n_discrete_crossfade_durations=5,
        )
        
        assert len(action_space.crossfade_durations) == 5
        assert action_space.crossfade_durations[0] >= 0.1
        assert action_space.crossfade_durations[-1] <= 2.0
        assert np.all(np.diff(action_space.crossfade_durations) > 0)  # Monotonic


class TestActionSpaceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_beat(self):
        """Test action space with single beat."""
        action_space = ActionSpace(n_beats=1)
        assert action_space.n_beats == 1
        assert action_space.n_keep == 1
        assert action_space.n_cut == 1
        assert action_space.n_loop == 4
        # No crossfade between different beats
        assert action_space.n_crossfade == 0
        # No reorder between different sections
        assert action_space.n_reorder == 0

    def test_two_beats(self):
        """Test action space with two beats."""
        action_space = ActionSpace(n_beats=2)
        assert action_space.n_crossfade > 0
        assert action_space.n_reorder == 2  # 2 * 1

    def test_large_beat_count(self):
        """Test action space with large beat count."""
        action_space = ActionSpace(n_beats=256)
        assert action_space.n_beats == 256
        assert action_space.n_discrete_actions > 0
        
        # Test random sampling works
        action_idx = action_space.sample()
        decoded = action_space.decode_action(action_idx)
        assert decoded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

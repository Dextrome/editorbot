"""Action space definitions for RL-based audio editor.

Defines 4 core actions (crossfades are applied automatically at edit boundaries):
- KEEP(beat_i): Include beat in output
- CUT(beat_i): Remove beat from output  
- LOOP(beat_i, n_times): Repeat a beat/section n times
- REORDER(section_a, section_b): Move sections around (non-linear editing)

Transitions between sections are handled automatically with crossfades.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


class ActionType(IntEnum):
    """Action type enumeration."""

    KEEP = 0
    CUT = 1
    LOOP = 2
    REORDER = 3


@dataclass
class Action:
    """Base action class."""

    action_type: ActionType
    beat_index: int

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.action_type.name}, beat={self.beat_index})"


@dataclass
class KeepAction(Action):
    """Keep beat action: include beat in output."""

    def __init__(self, beat_index: int) -> None:
        super().__init__(ActionType.KEEP, beat_index)


@dataclass
class CutAction(Action):
    """Cut beat action: remove beat from output."""

    def __init__(self, beat_index: int) -> None:
        super().__init__(ActionType.CUT, beat_index)


@dataclass
class LoopAction(Action):
    """Loop action: repeat a beat/section n times."""

    n_times: int

    def __init__(self, beat_index: int, n_times: int) -> None:
        super().__init__(ActionType.LOOP, beat_index)
        self.n_times = n_times


@dataclass
class ReorderAction(Action):
    """Reorder action: move sections around (non-linear editing)."""

    target_position: int  # Where to move this beat to

    def __init__(self, beat_index: int, target_position: int) -> None:
        super().__init__(ActionType.REORDER, beat_index)
        self.target_position = target_position


class ActionSpace:
    """Action space manager for RL editor.

    Simplified action space:
    - KEEP: Keep current beat (1 action)
    - CUT: Cut current beat (1 action)
    - LOOP: Loop current beat 2x, 3x, or 4x (3 actions)
    - REORDER: Move current beat to position +1, +2, +3, or +4 (4 actions)
    
    Total: 9 actions (much simpler than before!)
    
    Crossfades between segments are applied automatically during audio rendering.
    """

    def __init__(
        self,
        n_beats: int,
        max_loop_times: int = 4,
        default_crossfade_ms: int = 50,  # Auto-crossfade duration at edit boundaries
    ) -> None:
        """Initialize action space.

        Args:
            n_beats: Total number of beats in the track
            max_loop_times: Maximum loop repetitions
            default_crossfade_ms: Default crossfade duration for automatic transitions (ms)
        """
        self.n_beats = n_beats
        self.max_loop_times = max_loop_times
        self.default_crossfade_ms = default_crossfade_ms

        # Calculate action space dimensions
        self._compute_action_dimensions()

    def _compute_action_dimensions(self) -> None:
        """Compute total action space size.
        
        Simplified action space (9 actions total):
        - KEEP: 1 action (keep current beat)
        - CUT: 1 action (cut current beat)
        - LOOP: 3 actions (loop 2x, 3x, 4x)
        - REORDER: 4 actions (move to +1, +2, +3, +4 positions)
        
        Crossfades are automatic at edit boundaries, not explicit actions.
        """
        # KEEP: 1 action (keep current beat)
        self.n_keep = 1
        
        # CUT: 1 action (cut current beat)
        self.n_cut = 1
        
        # LOOP: max_loop_times - 1 options (loop current beat 2x, 3x, 4x)
        self.n_loop = self.max_loop_times - 1  # -1 because loop 1x = keep
        
        # REORDER: move current beat to nearby positions
        self.reorder_window = min(4, max(1, self.n_beats - 1))
        self.n_reorder = self.reorder_window

        # Total discrete action space size: 1 + 1 + 3 + 4 = 9
        self.n_discrete_actions = (
            self.n_keep + self.n_cut + self.n_loop + self.n_reorder
        )
        
        # Legacy fields for backwards compatibility
        self.n_crossfade = 0
        self.n_advance = 0

    def decode_action(self, action_idx: int, current_beat: int = 0) -> Action:
        """Decode discrete action index to Action object.
        
        Action layout:
        - 0: KEEP
        - 1: CUT
        - 2: LOOP 2x
        - 3: LOOP 3x
        - 4: LOOP 4x
        - 5: REORDER to +1
        - 6: REORDER to +2
        - 7: REORDER to +3
        - 8: REORDER to +4

        Args:
            action_idx: Integer action index
            current_beat: Current beat index (actions are relative to this)

        Returns:
            Decoded Action object

        Raises:
            ValueError: If action index is out of bounds
        """
        if action_idx < 0 or action_idx >= self.n_discrete_actions:
            raise ValueError(
                f"Action index {action_idx} out of bounds [0, {self.n_discrete_actions})"
            )

        offset = 0

        # KEEP action (action 0)
        if action_idx < offset + self.n_keep:
            return KeepAction(current_beat)
        offset += self.n_keep

        # CUT action (action 1)
        if action_idx < offset + self.n_cut:
            return CutAction(current_beat)
        offset += self.n_cut

        # LOOP actions (actions 2, 3, 4 = loop 2x, 3x, 4x)
        if action_idx < offset + self.n_loop:
            loop_idx = action_idx - offset
            n_times = loop_idx + 2  # loop_idx 0 -> 2x, 1 -> 3x, 2 -> 4x
            return LoopAction(current_beat, n_times)
        offset += self.n_loop

        # REORDER actions (actions 5, 6, 7, 8 = move to +1, +2, +3, +4)
        if action_idx < offset + self.n_reorder:
            reorder_idx = action_idx - offset
            target_position = min(current_beat + reorder_idx + 1, self.n_beats - 1)
            return ReorderAction(current_beat, target_position)
        offset += self.n_reorder
        
        raise ValueError(f"Failed to decode action index {action_idx}")

    def encode_action(self, action: Action, current_beat: int = 0) -> int:
        """Encode Action object to discrete action index.
        
        Action layout:
        - 0: KEEP
        - 1: CUT
        - 2-4: LOOP (2x, 3x, 4x)
        - 5-8: REORDER (to +1, +2, +3, +4)

        Args:
            action: Action object
            current_beat: Current beat index

        Returns:
            Discrete action index
        """
        offset = 0

        if isinstance(action, KeepAction):
            return offset  # Action 0
        offset += self.n_keep

        if isinstance(action, CutAction):
            return offset  # Action 1
        offset += self.n_cut

        if isinstance(action, LoopAction):
            # n_times 2 -> action 2, n_times 3 -> action 3, n_times 4 -> action 4
            return offset + (action.n_times - 2)
        offset += self.n_loop

        if isinstance(action, ReorderAction):
            # target +1 -> action 5, +2 -> action 6, etc.
            position_offset = action.target_position - current_beat - 1
            position_offset = max(0, min(position_offset, self.reorder_window - 1))
            return offset + position_offset
        offset += self.n_reorder

        raise ValueError(f"Unknown action type: {type(action)}")

    def get_action_mask(
        self,
        current_beat_index: int,
        remaining_duration: float,
        edited_beats: list,
        total_beats: int,
    ) -> np.ndarray:
        """Get action mask for valid actions given current state.
        
        Action layout:
        - 0: KEEP (valid if beat not edited)
        - 1: CUT (valid if beat not edited)
        - 2-4: LOOP (valid if beat not edited)
        - 5-8: REORDER (valid if target beat exists)

        Args:
            current_beat_index: Current beat index
            remaining_duration: Remaining duration budget in seconds
            edited_beats: List of beat indices already edited
            total_beats: Total number of beats in track

        Returns:
            Boolean mask of valid actions (True = valid)
        """
        mask = np.zeros(self.n_discrete_actions, dtype=bool)
        offset = 0
        edited_set = set(edited_beats)
        beat_valid = current_beat_index not in edited_set and current_beat_index < total_beats

        # KEEP action (0): valid if current beat not already edited
        if beat_valid:
            mask[offset] = True
        offset += self.n_keep

        # CUT action (1): valid if current beat not already edited
        if beat_valid:
            mask[offset] = True
        offset += self.n_cut

        # LOOP actions (2, 3, 4): valid if current beat not already edited
        if beat_valid:
            for i in range(self.n_loop):
                mask[offset + i] = True
        offset += self.n_loop

        # REORDER actions (5, 6, 7, 8): valid if target position exists
        for i in range(self.n_reorder):
            target_position = current_beat_index + i + 1
            if target_position < total_beats and beat_valid:
                mask[offset + i] = True

        return mask

    def get_keep_cut_only_mask(
        self,
        current_beat_index: int,
        edited_beats: list,
        total_beats: int,
    ) -> np.ndarray:
        """Get action mask that only allows KEEP and CUT actions.
        
        Use this for BC-based training where we only want KEEP/CUT decisions.

        Args:
            current_beat_index: Current beat index
            edited_beats: List of beat indices already edited
            total_beats: Total number of beats in track

        Returns:
            Boolean mask with only KEEP and CUT enabled
        """
        mask = np.zeros(self.n_discrete_actions, dtype=bool)
        edited_set = set(edited_beats)

        # KEEP action (index 0): valid if current beat not already edited
        if current_beat_index not in edited_set and current_beat_index < total_beats:
            mask[0] = True  # KEEP is always at index 0

        # CUT action (index 1): valid if current beat not already edited
        if current_beat_index not in edited_set and current_beat_index < total_beats:
            mask[1] = True  # CUT is always at index 1
        
        # All other actions (LOOP, REORDER, ADVANCE) are masked out

        return mask

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        """Sample random valid action.

        Args:
            mask: Optional boolean mask of valid actions

        Returns:
            Discrete action index
        """
        if mask is not None:
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                return np.random.randint(0, self.n_discrete_actions)
            return np.random.choice(valid_actions)
        return np.random.randint(0, self.n_discrete_actions)

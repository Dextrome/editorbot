"""Enhanced action space for RL-based audio editor (V2).

Key changes from V1:
1. Section-level actions (work on multiple beats at once)
2. Hierarchical decisions (phrase-level and beat-level)
3. More complex musical editing operations

Action Space Structure:
- KEEP_BEAT: Keep single beat (fine control)
- CUT_BEAT: Cut single beat (fine control)  
- KEEP_PHRASE: Keep entire 4-beat or 8-beat phrase
- CUT_PHRASE: Cut entire phrase
- LOOP_SECTION(n_beats, n_times): Loop a section
- JUMP_BACK(n_phrases): Jump back to repeat earlier content
- EXTEND_SECTION: Mark current section for extension (dynamic looping)
- TRANSITION_SOFT: Use soft crossfade at current edit point
- TRANSITION_HARD: Use hard cut (beat-aligned)

This makes the policy problem harder because:
1. Decisions affect multiple beats (longer horizon thinking needed)
2. More action types to choose from
3. Phrase-level structure must be learned
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple
import numpy as np


class ActionTypeV2(IntEnum):
    """Enhanced action type enumeration."""
    
    # Beat-level actions (fine control)
    KEEP_BEAT = 0
    CUT_BEAT = 1
    
    # Phrase-level actions (coarse control - 4 beats)
    KEEP_BAR = 2      # Keep next 4 beats
    CUT_BAR = 3       # Cut next 4 beats
    
    # Phrase-level actions (coarse control - 8 beats)
    KEEP_PHRASE = 4   # Keep next 8 beats
    CUT_PHRASE = 5    # Cut next 8 beats
    
    # Looping actions
    LOOP_2X = 6       # Loop current beat 2x
    LOOP_4X = 7       # Loop current beat 4x
    LOOP_BAR_2X = 8   # Loop current 4-beat bar 2x
    
    # Section control
    JUMP_BACK_4 = 9   # Jump back 4 beats (repeat section)
    JUMP_BACK_8 = 10  # Jump back 8 beats
    
    # Transition control (affects how edits are rendered)
    MARK_SOFT_TRANSITION = 11  # Mark for soft crossfade at next edit
    MARK_HARD_CUT = 12         # Mark for beat-aligned hard cut


@dataclass
class ActionV2:
    """Enhanced action class with section support."""
    
    action_type: ActionTypeV2
    beat_index: int
    n_beats_affected: int = 1  # How many beats this action affects
    
    def __repr__(self) -> str:
        return f"ActionV2({self.action_type.name}, beat={self.beat_index}, n_beats={self.n_beats_affected})"


class ActionSpaceV2:
    """Enhanced action space with section-level and phrase-level actions.
    
    Total: 13 discrete actions
    
    This makes the problem harder because:
    1. Phrase-level actions affect 4-8 beats at once
    2. Jump actions enable non-linear editing
    3. Transition markers affect rendering quality
    4. More complex action masking needed
    """
    
    # Action layout
    ACTION_KEEP_BEAT = 0
    ACTION_CUT_BEAT = 1
    ACTION_KEEP_BAR = 2
    ACTION_CUT_BAR = 3
    ACTION_KEEP_PHRASE = 4
    ACTION_CUT_PHRASE = 5
    ACTION_LOOP_2X = 6
    ACTION_LOOP_4X = 7
    ACTION_LOOP_BAR_2X = 8
    ACTION_JUMP_BACK_4 = 9
    ACTION_JUMP_BACK_8 = 10
    ACTION_MARK_SOFT = 11
    ACTION_MARK_HARD = 12
    
    N_ACTIONS = 13
    
    def __init__(
        self,
        n_beats: int,
        bar_size: int = 4,
        phrase_size: int = 8,
    ) -> None:
        """Initialize enhanced action space.
        
        Args:
            n_beats: Total number of beats in track
            bar_size: Beats per bar (default 4)
            phrase_size: Beats per phrase (default 8)
        """
        self.n_beats = n_beats
        self.bar_size = bar_size
        self.phrase_size = phrase_size
        self.n_discrete_actions = self.N_ACTIONS
        
        # Transition marker state (persists within episode)
        self.pending_soft_transition = False
        self.pending_hard_cut = False
    
    def reset(self) -> None:
        """Reset action space state (call at episode start)."""
        self.pending_soft_transition = False
        self.pending_hard_cut = False
    
    def decode_action(self, action_idx: int, current_beat: int = 0) -> ActionV2:
        """Decode action index to ActionV2 object.
        
        Args:
            action_idx: Integer action index [0, N_ACTIONS)
            current_beat: Current beat position
            
        Returns:
            ActionV2 object
        """
        if action_idx < 0 or action_idx >= self.N_ACTIONS:
            raise ValueError(f"Action index {action_idx} out of bounds [0, {self.N_ACTIONS})")
        
        remaining_beats = self.n_beats - current_beat
        
        if action_idx == self.ACTION_KEEP_BEAT:
            return ActionV2(ActionTypeV2.KEEP_BEAT, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_CUT_BEAT:
            return ActionV2(ActionTypeV2.CUT_BEAT, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_KEEP_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.KEEP_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_KEEP_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.KEEP_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_LOOP_2X:
            return ActionV2(ActionTypeV2.LOOP_2X, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_LOOP_4X:
            return ActionV2(ActionTypeV2.LOOP_4X, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_LOOP_BAR_2X:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.LOOP_BAR_2X, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_JUMP_BACK_4:
            return ActionV2(ActionTypeV2.JUMP_BACK_4, current_beat, n_beats_affected=0)
        
        elif action_idx == self.ACTION_JUMP_BACK_8:
            return ActionV2(ActionTypeV2.JUMP_BACK_8, current_beat, n_beats_affected=0)
        
        elif action_idx == self.ACTION_MARK_SOFT:
            self.pending_soft_transition = True
            self.pending_hard_cut = False
            return ActionV2(ActionTypeV2.MARK_SOFT_TRANSITION, current_beat, n_beats_affected=0)
        
        elif action_idx == self.ACTION_MARK_HARD:
            self.pending_hard_cut = True
            self.pending_soft_transition = False
            return ActionV2(ActionTypeV2.MARK_HARD_CUT, current_beat, n_beats_affected=0)
        
        raise ValueError(f"Unknown action index: {action_idx}")
    
    def get_action_mask(self, current_beat: int, edit_history: 'EditHistoryV2') -> np.ndarray:
        """Get valid action mask for current state.
        
        Some actions are invalid in certain states:
        - Can't KEEP_PHRASE if < 8 beats remaining
        - Can't JUMP_BACK if we're at the start
        - Can't LOOP if we've already looped too much
        
        Args:
            current_beat: Current beat position
            edit_history: Current edit history
            
        Returns:
            Boolean mask of valid actions [N_ACTIONS]
        """
        mask = np.ones(self.N_ACTIONS, dtype=bool)
        remaining_beats = self.n_beats - current_beat
        
        # Disable phrase-level actions if not enough beats
        if remaining_beats < self.phrase_size:
            mask[self.ACTION_KEEP_PHRASE] = False
            mask[self.ACTION_CUT_PHRASE] = False
        
        if remaining_beats < self.bar_size:
            mask[self.ACTION_KEEP_BAR] = False
            mask[self.ACTION_CUT_BAR] = False
            mask[self.ACTION_LOOP_BAR_2X] = False
        
        # Disable jump-back if not enough history
        if current_beat < 4:
            mask[self.ACTION_JUMP_BACK_4] = False
        if current_beat < 8:
            mask[self.ACTION_JUMP_BACK_8] = False
        
        # Disable excessive looping (check loop ratio)
        n_loops = len(edit_history.looped_beats)
        n_processed = len(edit_history.get_edited_beats())
        if n_processed > 0 and n_loops / n_processed > 0.15:
            mask[self.ACTION_LOOP_2X] = False
            mask[self.ACTION_LOOP_4X] = False
            mask[self.ACTION_LOOP_BAR_2X] = False
        
        return mask
    
    def encode_action(self, action: ActionV2) -> int:
        """Encode ActionV2 object to action index."""
        return action.action_type.value


class EditHistoryV2:
    """Enhanced edit history tracking section-level edits."""
    
    def __init__(self) -> None:
        self.kept_beats: set = set()
        self.cut_beats: set = set()
        self.looped_beats: dict = {}  # beat_idx -> n_times
        self.looped_sections: list = []  # [(start_beat, end_beat, n_times)]
        self.jump_points: list = []  # [(from_beat, to_beat)]
        self.transition_markers: dict = {}  # beat_idx -> 'soft' or 'hard'
        self.section_edits: list = []  # Track section-level decisions
    
    def add_keep(self, beat_idx: int) -> None:
        """Add single beat keep."""
        self.kept_beats.add(beat_idx)
        self.cut_beats.discard(beat_idx)
    
    def add_keep_section(self, start_beat: int, n_beats: int) -> None:
        """Keep multiple beats at once."""
        for i in range(n_beats):
            self.add_keep(start_beat + i)
        self.section_edits.append(('keep', start_beat, n_beats))
    
    def add_cut(self, beat_idx: int) -> None:
        """Add single beat cut."""
        self.cut_beats.add(beat_idx)
        self.kept_beats.discard(beat_idx)
    
    def add_cut_section(self, start_beat: int, n_beats: int) -> None:
        """Cut multiple beats at once."""
        for i in range(n_beats):
            self.add_cut(start_beat + i)
        self.section_edits.append(('cut', start_beat, n_beats))
    
    def add_loop(self, beat_idx: int, n_times: int) -> None:
        """Add beat loop."""
        self.looped_beats[beat_idx] = n_times
        self.add_keep(beat_idx)  # Looped beats are kept
    
    def add_section_loop(self, start_beat: int, n_beats: int, n_times: int) -> None:
        """Loop a section of beats."""
        self.looped_sections.append((start_beat, start_beat + n_beats, n_times))
        for i in range(n_beats):
            self.add_keep(start_beat + i)
    
    def add_jump(self, from_beat: int, to_beat: int) -> None:
        """Add jump point for non-linear editing."""
        self.jump_points.append((from_beat, to_beat))
    
    def set_transition_marker(self, beat_idx: int, marker_type: str) -> None:
        """Set transition marker at beat."""
        self.transition_markers[beat_idx] = marker_type
    
    def get_edited_beats(self) -> set:
        """Get all beats that have been edited."""
        return self.kept_beats | self.cut_beats
    
    def get_keep_ratio(self) -> float:
        """Get ratio of kept beats to edited beats."""
        edited = self.get_edited_beats()
        if not edited:
            return 0.0
        return len(self.kept_beats) / len(edited)
    
    def get_section_decision_count(self) -> int:
        """Count how many section-level (phrase/bar) decisions were made."""
        return len(self.section_edits)

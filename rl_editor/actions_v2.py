"""Enhanced action space for RL-based audio editor (V2).

Key changes from V1:
1. Section-level actions (work on multiple beats at once)
2. Hierarchical decisions (phrase-level and beat-level)
3. More complex musical editing operations

Action Space Structure (39 actions):
- KEEP_BEAT/CUT_BEAT: Single beat control
- KEEP_BAR/CUT_BAR: 4-beat bar control
- KEEP_PHRASE/CUT_PHRASE: 8-beat phrase control
- KEEP_2_BARS/CUT_2_BARS: 8-beat (2 bar) control
- KEEP_2_PHRASES/CUT_2_PHRASES: 16-beat (2 phrase) control
- LOOP_BEAT/LOOP_BAR/LOOP_PHRASE: Loop 2x
- JUMP_BACK_4/JUMP_BACK_8: Jump back to repeat earlier content
- SKIP_TO_NEXT_PHRASE: Jump forward to next phrase boundary
- REORDER_BEAT/REORDER_BAR/REORDER_PHRASE: Move content to end of output
- REPEAT_PREV_BAR/PHRASE: Copy previous bar/phrase here (DJ-style drop)
- SWAP_WITH_NEXT_BAR/PHRASE: Swap current section with next
- FADE_IN/FADE_OUT: Mark sections for fading
- DOUBLE_TIME/HALF_TIME: Speed up or slow down sections
- REVERSE_BAR/REVERSE_PHRASE: Play section backwards
- GAIN_UP_1/GAIN_DOWN_1: Fine volume control (+/-1dB)
- GAIN_UP_3/GAIN_DOWN_3: Volume control (+/-3dB)
- BOOST_LOW/CUT_LOW: Boost/cut bass frequencies
- BOOST_HIGH/CUT_HIGH: Boost/cut treble frequencies
- ADD_DISTORTION/ADD_REVERB: Apply audio effects

This makes the policy problem harder because:
1. Decisions affect multiple beats (longer horizon thinking needed)
2. More action types to choose from
3. Phrase-level structure must be learned
4. Reorder/reverse actions enable creative non-linear arrangements
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
    
    # Bar-level actions (4 beats)
    KEEP_BAR = 2      # Keep next 4 beats
    CUT_BAR = 3       # Cut next 4 beats
    
    # Phrase-level actions (8 beats)
    KEEP_PHRASE = 4   # Keep next 8 beats
    CUT_PHRASE = 5    # Cut next 8 beats
    
    # 2-bar actions (8 beats) - same as phrase but different semantic
    KEEP_2_BARS = 6   # Keep next 8 beats (2 bars)
    CUT_2_BARS = 7    # Cut next 8 beats (2 bars)
    
    # 2-phrase actions (16 beats)
    KEEP_2_PHRASES = 8   # Keep next 16 beats
    CUT_2_PHRASES = 9    # Cut next 16 beats
    
    # Looping actions (all loop 2x by default)
    LOOP_BEAT = 10    # Loop current beat 2x
    LOOP_BAR = 11     # Loop current 4-beat bar 2x
    LOOP_PHRASE = 12  # Loop current 8-beat phrase 2x
    
    # Navigation/Jump actions
    JUMP_BACK_4 = 13          # Jump back 4 beats (repeat section)
    JUMP_BACK_8 = 14          # Jump back 8 beats
    SKIP_TO_NEXT_PHRASE = 15  # Skip forward to next phrase boundary
    
    # Reorder actions (move content to end of output - deferred placement)
    REORDER_BEAT = 16   # Move current beat to end of output
    REORDER_BAR = 17    # Move current 4-beat bar to end of output
    REORDER_PHRASE = 18 # Move current 8-beat phrase to end of output
    
    # Fade actions (mark sections for volume fading)
    FADE_IN = 19       # Mark current bar for fade in
    FADE_OUT = 20      # Mark current bar for fade out
    
    # Time manipulation (speed changes)
    DOUBLE_TIME = 21   # Double speed of current bar (halves duration)
    HALF_TIME = 22     # Half speed of current bar (doubles duration)
    
    # Reverse actions (play backwards)
    REVERSE_BAR = 23     # Play current bar backwards
    REVERSE_PHRASE = 24  # Play current phrase backwards
    
    # Copy/Movement actions (DJ-style techniques)
    REPEAT_PREV_BAR = 25    # Copy previous 4 beats here (drop repeat)
    REPEAT_PREV_PHRASE = 26 # Copy previous 8 beats here
    SWAP_WITH_NEXT_BAR = 27    # Swap this bar with next bar
    SWAP_WITH_NEXT_PHRASE = 28 # Swap this phrase with next phrase
    
    # Volume/Gain actions (applied to 4-beat bars)
    GAIN_UP_1 = 29       # Increase volume +1dB (fine)
    GAIN_DOWN_1 = 30     # Decrease volume -1dB (fine)
    GAIN_UP_3 = 31       # Increase volume +3dB
    GAIN_DOWN_3 = 32     # Decrease volume -3dB
    
    # EQ actions (applied to 4-beat bars)
    BOOST_LOW = 33       # Boost bass frequencies
    CUT_LOW = 34         # Cut bass frequencies
    BOOST_HIGH = 35      # Boost treble frequencies
    CUT_HIGH = 36        # Cut treble frequencies
    
    # Effect actions (applied to 4-beat bars)
    ADD_DISTORTION = 37  # Add mild distortion/saturation
    ADD_REVERB = 38      # Add reverb effect


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
    
    Total: 25 discrete actions
    
    This makes the problem harder because:
    1. Phrase-level actions affect 4-16 beats at once
    2. Jump/skip actions enable non-linear editing
    3. Fade/reverse/time actions add creative possibilities
    4. More complex action masking needed
    """
    
    # Action layout
    ACTION_KEEP_BEAT = 0
    ACTION_CUT_BEAT = 1
    ACTION_KEEP_BAR = 2
    ACTION_CUT_BAR = 3
    ACTION_KEEP_PHRASE = 4
    ACTION_CUT_PHRASE = 5
    ACTION_KEEP_2_BARS = 6
    ACTION_CUT_2_BARS = 7
    ACTION_KEEP_2_PHRASES = 8
    ACTION_CUT_2_PHRASES = 9
    ACTION_LOOP_BEAT = 10
    ACTION_LOOP_BAR = 11
    ACTION_LOOP_PHRASE = 12
    ACTION_JUMP_BACK_4 = 13
    ACTION_JUMP_BACK_8 = 14
    ACTION_SKIP_TO_NEXT_PHRASE = 15
    ACTION_REORDER_BEAT = 16
    ACTION_REORDER_BAR = 17
    ACTION_REORDER_PHRASE = 18
    ACTION_FADE_IN = 19
    ACTION_FADE_OUT = 20
    ACTION_DOUBLE_TIME = 21
    ACTION_HALF_TIME = 22
    ACTION_REVERSE_BAR = 23
    ACTION_REVERSE_PHRASE = 24
    ACTION_REPEAT_PREV_BAR = 25
    ACTION_REPEAT_PREV_PHRASE = 26
    ACTION_SWAP_WITH_NEXT_BAR = 27
    ACTION_SWAP_WITH_NEXT_PHRASE = 28
    ACTION_GAIN_UP_1 = 29
    ACTION_GAIN_DOWN_1 = 30
    ACTION_GAIN_UP_3 = 31
    ACTION_GAIN_DOWN_3 = 32
    ACTION_BOOST_LOW = 33
    ACTION_CUT_LOW = 34
    ACTION_BOOST_HIGH = 35
    ACTION_CUT_HIGH = 36
    ACTION_ADD_DISTORTION = 37
    ACTION_ADD_REVERB = 38
    
    N_ACTIONS = 39
    
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
        self.two_bar_size = bar_size * 2  # 8 beats
        self.two_phrase_size = phrase_size * 2  # 16 beats
    
    def reset(self) -> None:
        """Reset action space state (call at episode start)."""
        pass  # No stateful tracking needed anymore
    
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
        
        elif action_idx == self.ACTION_KEEP_2_BARS:
            n_beats = min(self.two_bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.KEEP_2_BARS, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_2_BARS:
            n_beats = min(self.two_bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_2_BARS, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_KEEP_2_PHRASES:
            n_beats = min(self.two_phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.KEEP_2_PHRASES, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_2_PHRASES:
            n_beats = min(self.two_phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_2_PHRASES, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_LOOP_BEAT:
            return ActionV2(ActionTypeV2.LOOP_BEAT, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_LOOP_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.LOOP_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_LOOP_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.LOOP_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_JUMP_BACK_4:
            return ActionV2(ActionTypeV2.JUMP_BACK_4, current_beat, n_beats_affected=0)
        
        elif action_idx == self.ACTION_JUMP_BACK_8:
            return ActionV2(ActionTypeV2.JUMP_BACK_8, current_beat, n_beats_affected=0)
        
        elif action_idx == self.ACTION_SKIP_TO_NEXT_PHRASE:
            # Calculate beats to skip to reach next phrase boundary
            phrase_position = current_beat % self.phrase_size
            beats_to_skip = self.phrase_size - phrase_position if phrase_position > 0 else self.phrase_size
            beats_to_skip = min(beats_to_skip, remaining_beats)
            return ActionV2(ActionTypeV2.SKIP_TO_NEXT_PHRASE, current_beat, n_beats_affected=beats_to_skip)
        
        elif action_idx == self.ACTION_REORDER_BEAT:
            return ActionV2(ActionTypeV2.REORDER_BEAT, current_beat, n_beats_affected=1)
        
        elif action_idx == self.ACTION_REORDER_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.REORDER_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_REORDER_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.REORDER_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_FADE_IN:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.FADE_IN, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_FADE_OUT:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.FADE_OUT, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_DOUBLE_TIME:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.DOUBLE_TIME, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_HALF_TIME:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.HALF_TIME, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_REVERSE_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.REVERSE_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_REVERSE_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.REVERSE_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_REPEAT_PREV_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.REPEAT_PREV_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_REPEAT_PREV_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.REPEAT_PREV_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_SWAP_WITH_NEXT_BAR:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.SWAP_WITH_NEXT_BAR, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_SWAP_WITH_NEXT_PHRASE:
            n_beats = min(self.phrase_size, remaining_beats)
            return ActionV2(ActionTypeV2.SWAP_WITH_NEXT_PHRASE, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_GAIN_UP_1:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.GAIN_UP_1, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_GAIN_DOWN_1:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.GAIN_DOWN_1, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_GAIN_UP_3:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.GAIN_UP_3, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_GAIN_DOWN_3:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.GAIN_DOWN_3, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_BOOST_LOW:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.BOOST_LOW, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_LOW:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_LOW, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_BOOST_HIGH:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.BOOST_HIGH, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_CUT_HIGH:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.CUT_HIGH, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_ADD_DISTORTION:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.ADD_DISTORTION, current_beat, n_beats_affected=n_beats)
        
        elif action_idx == self.ACTION_ADD_REVERB:
            n_beats = min(self.bar_size, remaining_beats)
            return ActionV2(ActionTypeV2.ADD_REVERB, current_beat, n_beats_affected=n_beats)
        
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
        
        # Disable 2-phrase actions if not enough beats (16)
        if remaining_beats < self.two_phrase_size:
            mask[self.ACTION_KEEP_2_PHRASES] = False
            mask[self.ACTION_CUT_2_PHRASES] = False
        
        # Disable 2-bar actions if not enough beats (8)
        if remaining_beats < self.two_bar_size:
            mask[self.ACTION_KEEP_2_BARS] = False
            mask[self.ACTION_CUT_2_BARS] = False
        
        # Disable phrase-level actions if not enough beats (8)
        if remaining_beats < self.phrase_size:
            mask[self.ACTION_KEEP_PHRASE] = False
            mask[self.ACTION_CUT_PHRASE] = False
            mask[self.ACTION_LOOP_PHRASE] = False
            mask[self.ACTION_REORDER_PHRASE] = False
            mask[self.ACTION_REVERSE_PHRASE] = False
        
        # Disable bar-level actions if not enough beats (4)
        if remaining_beats < self.bar_size:
            mask[self.ACTION_KEEP_BAR] = False
            mask[self.ACTION_CUT_BAR] = False
            mask[self.ACTION_LOOP_BAR] = False
            mask[self.ACTION_REORDER_BAR] = False
            mask[self.ACTION_FADE_IN] = False
            mask[self.ACTION_FADE_OUT] = False
            mask[self.ACTION_DOUBLE_TIME] = False
            mask[self.ACTION_HALF_TIME] = False
            mask[self.ACTION_REVERSE_BAR] = False
            # Audio effect actions also need 4 beats
            mask[self.ACTION_GAIN_UP_1] = False
            mask[self.ACTION_GAIN_DOWN_1] = False
            mask[self.ACTION_GAIN_UP_3] = False
            mask[self.ACTION_GAIN_DOWN_3] = False
            mask[self.ACTION_BOOST_LOW] = False
            mask[self.ACTION_CUT_LOW] = False
            mask[self.ACTION_BOOST_HIGH] = False
            mask[self.ACTION_CUT_HIGH] = False
            mask[self.ACTION_ADD_DISTORTION] = False
            mask[self.ACTION_ADD_REVERB] = False
        
        # Disable repeat previous bar if no history (need 4 beats behind)
        if current_beat < self.bar_size:
            mask[self.ACTION_REPEAT_PREV_BAR] = False
        
        # Disable repeat previous phrase if not enough history (need 8 beats behind)
        if current_beat < self.phrase_size:
            mask[self.ACTION_REPEAT_PREV_PHRASE] = False
        
        # Disable swap with next bar if not enough beats ahead (need 8 beats: current bar + next bar)
        if remaining_beats < self.bar_size * 2:
            mask[self.ACTION_SWAP_WITH_NEXT_BAR] = False
        
        # Disable swap with next phrase if not enough beats ahead (need 16 beats)
        if remaining_beats < self.phrase_size * 2:
            mask[self.ACTION_SWAP_WITH_NEXT_PHRASE] = False
        
        # Disable jump-back if not enough history
        if current_beat < 4:
            mask[self.ACTION_JUMP_BACK_4] = False
        if current_beat < 8:
            mask[self.ACTION_JUMP_BACK_8] = False
        
        # Disable skip if we're at a phrase boundary already or near end
        phrase_position = current_beat % self.phrase_size
        if phrase_position == 0 or remaining_beats < 2:
            mask[self.ACTION_SKIP_TO_NEXT_PHRASE] = False
        
        # Disable excessive looping (check loop ratio)
        n_loops = len(edit_history.looped_beats)
        n_processed = len(edit_history.get_edited_beats())
        if n_processed > 0 and n_loops / n_processed > 0.15:
            mask[self.ACTION_LOOP_BEAT] = False
            mask[self.ACTION_LOOP_BAR] = False
            mask[self.ACTION_LOOP_PHRASE] = False
        
        # Disable excessive reordering (check reorder ratio)
        n_reordered = len(edit_history.reordered_sections)
        if n_processed > 0 and n_reordered / max(1, n_processed // 4) > 0.25:
            mask[self.ACTION_REORDER_BEAT] = False
            mask[self.ACTION_REORDER_BAR] = False
            mask[self.ACTION_REORDER_PHRASE] = False
        
        # Disable excessive fading (max 3 fade markers per episode)
        n_fades = len(getattr(edit_history, 'fade_markers', []))
        if n_fades >= 3:
            mask[self.ACTION_FADE_IN] = False
            mask[self.ACTION_FADE_OUT] = False
        
        # Disable excessive time changes (max 2 per episode)
        n_time_changes = len(getattr(edit_history, 'time_changes', []))
        if n_time_changes >= 2:
            mask[self.ACTION_DOUBLE_TIME] = False
            mask[self.ACTION_HALF_TIME] = False
        
        # Disable excessive reverses (max 2 per episode)
        n_reverses = len(getattr(edit_history, 'reversed_sections', []))
        if n_reverses >= 2:
            mask[self.ACTION_REVERSE_BAR] = False
            mask[self.ACTION_REVERSE_PHRASE] = False
        
        # Disable excessive gain changes (max 4 per episode)
        n_gain = len(getattr(edit_history, 'gain_changes', []))
        if n_gain >= 4:
            mask[self.ACTION_GAIN_UP_1] = False
            mask[self.ACTION_GAIN_DOWN_1] = False
            mask[self.ACTION_GAIN_UP_3] = False
            mask[self.ACTION_GAIN_DOWN_3] = False
        
        # Disable excessive repeats (max 3 per episode)
        n_repeats = len(getattr(edit_history, 'repeated_sections', []))
        if n_repeats >= 3:
            mask[self.ACTION_REPEAT_PREV_BAR] = False
            mask[self.ACTION_REPEAT_PREV_PHRASE] = False
        
        # Disable excessive swaps (max 2 per episode)
        n_swaps = len(getattr(edit_history, 'swapped_sections', []))
        if n_swaps >= 2:
            mask[self.ACTION_SWAP_WITH_NEXT_BAR] = False
            mask[self.ACTION_SWAP_WITH_NEXT_PHRASE] = False
        
        # Disable excessive EQ changes (max 4 per episode)
        n_eq = len(getattr(edit_history, 'eq_changes', []))
        if n_eq >= 4:
            mask[self.ACTION_BOOST_LOW] = False
            mask[self.ACTION_CUT_LOW] = False
            mask[self.ACTION_BOOST_HIGH] = False
            mask[self.ACTION_CUT_HIGH] = False
        
        # Disable excessive effects (max 3 per episode)
        n_effects = len(getattr(edit_history, 'audio_effects', []))
        if n_effects >= 3:
            mask[self.ACTION_ADD_DISTORTION] = False
            mask[self.ACTION_ADD_REVERB] = False
        
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
        self.skip_points: list = []  # [(from_beat, to_beat)] - forward skips
        self.transition_markers: dict = {}  # beat_idx -> 'soft' or 'hard'
        self.section_edits: list = []  # Track section-level decisions
        self.reordered_sections: list = []  # [(start_beat, n_beats)] - sections to append at end
        self.fade_markers: list = []  # [(start_beat, n_beats, 'in'|'out')]
        self.time_changes: list = []  # [(start_beat, n_beats, factor)] - factor: 0.5 or 2.0
        self.reversed_sections: list = []  # [(start_beat, n_beats)]
        self.gain_changes: list = []  # [(start_beat, n_beats, db_change)] - db: +1, -1, +3, or -3
        self.eq_changes: list = []  # [(start_beat, n_beats, eq_type)] - eq_type: 'boost_low', 'cut_low', etc.
        self.audio_effects: list = []  # [(start_beat, n_beats, effect_type)] - 'distortion', 'reverb'
        self.repeated_sections: list = []  # [(source_beat, n_beats)] - copy previous section here
        self.swapped_sections: list = []  # [(beat1, beat2, n_beats)] - swap two sections
    
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
    
    def add_reorder(self, start_beat: int, n_beats: int) -> None:
        """Add section to be reordered (placed at end of output).
        
        These beats are NOT kept in their original position - they will
        be appended to the end of the final output during rendering.
        """
        self.reordered_sections.append((start_beat, n_beats))
        # Mark these beats as processed but not kept in-place
        for i in range(n_beats):
            # Remove from kept/cut since they'll be placed elsewhere
            self.kept_beats.discard(start_beat + i)
            self.cut_beats.discard(start_beat + i)
    
    def set_transition_marker(self, beat_idx: int, marker_type: str) -> None:
        """Set transition marker at beat."""
        self.transition_markers[beat_idx] = marker_type
    
    def get_edited_beats(self) -> set:
        """Get all beats that have been edited (including reordered)."""
        reordered_beats = set()
        for start_beat, n_beats in self.reordered_sections:
            for i in range(n_beats):
                reordered_beats.add(start_beat + i)
        return self.kept_beats | self.cut_beats | reordered_beats
    
    def get_keep_ratio(self) -> float:
        """Get ratio of kept beats to edited beats."""
        edited = self.get_edited_beats()
        if not edited:
            return 0.0
        return len(self.kept_beats) / len(edited)
    
    def get_section_decision_count(self) -> int:
        """Count how many section-level (phrase/bar) decisions were made."""
        return len(self.section_edits)

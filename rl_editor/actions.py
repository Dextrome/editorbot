"""Factored Action Space for RL-based audio editor.

Instead of a single discrete action (39+ values), we use 3 output heads:
1. Action Type (20 types): WHAT to do
2. Action Size (5 sizes): HOW MANY beats
3. Action Amount (5 amounts): HOW MUCH (intensity/direction)

This gives us 20 * 5 * 5 = 500 possible action combinations from just 30 outputs.
The model learns "what" separately from "how much", making it easier to generalize.

Example combinations:
- (KEEP, PHRASE, NONE) -> Keep 8 beats
- (GAIN, BAR, +3dB) -> Boost 4 beats by 3dB
- (FADE_OUT, PHRASE, NONE) -> Fade out over 8 beats
- (JUMP_BACK, NONE, 8_BEATS) -> Jump back 8 beats
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple, Dict, NamedTuple
import numpy as np


class ActionType(IntEnum):
    """What kind of action to take (20 types)."""
    # Core editing
    KEEP = 0        # Keep audio segment
    CUT = 1         # Cut/remove audio segment
    LOOP = 2        # Loop segment 2x
    REORDER = 3     # Move to end of output
    
    # Navigation
    JUMP_BACK = 4   # Jump backwards (repeat earlier content)
    SKIP = 5        # Skip forward to next phrase boundary
    
    # Effects - Volume/Dynamics
    FADE_IN = 6     # Fade in effect
    FADE_OUT = 7    # Fade out effect
    GAIN = 8        # Volume change
    
    # Effects - Time/Speed (amount determines intensity)
    SPEED_UP = 9      # Speed up (amount: small=1.25x, large=2x)
    SPEED_DOWN = 10   # Slow down (amount: small=0.75x, large=0.5x)
    REVERSE = 11      # Play backwards
    
    # Effects - Pitch (amount determines intensity)
    PITCH_UP = 12     # Pitch up (amount: small=+3, large=+6 semitones)
    PITCH_DOWN = 13   # Pitch down (amount: small=-3, large=-6 semitones)
    
    # Effects - EQ
    EQ_LOW = 14     # Bass boost/cut
    EQ_HIGH = 15    # Treble boost/cut
    
    # Effects - Creative
    DISTORTION = 16 # Distortion effect
    REVERB = 17     # Reverb effect
    
    # Movement/Copy
    REPEAT_PREV = 18  # Copy previous section here (drop/repeat)
    SWAP_NEXT = 19    # Swap with next section


class ActionSize(IntEnum):
    """How many beats to affect (5 sizes)."""
    BEAT = 0       # 1 beat
    BAR = 1        # 4 beats (1 bar)
    PHRASE = 2     # 8 beats (1 phrase / 2 bars)
    TWO_BARS = 3   # 8 beats (semantically different from phrase)
    TWO_PHRASES = 4  # 16 beats (4 bars)


class ActionAmount(IntEnum):
    """Intensity or direction (5 amounts)."""
    # For GAIN: volume change in dB
    # For JUMP_BACK: how far back (4 or 8 beats)
    # For EQ: boost or cut
    NEG_LARGE = 0   # -3 dB, or back 8 beats, or CUT freq
    NEG_SMALL = 1   # -1 dB, or back 4 beats
    NEUTRAL = 2     # No change / default
    POS_SMALL = 3   # +1 dB
    POS_LARGE = 4   # +3 dB, or BOOST freq


# Constants for easy access
N_ACTION_TYPES = len(ActionType)      # 20
N_ACTION_SIZES = len(ActionSize)      # 5
N_ACTION_AMOUNTS = len(ActionAmount)  # 5

# Beat sizes for each ActionSize
SIZE_TO_BEATS = {
    ActionSize.BEAT: 1,
    ActionSize.BAR: 4,
    ActionSize.PHRASE: 8,
    ActionSize.TWO_BARS: 8,
    ActionSize.TWO_PHRASES: 16,
}

# Amount to dB mapping for GAIN
AMOUNT_TO_DB = {
    ActionAmount.NEG_LARGE: -3.0,
    ActionAmount.NEG_SMALL: -1.0,
    ActionAmount.NEUTRAL: 0.0,
    ActionAmount.POS_SMALL: 1.0,
    ActionAmount.POS_LARGE: 3.0,
}

# Amount to jump size mapping for JUMP_BACK
AMOUNT_TO_JUMP = {
    ActionAmount.NEG_LARGE: 8,  # Back 8 beats
    ActionAmount.NEG_SMALL: 4,  # Back 4 beats
    ActionAmount.NEUTRAL: 4,    # Default to 4
    ActionAmount.POS_SMALL: 4,
    ActionAmount.POS_LARGE: 8,
}

# Amount to speed factor mapping for SPEED_UP
AMOUNT_TO_SPEED_UP = {
    ActionAmount.NEG_LARGE: 1.1,   # Minimal speed up
    ActionAmount.NEG_SMALL: 1.15,
    ActionAmount.NEUTRAL: 1.25,    # Default
    ActionAmount.POS_SMALL: 1.5,
    ActionAmount.POS_LARGE: 2.0,   # Double speed
}

# Amount to speed factor mapping for SPEED_DOWN
AMOUNT_TO_SPEED_DOWN = {
    ActionAmount.NEG_LARGE: 0.5,   # Half speed
    ActionAmount.NEG_SMALL: 0.6,
    ActionAmount.NEUTRAL: 0.75,    # Default
    ActionAmount.POS_SMALL: 0.85,
    ActionAmount.POS_LARGE: 0.9,   # Minimal slow down
}

# Amount to semitones mapping for PITCH_UP
AMOUNT_TO_PITCH_UP = {
    ActionAmount.NEG_LARGE: 1,     # +1 semitone
    ActionAmount.NEG_SMALL: 2,     # +2 semitones
    ActionAmount.NEUTRAL: 3,       # +3 semitones (default)
    ActionAmount.POS_SMALL: 5,     # +5 semitones
    ActionAmount.POS_LARGE: 6,     # +6 semitones (tritone)
}

# Amount to semitones mapping for PITCH_DOWN
AMOUNT_TO_PITCH_DOWN = {
    ActionAmount.NEG_LARGE: -6,    # -6 semitones (tritone down)
    ActionAmount.NEG_SMALL: -5,    # -5 semitones
    ActionAmount.NEUTRAL: -3,      # -3 semitones (default)
    ActionAmount.POS_SMALL: -2,    # -2 semitones
    ActionAmount.POS_LARGE: -1,    # -1 semitone
}


class FactoredAction(NamedTuple):
    """A factored action with type, size, and amount."""
    action_type: ActionType
    action_size: ActionSize
    action_amount: ActionAmount
    beat_index: int = 0
    
    @property
    def n_beats(self) -> int:
        """Number of beats affected by this action."""
        return SIZE_TO_BEATS[self.action_size]
    
    @property
    def db_change(self) -> float:
        """dB change for GAIN actions."""
        return AMOUNT_TO_DB[self.action_amount]
    
    @property
    def jump_beats(self) -> int:
        """Jump distance for JUMP_BACK actions."""
        return AMOUNT_TO_JUMP[self.action_amount]
    
    @property
    def speed_factor(self) -> float:
        """Speed factor for SPEED_UP/SPEED_DOWN actions."""
        if self.action_type == ActionType.SPEED_UP:
            return AMOUNT_TO_SPEED_UP[self.action_amount]
        elif self.action_type == ActionType.SPEED_DOWN:
            return AMOUNT_TO_SPEED_DOWN[self.action_amount]
        return 1.0
    
    @property
    def pitch_semitones(self) -> int:
        """Pitch shift in semitones for PITCH_UP/PITCH_DOWN actions."""
        if self.action_type == ActionType.PITCH_UP:
            return AMOUNT_TO_PITCH_UP[self.action_amount]
        elif self.action_type == ActionType.PITCH_DOWN:
            return AMOUNT_TO_PITCH_DOWN[self.action_amount]
        return 0
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to integer tuple for network output."""
        return (self.action_type.value, self.action_size.value, self.action_amount.value)


@dataclass
class FactoredActionSpace:
    """Factored action space with 3 output heads.
    
    Handles:
    - Valid action masks for each head
    - Decoding (type, size, amount) tuples to actions
    - Tracking edit history
    """
    
    n_beats: int
    beat_size: int = 1
    bar_size: int = 4
    phrase_size: int = 8
    two_bar_size: int = 8
    two_phrase_size: int = 16
    
    def __post_init__(self):
        self.current_beat = 0
        self.reset()
    
    def reset(self):
        """Reset action space for new episode."""
        self.current_beat = 0
    
    def decode_action(
        self,
        action_type: int,
        action_size: int,
        action_amount: int,
        beat_index: int
    ) -> FactoredAction:
        """Decode network outputs to FactoredAction."""
        return FactoredAction(
            action_type=ActionType(action_type),
            action_size=ActionSize(action_size),
            action_amount=ActionAmount(action_amount),
            beat_index=beat_index,
        )
    
    def encode_action(self, action: FactoredAction) -> Tuple[int, int, int]:
        """Encode FactoredAction to (type, size, amount) integers.
        
        Args:
            action: FactoredAction to encode
            
        Returns:
            Tuple of (action_type, action_size, action_amount) as integers
        """
        return (
            action.action_type.value,
            action.action_size.value,
            action.action_amount.value,
        )
    
    def get_type_mask(
        self,
        current_beat: int,
        edit_history: 'EditHistoryFactored',
    ) -> np.ndarray:
        """Get mask of valid action types.
        
        Returns:
            Boolean mask of shape [N_ACTION_TYPES]
        """
        mask = np.ones(N_ACTION_TYPES, dtype=bool)
        remaining_beats = self.n_beats - current_beat
        
        # Disable JUMP_BACK if at start
        if current_beat < 4:
            mask[ActionType.JUMP_BACK] = False
        
        # Disable SKIP if at phrase boundary or near end
        phrase_position = current_beat % self.phrase_size
        if phrase_position == 0 or remaining_beats < 2:
            mask[ActionType.SKIP] = False
        
        # Disable REPEAT_PREV if no history
        if current_beat < self.bar_size:
            mask[ActionType.REPEAT_PREV] = False
        
        # Disable SWAP_NEXT if not enough beats ahead
        if remaining_beats < self.bar_size * 2:
            mask[ActionType.SWAP_NEXT] = False
        
        # Disable excessive looping
        n_loops = len(edit_history.looped_sections)
        n_processed = len(edit_history.kept_beats) + len(edit_history.cut_beats)
        if n_processed > 0 and n_loops / max(1, n_processed // 4) > 0.15:
            mask[ActionType.LOOP] = False
        
        # Disable excessive reordering (allow up to 1 reorder per 2 processed beats)
        n_reordered = len(edit_history.reordered_sections)
        if n_processed > 0 and n_reordered / max(1, n_processed // 4) > 2:
            mask[ActionType.REORDER] = False
        
        # Disable excessive effects (use counters from history)
        if len(edit_history.fade_markers) >= 8:
            mask[ActionType.FADE_IN] = False
            mask[ActionType.FADE_OUT] = False

        # Mask KEEP if projected keep ratio would exceed 0.95
        keep_ratio = edit_history.get_keep_ratio()
        if keep_ratio > 0.95:
            mask[ActionType.KEEP] = False
            mask[ActionType.LOOP] = False
            mask[ActionType.REPEAT_PREV] = False
            mask[ActionType.SKIP] = False                
            mask[ActionType.SPEED_DOWN] = False
        elif keep_ratio > 0.85:
            mask[ActionType.LOOP] = False
            mask[ActionType.REPEAT_PREV] = False
            mask[ActionType.SKIP] = False                
            mask[ActionType.SPEED_DOWN] = False
        
        return mask
    
    def get_size_mask(
        self,
        current_beat: int,
        action_type: ActionType,
    ) -> np.ndarray:
        """Get mask of valid sizes for given action type.
        
        Some action types only work with certain sizes.
        
        Returns:
            Boolean mask of shape [N_ACTION_SIZES]
        """
        mask = np.ones(N_ACTION_SIZES, dtype=bool)
        remaining_beats = self.n_beats - current_beat
        
        # Disable sizes that exceed remaining beats
        if remaining_beats < 1:
            mask[ActionSize.BEAT] = False
        if remaining_beats < self.bar_size:
            mask[ActionSize.BAR] = False
        if remaining_beats < self.phrase_size:
            mask[ActionSize.PHRASE] = False
        if remaining_beats < self.two_bar_size:
            mask[ActionSize.TWO_BARS] = False
        if remaining_beats < self.two_phrase_size:
            mask[ActionSize.TWO_PHRASES] = False
        
        # JUMP_BACK and SKIP ignore size (use amount instead)
        if action_type in (ActionType.JUMP_BACK, ActionType.SKIP):
            # Allow any size, but prefer BEAT as default (minimal effect)
            pass
        
        # SWAP_NEXT needs double the size available
        if action_type == ActionType.SWAP_NEXT:
            if remaining_beats < self.bar_size * 2:
                mask[ActionSize.BAR] = False
            if remaining_beats < self.phrase_size * 2:
                mask[ActionSize.PHRASE] = False
                mask[ActionSize.TWO_BARS] = False
            if remaining_beats < self.two_phrase_size * 2:
                mask[ActionSize.TWO_PHRASES] = False
        
        return mask
    
    def get_amount_mask(
        self,
        current_beat: int,
        action_type: ActionType,
    ) -> np.ndarray:
        """Get mask of valid amounts for given action type.
        
        Most actions ignore amount, so all are valid.
        For GAIN/EQ/JUMP, specific amounts are meaningful.
        
        Returns:
            Boolean mask of shape [N_ACTION_AMOUNTS]
        """
        mask = np.ones(N_ACTION_AMOUNTS, dtype=bool)
        
        # JUMP_BACK uses amount for distance
        if action_type == ActionType.JUMP_BACK:
            if current_beat < 8:
                mask[ActionAmount.NEG_LARGE] = False  # Can't jump back 8 if < 8 beats in
                mask[ActionAmount.POS_LARGE] = False
            if current_beat < 4:
                mask[ActionAmount.NEG_SMALL] = False
                mask[ActionAmount.POS_SMALL] = False
                mask[ActionAmount.NEUTRAL] = False
        
        return mask
    
    def get_combined_mask(
        self,
        current_beat: int,
        edit_history: 'EditHistoryFactored',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all three masks at once.
        
        Returns:
            Tuple of (type_mask, size_mask, amount_mask)
            Note: size_mask and amount_mask are computed for all types,
            but will be refined based on actual type selected.
        """
        type_mask = self.get_type_mask(current_beat, edit_history)
        
        # Default size/amount masks (will be refined based on type)
        # Use most permissive masks here
        remaining_beats = self.n_beats - current_beat
        
        size_mask = np.ones(N_ACTION_SIZES, dtype=bool)
        if remaining_beats < 1:
            size_mask[ActionSize.BEAT] = False
        if remaining_beats < self.bar_size:
            size_mask[ActionSize.BAR] = False
        if remaining_beats < self.phrase_size:
            size_mask[ActionSize.PHRASE] = False
        if remaining_beats < self.two_bar_size:
            size_mask[ActionSize.TWO_BARS] = False
        if remaining_beats < self.two_phrase_size:
            size_mask[ActionSize.TWO_PHRASES] = False
        
        amount_mask = np.ones(N_ACTION_AMOUNTS, dtype=bool)
        if current_beat < 8:
            # Restrict large jumps
            pass  # Keep all enabled for simplicity
        
        return type_mask, size_mask, amount_mask


class EditHistoryFactored:
    """Edit history tracking for factored actions."""
    
    def __init__(self) -> None:
        self.kept_beats: set = set()
        self.cut_beats: set = set()
        self.looped_beats: dict = {}  # beat_idx -> n_times (individual beat loops)
        self.looped_sections: list = []  # [(start_beat, n_beats, times)]
        self.jump_points: list = []  # [(from_beat, to_beat)]
        self.skip_points: list = []  # [(from_beat, to_beat)]
        self.reordered_sections: list = []  # [(start_beat, n_beats)]
        self.reordered_beats: set = set()  # Track reordered beats for get_edited_beats
        self.fade_markers: list = []  # [(start_beat, n_beats, 'in'|'out')]
        self.time_changes: list = []  # [(start_beat, n_beats, factor)]
        self.pitch_changes: list = []  # [(start_beat, n_beats, semitones)]
        self.reversed_sections: list = []  # [(start_beat, n_beats)]
        self.gain_changes: list = []  # [(start_beat, n_beats, db_change)]
        self.eq_changes: list = []  # [(start_beat, n_beats, eq_type)]
        self.audio_effects: list = []  # [(start_beat, n_beats, effect_type)]
        self.repeated_sections: list = []  # [(source_beat, n_beats)]
        self.swapped_sections: list = []  # [(beat1, beat2, n_beats)]
        self.section_edits: list = []  # Track section-level decisions
        self.transition_markers: dict = {}  # beat_idx -> 'soft' or 'hard'
    
    def add_keep(self, beat_idx: int) -> None:
        self.kept_beats.add(beat_idx)
        self.cut_beats.discard(beat_idx)
    
    def add_cut(self, beat_idx: int) -> None:
        self.cut_beats.add(beat_idx)
        self.kept_beats.discard(beat_idx)
    
    def add_keep_section(self, start_beat: int, n_beats: int) -> None:
        for i in range(n_beats):
            self.add_keep(start_beat + i)
        self.section_edits.append(("KEEP", start_beat, n_beats))
    
    def add_cut_section(self, start_beat: int, n_beats: int) -> None:
        for i in range(n_beats):
            self.add_cut(start_beat + i)
        self.section_edits.append(("CUT", start_beat, n_beats))
    
    def add_loop(self, start_beat: int, n_beats: int, times: int = 2) -> None:
        self.looped_sections.append((start_beat, n_beats, times))
        for i in range(n_beats):
            self.add_keep(start_beat + i)
    
    def add_jump(self, from_beat: int, to_beat: int) -> None:
        self.jump_points.append((from_beat, to_beat))
    
    def add_reorder(self, start_beat: int, n_beats: int) -> None:
        self.reordered_sections.append((start_beat, n_beats))
        # Reordered beats are neither kept nor cut in place
        for i in range(n_beats):
            beat_idx = start_beat + i
            self.reordered_beats.add(beat_idx)
            self.kept_beats.discard(beat_idx)
            self.cut_beats.discard(beat_idx)
    
    def add_section_loop(self, start_beat: int, n_beats: int, n_times: int) -> None:
        """Add a section loop (separate from single beat loops)."""
        self.looped_sections.append((start_beat, n_beats, n_times))
        for i in range(n_beats):
            self.add_keep(start_beat + i)
    
    def set_transition_marker(self, beat_idx: int, marker_type: str) -> None:
        """Set transition marker at beat.
        
        Args:
            beat_idx: Beat index
            marker_type: 'soft' or 'hard'
        """
        self.transition_markers[beat_idx] = marker_type
    
    def get_edited_beats(self) -> set:
        return self.kept_beats | self.cut_beats | self.reordered_beats
    
    def get_keep_ratio(self) -> float:
        total = len(self.kept_beats) + len(self.cut_beats)
        if total == 0:
            # No decisions made yet - return neutral ratio to avoid biased penalties
            return 0.5
        return len(self.kept_beats) / total
    
    def get_section_decision_count(self) -> int:
        """Get count of section-level decisions made."""
        return len(self.section_edits)
    
    def reset(self) -> None:
        self.__init__()


def apply_factored_action(
    action: FactoredAction,
    current_beat: int,
    edit_history: EditHistoryFactored,
    n_beats: int,
    action_space: FactoredActionSpace,
) -> int:
    """Apply a factored action and return beats to advance.
    
    Args:
        action: The factored action to apply
        current_beat: Current beat position
        edit_history: Edit history to update
        n_beats: Total beats in track
        action_space: Action space reference
    
    Returns:
        Number of beats to advance
    """
    atype = action.action_type
    asize = action.action_size
    aamount = action.action_amount
    n_beats_affected = action.n_beats
    
    # Clamp to remaining beats
    remaining = n_beats - current_beat
    n_beats_affected = min(n_beats_affected, remaining)
    if n_beats_affected <= 0:
        return 1
    
    # KEEP
    if atype == ActionType.KEEP:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        return n_beats_affected
    
    # CUT
    elif atype == ActionType.CUT:
        edit_history.add_cut_section(current_beat, n_beats_affected)
        return n_beats_affected
    
    # LOOP
    elif atype == ActionType.LOOP:
        edit_history.add_loop(current_beat, n_beats_affected, times=2)
        return n_beats_affected
    
    # REORDER
    elif atype == ActionType.REORDER:
        edit_history.add_reorder(current_beat, n_beats_affected)
        return n_beats_affected
    
    # JUMP_BACK
    elif atype == ActionType.JUMP_BACK:
        jump_dist = action.jump_beats
        target = max(0, current_beat - jump_dist)
        edit_history.add_jump(current_beat, target)
        return 1  # Minimal advance, jump is for rendering
    
    # SKIP
    elif atype == ActionType.SKIP:
        # Skip to next phrase boundary
        phrase_size = action_space.phrase_size
        next_phrase = ((current_beat // phrase_size) + 1) * phrase_size
        skip_beats = min(next_phrase - current_beat, remaining)
        for i in range(skip_beats):
            edit_history.add_cut(current_beat + i)
        edit_history.skip_points.append((current_beat, current_beat + skip_beats))
        return skip_beats
    
    # FADE_IN
    elif atype == ActionType.FADE_IN:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.fade_markers.append((current_beat, n_beats_affected, 'in'))
        return n_beats_affected
    
    # FADE_OUT
    elif atype == ActionType.FADE_OUT:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.fade_markers.append((current_beat, n_beats_affected, 'out'))
        return n_beats_affected
    
    # GAIN
    elif atype == ActionType.GAIN:
        db = action.db_change
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.gain_changes.append((current_beat, n_beats_affected, db))
        return n_beats_affected
    
    # SPEED_UP
    elif atype == ActionType.SPEED_UP:
        speed_factor = action.speed_factor
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.time_changes.append((current_beat, n_beats_affected, speed_factor))
        return n_beats_affected
    
    # SPEED_DOWN
    elif atype == ActionType.SPEED_DOWN:
        speed_factor = action.speed_factor
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.time_changes.append((current_beat, n_beats_affected, speed_factor))
        return n_beats_affected
    
    # REVERSE
    elif atype == ActionType.REVERSE:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.reversed_sections.append((current_beat, n_beats_affected))
        return n_beats_affected
    
    # PITCH_UP
    elif atype == ActionType.PITCH_UP:
        semitones = action.pitch_semitones
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.pitch_changes.append((current_beat, n_beats_affected, semitones))
        return n_beats_affected
    
    # PITCH_DOWN
    elif atype == ActionType.PITCH_DOWN:
        semitones = action.pitch_semitones
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.pitch_changes.append((current_beat, n_beats_affected, semitones))
        return n_beats_affected
    
    # EQ_LOW
    elif atype == ActionType.EQ_LOW:
        eq_type = 'boost_low' if aamount >= ActionAmount.NEUTRAL else 'cut_low'
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.eq_changes.append((current_beat, n_beats_affected, eq_type))
        return n_beats_affected
    
    # EQ_HIGH
    elif atype == ActionType.EQ_HIGH:
        eq_type = 'boost_high' if aamount >= ActionAmount.NEUTRAL else 'cut_high'
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.eq_changes.append((current_beat, n_beats_affected, eq_type))
        return n_beats_affected
    
    # DISTORTION
    elif atype == ActionType.DISTORTION:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.audio_effects.append((current_beat, n_beats_affected, 'distortion'))
        return n_beats_affected
    
    # REVERB
    elif atype == ActionType.REVERB:
        edit_history.add_keep_section(current_beat, n_beats_affected)
        edit_history.audio_effects.append((current_beat, n_beats_affected, 'reverb'))
        return n_beats_affected
    
    # REPEAT_PREV
    elif atype == ActionType.REPEAT_PREV:
        # Copy previous section
        source_beat = current_beat - n_beats_affected
        if source_beat >= 0:
            edit_history.repeated_sections.append((source_beat, n_beats_affected))
        edit_history.add_keep_section(current_beat, n_beats_affected)
        return n_beats_affected
    
    # SWAP_NEXT
    elif atype == ActionType.SWAP_NEXT:
        # Swap current section with next
        next_beat = current_beat + n_beats_affected
        if next_beat + n_beats_affected <= n_beats:
            edit_history.swapped_sections.append((current_beat, next_beat, n_beats_affected))
            edit_history.add_keep_section(current_beat, n_beats_affected * 2)
            return n_beats_affected * 2
        else:
            edit_history.add_keep_section(current_beat, n_beats_affected)
            return n_beats_affected
    
    return 1  # Default advance

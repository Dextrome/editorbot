"""Inference script for V2 RL audio editor.

Load a trained V2 model and edit audio files using section-level actions.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter

import numpy as np
import torch
import librosa
import soundfile as sf

from rl_editor.config import get_default_config, Config
from rl_editor.agent import Agent
from rl_editor.state import AudioState
from rl_editor.environment_v2 import AudioEditingEnvV2
from rl_editor.actions_v2 import ActionSpaceV2, ActionTypeV2

# Import utility functions
from rl_editor.infer_utils import load_and_process_audio, apply_crossfade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference_v2(
    agent: Agent,
    config: Config,
    audio_state: AudioState,
    deterministic: bool = True
) -> Tuple[List[int], List[str], List[Dict]]:
    """Run inference on audio state using V2 environment.
    
    Args:
        agent: Trained V2 agent
        config: Configuration
        audio_state: Audio state to process
        deterministic: Whether to use deterministic policy
        
    Returns:
        Tuple of (action_indices, action_names, action_details)
    """
    agent.eval()
    env = AudioEditingEnvV2(config, audio_state=audio_state)
    
    obs, info = env.reset()
    done = False
    actions = []
    action_names = []
    action_details = []
    
    while not done:
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        mask = env.get_action_mask()
        mask_tensor = torch.from_numpy(mask).bool().to(agent.device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = agent.select_action_batch(state_tensor, mask_tensor, deterministic=deterministic)
            action = action[0].item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record action
        actions.append(action)
        action_name = ActionTypeV2(action).name if action < len(ActionTypeV2) else f"ACTION_{action}"
        action_names.append(action_name)
        action_details.append({
            "beat": info.get("beat", 0),
            "action": action_name,
            "reward": reward,
        })
        
        done = terminated or truncated
    
    return actions, action_names, action_details


def decode_action_v2(action_idx: int, current_beat: int, n_beats: int) -> Dict:
    """Decode V2 action index to action info for audio processing.
    
    V2 Actions (39 total):
    - 0: KEEP_BEAT
    - 1: CUT_BEAT
    - 2: KEEP_BAR (4 beats)
    - 3: CUT_BAR (4 beats)
    - 4: KEEP_PHRASE (8 beats)
    - 5: CUT_PHRASE (8 beats)
    - 6: KEEP_2_BARS (8 beats)
    - 7: CUT_2_BARS (8 beats)
    - 8: KEEP_2_PHRASES (16 beats)
    - 9: CUT_2_PHRASES (16 beats)
    - 10: LOOP_BEAT (loop 2x)
    - 11: LOOP_BAR (4 beats looped 2x)
    - 12: LOOP_PHRASE (8 beats looped 2x)
    - 13: JUMP_BACK_4
    - 14: JUMP_BACK_8
    - 15: SKIP_TO_NEXT_PHRASE
    - 16: REORDER_BEAT
    - 17: REORDER_BAR (4 beats)
    - 18: REORDER_PHRASE (8 beats)
    - 19: FADE_IN (4 beats)
    - 20: FADE_OUT (4 beats)
    - 21: DOUBLE_TIME (4 beats)
    - 22: HALF_TIME (4 beats)
    - 23: REVERSE_BAR (4 beats)
    - 24: REVERSE_PHRASE (8 beats)
    - 25: REPEAT_PREV_BAR (copy previous 4 beats)
    - 26: REPEAT_PREV_PHRASE (copy previous 8 beats)
    - 27: SWAP_WITH_NEXT_BAR (swap with next 4 beats)
    - 28: SWAP_WITH_NEXT_PHRASE (swap with next 8 beats)
    - 29: GAIN_UP_1 (+1dB, 4 beats)
    - 30: GAIN_DOWN_1 (-1dB, 4 beats)
    - 31: GAIN_UP_3 (+3dB, 4 beats)
    - 32: GAIN_DOWN_3 (-3dB, 4 beats)
    - 33: BOOST_LOW (bass boost, 4 beats)
    - 34: CUT_LOW (bass cut, 4 beats)
    - 35: BOOST_HIGH (treble boost, 4 beats)
    - 36: CUT_HIGH (treble cut, 4 beats)
    - 37: ADD_DISTORTION (4 beats)
    - 38: ADD_REVERB (4 beats)
    """
    if action_idx == ActionSpaceV2.ACTION_KEEP_BEAT:
        return {"type": "KEEP", "beats": [current_beat]}
    elif action_idx == ActionSpaceV2.ACTION_CUT_BEAT:
        return {"type": "CUT", "beats": [current_beat]}
    elif action_idx == ActionSpaceV2.ACTION_KEEP_BAR:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "KEEP", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_CUT_BAR:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "CUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_KEEP_2_BARS:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "KEEP", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_CUT_2_BARS:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "CUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_KEEP_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "KEEP", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_CUT_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "CUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_KEEP_2_PHRASES:
        end_beat = min(current_beat + 16, n_beats)
        return {"type": "KEEP", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_CUT_2_PHRASES:
        end_beat = min(current_beat + 16, n_beats)
        return {"type": "CUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_BEAT:
        return {"type": "LOOP", "beats": [current_beat], "times": 2}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_BAR:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "LOOP", "beats": list(range(current_beat, end_beat)), "times": 2}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "LOOP", "beats": list(range(current_beat, end_beat)), "times": 2}
    elif action_idx == ActionSpaceV2.ACTION_JUMP_BACK_4:
        return {"type": "JUMP_BACK", "beats": 4}
    elif action_idx == ActionSpaceV2.ACTION_JUMP_BACK_8:
        return {"type": "JUMP_BACK", "beats": 8}
    elif action_idx == ActionSpaceV2.ACTION_SKIP_TO_NEXT_PHRASE:
        # Skip to next phrase boundary (8-beat aligned)
        next_phrase = ((current_beat // 8) + 1) * 8
        beats_to_skip = min(next_phrase - current_beat, n_beats - current_beat)
        return {"type": "SKIP", "beats": list(range(current_beat, current_beat + beats_to_skip))}
    elif action_idx == ActionSpaceV2.ACTION_FADE_IN:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "FADE_IN", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_FADE_OUT:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "FADE_OUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_DOUBLE_TIME:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "DOUBLE_TIME", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_HALF_TIME:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "HALF_TIME", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_REVERSE_BAR:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "REVERSE", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_REVERSE_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "REVERSE", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_REORDER_BEAT:
        return {"type": "REORDER", "beats": [current_beat]}
    elif action_idx == ActionSpaceV2.ACTION_REORDER_BAR:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "REORDER", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_REORDER_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "REORDER", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_REPEAT_PREV_BAR:
        # Copy previous 4 beats and insert here (also keep current 4 beats)
        end_beat = min(current_beat + 4, n_beats)
        source_start = max(0, current_beat - 4)
        return {"type": "REPEAT", "beats": list(range(current_beat, end_beat)), 
                "source_beats": list(range(source_start, source_start + 4))}
    elif action_idx == ActionSpaceV2.ACTION_REPEAT_PREV_PHRASE:
        # Copy previous 8 beats and insert here
        end_beat = min(current_beat + 8, n_beats)
        source_start = max(0, current_beat - 8)
        return {"type": "REPEAT", "beats": list(range(current_beat, end_beat)),
                "source_beats": list(range(source_start, source_start + 8))}
    elif action_idx == ActionSpaceV2.ACTION_SWAP_WITH_NEXT_BAR:
        # Swap current 4 beats with next 4 beats
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "SWAP", "beats": list(range(current_beat, end_beat)), "swap_size": 4}
    elif action_idx == ActionSpaceV2.ACTION_SWAP_WITH_NEXT_PHRASE:
        # Swap current 8 beats with next 8 beats
        end_beat = min(current_beat + 16, n_beats)
        return {"type": "SWAP", "beats": list(range(current_beat, end_beat)), "swap_size": 8}
    elif action_idx == ActionSpaceV2.ACTION_GAIN_UP_1:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "GAIN", "beats": list(range(current_beat, end_beat)), "db": 1.0}
    elif action_idx == ActionSpaceV2.ACTION_GAIN_DOWN_1:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "GAIN", "beats": list(range(current_beat, end_beat)), "db": -1.0}
    elif action_idx == ActionSpaceV2.ACTION_GAIN_UP_3:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "GAIN", "beats": list(range(current_beat, end_beat)), "db": 3.0}
    elif action_idx == ActionSpaceV2.ACTION_GAIN_DOWN_3:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "GAIN", "beats": list(range(current_beat, end_beat)), "db": -3.0}
    elif action_idx == ActionSpaceV2.ACTION_BOOST_LOW:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EQ", "beats": list(range(current_beat, end_beat)), "eq_type": "boost_low"}
    elif action_idx == ActionSpaceV2.ACTION_CUT_LOW:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EQ", "beats": list(range(current_beat, end_beat)), "eq_type": "cut_low"}
    elif action_idx == ActionSpaceV2.ACTION_BOOST_HIGH:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EQ", "beats": list(range(current_beat, end_beat)), "eq_type": "boost_high"}
    elif action_idx == ActionSpaceV2.ACTION_CUT_HIGH:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EQ", "beats": list(range(current_beat, end_beat)), "eq_type": "cut_high"}
    elif action_idx == ActionSpaceV2.ACTION_ADD_DISTORTION:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EFFECT", "beats": list(range(current_beat, end_beat)), "effect": "distortion"}
    elif action_idx == ActionSpaceV2.ACTION_ADD_REVERB:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "EFFECT", "beats": list(range(current_beat, end_beat)), "effect": "reverb"}
    else:
        return {"type": "KEEP", "beats": [current_beat]}


def create_edited_audio_v2(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    actions: List[int],
    crossfade_ms: float = 50.0
) -> np.ndarray:
    """Create edited audio based on V2 actions.
    
    Args:
        audio: Original audio array
        sr: Sample rate
        beat_times: Array of beat times in seconds
        actions: List of action indices
        crossfade_ms: Crossfade duration in milliseconds
        
    Returns:
        Edited audio array
    """
    # Get beat boundaries (samples)
    beat_samples = (beat_times * sr).astype(int)
    beat_samples = np.append(beat_samples, len(audio))
    n_beats = len(beat_times)
    
    crossfade_samples = int(crossfade_ms * sr / 1000)
    
    # Track which beats to keep/loop
    # Default: all beats are initially undecided
    beat_status = {}  # beat_idx -> "KEEP" | "CUT" | ("LOOP", times)
    transition_markers = {}  # beat_idx -> "SOFT" | "HARD"
    reordered_sections = []  # List of (start_beat, n_beats) to append at end
    fade_markers = {}  # beat_idx -> ("IN", n_beats) or ("OUT", n_beats)
    time_changes = {}  # beat_idx -> (n_beats, factor) where factor is 2.0 or 0.5
    reverse_sections = {}  # beat_idx -> n_beats
    gain_changes = {}  # beat_idx -> (n_beats, db_change)
    eq_changes = {}  # beat_idx -> (n_beats, eq_type)
    audio_effects = {}  # beat_idx -> (n_beats, effect_type)
    
    current_beat = 0
    for action_idx in actions:
        if current_beat >= n_beats:
            break
            
        action_info = decode_action_v2(action_idx, current_beat, n_beats)
        action_type = action_info["type"]
        
        if action_type == "KEEP":
            for b in action_info["beats"]:
                if b < n_beats and b not in beat_status:
                    beat_status[b] = "KEEP"
            current_beat = max(action_info["beats"]) + 1 if action_info["beats"] else current_beat + 1
            
        elif action_type == "CUT":
            for b in action_info["beats"]:
                if b < n_beats:
                    beat_status[b] = "CUT"
            current_beat = max(action_info["beats"]) + 1 if action_info["beats"] else current_beat + 1
        
        elif action_type == "SKIP":
            # Skip marks beats as cut (similar to CUT but for phrase-aligned skips)
            for b in action_info["beats"]:
                if b < n_beats:
                    beat_status[b] = "CUT"
            current_beat = max(action_info["beats"]) + 1 if action_info["beats"] else current_beat + 1
            
        elif action_type == "LOOP":
            loop_times = action_info["times"]
            for b in action_info["beats"]:
                if b < n_beats:
                    beat_status[b] = ("LOOP", loop_times)
            current_beat = max(action_info["beats"]) + 1 if action_info["beats"] else current_beat + 1
            
        elif action_type == "REORDER":
            # Mark these beats as reordered (will be appended at end)
            beats = action_info["beats"]
            if beats:
                reordered_sections.append((min(beats), len(beats)))
                for b in beats:
                    if b < n_beats:
                        beat_status[b] = "REORDER"  # Don't keep in place
            current_beat = max(beats) + 1 if beats else current_beat + 1
            
        elif action_type == "JUMP_BACK":
            # Jump back means we'll potentially revisit earlier beats
            jump_beats = action_info["beats"]
            current_beat = max(0, current_beat - jump_beats)
        
        elif action_type == "FADE_IN":
            # Mark section for fade in effect
            beats = action_info["beats"]
            if beats:
                fade_markers[min(beats)] = ("IN", len(beats))
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "FADE_OUT":
            # Mark section for fade out effect
            beats = action_info["beats"]
            if beats:
                fade_markers[min(beats)] = ("OUT", len(beats))
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "DOUBLE_TIME":
            # Mark section for 2x speed
            beats = action_info["beats"]
            if beats:
                time_changes[min(beats)] = (len(beats), 2.0)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "HALF_TIME":
            # Mark section for 0.5x speed
            beats = action_info["beats"]
            if beats:
                time_changes[min(beats)] = (len(beats), 0.5)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "REVERSE":
            # Mark section to be played in reverse
            beats = action_info["beats"]
            if beats:
                reverse_sections[min(beats)] = len(beats)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "GAIN":
            # Mark section for gain change
            beats = action_info["beats"]
            db = action_info.get("db", 0.0)
            if beats:
                gain_changes[min(beats)] = (len(beats), db)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "EQ":
            # Mark section for EQ change
            beats = action_info["beats"]
            eq_type = action_info.get("eq_type", "")
            if beats:
                eq_changes[min(beats)] = (len(beats), eq_type)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "EFFECT":
            # Mark section for audio effect
            beats = action_info["beats"]
            effect = action_info.get("effect", "")
            if beats:
                audio_effects[min(beats)] = (len(beats), effect)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "REPEAT":
            # Copy source beats and also keep current beats
            beats = action_info["beats"]
            source_beats = action_info.get("source_beats", [])
            if beats and source_beats:
                # Mark that we want to insert source_beats before these beats
                # We'll store this in beat_status as a special tuple
                beat_status[min(beats)] = ("REPEAT", source_beats)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
        
        elif action_type == "SWAP":
            # Swap two sections
            beats = action_info["beats"]
            swap_size = action_info.get("swap_size", 4)
            if beats:
                first_start = min(beats)
                second_start = first_start + swap_size
                # Mark as swap - during rendering, we'll output second section first, then first
                beat_status[first_start] = ("SWAP", swap_size)
                # Mark all beats in both sections as part of the swap (keep them)
                for b in beats:
                    if b < n_beats and b not in beat_status:
                        beat_status[b] = "KEEP"
            current_beat = max(beats) + 1 if beats else current_beat + 1
            
        elif action_type == "MARK_SOFT":
            for b in action_info["beats"]:
                transition_markers[b] = "SOFT"
            current_beat += 1
            
        elif action_type == "MARK_HARD":
            for b in action_info["beats"]:
                transition_markers[b] = "HARD"
            current_beat += 1
    
    # Fill in any remaining beats as KEEP by default
    for i in range(n_beats):
        if i not in beat_status:
            beat_status[i] = "KEEP"
    
    # Helper function to apply time stretch (simple resampling)
    def apply_time_stretch(segment: np.ndarray, factor: float) -> np.ndarray:
        """Stretch audio by factor (2.0 = double speed/half length, 0.5 = half speed/double length)."""
        if factor == 1.0:
            return segment
        # Simple linear interpolation for speed change
        original_len = len(segment)
        new_len = int(original_len / factor)
        if new_len <= 0:
            return segment
        indices = np.linspace(0, original_len - 1, new_len)
        return np.interp(indices, np.arange(original_len), segment)
    
    # Helper function to apply fade
    def apply_fade(segment: np.ndarray, fade_type: str) -> np.ndarray:
        """Apply fade in or fade out to segment."""
        n = len(segment)
        if n == 0:
            return segment
        if fade_type == "IN":
            fade = np.linspace(0, 1, n)
        else:  # OUT
            fade = np.linspace(1, 0, n)
        return segment * fade
    
    # Helper function to apply gain change
    def apply_gain(segment: np.ndarray, db: float) -> np.ndarray:
        """Apply gain change in dB to segment."""
        if db == 0.0:
            return segment
        linear_gain = 10 ** (db / 20.0)
        return segment * linear_gain
    
    # Helper function to apply simple EQ (using FFT-based filtering)
    def apply_eq(segment: np.ndarray, eq_type: str) -> np.ndarray:
        """Apply simple EQ to segment."""
        if len(segment) < 256:
            return segment
        
        # Simple biquad-style filtering approximation using FFT
        n = len(segment)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        
        if eq_type == "boost_low":
            # Boost frequencies below 300Hz by 6dB
            boost = np.where(freqs < 300, 2.0, 1.0)
            # Smooth transition
            transition = (freqs >= 200) & (freqs < 400)
            boost[transition] = 2.0 - (freqs[transition] - 200) / 200
        elif eq_type == "cut_low":
            # Cut frequencies below 300Hz by 6dB
            boost = np.where(freqs < 300, 0.5, 1.0)
            transition = (freqs >= 200) & (freqs < 400)
            boost[transition] = 0.5 + (freqs[transition] - 200) / 400
        elif eq_type == "boost_high":
            # Boost frequencies above 4kHz by 6dB
            boost = np.where(freqs > 4000, 2.0, 1.0)
            transition = (freqs >= 3000) & (freqs <= 5000)
            boost[transition] = 1.0 + (freqs[transition] - 3000) / 2000
        elif eq_type == "cut_high":
            # Cut frequencies above 4kHz by 6dB
            boost = np.where(freqs > 4000, 0.5, 1.0)
            transition = (freqs >= 3000) & (freqs <= 5000)
            boost[transition] = 1.0 - (freqs[transition] - 3000) / 4000
        else:
            return segment
        
        fft_modified = fft * boost
        result = np.fft.irfft(fft_modified, n)
        return result.astype(segment.dtype)
    
    # Helper function to apply distortion
    def apply_distortion(segment: np.ndarray, drive: float = 0.5) -> np.ndarray:
        """Apply soft-clipping distortion to segment."""
        # Normalize, apply tanh saturation, then restore level
        max_val = np.max(np.abs(segment)) + 1e-10
        normalized = segment / max_val
        # Soft clipping with tanh
        driven = np.tanh(normalized * (1 + drive * 3))
        return driven * max_val
    
    # Helper function to apply simple reverb (convolution with decay)
    def apply_reverb(segment: np.ndarray, decay: float = 0.3, length_ms: float = 100) -> np.ndarray:
        """Apply simple reverb effect using exponential decay."""
        reverb_samples = int(length_ms * sr / 1000)
        if reverb_samples < 10:
            return segment
        
        # Create simple impulse response (exponential decay)
        t = np.arange(reverb_samples) / sr
        impulse = np.exp(-t * 20) * decay
        impulse[0] = 1.0  # Direct signal
        
        # Convolve (this will extend the audio slightly)
        wet = np.convolve(segment, impulse, mode='same')
        return wet.astype(segment.dtype)
    
    # Build output audio
    segments = []
    prev_beat_idx = -1
    i = 0
    
    while i < n_beats:
        status = beat_status.get(i, "KEEP")
        
        if status == "CUT" or status == "REORDER":
            # Skip cut beats and reordered beats (reordered will be added at end)
            i += 1
            continue
        
        # Check if this beat starts a special effect section
        if i in fade_markers:
            fade_type, n_fade_beats = fade_markers[i]
            # Collect all beats in the fade section
            fade_segment_parts = []
            for j in range(n_fade_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                fade_segment_parts.append(audio[start_sample:end_sample].copy())
            if fade_segment_parts:
                combined_segment = np.concatenate(fade_segment_parts)
                segment = apply_fade(combined_segment, fade_type)
                logger.debug(f"Beats {i}-{i+n_fade_beats-1}: FADE {fade_type}")
                segments.append(segment)
                prev_beat_idx = i + n_fade_beats - 1
            i += n_fade_beats
            continue
        
        if i in time_changes:
            n_time_beats, factor = time_changes[i]
            # Collect all beats in the time change section
            time_segment_parts = []
            for j in range(n_time_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                time_segment_parts.append(audio[start_sample:end_sample].copy())
            if time_segment_parts:
                combined_segment = np.concatenate(time_segment_parts)
                segment = apply_time_stretch(combined_segment, factor)
                speed_label = "DOUBLE_TIME" if factor == 2.0 else "HALF_TIME"
                logger.debug(f"Beats {i}-{i+n_time_beats-1}: {speed_label}")
                segments.append(segment)
                prev_beat_idx = i + n_time_beats - 1
            i += n_time_beats
            continue
        
        if i in reverse_sections:
            n_reverse_beats = reverse_sections[i]
            # Collect all beats in the reverse section
            reverse_segment_parts = []
            for j in range(n_reverse_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                reverse_segment_parts.append(audio[start_sample:end_sample].copy())
            if reverse_segment_parts:
                combined_segment = np.concatenate(reverse_segment_parts)
                segment = combined_segment[::-1]  # Reverse the audio
                logger.debug(f"Beats {i}-{i+n_reverse_beats-1}: REVERSE")
                segments.append(segment)
                prev_beat_idx = i + n_reverse_beats - 1
            i += n_reverse_beats
            continue
        
        if i in gain_changes:
            n_gain_beats, db = gain_changes[i]
            # Collect all beats in the gain section
            gain_segment_parts = []
            for j in range(n_gain_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                gain_segment_parts.append(audio[start_sample:end_sample].copy())
            if gain_segment_parts:
                combined_segment = np.concatenate(gain_segment_parts)
                segment = apply_gain(combined_segment, db)
                gain_label = f"GAIN +{db}dB" if db > 0 else f"GAIN {db}dB"
                logger.debug(f"Beats {i}-{i+n_gain_beats-1}: {gain_label}")
                segments.append(segment)
                prev_beat_idx = i + n_gain_beats - 1
            i += n_gain_beats
            continue
        
        if i in eq_changes:
            n_eq_beats, eq_type = eq_changes[i]
            # Collect all beats in the EQ section
            eq_segment_parts = []
            for j in range(n_eq_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                eq_segment_parts.append(audio[start_sample:end_sample].copy())
            if eq_segment_parts:
                combined_segment = np.concatenate(eq_segment_parts)
                segment = apply_eq(combined_segment, eq_type)
                logger.debug(f"Beats {i}-{i+n_eq_beats-1}: EQ {eq_type}")
                segments.append(segment)
                prev_beat_idx = i + n_eq_beats - 1
            i += n_eq_beats
            continue
        
        if i in audio_effects:
            n_fx_beats, effect_type = audio_effects[i]
            # Collect all beats in the effect section
            fx_segment_parts = []
            for j in range(n_fx_beats):
                beat_idx = i + j
                if beat_idx >= n_beats:
                    break
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                fx_segment_parts.append(audio[start_sample:end_sample].copy())
            if fx_segment_parts:
                combined_segment = np.concatenate(fx_segment_parts)
                if effect_type == "distortion":
                    segment = apply_distortion(combined_segment)
                elif effect_type == "reverb":
                    segment = apply_reverb(combined_segment)
                else:
                    segment = combined_segment
                logger.debug(f"Beats {i}-{i+n_fx_beats-1}: EFFECT {effect_type}")
                segments.append(segment)
                prev_beat_idx = i + n_fx_beats - 1
            i += n_fx_beats
            continue
        
        # Check for REPEAT action (copy previous section before current)
        if isinstance(status, tuple) and status[0] == "REPEAT":
            source_beats = status[1]
            # First, insert the source beats (copy from earlier in the track)
            repeat_segment_parts = []
            for src_b in source_beats:
                if 0 <= src_b < n_beats:
                    start_sample = beat_samples[src_b]
                    end_sample = beat_samples[src_b + 1]
                    repeat_segment_parts.append(audio[start_sample:end_sample].copy())
            if repeat_segment_parts:
                repeated_segment = np.concatenate(repeat_segment_parts)
                logger.debug(f"Beat {i}: REPEAT (copying beats {min(source_beats)}-{max(source_beats)})")
                segments.append(repeated_segment)
            
            # Then continue with current beat (which is marked as KEEP)
            start_sample = beat_samples[i]
            end_sample = beat_samples[i + 1]
            segment = audio[start_sample:end_sample].copy()
            logger.debug(f"Beat {i}: KEEP (after repeat)")
            segments.append(segment)
            prev_beat_idx = i
            i += 1
            continue
        
        # Check for SWAP action (swap current section with next section)
        if isinstance(status, tuple) and status[0] == "SWAP":
            swap_size = status[1]
            first_start = i
            second_start = i + swap_size
            
            # Output second section first
            second_segment_parts = []
            for j in range(swap_size):
                beat_idx = second_start + j
                if beat_idx < n_beats:
                    start_sample = beat_samples[beat_idx]
                    end_sample = beat_samples[beat_idx + 1]
                    second_segment_parts.append(audio[start_sample:end_sample].copy())
            
            # Then output first section
            first_segment_parts = []
            for j in range(swap_size):
                beat_idx = first_start + j
                if beat_idx < n_beats:
                    start_sample = beat_samples[beat_idx]
                    end_sample = beat_samples[beat_idx + 1]
                    first_segment_parts.append(audio[start_sample:end_sample].copy())
            
            if second_segment_parts:
                segments.append(np.concatenate(second_segment_parts))
                logger.debug(f"Beats {second_start}-{second_start+swap_size-1}: SWAP (moved earlier)")
            if first_segment_parts:
                segments.append(np.concatenate(first_segment_parts))
                logger.debug(f"Beats {first_start}-{first_start+swap_size-1}: SWAP (moved later)")
            
            prev_beat_idx = i + swap_size * 2 - 1
            i += swap_size * 2  # Skip both swapped sections
            continue
            
        # Normal beat processing
        start_sample = beat_samples[i]
        end_sample = beat_samples[i + 1]
        segment = audio[start_sample:end_sample].copy()
        
        if isinstance(status, tuple) and status[0] == "LOOP":
            loop_times = status[1]
            segment = np.tile(segment, loop_times)
            logger.debug(f"Beat {i}: LOOP {loop_times}x")
        else:
            logger.debug(f"Beat {i}: KEEP")
        
        # Check if we need crossfade (non-consecutive beats or soft marker)
        use_crossfade = False
        if prev_beat_idx >= 0:
            is_non_consecutive = (i != prev_beat_idx + 1)
            is_soft_marker = i in transition_markers and transition_markers[i] == "SOFT"
            use_crossfade = is_non_consecutive or is_soft_marker
        
        if segments and use_crossfade and crossfade_samples > 0:
            # Pop last segment and apply crossfade
            prev_audio = segments.pop()
            combined = apply_crossfade(prev_audio, segment, crossfade_samples)
            segments.append(combined)
        else:
            segments.append(segment)
        
        prev_beat_idx = i
        i += 1
    
    # Append reordered sections at the end
    for start_beat, n_reorder_beats in reordered_sections:
        for i in range(n_reorder_beats):
            beat_idx = start_beat + i
            if beat_idx >= n_beats:
                continue
            start_sample = beat_samples[beat_idx]
            end_sample = beat_samples[beat_idx + 1]
            segment = audio[start_sample:end_sample].copy()
            logger.debug(f"Beat {beat_idx}: REORDER (appended at end)")
            
            # Apply crossfade when appending reordered content
            if segments and crossfade_samples > 0:
                prev_audio = segments.pop()
                combined = apply_crossfade(prev_audio, segment, crossfade_samples)
                segments.append(combined)
            else:
                segments.append(segment)
    
    if not segments:
        logger.warning("No segments kept! Returning original audio.")
        return audio
    
    # Concatenate all segments
    edited = np.concatenate(segments)

    # --- Accurate output beat statistics and duration ratios ---
    # Track how many times each beat index is included in the output, and sum their durations
    output_beat_counts = np.zeros(n_beats, dtype=int)
    output_beat_samples = np.zeros(n_beats, dtype=int)

    # For main pass (kept/looped beats)
    for i in range(n_beats):
        status = beat_status.get(i, "KEEP")
        if status == "CUT" or status == "REORDER":
            continue
        start_sample = beat_samples[i]
        end_sample = beat_samples[i + 1]
        seg_len = end_sample - start_sample
        if isinstance(status, tuple) and status[0] == "LOOP":
            loop_times = status[1]
            output_beat_counts[i] += loop_times
            output_beat_samples[i] += seg_len * loop_times
        else:
            output_beat_counts[i] += 1
            output_beat_samples[i] += seg_len
    # For reordered sections appended at end
    for start_beat, n_reorder_beats in reordered_sections:
        for j in range(n_reorder_beats):
            beat_idx = start_beat + j
            if beat_idx < n_beats:
                start_sample = beat_samples[beat_idx]
                end_sample = beat_samples[beat_idx + 1]
                seg_len = end_sample - start_sample
                output_beat_counts[beat_idx] += 1
                output_beat_samples[beat_idx] += seg_len

    kept_samples = np.sum(output_beat_samples)
    cut_samples = len(audio) - kept_samples
    original_duration = len(audio) / sr
    edited_duration = kept_samples / sr
    cut_duration = cut_samples / sr

    # Calculate total duration of cut beats
    cut_beat_samples = 0
    for i in range(n_beats):
        if output_beat_counts[i] == 0:
            start_sample = beat_samples[i]
            end_sample = beat_samples[i + 1]
            cut_beat_samples += (end_sample - start_sample)
    cut_beat_duration = cut_beat_samples / sr

    logger.info(f"Audio: {original_duration:.2f}s -> {edited_duration:.2f}s ({100*edited_duration/original_duration:.1f}%)")
    logger.info(f"Kept duration: {edited_duration:.2f}s ({100*edited_duration/original_duration:.1f}%), Cut duration: {cut_duration:.2f}s ({100*cut_duration/original_duration:.1f}%)")
    logger.info(f"Output beats: {np.sum(output_beat_counts)} total (kept: {np.sum(output_beat_counts == 1)}, looped/reordered: {np.sum(output_beat_counts > 1)}, cut: {np.sum(output_beat_counts == 0)}, of {n_beats} unique beats)")
    logger.info(f"Total duration of cut beats: {cut_beat_duration:.2f}s ({100*cut_beat_duration/original_duration:.1f}%)")

    # --- Debug: Print and plot beat durations ---
    import matplotlib.pyplot as plt
    beat_durations_sec = np.diff(beat_times)
    print("Beat durations (seconds):", beat_durations_sec)
    print(f"Min: {beat_durations_sec.min():.4f}s, Max: {beat_durations_sec.max():.4f}s, Mean: {beat_durations_sec.mean():.4f}s, Std: {beat_durations_sec.std():.4f}s")
    #plt.figure(figsize=(10, 4))
    #plt.plot(beat_durations_sec, marker='o')
    #plt.title('Beat Durations (seconds)')
    #plt.xlabel('Beat Index')
    #plt.ylabel('Duration (s)')
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    return edited


def main():
    parser = argparse.ArgumentParser(description="V2 RL Audio Editor Inference")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, default=None, help="Output audio file")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="V2 model checkpoint path")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy (default: True)")
    parser.add_argument("--max-beats", type=int, default=0, help="Maximum beats to process")
    parser.add_argument("--crossfade-ms", type=float, default=50.0, 
                       help="Crossfade duration in ms at edit boundaries")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = get_default_config()
    config.training.device = str(device)
    
    # Load audio and create state
    audio, sr, audio_state = load_and_process_audio(
        args.input,
        config=config,
        max_beats=args.max_beats,
        cache_dir=config.data.cache_dir,
    )
    
    # Create temp environment to get observation dimension
    temp_env = AudioEditingEnvV2(config, audio_state=audio_state)
    obs, _ = temp_env.reset()
    input_dim = len(obs)
    logger.info(f"State dimension: {input_dim}")
    
    # Initialize agent with V2 action space
    n_actions = ActionSpaceV2.N_ACTIONS
    logger.info(f"Using V2 action space: {n_actions} actions")
    agent = Agent(config, input_dim=input_dim, n_actions=n_actions)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading V2 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model weights
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.value_net.load_state_dict(checkpoint["value_state_dict"])
    
    best_reward = checkpoint.get('best_reward', 'N/A')
    epoch = checkpoint.get('current_epoch', 'N/A')
    logger.info(f"Loaded checkpoint: epoch={epoch}, best_reward={best_reward}")
    
    # Run inference
    logger.info("Running V2 inference...")
    actions, action_names, action_details = run_inference_v2(
        agent, config, audio_state, deterministic=args.deterministic
    )
    
    # Analyze actions
    action_counter = Counter(action_names)
    logger.info(f"Action distribution: {dict(action_counter)}")

    # Create edited audio
    edited_audio = create_edited_audio_v2(
        audio, sr, 
        audio_state.beat_times, 
        actions,
        crossfade_ms=args.crossfade_ms
    )
    
    # Save output
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_edited_v2{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), edited_audio, sr)
    logger.info(f"Saved edited audio to: {output_path}")


if __name__ == "__main__":
    main()
    

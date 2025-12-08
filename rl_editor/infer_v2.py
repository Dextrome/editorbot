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

# Import from original infer.py
from rl_editor.infer import load_and_process_audio, apply_crossfade

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
    
    V2 Actions:
    - 0: KEEP_BEAT
    - 1: CUT_BEAT
    - 2: KEEP_BAR (4 beats)
    - 3: CUT_BAR (4 beats)
    - 4: KEEP_PHRASE (8 beats)
    - 5: CUT_PHRASE (8 beats)
    - 6: LOOP_2X
    - 7: LOOP_4X
    - 8: LOOP_BAR_2X (4 beats looped)
    - 9: JUMP_BACK_4
    - 10: JUMP_BACK_8
    - 11: MARK_SOFT_TRANSITION
    - 12: MARK_HARD_CUT
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
    elif action_idx == ActionSpaceV2.ACTION_KEEP_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "KEEP", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_CUT_PHRASE:
        end_beat = min(current_beat + 8, n_beats)
        return {"type": "CUT", "beats": list(range(current_beat, end_beat))}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_2X:
        return {"type": "LOOP", "beats": [current_beat], "times": 2}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_4X:
        return {"type": "LOOP", "beats": [current_beat], "times": 4}
    elif action_idx == ActionSpaceV2.ACTION_LOOP_BAR_2X:
        end_beat = min(current_beat + 4, n_beats)
        return {"type": "LOOP", "beats": list(range(current_beat, end_beat)), "times": 2}
    elif action_idx == ActionSpaceV2.ACTION_JUMP_BACK_4:
        return {"type": "JUMP_BACK", "beats": 4}
    elif action_idx == ActionSpaceV2.ACTION_JUMP_BACK_8:
        return {"type": "JUMP_BACK", "beats": 8}
    elif action_idx == ActionSpaceV2.ACTION_MARK_SOFT:
        return {"type": "MARK_SOFT", "beats": [current_beat]}
    elif action_idx == ActionSpaceV2.ACTION_MARK_HARD:
        return {"type": "MARK_HARD", "beats": [current_beat]}
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
            
        elif action_type == "LOOP":
            loop_times = action_info["times"]
            for b in action_info["beats"]:
                if b < n_beats:
                    beat_status[b] = ("LOOP", loop_times)
            current_beat = max(action_info["beats"]) + 1 if action_info["beats"] else current_beat + 1
            
        elif action_type == "JUMP_BACK":
            # Jump back means we'll potentially revisit earlier beats
            jump_beats = action_info["beats"]
            current_beat = max(0, current_beat - jump_beats)
            
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
    
    # Build output audio
    segments = []
    prev_beat_idx = -1
    
    for i in range(n_beats):
        status = beat_status.get(i, "KEEP")
        
        if status == "CUT":
            continue
            
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
    
    if not segments:
        logger.warning("No segments kept! Returning original audio.")
        return audio
    
    # Concatenate all segments
    edited = np.concatenate(segments)
    
    # Log statistics
    original_duration = len(audio) / sr
    edited_duration = len(edited) / sr
    
    keep_count = sum(1 for s in beat_status.values() if s == "KEEP")
    cut_count = sum(1 for s in beat_status.values() if s == "CUT")
    loop_count = sum(1 for s in beat_status.values() if isinstance(s, tuple))
    
    logger.info(f"Audio: {original_duration:.2f}s -> {edited_duration:.2f}s ({100*edited_duration/original_duration:.1f}%)")
    logger.info(f"Beats: {keep_count} kept, {cut_count} cut, {loop_count} looped (of {n_beats} total)")
    
    return edited


def main():
    parser = argparse.ArgumentParser(description="V2 RL Audio Editor Inference")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, default=None, help="Output audio file")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="V2 model checkpoint path")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy (default: True)")
    parser.add_argument("--max-beats", type=int, default=500, help="Maximum beats to process")
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

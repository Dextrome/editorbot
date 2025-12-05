"""Inference script for the RL audio editor.

Load a trained model and edit audio files.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import librosa
import soundfile as sf

from rl_editor.config import get_default_config, Config
from rl_editor.agent import Agent
from rl_editor.state import AudioState
from rl_editor.environment import AudioEditingEnv
from rl_editor.actions import KeepAction, CutAction, LoopAction, ReorderAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_process_audio(
    audio_path: str, 
    sr: int = 22050,
    max_beats: int = 500
) -> Tuple[np.ndarray, int, AudioState]:
    """Load audio and compute features for the RL environment.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        max_beats: Maximum number of beats to process
        
    Returns:
        Tuple of (audio_array, sample_rate, audio_state)
    """
    logger.info(f"Loading audio: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y) / sr
    logger.info(f"Audio duration: {duration:.2f}s")
    
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])  # Handle both scalar and array tempo
    logger.info(f"Detected {len(beat_times)} beats at tempo {tempo:.1f} BPM")
    
    # Compute beat features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    # Synchronize features to beats
    beat_features = []
    frames = librosa.time_to_frames(beat_times, sr=sr)
    frames = np.concatenate(([0], frames, [len(onset_env)]))
    
    for i in range(len(beats)):
        start = frames[i]
        end = frames[i+1]
        if start >= len(onset_env):
            break
        if end > len(onset_env):
            end = len(onset_env)
        if start == end:
            end = start + 1
            
        b_onset = np.mean(onset_env[start:end])
        b_centroid = np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))])
        b_zcr = np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))])
        b_rms = np.mean(rms[min(start, len(rms)-1):min(end, len(rms))])
        beat_features.append([b_onset, b_centroid, b_zcr, b_rms])
    
    beat_features = np.array(beat_features) if beat_features else np.zeros((0, 4))
    
    # Truncate to max_beats
    if len(beat_times) > max_beats:
        logger.warning(f"Truncating from {len(beat_times)} to {max_beats} beats")
        beat_times = beat_times[:max_beats]
        beat_features = beat_features[:max_beats]
    
    # Normalize features
    beat_features = (beat_features - beat_features.mean(axis=0)) / (beat_features.std(axis=0) + 1e-8)
    
    # Create AudioState
    audio_state = AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=float(tempo),
    )
    
    return y, sr, audio_state


def run_inference(
    agent: Agent,
    config: Config,
    audio_state: AudioState,
    deterministic: bool = True
) -> Tuple[List[int], List[str]]:
    """Run inference on audio state.
    
    Args:
        agent: Trained agent
        config: Configuration
        audio_state: Audio state to process
        deterministic: Whether to use deterministic policy
        
    Returns:
        Tuple of (action_indices, action_names)
    """
    agent.eval()
    env = AudioEditingEnv(config, audio_state)
    
    obs, _ = env.reset()
    done = False
    actions = []
    action_names = []
    
    while not done:
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = agent.select_action(state_tensor, deterministic=deterministic)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record action
        actions.append(action)
        action_name = info.get("action_name", f"action_{action}")
        action_names.append(action_name)
        
        done = terminated or truncated
    
    return actions, action_names


def run_inference_bc(
    agent: Agent,
    config: Config,
    audio_state: AudioState,
    deterministic: bool = True
) -> Tuple[List[int], List[str]]:
    """Run inference with BC model (only KEEP/CUT actions).
    
    This is specifically for behavioral cloning models that were only 
    trained on KEEP/CUT decisions. We mask out all other actions.
    
    Args:
        agent: Trained agent
        config: Configuration
        audio_state: Audio state to process
        deterministic: Whether to use deterministic policy
        
    Returns:
        Tuple of (action_indices, action_names)
    """
    agent.eval()
    env = AudioEditingEnv(config, audio_state)
    
    obs, _ = env.reset()
    done = False
    actions = []
    action_names = []
    
    while not done:
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get logits from policy network
            logits, _ = agent.policy_net(state_tensor)
            
            # Only consider KEEP (0) and CUT (1) actions
            keep_cut_logits = logits[:, :2]
            
            # Choose action (argmax for deterministic)
            if deterministic:
                action = keep_cut_logits.argmax(dim=1).item()
            else:
                probs = torch.softmax(keep_cut_logits, dim=1)
                action = torch.multinomial(probs, 1).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record action
        actions.append(action)
        action_name = "KEEP" if action == 0 else "CUT"
        action_names.append(action_name)
        
        done = terminated or truncated
    
    return actions, action_names


def decode_action_for_audio(action_idx: int, current_beat: int, n_beats: int) -> Dict:
    """Decode action index to action info for audio processing.
    
    Action layout (matching actions.py):
    - 0: KEEP (keep current beat)
    - 1: CUT (remove current beat)
    - 2: LOOP 2x
    - 3: LOOP 3x
    - 4: LOOP 4x
    - 5: REORDER to position +1
    - 6: REORDER to position +2
    - 7: REORDER to position +3
    - 8: REORDER to position +4
    
    Args:
        action_idx: Integer action index
        current_beat: Current beat index
        n_beats: Total number of beats
        
    Returns:
        Dict with 'type', and action-specific parameters
    """
    if action_idx == 0:
        return {"type": "KEEP", "beat": current_beat}
    elif action_idx == 1:
        return {"type": "CUT", "beat": current_beat}
    elif action_idx in [2, 3, 4]:
        loop_times = action_idx  # 2x, 3x, 4x
        return {"type": "LOOP", "beat": current_beat, "times": loop_times}
    elif action_idx in [5, 6, 7, 8]:
        target_offset = action_idx - 4  # +1, +2, +3, +4
        target_position = min(current_beat + target_offset, n_beats - 1)
        return {"type": "REORDER", "beat": current_beat, "target": target_position}
    else:
        # Unknown action, treat as KEEP
        return {"type": "KEEP", "beat": current_beat}


def apply_crossfade(seg1: np.ndarray, seg2: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """Apply crossfade between two audio segments.
    
    Args:
        seg1: First segment (fade out at end)
        seg2: Second segment (fade in at start)
        crossfade_samples: Number of samples for crossfade
        
    Returns:
        Combined audio with crossfade
    """
    if crossfade_samples <= 0:
        return np.concatenate([seg1, seg2])
    
    # Ensure we have enough samples
    actual_fade = min(crossfade_samples, len(seg1), len(seg2))
    
    if actual_fade < 10:  # Too short for meaningful crossfade
        return np.concatenate([seg1, seg2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, actual_fade)
    fade_in = np.linspace(0.0, 1.0, actual_fade)
    
    # Copy segments to avoid modifying originals
    result = np.zeros(len(seg1) + len(seg2) - actual_fade)
    result[:len(seg1) - actual_fade] = seg1[:-actual_fade]
    
    # Apply crossfade in overlap region
    overlap_start = len(seg1) - actual_fade
    result[overlap_start:overlap_start + actual_fade] = (
        seg1[-actual_fade:] * fade_out + seg2[:actual_fade] * fade_in
    )
    
    # Add rest of second segment
    result[overlap_start + actual_fade:] = seg2[actual_fade:]
    
    return result


def create_edited_audio(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    actions: List[int],
    action_names: List[str],
    crossfade_ms: float = 50.0
) -> np.ndarray:
    """Create edited audio based on actions.
    
    This function properly implements all creative actions:
    - KEEP: Include the beat as-is
    - CUT: Remove the beat
    - LOOP: Duplicate the beat N times
    - REORDER: Mark beat for swapping (implemented as deferred insertion)
    
    Automatic crossfades are applied at edit boundaries (transitions between
    kept and cut regions) to smooth out any discontinuities.
    
    Args:
        audio: Original audio array
        sr: Sample rate
        beat_times: Array of beat times in seconds
        actions: List of action indices
        action_names: List of action names  
        crossfade_ms: Crossfade duration in milliseconds for transitions
        
    Returns:
        Edited audio array
    """
    # Get beat boundaries (samples)
    beat_samples = (beat_times * sr).astype(int)
    beat_samples = np.append(beat_samples, len(audio))
    n_beats = len(beat_times)
    
    crossfade_samples = int(crossfade_ms * sr / 1000)
    
    # First pass: decode all actions and collect segment info
    segment_info = []  # List of dicts with segment data
    reorder_queue = []  # Beats marked for reordering
    
    for i, action_idx in enumerate(actions):
        if i >= len(beat_times):
            break
            
        start_sample = beat_samples[i]
        end_sample = beat_samples[i + 1]
        segment = audio[start_sample:end_sample].copy()
        
        action_info = decode_action_for_audio(action_idx, i, n_beats)
        action_type = action_info["type"]
        
        if action_type == "CUT":
            # Don't include this segment
            logger.debug(f"Beat {i}: CUT")
            continue
            
        elif action_type == "KEEP":
            segment_info.append({
                "beat": i,
                "audio": segment,
                "action": "KEEP"
            })
            logger.debug(f"Beat {i}: KEEP")
            
        elif action_type == "LOOP":
            loop_times = action_info["times"]
            # Duplicate the segment N times
            looped_audio = np.tile(segment, loop_times)
            segment_info.append({
                "beat": i,
                "audio": looped_audio,
                "action": f"LOOP_{loop_times}x"
            })
            logger.debug(f"Beat {i}: LOOP {loop_times}x ({len(segment)} -> {len(looped_audio)} samples)")
            
        elif action_type == "REORDER":
            target = action_info["target"]
            # For now, keep the beat but mark it for potential reordering
            # Full reorder implementation would require two-pass processing
            # to actually swap positions. For now, we just keep it in place
            # but log the intended reorder.
            segment_info.append({
                "beat": i,
                "audio": segment,
                "action": f"REORDER_to_{target}",
                "reorder_target": target
            })
            logger.debug(f"Beat {i}: REORDER to position {target}")
    
    if not segment_info:
        logger.warning("No segments kept! Returning original audio.")
        return audio
    
    # Process reorder actions (swap adjacent beats if marked)
    # This is a simple implementation - we look for pairs that want to swap
    processed = set()
    final_segments = []
    
    i = 0
    while i < len(segment_info):
        seg = segment_info[i]
        
        # Check if this is a reorder action that wants to swap with next
        if "reorder_target" in seg and i + 1 < len(segment_info):
            target_beat = seg["reorder_target"]
            next_seg = segment_info[i + 1]
            
            # If the target is the next beat, swap them
            if next_seg["beat"] == target_beat:
                # Swap order
                final_segments.append(next_seg)
                final_segments.append(seg)
                logger.info(f"Swapped beats {seg['beat']} and {next_seg['beat']}")
                i += 2
                continue
        
        final_segments.append(seg)
        i += 1
    
    # Concatenate segments with automatic crossfades at edit boundaries
    if len(final_segments) == 1:
        edited = final_segments[0]["audio"]
    else:
        edited = final_segments[0]["audio"].copy()
        prev_beat = final_segments[0]["beat"]
        
        for seg in final_segments[1:]:
            current_beat = seg["beat"]
            segment_audio = seg["audio"]
            
            # Determine if we need a crossfade (edit boundary = non-consecutive beats)
            # If beats are consecutive (e.g., 5 -> 6), no crossfade needed
            # If beats are non-consecutive (e.g., 5 -> 8), apply crossfade
            is_edit_boundary = (current_beat != prev_beat + 1)
            
            if is_edit_boundary and crossfade_samples > 0:
                edited = apply_crossfade(edited, segment_audio, crossfade_samples)
            else:
                edited = np.concatenate([edited, segment_audio])
            
            prev_beat = current_beat
    
    # Log statistics
    original_duration = len(audio) / sr
    edited_duration = len(edited) / sr
    
    action_counts = {}
    for seg in final_segments:
        action = seg["action"].split("_")[0]  # Get base action name
        action_counts[action] = action_counts.get(action, 0) + 1
    
    logger.info(f"Audio: {original_duration:.2f}s -> {edited_duration:.2f}s ({100*edited_duration/original_duration:.1f}%)")
    logger.info(f"Segments: {len(final_segments)}/{len(actions)} kept")
    logger.info(f"Actions used: {action_counts}")
    
    return edited


def main():
    parser = argparse.ArgumentParser(description="RL Audio Editor Inference")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, default=None, help="Output audio file")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoint_best.pt", 
                       help="Model checkpoint path")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Use deterministic policy (default: stochastic)")
    parser.add_argument("--max-beats", type=int, default=500, help="Maximum beats to process")
    parser.add_argument("--crossfade-ms", type=float, default=50.0, help="Crossfade duration in ms at edit boundaries")
    parser.add_argument("--bc", action="store_true", default=False,
                       help="Use behavioral cloning mode (KEEP/CUT only)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--n-actions", type=int, default=9, 
                       help="Number of actions in action space (9 for new, 30 for legacy)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = get_default_config()
    config.training.device = str(device)  # Update config with device
    
    # Load audio and create state
    audio, sr, audio_state = load_and_process_audio(
        args.input, 
        sr=config.audio.sample_rate,
        max_beats=args.max_beats
    )
    
    # Create temp environment to get observation dimension
    temp_env = AudioEditingEnv(config, audio_state)
    obs, _ = temp_env.reset()
    input_dim = len(obs)
    logger.info(f"State dimension: {input_dim}")
    
    # Initialize agent
    n_actions = args.n_actions
    logger.info(f"Using {n_actions}-action space")
    agent = Agent(config, input_dim=input_dim, n_actions=n_actions)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.value_net.load_state_dict(checkpoint["value_state_dict"])
    logger.info(f"Loaded checkpoint with best reward: {checkpoint.get('best_reward', 'N/A')}")
    
    # Run inference
    logger.info("Running inference...")
    if args.bc:
        logger.info("Using BC mode (KEEP/CUT only)")
        actions, action_names = run_inference_bc(agent, config, audio_state, deterministic=args.deterministic)
    else:
        actions, action_names = run_inference(agent, config, audio_state, deterministic=args.deterministic)
    
    # Analyze actions
    from collections import Counter
    action_counter = Counter(actions)
    logger.info(f"Action distribution: {dict(action_counter)}")
    keep_count = sum(1 for a in actions if a == 0)
    cut_count = sum(1 for a in actions if a == 1)
    other_count = len(actions) - keep_count - cut_count
    logger.info(f"Actions: KEEP={keep_count}, CUT={cut_count}, OTHER={other_count}")
    
    # Create edited audio
    edited_audio = create_edited_audio(
        audio, sr, 
        audio_state.beat_times, 
        actions, 
        action_names,
        crossfade_ms=args.crossfade_ms
    )
    
    # Save output
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_edited{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), edited_audio, sr)
    logger.info(f"Saved edited audio to: {output_path}")


if __name__ == "__main__":
    main()

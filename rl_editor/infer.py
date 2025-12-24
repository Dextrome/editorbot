"""Inference script for the RL audio editor.

Load a trained model and edit audio files.

Uses the same feature extraction as training to ensure
the model receives consistent input dimensions.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import librosa
import soundfile as sf
from collections import Counter
import random
import shutil

from rl_editor.config import get_default_config, Config
from rl_editor.agent import Agent
from rl_editor.state import AudioState
from rl_editor.environment import AudioEditingEnvFactored
from rl_editor.actions import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
    SIZE_TO_BEATS, AMOUNT_TO_DB, AMOUNT_TO_JUMP,
)

# Import feature extraction components used by training
try:
    from rl_editor.features import (
        BeatFeatureExtractor,
        get_enhanced_feature_config,
        get_basic_feature_config,
        StemProcessor,
    )
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    StemProcessor = None

# Import feature cache for loading pre-cached features
try:
    from rl_editor.cache import FeatureCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_process_audio(
    audio_path: str, 
    config: Config,
    max_beats: int = 0,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, int, AudioState]:
    """Load audio and compute features matching the training pipeline.
    
    Uses the same feature extraction as training to ensure consistent
    input dimensions for the model.
    
    Args:
        audio_path: Path to audio file
        config: Configuration object (determines feature mode)
        max_beats: Maximum number of beats to process
        cache_dir: Directory to check for cached features
        
    Returns:
        Tuple of (audio_array, sample_rate, audio_state)
    """
    sr = config.audio.sample_rate
    audio_path = Path(audio_path)
    logger.info(f"Loading audio: {audio_path}")
    
    # Check if features are cached
    feature_cache = None
    cached_features = None
    if HAS_CACHE and cache_dir:
        feature_cache = FeatureCache(cache_dir=Path(cache_dir), enabled=True)
        cached_features = feature_cache.load_full(audio_path)
    
    if cached_features is not None:
        logger.info("Using cached features")
        beat_times = cached_features["beat_times"]
        beat_features = cached_features["beat_features"]
        tempo = float(cached_features["tempo"])
        
        # Still need to load audio for output generation
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    else:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        duration = len(y) / sr
        logger.info(f"Audio duration: {duration:.2f}s")
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        logger.info(f"Detected {len(beat_times)} beats at tempo {tempo:.1f} BPM")
        
        # Use enhanced feature extractor if available (matching training)
        if HAS_ENHANCED_FEATURES:
            feature_mode = getattr(config.features, 'feature_mode', 'basic') if hasattr(config, 'features') else 'basic'
            
            if feature_mode == "full":
                feat_config = get_enhanced_feature_config()
                feat_config.use_stem_features = True
            elif feature_mode == "enhanced":
                feat_config = get_enhanced_feature_config()
            else:
                feat_config = get_basic_feature_config()
            
            feature_extractor = BeatFeatureExtractor(
                sr=sr,
                hop_length=config.audio.hop_length,
                n_fft=config.audio.n_fft,
                n_mels=config.audio.n_mels,
                config=feat_config,
            )
            
            logger.info(f"Using {feature_mode} features: {feature_extractor.get_feature_dim()} dims")
            beat_features = feature_extractor.extract_features(y, beats, beat_times, tempo)
            
            # If full mode, also add stem features (requires Demucs separation)
            if feature_mode == "full" and config.data.use_stems:
                logger.info("Extracting stem features (this may take a moment)...")
                try:
                    stem_processor = StemProcessor(
                        cache_dir=Path(cache_dir) if cache_dir else None,
                        sr=sr,
                    )
                    
                    # Check for cached stems first
                    cached_stems = None
                    if feature_cache:
                        cached_stems = feature_cache.load_stems(audio_path)
                    
                    if cached_stems is not None:
                        stems = cached_stems
                        logger.info("Using cached stems")
                    else:
                        stems = stem_processor.separate(str(audio_path))
                        # Cache for future use
                        if feature_cache and stems:
                            feature_cache.save_stems(audio_path, stems)
                    
                    if stems:
                        stem_features = stem_processor.get_stem_features(
                            stems, beats, hop_length=config.audio.hop_length
                        )
                        logger.info(f"Stem features shape: {stem_features.shape}")
                        beat_features = np.concatenate([beat_features, stem_features], axis=1)
                    else:
                        logger.warning("Could not extract stems - padding with zeros")
                        # Pad with zeros for stem features (12 dims: 4 stems × 3 features)
                        stem_padding = np.zeros((len(beat_features), 12))
                        beat_features = np.concatenate([beat_features, stem_padding], axis=1)
                except Exception as e:
                    logger.warning(f"Stem extraction failed: {e} - padding with zeros")
                    stem_padding = np.zeros((len(beat_features), 12))
                    beat_features = np.concatenate([beat_features, stem_padding], axis=1)
        else:
            # Fall back to basic features
            logger.warning("Enhanced features not available - using basic features")
            beat_features = _extract_basic_features(y, sr, beats, beat_times)
    
    logger.info(f"Beat features shape: {beat_features.shape}")
    
    # Truncate to max_beats
    if max_beats > 0 and len(beat_times) > max_beats:
        logger.warning(f"Truncating from {len(beat_times)} to {max_beats} beats")
        beat_times = beat_times[:max_beats]
        beat_features = beat_features[:max_beats]
    
    # Normalize features (per-feature normalization)
    mean = beat_features.mean(axis=0, keepdims=True)
    std = beat_features.std(axis=0, keepdims=True) + 1e-8
    beat_features = (beat_features - mean) / std
    
    # Create AudioState
    audio_state = AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=float(tempo),
    )
    
    return y, sr, audio_state


def _extract_basic_features(y: np.ndarray, sr: int, beats: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Extract basic 4-dimensional beat features (fallback)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    
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
    
    return np.array(beat_features) if beat_features else np.zeros((0, 4))


def run_inference(
    agent: Agent,
    config: Config,
    audio_state: AudioState,
    deterministic: bool = True,
    verbose: bool = True,
    collect_aux: bool = False,
) -> Tuple[List[FactoredAction], List[str]]:
    """Run inference on audio state with factored action space.
    
    Args:
        agent: Trained factored agent
        config: Configuration
        audio_state: Audio state to process
        deterministic: Whether to use deterministic policy
        verbose: Whether to log action details
        
    Returns:
        Tuple of (actions, action_names)
    """
    agent.eval()
    env = AudioEditingEnvFactored(config, audio_state)
    
    obs, info = env.reset()
    done = False
    actions = []
    action_names = []
    aux_preds = []
    step = 0
    
    total_reward = 0.0
    while not done:
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        
        # Get action masks from environment
        type_mask, size_mask, amount_mask = env.get_action_masks()
        
        # Convert masks to tensors
        type_mask_t = torch.from_numpy(type_mask).bool().to(agent.device).unsqueeze(0)
        size_mask_t = torch.from_numpy(size_mask).bool().to(agent.device).unsqueeze(0)
        amount_mask_t = torch.from_numpy(amount_mask).bool().to(agent.device).unsqueeze(0)
        
        with torch.no_grad():
            action_tuple, log_prob = agent.select_action(
                state_tensor,
                type_mask=type_mask_t,
                size_mask=size_mask_t,
                amount_mask=amount_mask_t,
                deterministic=deterministic,
            )
        
        type_idx, size_idx, amount_idx = action_tuple
        
        # Decode to FactoredAction
        factored_action = FactoredAction(
            action_type=ActionType(type_idx),
            action_size=ActionSize(size_idx),
            action_amount=ActionAmount(amount_idx),
            beat_index=info.get("current_beat", step),
        )
        
        obs, reward, terminated, truncated, info = env.step(action_tuple)
        total_reward += float(reward)
        
        # Record action
        actions.append(factored_action)
        action_name = f"{factored_action.action_type.name}_{factored_action.action_size.name}"
        if factored_action.action_type in [ActionType.GAIN, ActionType.EQ_LOW, ActionType.EQ_HIGH]:
            action_name += f"_{factored_action.action_amount.name}"
        action_names.append(action_name)

        # collect auxiliary predictions if requested and available
        if collect_aux and hasattr(agent, "auxiliary_module") and agent.auxiliary_module is not None:
            try:
                preds = agent.get_auxiliary_predictions(state_tensor)
                # convert tensors to numpy for portability
                preds_cpu = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
            except Exception:
                preds_cpu = {}
            aux_preds.append({"beat": info.get("current_beat", step), "preds": preds_cpu})  # Keep aux_preds for later use
        
        if verbose:
            logger.debug(f"Step {step}: {action_name} at beat {factored_action.beat_index}")
        
        done = terminated or truncated
        step += 1

    # Final per-beat keep ratio from environment edit history
    try:
        final_keep_ratio = float(env.edit_history.get_keep_ratio())
    except Exception:
        final_keep_ratio = float(0.0)

    return actions, action_names, total_reward, aux_preds, final_keep_ratio



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
    
    # Create smooth cosine (Hann-like) fade curves for fewer clicks
    t = np.linspace(0.0, 1.0, actual_fade)
    # fade_out goes from 1 -> 0, fade_in from 0 -> 1 using raised-cosine
    fade_in = 0.5 * (1 - np.cos(np.pi * t))
    fade_out = 0.5 * (1 + np.cos(np.pi * t))
    
    # Copy segments to avoid modifying originals
    # Pre-allocate and copy non-overlap parts
    result = np.zeros(len(seg1) + len(seg2) - actual_fade, dtype=seg1.dtype)
    pre_len = len(seg1) - actual_fade
    if pre_len > 0:
        result[:pre_len] = seg1[:pre_len]

    # Apply crossfade in overlap region
    overlap_start = pre_len
    result[overlap_start:overlap_start + actual_fade] = (
        seg1[-actual_fade:] * fade_out + seg2[:actual_fade] * fade_in
    )

    # Append remainder of second segment
    if len(seg2) > actual_fade:
        result[overlap_start + actual_fade:] = seg2[actual_fade:]
    
    return result


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    """Apply gain change in dB."""
    linear_gain = 10 ** (db / 20.0)
    return audio * linear_gain


def apply_fade_in(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply fade-in effect."""
    if n_samples > len(audio):
        n_samples = len(audio)
    if n_samples < 1:
        return audio
    
    result = audio.copy()
    fade = np.linspace(0.0, 1.0, n_samples)
    result[:n_samples] *= fade
    return result


def apply_fade_out(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply fade-out effect."""
    if n_samples > len(audio):
        n_samples = len(audio)
    if n_samples < 1:
        return audio
    
    result = audio.copy()
    fade = np.linspace(1.0, 0.0, n_samples)
    result[-n_samples:] *= fade
    return result


def apply_time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Apply time stretch (librosa)."""
    try:
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception as e:
        logger.warning(f"Time stretch failed: {e}")
        return audio


def apply_pitch_shift(audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    """Apply pitch shift in semitones (librosa)."""
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        logger.warning(f"Pitch shift failed: {e}")
        return audio


def create_edited_audio(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    actions: List[FactoredAction],
    crossfade_ms: float = 50.0,
) -> np.ndarray:
    """Create edited audio based on factored actions.
    
    Implements all factored action types:
    - KEEP: Include the beat segment(s) as-is
    - CUT: Remove the beat segment(s)
    - LOOP: Duplicate the segment 2x
    - FADE_IN/OUT: Apply fade effects
    - GAIN: Volume change
    - SPEED_UP/SPEED_DOWN: Time stretch (amount determines intensity)
    - PITCH_UP/PITCH_DOWN: Pitch shift (amount determines semitones)
    - REVERSE: Play backwards
    - And more...
    
    Args:
        audio: Original audio array
        sr: Sample rate
        beat_times: Array of beat times in seconds
        actions: List of FactoredAction objects
        crossfade_ms: Crossfade duration in milliseconds
        
    Returns:
        Edited audio array
    """
    # Get beat boundaries (samples)
    beat_samples = (beat_times * sr).astype(int)
    beat_samples = np.append(beat_samples, len(audio))
    n_beats = len(beat_times)
    
    crossfade_samples = int(crossfade_ms * sr / 1000)
    
    # Process actions and collect segments
    segments = []  # List of (beat_idx, audio_segment, action_type)
    current_beat = 0
    action_idx = 0
    
    while current_beat < n_beats and action_idx < len(actions):
        action = actions[action_idx]
        n_action_beats = action.n_beats
        
        # Clamp to remaining beats
        end_beat = min(current_beat + n_action_beats, n_beats)
        
        # Extract segment for these beats
        start_sample = beat_samples[current_beat]
        end_sample = beat_samples[end_beat]
        segment = audio[start_sample:end_sample].copy()
        
        # Process based on action type
        if action.action_type == ActionType.CUT:
            # Skip this segment (don't add to output)
            logger.debug(f"Beat {current_beat}-{end_beat}: CUT")
            
        elif action.action_type == ActionType.KEEP:
            segments.append({
                "beat": current_beat,
                "audio": segment,
                "action": "KEEP"
            })
            logger.debug(f"Beat {current_beat}-{end_beat}: KEEP")
            
        elif action.action_type == ActionType.LOOP:
            # Loop the segment 2x
            looped = np.tile(segment, 2)
            segments.append({
                "beat": current_beat,
                "audio": looped,
                "action": "LOOP"
            })
            logger.debug(f"Beat {current_beat}-{end_beat}: LOOP 2x")
            
        elif action.action_type == ActionType.FADE_IN:
            fade_samples = len(segment) // 2  # Fade for half the segment
            faded = apply_fade_in(segment, fade_samples)
            segments.append({
                "beat": current_beat,
                "audio": faded,
                "action": "FADE_IN"
            })
            
        elif action.action_type == ActionType.FADE_OUT:
            fade_samples = len(segment) // 2
            faded = apply_fade_out(segment, fade_samples)
            segments.append({
                "beat": current_beat,
                "audio": faded,
                "action": "FADE_OUT"
            })
            
        elif action.action_type == ActionType.GAIN:
            db = action.db_change
            gained = apply_gain(segment, db)
            segments.append({
                "beat": current_beat,
                "audio": gained,
                "action": f"GAIN_{db}dB"
            })
            
        elif action.action_type == ActionType.SPEED_UP:
            speed_factor = action.speed_factor
            stretched = apply_time_stretch(segment, sr, speed_factor)
            segments.append({
                "beat": current_beat,
                "audio": stretched,
                "action": f"SPEED_UP_{speed_factor}x"
            })
            
        elif action.action_type == ActionType.SPEED_DOWN:
            speed_factor = action.speed_factor
            stretched = apply_time_stretch(segment, sr, speed_factor)
            segments.append({
                "beat": current_beat,
                "audio": stretched,
                "action": f"SPEED_DOWN_{speed_factor}x"
            })
            
        elif action.action_type == ActionType.REVERSE:
            reversed_seg = segment[::-1].copy()
            segments.append({
                "beat": current_beat,
                "audio": reversed_seg,
                "action": "REVERSE"
            })
            
        elif action.action_type == ActionType.PITCH_UP:
            semitones = action.pitch_semitones
            pitched = apply_pitch_shift(segment, sr, semitones)
            segments.append({
                "beat": current_beat,
                "audio": pitched,
                "action": f"PITCH_UP_{semitones}"
            })
            
        elif action.action_type == ActionType.PITCH_DOWN:
            semitones = action.pitch_semitones
            pitched = apply_pitch_shift(segment, sr, semitones)
            segments.append({
                "beat": current_beat,
                "audio": pitched,
                "action": f"PITCH_DOWN_{semitones}"
            })
            
        elif action.action_type == ActionType.REORDER:
            # Keep for now but mark for potential reordering
            segments.append({
                "beat": current_beat,
                "audio": segment,
                "action": "REORDER",
                "reorder": True
            })
            
        elif action.action_type == ActionType.JUMP_BACK:
            # Jump back in the audio (create a repeat)
            jump_beats = action.jump_beats
            target_beat = max(0, current_beat - jump_beats)
            target_start = beat_samples[target_beat]
            target_end = beat_samples[min(target_beat + n_action_beats, n_beats)]
            jumped_seg = audio[target_start:target_end].copy()
            segments.append({
                "beat": current_beat,
                "audio": jumped_seg,
                "action": f"JUMP_BACK_{jump_beats}"
            })
            
        elif action.action_type == ActionType.SKIP:
            # Skip to next phrase (just cut this segment)
            logger.debug(f"Beat {current_beat}-{end_beat}: SKIP")
            
        elif action.action_type == ActionType.REPEAT_PREV:
            # Repeat the previous segment
            if len(segments) > 0:
                prev_seg = segments[-1]["audio"].copy()
                segments.append({
                    "beat": current_beat,
                    "audio": prev_seg,
                    "action": "REPEAT_PREV"
                })
            else:
                # No previous, just keep
                segments.append({
                    "beat": current_beat,
                    "audio": segment,
                    "action": "KEEP"
                })
                
        elif action.action_type == ActionType.SWAP_NEXT:
            # For now, just keep (swap would require look-ahead)
            segments.append({
                "beat": current_beat,
                "audio": segment,
                "action": "SWAP_NEXT"
            })
            
        elif action.action_type in [ActionType.EQ_LOW, ActionType.EQ_HIGH, ActionType.DISTORTION, ActionType.REVERB]:
            # Effects not implemented - just keep the audio
            segments.append({
                "beat": current_beat,
                "audio": segment,
                "action": action.action_type.name
            })
        else:
            # Default: keep
            segments.append({
                "beat": current_beat,
                "audio": segment,
                "action": action.action_type.name
            })
        
        # Advance position
        current_beat = end_beat
        action_idx += 1
    
    if not segments:
        logger.warning("No segments kept! Returning original audio.")
        return audio
    
    # Concatenate segments with crossfades at edit boundaries
    if len(segments) == 1:
        edited = segments[0]["audio"]
    else:
        edited = segments[0]["audio"].copy()
        prev_beat = segments[0]["beat"]
        
        for seg in segments[1:]:
            current_beat = seg["beat"]
            segment_audio = seg["audio"]

            # Prefer applying a short crossfade at every boundary when possible
            use_crossfade = (
                crossfade_samples > 0
                and len(edited) > crossfade_samples
                and len(segment_audio) > crossfade_samples
            )

            if use_crossfade:
                edited = apply_crossfade(edited, segment_audio, crossfade_samples)
            else:
                edited = np.concatenate([edited, segment_audio])

            prev_beat = current_beat
    
    # Log statistics
    original_duration = len(audio) / sr
    edited_duration = len(edited) / sr
    
    action_counts = Counter([seg["action"].split("_")[0] for seg in segments])
    
    logger.info(f"Audio: {original_duration:.2f}s -> {edited_duration:.2f}s ({100*edited_duration/original_duration:.1f}%)")
    logger.info(f"Segments: {len(segments)} kept from {len(actions)} actions")
    logger.info(f"Actions used: {dict(action_counts)}")
    
    return edited


def main():
    parser = argparse.ArgumentParser(description="RL Audio Editor Inference (Factored Action Space)")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, default=None, help="Output audio file")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoint_best.pt", 
                       help="Model checkpoint path")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Use deterministic policy (default: stochastic)")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed for stochastic inference")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of stochastic samples to run and pick best by reward")
    parser.add_argument("--max-beats", type=int, default=0, help="Maximum beats to process (0 = no truncation)")
    parser.add_argument("--crossfade-ms", type=float, default=50.0, help="Crossfade duration in ms at edit boundaries")
    parser.add_argument("--use-auxiliary", action="store_true", default=False,
                       help="Load and run auxiliary task module during inference (mel recon, good/bad)")
    parser.add_argument("--save-aux-preds", type=str, default=None,
                       help="Path to save per-beat auxiliary predictions (npz). Saved for chosen best sample.")
    
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
    temp_env = AudioEditingEnvFactored(config, audio_state)
    obs, _ = temp_env.reset()
    input_dim = len(obs)
    logger.info(f"State dimension: {input_dim}")
    
    # Initialize agent
    agent = Agent(
        config,
        input_dim=input_dim,
        beat_feature_dim=audio_state.beat_features.shape[1],
        use_auxiliary_tasks=bool(args.use_auxiliary),
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    agent.load(str(checkpoint_path))
    
    # Run inference
    logger.info("Running inference with factored action space...")

    def set_seeds(s: int):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    n_samples = max(1, int(args.n_samples))
    best_score = float("-inf")
    best_audio = None
    best_actions = None
    best_action_names = None
    chosen_sample_idx = 0

    for i in range(n_samples):
        # choose seed for this sample
        if args.seed is not None:
            seed_i = args.seed + i
            set_seeds(seed_i)
            logger.info(f"Using seed={seed_i} for sample {i}")
        else:
            # randomize torch/np/random using python random
            seed_i = random.randrange(0, 2**31 - 1)
            set_seeds(seed_i)

        actions, action_names, total_reward, aux_preds, final_keep_ratio = run_inference(
            agent, config, audio_state,
            deterministic=args.deterministic,
            collect_aux=bool(args.use_auxiliary),
        )

        logger.info(f"Sample {i}: cumulative reward={total_reward:.4f} per-beat-keep_ratio={final_keep_ratio:.3f}")

        # Create edited audio for this sample
        edited_audio = create_edited_audio(
            audio, sr,
            audio_state.beat_times,
            actions,
            crossfade_ms=args.crossfade_ms,
        )

        # Save each sample temporarily
        input_path = Path(args.input)
        sample_out = None
        if args.output is None:
            sample_out = input_path.parent / f"{input_path.stem}_edited_sample{i}{input_path.suffix}"
        else:
            out_p = Path(args.output)
            sample_out = out_p.parent / f"{out_p.stem}_sample{i}{out_p.suffix}"

        sample_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(sample_out), edited_audio, sr)
        logger.info(f"Saved sample {i} to: {sample_out}")

        # choose best by cumulative reward
        if total_reward > best_score:
            best_score = total_reward
            best_audio = edited_audio
            best_actions = actions
            best_action_names = action_names
            best_keep_ratio = final_keep_ratio
            chosen_sample_idx = i

    # After sampling, determine final output path and save chosen sample
    input_path = Path(args.input)
    if args.output is None:
        final_out = input_path.parent / f"{input_path.stem}_edited{input_path.suffix}"
    else:
        final_out = Path(args.output)

    # Safety: avoid writing directly over the input file
    try:
        if final_out.resolve() == input_path.resolve():
            alt = input_path.parent / f"{input_path.stem}_edited{input_path.suffix}"
            counter = 1
            while alt.exists() and alt.resolve() == input_path.resolve():
                alt = input_path.parent / f"{input_path.stem}_edited{counter}{input_path.suffix}"
                counter += 1
            logger.warning(f"Requested output path is the same as input. Writing to {alt} instead to avoid overwriting input.")
            final_out = alt
    except Exception:
        if str(final_out) == str(input_path):
            final_out = input_path.parent / f"{input_path.stem}_edited{input_path.suffix}"
            logger.warning(f"Requested output path equals input; using {final_out} instead.")

    if best_audio is not None:
        final_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(final_out), best_audio, sr)
        try:
            keep_ratio_msg = f" per-beat-keep_ratio={best_keep_ratio:.3f}"
        except Exception:
            keep_ratio_msg = ""
        logger.info(f"Saved chosen best sample {chosen_sample_idx} (score={best_score:.4f}){keep_ratio_msg} to: {final_out}")
    else:
        logger.error("No samples produced - nothing saved")

    # expose the chosen actions distribution in logs
    actions = best_actions or []
    action_names = best_action_names or []

    # Analyze actions
    type_counter = Counter([a.action_type.name for a in actions])
    size_counter = Counter([a.action_size.name for a in actions])
    logger.info(f"Action type distribution: {dict(type_counter)}")
    logger.info(f"Action size distribution: {dict(size_counter)}")


if __name__ == "__main__":
    main()

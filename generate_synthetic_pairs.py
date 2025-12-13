"""Generate synthetic training pairs from reference tracks.

For each reference track (finished song), generate a "corrupted" input version by:
1. Randomly reordering sections (phrases/bars)
2. Looping some beats/bars
3. Adding extra repeated sections
4. Applying minor audio augmentations (gain, EQ, pitch shift)

The original reference becomes the "desired output" and the corrupted version
becomes the "input" - teaching the model to recognize and fix these issues.
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_beats(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Detect beats and return beat times and tempo."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, float(tempo)


def get_beat_segments(y: np.ndarray, sr: int, beat_times: np.ndarray) -> List[np.ndarray]:
    """Split audio into beat-aligned segments."""
    segments = []
    beat_samples = librosa.time_to_samples(beat_times, sr=sr)
    
    for i in range(len(beat_samples)):
        start = beat_samples[i]
        end = beat_samples[i + 1] if i + 1 < len(beat_samples) else len(y)
        segments.append(y[start:end])
    
    return segments


def get_bar_segments(y: np.ndarray, sr: int, beat_times: np.ndarray, beats_per_bar: int = 4) -> List[np.ndarray]:
    """Split audio into bar-aligned segments (typically 4 beats per bar)."""
    segments = []
    beat_samples = librosa.time_to_samples(beat_times, sr=sr)
    
    for i in range(0, len(beat_samples), beats_per_bar):
        start = beat_samples[i]
        end_idx = min(i + beats_per_bar, len(beat_samples))
        end = beat_samples[end_idx] if end_idx < len(beat_samples) else len(y)
        segments.append(y[start:end])
    
    return segments


def get_phrase_segments(y: np.ndarray, sr: int, beat_times: np.ndarray, beats_per_phrase: int = 8) -> List[np.ndarray]:
    """Split audio into phrase-aligned segments (typically 8 beats per phrase)."""
    segments = []
    beat_samples = librosa.time_to_samples(beat_times, sr=sr)
    
    for i in range(0, len(beat_samples), beats_per_phrase):
        start = beat_samples[i]
        end_idx = min(i + beats_per_phrase, len(beat_samples))
        end = beat_samples[end_idx] if end_idx < len(beat_samples) else len(y)
        segments.append(y[start:end])
    
    return segments


def corrupt_audio(
    y: np.ndarray, 
    sr: int,
    beat_times: np.ndarray,
    reorder_prob: float = 0.3,
    loop_prob: float = 0.2,
    extra_section_prob: float = 0.15,
    target_expansion: Tuple[float, float] = (1.65, 2.85),  # Original should be 35-60% of corrupted
) -> np.ndarray:
    """Corrupt audio by reordering sections, looping beats, adding extra sections.
    
    The corrupted version will be LONGER than the original, so that the original
    is 35-60% of the corrupted length (meaning the model needs to cut 40-65%).
    
    Args:
        y: Audio signal
        sr: Sample rate
        beat_times: Beat times in seconds
        reorder_prob: Probability of reordering a phrase
        loop_prob: Probability of looping a bar
        extra_section_prob: Probability of duplicating a phrase
        target_expansion: Range for length multiplier (1.65-2.85x means original is 35-60% of result)
    
    Returns:
        Corrupted audio signal
    """
    # Get different segment types
    beat_segments = get_beat_segments(y, sr, beat_times)
    bar_segments = get_bar_segments(y, sr, beat_times, beats_per_bar=4)
    phrase_segments = get_phrase_segments(y, sr, beat_times, beats_per_phrase=8)
    
    if len(phrase_segments) < 4:
        logger.warning("Track too short for meaningful corruption, using bars instead")
        phrase_segments = bar_segments
    
    corrupted_segments = []
    original_length = len(y)
    
    # Pick a random expansion factor so original is 35-60% of corrupted
    expansion = random.uniform(*target_expansion)
    target_length = int(original_length * expansion)
    current_length = 0
    
    # Work with phrases as the main unit
    phrase_indices = list(range(len(phrase_segments)))
    
    # Reorder some phrases
    if random.random() < reorder_prob and len(phrase_indices) > 3:
        # Swap 2-3 random phrase pairs
        n_swaps = random.randint(1, min(3, len(phrase_indices) // 2))
        for _ in range(n_swaps):
            i, j = random.sample(range(len(phrase_indices)), 2)
            phrase_indices[i], phrase_indices[j] = phrase_indices[j], phrase_indices[i]
    
    # Build corrupted audio - keep adding until we hit target length
    pass_count = 0
    max_passes = 5  # Prevent infinite loops
    
    while current_length < target_length and pass_count < max_passes:
        for idx in phrase_indices:
            if current_length >= target_length:
                break
                
            phrase = phrase_segments[idx]
            corrupted_segments.append(phrase)
            current_length += len(phrase)
            
            # Maybe loop this phrase (more aggressive looping to hit length target)
            loop_chance = loop_prob * (1 + (target_length - current_length) / target_length)
            if random.random() < loop_chance and current_length < target_length:
                # Loop 1-3 times
                n_loops = random.randint(1, 3)
                for _ in range(n_loops):
                    if current_length >= target_length:
                        break
                    corrupted_segments.append(phrase)
                    current_length += len(phrase)
            
            # Maybe add an extra copy of a random earlier phrase
            extra_chance = extra_section_prob * (1 + (target_length - current_length) / target_length)
            if random.random() < extra_chance and len(corrupted_segments) > 2 and current_length < target_length:
                random_earlier_idx = random.randint(0, len(corrupted_segments) - 2)
                extra_segment = corrupted_segments[random_earlier_idx]
                corrupted_segments.append(extra_segment)
                current_length += len(extra_segment)
        
        pass_count += 1
    
    # Concatenate
    corrupted = np.concatenate(corrupted_segments)
    
    # Log the actual ratio achieved
    actual_ratio = original_length / len(corrupted)
    logger.debug(f"  Target expansion: {expansion:.2f}x, Actual: {len(corrupted)/original_length:.2f}x, Original is {actual_ratio:.1%} of corrupted")
    
    return corrupted


def apply_audio_augmentation(
    y: np.ndarray,
    sr: int,
    gain_range: Tuple[float, float] = (-6.0, 6.0),  # dB
    pitch_range: Tuple[float, float] = (-1.0, 1.0),  # semitones
    eq_bands: int = 3,
    eq_gain_range: Tuple[float, float] = (-4.0, 4.0),  # dB
) -> np.ndarray:
    """Apply minor audio augmentations.
    
    Args:
        y: Audio signal
        sr: Sample rate
        gain_range: Random gain adjustment in dB
        pitch_range: Random pitch shift in semitones
        eq_bands: Number of EQ bands to adjust
        eq_gain_range: Random EQ gain per band in dB
    
    Returns:
        Augmented audio signal
    """
    augmented = y.copy()
    
    # Random gain
    gain_db = random.uniform(*gain_range)
    gain_linear = 10 ** (gain_db / 20)
    augmented = augmented * gain_linear
    
    # Random pitch shift (small, to preserve musicality)
    if random.random() < 0.5:  # 50% chance of pitch shift
        pitch_shift = random.uniform(*pitch_range)
        if abs(pitch_shift) > 0.1:  # Only apply if significant
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_shift)
    
    # Simple EQ simulation using filtering
    # Apply random boosts/cuts to low, mid, high frequencies
    if random.random() < 0.7:  # 70% chance of EQ
        # Low shelf (below 200Hz)
        low_gain = random.uniform(*eq_gain_range)
        # This is a simplified EQ - just amplitude scaling for different frequency ranges
        # A proper implementation would use biquad filters
        
        # For simplicity, we'll just add some coloration via mild filtering
        if abs(low_gain) > 1.0:
            # Boost or cut lows using a simple lowpass/highpass blend
            from scipy import signal
            
            # Low frequency component
            b_low, a_low = signal.butter(2, 200 / (sr / 2), btype='low')
            low_component = signal.filtfilt(b_low, a_low, augmented)
            
            # Apply gain to low component and blend back
            low_linear = 10 ** (low_gain / 20)
            augmented = augmented + low_component * (low_linear - 1)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(augmented))
    if max_val > 0.99:
        augmented = augmented * (0.99 / max_val)
    
    return augmented


def generate_synthetic_pair(
    reference_path: Path,
    output_dir: Path,
    sr: int = 22050,
    n_variations: int = 1,
) -> List[Tuple[Path, Path]]:
    """Generate synthetic input/output pairs from a reference track.
    
    Args:
        reference_path: Path to reference (finished) audio file
        output_dir: Directory to save generated pairs
        sr: Sample rate
        n_variations: Number of variations to generate per reference
    
    Returns:
        List of (input_path, output_path) tuples
    """
    # Load reference audio
    logger.info(f"Processing: {reference_path.name}")
    y, sr_orig = librosa.load(reference_path, sr=sr, mono=True)
    
    # Detect beats
    beat_times, tempo = detect_beats(y, sr)
    logger.info(f"  Detected {len(beat_times)} beats at {tempo:.1f} BPM")
    
    if len(beat_times) < 16:
        logger.warning(f"  Skipping - too few beats ({len(beat_times)})")
        return []
    
    pairs = []
    base_name = reference_path.stem
    
    # Create output directories
    input_dir = output_dir / "input"
    desired_dir = output_dir / "desired_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    desired_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_variations):
        variation_name = f"{base_name}_synth{i+1}" if n_variations > 1 else f"{base_name}_synth"
        
        # Corrupt the audio (target: original should be 35-60% of corrupted)
        corrupted = corrupt_audio(
            y, sr, beat_times,
            reorder_prob=random.uniform(0.2, 0.5),
            loop_prob=random.uniform(0.15, 0.35),
            extra_section_prob=random.uniform(0.1, 0.25),
            target_expansion=(1.65, 2.85),  # Original will be 35-60% of corrupted
        )
        
        # Apply augmentations to the corrupted version
        corrupted = apply_audio_augmentation(
            corrupted, sr,
            gain_range=(-4.0, 4.0),
            pitch_range=(-0.5, 0.5),
            eq_gain_range=(-3.0, 3.0),
        )
        
        # Save input (corrupted) and output (original)
        input_path = input_dir / f"{variation_name}_raw.wav"
        output_path = desired_dir / f"{variation_name}_edit.wav"
        
        sf.write(input_path, corrupted, sr)
        sf.write(output_path, y, sr)
        
        original_ratio = len(y) / len(corrupted)
        logger.info(f"  Generated: {variation_name}")
        logger.info(f"    Input (corrupted):  {len(corrupted)/sr:.1f}s")
        logger.info(f"    Output (original):  {len(y)/sr:.1f}s ({original_ratio:.1%} of input, need to cut {1-original_ratio:.1%})")
        
        pairs.append((input_path, output_path))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs from reference tracks")
    parser.add_argument("--reference_dir", type=str, default="./training_data/reference",
                        help="Directory containing reference tracks")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                        help="Output directory (will create input/ and desired_output/ subdirs)")
    parser.add_argument("--n_variations", type=int, default=1,
                        help="Number of variations to generate per reference track")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--max_tracks", type=int, default=None,
                        help="Maximum number of reference tracks to process")
    
    args = parser.parse_args()
    
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    
    if not reference_dir.exists():
        logger.error(f"Reference directory not found: {reference_dir}")
        return
    
    # Find all reference tracks
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    reference_files = [f for f in reference_dir.iterdir() if f.suffix.lower() in extensions]
    
    if args.max_tracks:
        reference_files = reference_files[:args.max_tracks]
    
    logger.info(f"Found {len(reference_files)} reference tracks")
    
    all_pairs = []
    for ref_path in reference_files:
        try:
            pairs = generate_synthetic_pair(
                ref_path,
                output_dir,
                sr=args.sr,
                n_variations=args.n_variations,
            )
            all_pairs.extend(pairs)
        except Exception as e:
            logger.error(f"Error processing {ref_path}: {e}")
            continue
    
    logger.info(f"\nGenerated {len(all_pairs)} synthetic pairs")
    logger.info(f"Input files saved to: {output_dir / 'input'}")
    logger.info(f"Output files saved to: {output_dir / 'desired_output'}")


if __name__ == "__main__":
    main()

"""Generate synthetic raw audio from edited tracks for training data augmentation.

Takes edited audio tracks and adds synthetic "imperfections" to create fake raw inputs:
- Duplicate sections (fake retakes/false starts)
- Silence gaps (fake pauses between takes)
- Repeated phrases (fake loops/warmups)
- Random section reordering (fake out-of-order recording)
- Extra intro/outro (fake count-ins, noodling)

The pointer network then learns to map these synthetic "raw" recordings back to the
clean edited version.

Usage:
    python -m pointer_network.generate_synthetic_data --edit-dir training_data/desired_output --output-dir training_data
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass, field


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    # Probability of each augmentation type
    duplicate_prob: float = 0.7  # Duplicate random sections
    silence_prob: float = 0.5  # Add silence gaps
    repeat_prob: float = 0.6  # Repeat phrases
    reorder_prob: float = 0.4  # Reorder sections
    intro_outro_prob: float = 0.5  # Add extra intro/outro

    # Augmentation parameters
    min_duplicates: int = 1
    max_duplicates: int = 5
    min_duplicate_len_sec: float = 2.0
    max_duplicate_len_sec: float = 15.0

    min_silences: int = 1
    max_silences: int = 3
    min_silence_len_sec: float = 0.5
    max_silence_len_sec: float = 3.0

    min_repeats: int = 1
    max_repeats: int = 3
    min_repeat_len_sec: float = 1.0
    max_repeat_len_sec: float = 8.0
    repeat_times: int = 2  # How many times to repeat each phrase

    reorder_chunk_sec: float = 10.0  # Size of chunks for reordering

    min_intro_len_sec: float = 2.0
    max_intro_len_sec: float = 10.0
    min_outro_len_sec: float = 2.0
    max_outro_len_sec: float = 10.0

    # Audio parameters
    sr: int = 22050
    hop_length: int = 256


@dataclass
class SyntheticResult:
    """Result of synthetic data generation."""

    raw_audio: np.ndarray  # Synthetic "raw" audio
    edit_audio: np.ndarray  # Original edited audio
    pointer_sequence: np.ndarray  # Frame-by-frame mapping: edit_frame -> raw_frame
    operations: List[dict]  # List of operations applied


def samples_to_frames(samples: int, hop_length: int) -> int:
    """Convert audio samples to mel spectrogram frames."""
    return samples // hop_length


def frames_to_samples(frames: int, hop_length: int) -> int:
    """Convert mel spectrogram frames to audio samples."""
    return frames * hop_length


def generate_silence(duration_sec: float, sr: int) -> np.ndarray:
    """Generate silence of given duration."""
    return np.zeros(int(duration_sec * sr))


def generate_noise(duration_sec: float, sr: int, amplitude: float = 0.001) -> np.ndarray:
    """Generate low-level noise (simulates room tone)."""
    return np.random.randn(int(duration_sec * sr)) * amplitude


class SyntheticDataGenerator:
    """Generates synthetic raw audio from edited tracks."""

    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()

    def generate(self, edit_audio: np.ndarray, sr: int) -> SyntheticResult:
        """Generate synthetic raw audio from edited audio.

        Args:
            edit_audio: Original edited audio
            sr: Sample rate

        Returns:
            SyntheticResult with synthetic raw, edit, pointers, and operations
        """
        # Start with a copy of the edit as the base
        # We'll build the synthetic raw by inserting/duplicating sections
        operations = []

        # Track segments: list of (audio_chunk, is_edit_content, edit_start_sample, edit_end_sample)
        # is_edit_content=True means this chunk maps to edit audio
        # is_edit_content=False means this chunk is synthetic (silence, duplicate)
        segments = [(edit_audio.copy(), True, 0, len(edit_audio))]

        # Apply augmentations
        if random.random() < self.config.intro_outro_prob:
            segments, op = self._add_intro_outro(segments, edit_audio, sr)
            operations.extend(op)

        if random.random() < self.config.duplicate_prob:
            segments, op = self._add_duplicates(segments, edit_audio, sr)
            operations.extend(op)

        if random.random() < self.config.silence_prob:
            segments, op = self._add_silences(segments, sr)
            operations.extend(op)

        if random.random() < self.config.repeat_prob:
            segments, op = self._add_repeats(segments, edit_audio, sr)
            operations.extend(op)

        if random.random() < self.config.reorder_prob:
            segments, op = self._reorder_segments(segments, sr)
            operations.extend(op)

        # Build final raw audio and pointer sequence
        raw_audio, pointer_sequence = self._build_output(segments, sr)

        return SyntheticResult(
            raw_audio=raw_audio,
            edit_audio=edit_audio,
            pointer_sequence=pointer_sequence,
            operations=operations,
        )

    def _add_intro_outro(
        self,
        segments: List[Tuple],
        edit_audio: np.ndarray,
        sr: int,
    ) -> Tuple[List[Tuple], List[dict]]:
        """Add synthetic intro and/or outro from edit audio content."""
        operations = []
        new_segments = list(segments)

        # Add intro (copy from somewhere in the edit)
        if random.random() < 0.7:
            intro_len = random.uniform(
                self.config.min_intro_len_sec, self.config.max_intro_len_sec
            )
            intro_samples = int(intro_len * sr)

            # Pick a random section from edit to use as intro
            if len(edit_audio) > intro_samples * 2:
                start = random.randint(0, len(edit_audio) - intro_samples)
                intro_audio = edit_audio[start : start + intro_samples].copy()

                # Add some variation (fade, slight pitch shift simulation via noise)
                intro_audio = self._apply_fade(intro_audio, fade_in=True, fade_out=True)
                intro_audio *= random.uniform(0.7, 1.0)  # Slightly quieter

                # Insert at beginning (this is synthetic content, not mapping to edit)
                new_segments.insert(0, (intro_audio, False, -1, -1))
                operations.append(
                    {"type": "INTRO", "duration_sec": intro_len, "source_start": start / sr}
                )

        # Add outro
        if random.random() < 0.5:
            outro_len = random.uniform(
                self.config.min_outro_len_sec, self.config.max_outro_len_sec
            )
            outro_samples = int(outro_len * sr)

            if len(edit_audio) > outro_samples * 2:
                start = random.randint(0, len(edit_audio) - outro_samples)
                outro_audio = edit_audio[start : start + outro_samples].copy()
                outro_audio = self._apply_fade(outro_audio, fade_in=True, fade_out=True)
                outro_audio *= random.uniform(0.6, 0.9)

                new_segments.append((outro_audio, False, -1, -1))
                operations.append(
                    {"type": "OUTRO", "duration_sec": outro_len, "source_start": start / sr}
                )

        return new_segments, operations

    def _add_duplicates(
        self,
        segments: List[Tuple],
        edit_audio: np.ndarray,
        sr: int,
    ) -> Tuple[List[Tuple], List[dict]]:
        """Add duplicate sections (fake retakes/false starts)."""
        operations = []
        new_segments = []

        n_duplicates = random.randint(self.config.min_duplicates, self.config.max_duplicates)

        for segment in segments:
            audio, is_edit, edit_start, edit_end = segment

            # Only duplicate edit content segments
            if not is_edit:
                new_segments.append(segment)
                continue

            # Decide where to insert duplicates in this segment
            segment_len = len(audio)
            min_dup_samples = int(self.config.min_duplicate_len_sec * sr)
            max_dup_samples = int(self.config.max_duplicate_len_sec * sr)

            if segment_len < min_dup_samples * 2:
                new_segments.append(segment)
                continue

            # Insert duplicates at random positions
            insert_points = sorted(
                random.sample(range(min_dup_samples, segment_len - min_dup_samples),
                              min(n_duplicates, (segment_len - 2 * min_dup_samples) // min_dup_samples))
            ) if segment_len > 3 * min_dup_samples else []

            current_pos = 0
            for insert_point in insert_points:
                # Add segment up to insert point
                if insert_point > current_pos:
                    chunk = audio[current_pos:insert_point]
                    chunk_edit_start = edit_start + current_pos if edit_start >= 0 else -1
                    chunk_edit_end = edit_start + insert_point if edit_start >= 0 else -1
                    new_segments.append((chunk, True, chunk_edit_start, chunk_edit_end))

                # Create duplicate (copy from nearby in edit)
                max_dup_allowed = min(max_dup_samples, segment_len // 3)
                if max_dup_allowed < min_dup_samples:
                    max_dup_allowed = min_dup_samples  # Fallback to min
                dup_len = random.randint(min_dup_samples, max_dup_allowed)
                dup_start = max(0, insert_point - dup_len)
                dup_audio = audio[dup_start : dup_start + dup_len].copy()

                # Add variation to make it sound like a retake
                dup_audio = self._apply_variation(dup_audio)

                new_segments.append((dup_audio, False, -1, -1))
                operations.append({
                    "type": "DUPLICATE",
                    "duration_sec": dup_len / sr,
                    "insert_at_sec": insert_point / sr,
                })

                current_pos = insert_point

            # Add remaining segment
            if current_pos < segment_len:
                chunk = audio[current_pos:]
                chunk_edit_start = edit_start + current_pos if edit_start >= 0 else -1
                chunk_edit_end = edit_end
                new_segments.append((chunk, True, chunk_edit_start, chunk_edit_end))

        return new_segments, operations

    def _add_silences(
        self,
        segments: List[Tuple],
        sr: int,
    ) -> Tuple[List[Tuple], List[dict]]:
        """Add silence gaps between segments."""
        operations = []
        new_segments = []

        n_silences = random.randint(self.config.min_silences, self.config.max_silences)

        # Pick random positions to insert silence
        if len(segments) > 1:
            silence_positions = sorted(
                random.sample(range(1, len(segments)), min(n_silences, len(segments) - 1))
            )
        else:
            silence_positions = []

        for i, segment in enumerate(segments):
            new_segments.append(segment)

            if i + 1 in silence_positions:
                silence_len = random.uniform(
                    self.config.min_silence_len_sec, self.config.max_silence_len_sec
                )
                # Use low-level noise instead of pure silence (more realistic)
                silence = generate_noise(silence_len, sr, amplitude=0.0005)
                new_segments.append((silence, False, -1, -1))
                operations.append({"type": "SILENCE", "duration_sec": silence_len})

        return new_segments, operations

    def _add_repeats(
        self,
        segments: List[Tuple],
        edit_audio: np.ndarray,
        sr: int,
    ) -> Tuple[List[Tuple], List[dict]]:
        """Add repeated phrases (fake warmup loops)."""
        operations = []
        new_segments = []

        n_repeats = random.randint(self.config.min_repeats, self.config.max_repeats)
        min_repeat_samples = int(self.config.min_repeat_len_sec * sr)
        max_repeat_samples = int(self.config.max_repeat_len_sec * sr)

        for segment in segments:
            audio, is_edit, edit_start, edit_end = segment

            if not is_edit or len(audio) < min_repeat_samples * 3:
                new_segments.append(segment)
                continue

            # Pick a section to repeat
            max_allowed = min(max_repeat_samples, len(audio) // 4)
            if random.random() < 0.5 and n_repeats > 0 and max_allowed >= min_repeat_samples:
                repeat_len = random.randint(min_repeat_samples, max_allowed)
                repeat_start = random.randint(0, len(audio) - repeat_len)

                repeat_audio = audio[repeat_start : repeat_start + repeat_len].copy()

                # Add the repeated section before the main segment
                for _ in range(self.config.repeat_times):
                    varied = self._apply_variation(repeat_audio.copy())
                    new_segments.append((varied, False, -1, -1))

                operations.append({
                    "type": "REPEAT",
                    "duration_sec": repeat_len / sr,
                    "times": self.config.repeat_times,
                })
                n_repeats -= 1

            new_segments.append(segment)

        return new_segments, operations

    def _reorder_segments(
        self,
        segments: List[Tuple],
        sr: int,
    ) -> Tuple[List[Tuple], List[dict]]:
        """Reorder some segments (simulate out-of-order recording)."""
        operations = []

        if len(segments) < 3:
            return segments, operations

        # Find edit content segments that can be reordered
        edit_indices = [i for i, (_, is_edit, _, _) in enumerate(segments) if is_edit]

        if len(edit_indices) < 3:
            return segments, operations

        # Swap a few segments
        n_swaps = random.randint(1, min(3, len(edit_indices) // 2))

        new_segments = list(segments)
        for _ in range(n_swaps):
            if len(edit_indices) < 2:
                break

            i, j = random.sample(edit_indices, 2)
            new_segments[i], new_segments[j] = new_segments[j], new_segments[i]

            operations.append({"type": "REORDER", "swap": [i, j]})

        return new_segments, operations

    def _apply_fade(
        self, audio: np.ndarray, fade_in: bool = False, fade_out: bool = False, fade_len: int = 1000
    ) -> np.ndarray:
        """Apply fade in/out to audio."""
        audio = audio.copy()
        fade_len = min(fade_len, len(audio) // 4)

        if fade_in and fade_len > 0:
            fade_curve = np.linspace(0, 1, fade_len)
            audio[:fade_len] *= fade_curve

        if fade_out and fade_len > 0:
            fade_curve = np.linspace(1, 0, fade_len)
            audio[-fade_len:] *= fade_curve

        return audio

    def _apply_variation(self, audio: np.ndarray) -> np.ndarray:
        """Apply slight variation to audio (simulates different take)."""
        audio = audio.copy()

        # Slight volume change
        audio *= random.uniform(0.85, 1.15)

        # Slight noise addition
        audio += np.random.randn(len(audio)) * 0.001

        # Fade edges
        audio = self._apply_fade(audio, fade_in=True, fade_out=True, fade_len=500)

        return audio

    def _build_output(
        self, segments: List[Tuple], sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build final raw audio and pointer sequence from segments.

        Returns:
            (raw_audio, pointer_sequence) where pointer_sequence maps
            each edit frame to its corresponding raw frame
        """
        hop_length = self.config.hop_length

        # Concatenate all segments to create raw audio
        raw_parts = []
        raw_sample_offset = 0

        # Track mapping: for each sample in edit audio, which sample in raw audio?
        # We'll convert to frames later
        edit_to_raw_samples = {}

        for audio, is_edit, edit_start, edit_end in segments:
            raw_parts.append(audio)

            if is_edit and edit_start >= 0:
                # This segment maps to edit audio
                for i, edit_sample in enumerate(range(edit_start, edit_end)):
                    if edit_sample < edit_end:  # Bounds check
                        edit_to_raw_samples[edit_sample] = raw_sample_offset + i

            raw_sample_offset += len(audio)

        raw_audio = np.concatenate(raw_parts)

        # Find the total edit length from the original segments
        edit_length = 0
        for _, is_edit, edit_start, edit_end in segments:
            if is_edit and edit_end > edit_length:
                edit_length = edit_end

        # Convert sample mapping to frame mapping
        n_edit_frames = samples_to_frames(edit_length, hop_length)
        n_raw_frames = samples_to_frames(len(raw_audio), hop_length)

        pointer_sequence = np.zeros(n_edit_frames, dtype=np.int64)

        for edit_frame in range(n_edit_frames):
            edit_sample = frames_to_samples(edit_frame, hop_length)

            # Find corresponding raw sample
            if edit_sample in edit_to_raw_samples:
                raw_sample = edit_to_raw_samples[edit_sample]
            else:
                # Interpolate from nearest known mapping
                known_samples = sorted(edit_to_raw_samples.keys())
                if not known_samples:
                    raw_sample = edit_sample  # Fallback
                elif edit_sample < known_samples[0]:
                    raw_sample = edit_to_raw_samples[known_samples[0]]
                elif edit_sample > known_samples[-1]:
                    raw_sample = edit_to_raw_samples[known_samples[-1]]
                else:
                    # Find nearest
                    idx = np.searchsorted(known_samples, edit_sample)
                    raw_sample = edit_to_raw_samples[known_samples[min(idx, len(known_samples) - 1)]]

            raw_frame = samples_to_frames(raw_sample, hop_length)
            raw_frame = max(0, min(raw_frame, n_raw_frames - 1))  # Clamp to valid range
            pointer_sequence[edit_frame] = raw_frame

        return raw_audio, pointer_sequence


def process_track(
    edit_path: Path,
    output_dir: Path,
    cache_dir: Path,
    generator: SyntheticDataGenerator,
    sr: int = 22050,
) -> Optional[dict]:
    """Process a single edited track to generate synthetic data.

    Args:
        edit_path: Path to edited audio file
        output_dir: Base output directory
        cache_dir: Directory for mel spectrograms
        generator: SyntheticDataGenerator instance
        sr: Sample rate

    Returns:
        Info dict if successful, None if failed
    """
    print(f"Processing: {edit_path.name}")

    # Load edited audio
    try:
        edit_audio, file_sr = librosa.load(edit_path, sr=sr)
    except Exception as e:
        print(f"  Error loading audio: {e}")
        return None

    # Generate synthetic raw
    result = generator.generate(edit_audio, sr)

    # Create output paths
    base_name = edit_path.stem.replace("_edit", "")
    synth_name = f"{base_name}_synth"

    raw_output_path = output_dir / "input" / f"{synth_name}_raw.wav"
    edit_output_path = output_dir / "desired_output" / f"{synth_name}_edit.wav"
    pointer_output_path = output_dir / "pointer_sequences" / f"{synth_name}_pointers.npy"
    info_output_path = output_dir / "pointer_sequences" / f"{synth_name}_info.json"

    # Ensure directories exist
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    edit_output_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save audio files
    sf.write(raw_output_path, result.raw_audio, sr)
    sf.write(edit_output_path, result.edit_audio, sr)

    # Save pointer sequence
    np.save(pointer_output_path, result.pointer_sequence)

    # Compute mel spectrograms and cache them
    hop_length = generator.config.hop_length
    n_mels = 128

    raw_mel = librosa.feature.melspectrogram(
        y=result.raw_audio, sr=sr, hop_length=hop_length, n_mels=n_mels
    )
    raw_mel_db = librosa.power_to_db(raw_mel, ref=np.max)

    edit_mel = librosa.feature.melspectrogram(
        y=result.edit_audio, sr=sr, hop_length=hop_length, n_mels=n_mels
    )
    edit_mel_db = librosa.power_to_db(edit_mel, ref=np.max)

    # Save mel spectrograms to cache
    features_dir = cache_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    raw_mel_path = features_dir / f"{synth_name}_raw.npz"
    edit_mel_path = features_dir / f"{synth_name}_edit.npz"

    np.savez(raw_mel_path, mel=raw_mel_db)
    np.savez(edit_mel_path, mel=edit_mel_db)

    # Create info file
    info = {
        "raw_path": str(raw_output_path),
        "edit_path": str(edit_output_path),
        "raw_frames": raw_mel_db.shape[1],
        "edit_frames": edit_mel_db.shape[1],
        "aligned_frames": len(result.pointer_sequence),
        "synthetic": True,
        "source_edit": str(edit_path),
        "operations": result.operations,
    }

    with open(info_output_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Generated: {synth_name}")
    print(f"    Raw frames: {info['raw_frames']}, Edit frames: {info['edit_frames']}")
    print(f"    Operations: {[op['type'] for op in result.operations]}")

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic raw audio from edited tracks"
    )
    parser.add_argument(
        "--edit-dir",
        type=str,
        default="F:/editorbot/training_data/desired_output",
        help="Directory containing edited audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="F:/editorbot/training_data",
        help="Base output directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="F:/editorbot/cache",
        help="Directory for mel spectrograms cache",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=1,
        help="Number of synthetic variants to generate per track",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tracks that already have synthetic versions",
    )

    args = parser.parse_args()

    edit_dir = Path(args.edit_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    # Find all edited tracks
    edit_files = list(edit_dir.glob("*_edit.wav"))
    print(f"Found {len(edit_files)} edited tracks")

    # Filter out tracks that already have synthetic versions
    if args.skip_existing:
        pointer_dir = output_dir / "pointer_sequences"
        existing = set()
        if pointer_dir.exists():
            for f in pointer_dir.glob("*_synth_info.json"):
                # Extract base name
                base = f.stem.replace("_synth_info", "").replace("_info", "")
                existing.add(base)

        edit_files = [
            f for f in edit_files
            if f.stem.replace("_edit", "") not in existing
        ]
        print(f"After filtering existing: {len(edit_files)} tracks to process")

    generator = SyntheticDataGenerator()

    successful = 0
    failed = 0

    for edit_path in edit_files:
        for variant in range(args.num_variants):
            # Add variant suffix if generating multiple
            if args.num_variants > 1:
                variant_name = f"{edit_path.stem}_v{variant}"
                # Would need to modify process_track to handle variants
                pass

            result = process_track(
                edit_path=edit_path,
                output_dir=output_dir,
                cache_dir=cache_dir,
                generator=generator,
                sr=args.sr,
            )

            if result:
                successful += 1
            else:
                failed += 1

    print(f"\nDone! Successful: {successful}, Failed: {failed}")


if __name__ == "__main__":
    main()

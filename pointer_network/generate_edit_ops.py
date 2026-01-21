"""Generate frame-level edit operation labels from pointer sequences.

For each frame in the edited audio, determine what edit operation is occurring:
- COPY: Normal 1:1 progression (pointer increments by 1)
- LOOP_START: Beginning of a repeated section
- LOOP_END: End of loop, jumping back to repeat
- SKIP: Jumped forward in raw audio (cut out content)
- FADE_IN: First frames of a new section (after SKIP or LOOP_END)
- FADE_OUT: Last frames before a jump (before SKIP or LOOP_START)
- STOP: End of sequence

Usage:
    # Generate ops for all existing pointer sequences
    python -m pointer_network.generate_edit_ops

    # Generate ops for specific samples
    python -m pointer_network.generate_edit_ops sample1 sample2
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse

# Import EditOp enum from model
import sys
sys.path.insert(0, '.')


class EditOp:
    """Edit operation types - must match pointer_network/models/pointer_network.py"""
    COPY = 0
    LOOP_START = 1
    LOOP_END = 2
    SKIP = 3
    FADE_IN = 4
    FADE_OUT = 5
    STOP = 6
    NUM_OPS = 7

    @classmethod
    def names(cls):
        return ['COPY', 'LOOP_START', 'LOOP_END', 'SKIP', 'FADE_IN', 'FADE_OUT', 'STOP']


def detect_edit_ops(pointers: np.ndarray,
                    skip_threshold: int = 50,
                    loop_threshold: int = 10,
                    fade_frames: int = 10) -> np.ndarray:
    """Generate frame-level edit operation labels from pointer sequence.

    Args:
        pointers: (T,) array of frame indices mapping edit -> raw
        skip_threshold: minimum forward jump to count as SKIP (cut)
        loop_threshold: minimum backward jump to count as LOOP
        fade_frames: number of frames to mark as FADE_IN/FADE_OUT around transitions

    Returns:
        ops: (T,) array of EditOp values
    """
    T = len(pointers)
    ops = np.full(T, EditOp.COPY, dtype=np.int32)

    if T == 0:
        return ops

    # Compute pointer deltas (how much pointer changes between consecutive frames)
    deltas = np.zeros(T, dtype=np.int32)
    deltas[1:] = pointers[1:] - pointers[:-1]
    deltas[0] = 1  # First frame is always "normal"

    # Track loop regions
    loop_starts = []  # Stack of (edit_frame, raw_frame) for open loops

    for i in range(T):
        delta = deltas[i]

        if i == T - 1:
            # Last frame
            ops[i] = EditOp.STOP

        elif delta < -loop_threshold:
            # Backward jump -> LOOP_END (going back to repeat a section)
            ops[i] = EditOp.LOOP_END

            # Mark the start of this loop (where we're jumping back to)
            target_raw = pointers[i]
            # Find where this target was first played (approximate loop start)
            for j in range(i):
                if abs(pointers[j] - target_raw) < loop_threshold:
                    # Found approximate loop start
                    if ops[j] == EditOp.COPY:
                        ops[j] = EditOp.LOOP_START
                    break

        elif delta > skip_threshold:
            # Forward jump -> SKIP (cut out content from raw)
            ops[i] = EditOp.SKIP

        elif delta == 0 or abs(delta) == 1:
            # Normal copy (delta of 0 or 1 is expected for continuous playback)
            ops[i] = EditOp.COPY

        else:
            # Small jump (2-49 frames) - still COPY but with minor drift
            ops[i] = EditOp.COPY

    # Add FADE_IN/FADE_OUT around transitions
    # Find transition points (SKIP, LOOP_START, LOOP_END)
    transitions = np.where(
        (ops == EditOp.SKIP) |
        (ops == EditOp.LOOP_START) |
        (ops == EditOp.LOOP_END)
    )[0]

    for t_idx in transitions:
        op_type = ops[t_idx]

        if op_type == EditOp.SKIP or op_type == EditOp.LOOP_END:
            # Mark frames BEFORE the jump as FADE_OUT
            start = max(0, t_idx - fade_frames)
            for j in range(start, t_idx):
                if ops[j] == EditOp.COPY:
                    ops[j] = EditOp.FADE_OUT

            # Mark frames AFTER the jump as FADE_IN
            end = min(T, t_idx + fade_frames + 1)
            for j in range(t_idx + 1, end):
                if ops[j] == EditOp.COPY:
                    ops[j] = EditOp.FADE_IN

        elif op_type == EditOp.LOOP_START:
            # Mark frames after loop start as FADE_IN (entering loop region)
            end = min(T, t_idx + fade_frames + 1)
            for j in range(t_idx + 1, end):
                if ops[j] == EditOp.COPY:
                    ops[j] = EditOp.FADE_IN

    return ops


def generate_ops_for_sample(pointer_file: Path, output_dir: Path = None) -> dict:
    """Generate edit ops for a single sample.

    Args:
        pointer_file: Path to {name}_pointers.npy file
        output_dir: Where to save ops (default: same dir as pointers)

    Returns:
        dict with statistics about the generated ops
    """
    if output_dir is None:
        output_dir = pointer_file.parent

    base_name = pointer_file.stem.replace("_pointers", "")

    # Load pointers
    pointers = np.load(pointer_file)

    # Generate ops
    ops = detect_edit_ops(pointers)

    # Save ops
    ops_file = output_dir / f"{base_name}_ops.npy"
    np.save(ops_file, ops)

    # Compute statistics
    op_counts = {}
    for i, name in enumerate(EditOp.names()):
        op_counts[name] = int(np.sum(ops == i))

    stats = {
        'total_frames': len(ops),
        'op_counts': op_counts,
        'ops_file': str(ops_file),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate edit ops from pointer sequences")
    parser.add_argument('samples', nargs='*', help='Specific sample names to process (default: all)')
    parser.add_argument('--pointer-dir', type=str,
                        default='F:/editorbot/training_data/pointer_sequences',
                        help='Directory containing pointer sequences')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate ops even if they already exist')
    args = parser.parse_args()

    pointer_dir = Path(args.pointer_dir)

    # Find pointer files
    if args.samples:
        pointer_files = [pointer_dir / f"{name}_pointers.npy" for name in args.samples]
        pointer_files = [f for f in pointer_files if f.exists()]
    else:
        pointer_files = list(pointer_dir.glob("*_pointers.npy"))

    print(f"Found {len(pointer_files)} pointer files")
    print("=" * 60)

    all_stats = []
    total_ops = {name: 0 for name in EditOp.names()}

    for pointer_file in tqdm(pointer_files, desc="Generating ops"):
        base_name = pointer_file.stem.replace("_pointers", "")
        ops_file = pointer_dir / f"{base_name}_ops.npy"

        # Skip if already exists and not forcing
        if ops_file.exists() and not args.force:
            # Load existing to count
            ops = np.load(ops_file)
            for i, name in enumerate(EditOp.names()):
                total_ops[name] += int(np.sum(ops == i))
            continue

        try:
            stats = generate_ops_for_sample(pointer_file)
            all_stats.append(stats)

            for name, count in stats['op_counts'].items():
                total_ops[name] += count

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("EDIT OPS SUMMARY")
    print("=" * 60)
    print(f"Processed {len(pointer_files)} samples")
    print("\nOperation counts:")
    total_frames = sum(total_ops.values())
    for name in EditOp.names():
        count = total_ops[name]
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"  {name:12s}: {count:8d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':12s}: {total_frames:8d}")
    print("=" * 60)


if __name__ == "__main__":
    main()

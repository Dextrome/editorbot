"""Generate ground truth pointer sequences by aligning raw ↔ edited audio.

For each frame in edited audio, find the corresponding frame in raw audio.
This creates training data for the pointer network.

Approach:
1. Extract mel spectrograms from both
2. For each chunk of edited, find best match in raw via cross-correlation
3. Build pointer sequence: edited_frame_i → raw_frame_j
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import json

# =============================================================================
# CONFIG
# =============================================================================
sr = 22050
n_fft = 2048
hop_length = 256
n_mels = 128

# Alignment params
chunk_seconds = 2.0  # Size of chunks to align
chunk_frames = int(chunk_seconds * sr / hop_length)
hop_seconds = 0.5  # How much to advance between chunks
hop_frames_align = int(hop_seconds * sr / hop_length)

# Search params
search_window_seconds = 30  # How far to search in raw audio (before/after expected position)
search_window_frames = int(search_window_seconds * sr / hop_length)

# Correlation threshold (reject matches below this)
min_correlation = 0.5


def extract_mel(audio_path):
    """Extract mel spectrogram from audio file."""
    print(f"Loading {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80
    mel_db = np.clip(mel_db, 0, 1)

    return mel_db.T, audio  # (T, n_mels), audio


def normalize_chunk(chunk):
    """Normalize chunk for correlation."""
    chunk = chunk - chunk.mean()
    norm = np.linalg.norm(chunk)
    if norm > 1e-8:
        chunk = chunk / norm
    return chunk


def find_best_match(edit_chunk, raw_mel, search_start, search_end):
    """Find best matching position in raw audio for an edited chunk.

    Returns: (best_position, correlation_score)
    """
    chunk_len = len(edit_chunk)

    # Clamp search range
    search_start = max(0, search_start)
    search_end = min(len(raw_mel) - chunk_len, search_end)

    if search_end <= search_start:
        return None, 0.0

    # Flatten and normalize edit chunk
    edit_flat = normalize_chunk(edit_chunk.flatten())

    best_pos = None
    best_corr = -1

    # Slide through search window
    for pos in range(search_start, search_end):
        raw_chunk = raw_mel[pos:pos + chunk_len]
        raw_flat = normalize_chunk(raw_chunk.flatten())

        # Compute correlation
        corr = np.dot(edit_flat, raw_flat)

        if corr > best_corr:
            best_corr = corr
            best_pos = pos

    return best_pos, best_corr


def align_audio_pair(raw_path, edit_path):
    """Align edited audio to raw audio and generate pointer sequence.

    Returns:
        pointers: array of shape (edit_frames,) mapping each edit frame to raw frame
        alignment_info: dict with metadata
    """
    # Load audio
    raw_mel, raw_audio = extract_mel(raw_path)
    edit_mel, edit_audio = extract_mel(edit_path)

    print(f"Raw: {len(raw_mel)} frames ({len(raw_mel) * hop_length / sr:.1f}s)")
    print(f"Edit: {len(edit_mel)} frames ({len(edit_mel) * hop_length / sr:.1f}s)")

    # Initialize pointer array (-1 = unaligned)
    pointers = np.full(len(edit_mel), -1, dtype=np.int32)

    # Track alignment quality
    correlations = []

    # Expected position tracking (assume roughly sequential with jumps)
    expected_raw_pos = 0

    print(f"\nAligning chunks (chunk={chunk_seconds}s, hop={hop_seconds}s)...")

    # Align each chunk of edited audio
    chunk_starts = list(range(0, len(edit_mel) - chunk_frames + 1, hop_frames_align))

    for edit_start in tqdm(chunk_starts):
        edit_end = edit_start + chunk_frames
        edit_chunk = edit_mel[edit_start:edit_end]

        # Search window centered on expected position
        search_start = expected_raw_pos - search_window_frames
        search_end = expected_raw_pos + search_window_frames

        # Find best match
        raw_pos, corr = find_best_match(edit_chunk, raw_mel, search_start, search_end)

        if raw_pos is not None and corr >= min_correlation:
            # Fill in pointers for this chunk
            for i in range(chunk_frames):
                edit_idx = edit_start + i
                raw_idx = raw_pos + i
                if edit_idx < len(pointers) and raw_idx < len(raw_mel):
                    if pointers[edit_idx] == -1:  # Don't overwrite
                        pointers[edit_idx] = raw_idx

            # Update expected position (allow for cuts)
            expected_raw_pos = raw_pos + chunk_frames
            correlations.append(corr)
        else:
            # No good match found - might be generated content or heavy processing
            correlations.append(corr if corr else 0.0)
            # Don't update expected position

    # Interpolate any remaining gaps
    pointers = interpolate_gaps(pointers)

    # Compute statistics
    valid_pointers = pointers[pointers >= 0]
    alignment_info = {
        'raw_path': str(raw_path),
        'edit_path': str(edit_path),
        'raw_frames': int(len(raw_mel)),
        'edit_frames': int(len(edit_mel)),
        'aligned_frames': int(len(valid_pointers)),
        'alignment_ratio': float(len(valid_pointers) / len(edit_mel)),
        'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
        'min_correlation': float(np.min(correlations)) if correlations else 0.0,
    }

    return pointers, alignment_info


def interpolate_gaps(pointers):
    """Fill small gaps in pointer sequence via interpolation."""
    pointers = pointers.copy()

    # Find gaps
    gap_start = None
    for i in range(len(pointers)):
        if pointers[i] == -1:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_end = i
                gap_len = gap_end - gap_start

                # Only interpolate small gaps
                if gap_len <= 50 and gap_start > 0:
                    start_val = pointers[gap_start - 1]
                    end_val = pointers[gap_end]

                    # Linear interpolation
                    for j in range(gap_len):
                        t = (j + 1) / (gap_len + 1)
                        pointers[gap_start + j] = int(start_val + t * (end_val - start_val))

                gap_start = None

    return pointers


def visualize_alignment(pointers, raw_frames, edit_frames, output_path=None):
    """Create visualization of the alignment."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Pointer values over time
    ax1 = axes[0]
    valid_mask = pointers >= 0
    edit_times = np.arange(len(pointers)) * hop_length / sr

    ax1.scatter(edit_times[valid_mask], pointers[valid_mask] * hop_length / sr,
                s=1, alpha=0.5, c='blue')
    ax1.set_xlabel('Edited audio time (s)')
    ax1.set_ylabel('Raw audio time (s)')
    ax1.set_title('Alignment: Edit frame → Raw frame')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Derivative (shows cuts/jumps)
    ax2 = axes[1]
    pointer_diff = np.diff(pointers[valid_mask].astype(float))
    times_diff = edit_times[valid_mask][1:]

    ax2.plot(times_diff, pointer_diff, linewidth=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Normal (1:1)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Repeat')
    ax2.set_xlabel('Edited audio time (s)')
    ax2.set_ylabel('Pointer delta (frames)')
    ax2.set_title('Pointer changes (jumps = cuts in raw, 0 = loops)')
    ax2.set_ylim(-100, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def detect_edit_points(pointers, min_jump=50):
    """Detect where cuts/jumps occur in the edit."""
    edit_points = []

    valid_mask = pointers >= 0
    valid_indices = np.where(valid_mask)[0]
    valid_pointers = pointers[valid_mask]

    for i in range(1, len(valid_pointers)):
        jump = valid_pointers[i] - valid_pointers[i-1]

        if abs(jump) > min_jump or jump < 0:
            edit_idx = valid_indices[i]
            edit_time = edit_idx * hop_length / sr
            raw_from = valid_pointers[i-1] * hop_length / sr
            raw_to = valid_pointers[i] * hop_length / sr

            if jump > min_jump:
                edit_type = "CUT"  # Skipped forward in raw
            elif jump < -min_jump:
                edit_type = "JUMP_BACK"  # Went backwards (loop/repeat)
            else:
                edit_type = "SMALL_JUMP"

            edit_points.append({
                'edit_time': float(edit_time),
                'raw_from': float(raw_from),
                'raw_to': float(raw_to),
                'jump_frames': int(jump),
                'type': edit_type
            })

    return edit_points


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Find audio pairs
    input_dir = Path('F:/editorbot/training_data/input')
    output_dir = Path('F:/editorbot/training_data/desired_output')
    save_dir = Path('F:/editorbot/training_data/pointer_sequences')
    save_dir.mkdir(exist_ok=True)

    # Find matching pairs
    raw_files = list(input_dir.glob('*_raw.*'))

    print(f"Found {len(raw_files)} raw files")
    print("=" * 60)

    all_results = []

    for raw_path in raw_files:
        # Find corresponding edit file
        base_name = raw_path.stem.replace('_raw', '')

        # Skip if already processed
        if (save_dir / f"{base_name}_pointers.npy").exists():
            print(f"Skipping {base_name} (already processed)")
            continue

        # Try different edit file patterns
        edit_patterns = [
            f"{base_name}_edit.wav",
            f"{base_name}_edit.mp3",
            f"{base_name}.wav",
            f"{base_name}.mp3",
        ]

        edit_path = None
        for pattern in edit_patterns:
            candidate = output_dir / pattern
            if candidate.exists():
                edit_path = candidate
                break

        if edit_path is None:
            print(f"No edit file found for {raw_path.name}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {raw_path.name} <-> {edit_path.name}")
        print(f"{'='*60}")

        try:
            # Align
            pointers, info = align_audio_pair(raw_path, edit_path)

            # Detect edit points
            edit_points = detect_edit_points(pointers)
            info['edit_points'] = edit_points
            info['n_cuts'] = len([e for e in edit_points if e['type'] == 'CUT'])
            info['n_loops'] = len([e for e in edit_points if e['type'] == 'JUMP_BACK'])

            # Save
            pair_name = base_name
            np.save(save_dir / f"{pair_name}_pointers.npy", pointers)

            with open(save_dir / f"{pair_name}_info.json", 'w') as f:
                json.dump(info, f, indent=2)

            # Visualize
            visualize_alignment(
                pointers, info['raw_frames'], info['edit_frames'],
                save_dir / f"{pair_name}_alignment.png"
            )

            # Print summary
            print(f"\nResults for {pair_name}:")
            print(f"  Alignment ratio: {info['alignment_ratio']:.1%}")
            print(f"  Mean correlation: {info['mean_correlation']:.3f}")
            print(f"  Detected cuts: {info['n_cuts']}")
            print(f"  Detected loops: {info['n_loops']}")

            if edit_points:
                print(f"\n  First 5 edit points:")
                for ep in edit_points[:5]:
                    print(f"    {ep['edit_time']:.1f}s: {ep['type']} "
                          f"(raw {ep['raw_from']:.1f}s -> {ep['raw_to']:.1f}s)")

            all_results.append(info)

        except Exception as e:
            print(f"Error processing {raw_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed {len(all_results)} pairs")

    if all_results:
        avg_corr = np.mean([r['mean_correlation'] for r in all_results])
        avg_align = np.mean([r['alignment_ratio'] for r in all_results])
        total_cuts = sum(r['n_cuts'] for r in all_results)
        total_loops = sum(r['n_loops'] for r in all_results)

        print(f"Average correlation: {avg_corr:.3f}")
        print(f"Average alignment ratio: {avg_align:.1%}")
        print(f"Total cuts detected: {total_cuts}")
        print(f"Total loops detected: {total_loops}")

    print(f"\nPointer sequences saved to: {save_dir}")

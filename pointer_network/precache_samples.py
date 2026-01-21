"""Precache mel spectrograms and generate pointer sequences for new samples.

Usage:
    python -m pointer_network.precache_samples [sample_names...]

If no sample names provided, processes all samples not yet cached.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
from pathlib import Path
import argparse

# Audio params (must match training)
SR = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128


def extract_and_cache_mel(audio_path: Path, cache_dir: Path) -> np.ndarray:
    """Extract mel spectrogram and save to cache.

    Returns:
        mel: (n_mels, time) array
    """
    cache_path = cache_dir / "features" / f"{audio_path.stem}.npz"

    # Check if already cached
    if cache_path.exists():
        print(f"  Loading cached mel: {cache_path.name}")
        data = np.load(cache_path)
        return data['mel']

    print(f"  Extracting mel: {audio_path.name}")
    audio, _ = librosa.load(audio_path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80  # Normalize to [0, 1]
    mel_db = np.clip(mel_db, 0, 1)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mel=mel_db)
    print(f"  Cached: {cache_path.name} ({mel_db.shape[1]} frames)")

    return mel_db


def main():
    parser = argparse.ArgumentParser(description="Precache mel spectrograms for pointer network")
    parser.add_argument("samples", nargs="*", help="Sample names to process (without _raw/_edit suffix)")
    parser.add_argument("--cache-dir", default="F:/editorbot/cache", help="Cache directory")
    parser.add_argument("--input-dir", default="F:/editorbot/training_data/input", help="Raw audio directory")
    parser.add_argument("--output-dir", default="F:/editorbot/training_data/desired_output", help="Edited audio directory")
    parser.add_argument("--pointer-dir", default="F:/editorbot/training_data/pointer_sequences", help="Pointer sequences directory")
    parser.add_argument("--skip-pointers", action="store_true", help="Only cache mels, skip pointer generation")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pointer_dir = Path(args.pointer_dir)

    # Find samples to process
    if args.samples:
        sample_names = args.samples
    else:
        # Find all raw files
        raw_files = list(input_dir.glob("*_raw.*"))
        sample_names = [f.stem.replace("_raw", "") for f in raw_files]

    print(f"Processing {len(sample_names)} samples")
    print("=" * 60)

    for name in sample_names:
        print(f"\n[{name}]")

        # Find raw file
        raw_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            candidate = input_dir / f"{name}_raw{ext}"
            if candidate.exists():
                raw_path = candidate
                break

        if raw_path is None:
            print(f"  WARNING: No raw file found for {name}")
            continue

        # Find edit file
        edit_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            for suffix in ["_edit", ""]:
                candidate = output_dir / f"{name}{suffix}{ext}"
                if candidate.exists():
                    edit_path = candidate
                    break
            if edit_path:
                break

        if edit_path is None:
            print(f"  WARNING: No edit file found for {name}")
            continue

        # Cache mel spectrograms
        print(f"  Raw: {raw_path.name}")
        extract_and_cache_mel(raw_path, cache_dir)

        print(f"  Edit: {edit_path.name}")
        extract_and_cache_mel(edit_path, cache_dir)

        # Check if pointer sequence already exists
        pointer_path = pointer_dir / f"{name}_pointers.npy"
        if pointer_path.exists() and not args.skip_pointers:
            print(f"  Pointer sequence already exists: {pointer_path.name}")
        elif not args.skip_pointers:
            print(f"  Generating pointer sequence...")
            # Import here to avoid circular imports
            from pointer_network.generate_pointer_sequences import align_audio_pair, detect_edit_points, visualize_alignment
            import json

            try:
                pointers, info = align_audio_pair(raw_path, edit_path)
                edit_points = detect_edit_points(pointers)
                info['edit_points'] = edit_points
                info['n_cuts'] = len([e for e in edit_points if e['type'] == 'CUT'])
                info['n_loops'] = len([e for e in edit_points if e['type'] == 'JUMP_BACK'])

                # Save
                pointer_dir.mkdir(parents=True, exist_ok=True)
                np.save(pointer_path, pointers)

                with open(pointer_dir / f"{name}_info.json", 'w') as f:
                    json.dump(info, f, indent=2)

                visualize_alignment(
                    pointers, info['raw_frames'], info['edit_frames'],
                    pointer_dir / f"{name}_alignment.png"
                )

                print(f"  Alignment ratio: {info['alignment_ratio']:.1%}")
                print(f"  Detected cuts: {info['n_cuts']}, loops: {info['n_loops']}")

            except Exception as e:
                print(f"  ERROR generating pointers: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()

"""Regenerate super_editor cache with all audio pairs."""

import os
from pathlib import Path
from super_editor.config import AudioConfig
from super_editor.data.preprocessing import process_audio_pair

def main():
    input_dir = Path("training_data/input")
    output_dir_base = Path("training_data/desired_output")
    cache_dir = Path("training_data/super_editor_cache")

    config = AudioConfig()

    # Find all pairs (support both wav and mp3)
    pairs = []
    for ext in ["*.wav", "*.mp3"]:
        for raw_file in input_dir.glob(f"*_raw{ext[1:]}"):
            pair_id = raw_file.stem.replace("_raw", "")

            # Try both wav and mp3 for edit file
            edit_file = None
            for edit_ext in [".wav", ".mp3"]:
                candidate = output_dir_base / f"{pair_id}_edit{edit_ext}"
                if candidate.exists():
                    edit_file = candidate
                    break

            if edit_file:
                pairs.append((str(raw_file), str(edit_file), pair_id))
            else:
                print(f"No edit file for {pair_id}")

    print(f"Found {len(pairs)} pairs to process")

    # Check what's already cached
    features_dir = cache_dir / "features"
    existing = set()
    if features_dir.exists():
        for f in features_dir.glob("*_raw.npz"):
            existing.add(f.stem.replace("_raw", ""))

    # Process only new pairs
    new_pairs = [(r, e, p) for r, e, p in pairs if p not in existing]
    print(f"Already cached: {len(existing)}, new to process: {len(new_pairs)}")

    if not new_pairs:
        print("All pairs already cached!")
        return

    # Process each new pair
    for i, (raw_path, edit_path, pair_id) in enumerate(new_pairs):
        print(f"\n[{i+1}/{len(new_pairs)}] Processing {pair_id}...")
        try:
            process_audio_pair(
                raw_path, edit_path, str(cache_dir), pair_id, config, infer_labels=True
            )
            print(f"  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    # Final count
    final_count = len(list(features_dir.glob("*_raw.npz")))
    print(f"\nTotal cached pairs: {final_count}")

if __name__ == "__main__":
    main()

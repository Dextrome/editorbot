"""Precompute and cache mel spectrograms and per-beat mel summaries for paired data.

Usage:
    python -m scripts.precache_mels --data_dir training_data --cache_dir feature_cache/paired_mels --force

This script uses the project's `PairedAudioDataset` processing to ensure cached
features match training preprocessing. It saves cache files via
`PairedAudioDataset._save_to_cache` using a `mel_` prefix so they integrate with
the existing cache layout.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import librosa
import torch

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset

logger = logging.getLogger("precache_mels")


def compute_per_beat_mel(mel: np.ndarray, beat_times: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Aggregate mel spectrogram into per-beat mel vectors.

    mel: (n_mels, n_frames)
    beat_times: (n_beats,)
    Returns (n_beats, n_mels)
    """
    if mel is None or mel.size == 0 or len(beat_times) == 0:
        return np.zeros((0, mel.shape[0] if mel is not None else 0), dtype=np.float32)

    n_mels, n_frames = mel.shape
    # Convert beat_times (seconds) to frames used by mel (hop_length)
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    frames = np.concatenate(([0], frames, [n_frames]))
    n_beats = len(beat_times)
    per_beat = np.zeros((n_beats, n_mels), dtype=np.float32)

    for i in range(n_beats):
        s = int(frames[i])
        e = int(frames[i + 1])
        if s >= n_frames:
            per_beat[i] = mel[:, -1]
            continue
        if e <= s:
            e = min(s + 1, n_frames)
        seg = mel[:, s:e]
        if seg.size == 0:
            per_beat[i] = mel[:, min(s, n_frames - 1)]
        else:
            per_beat[i] = seg.mean(axis=1)

    return per_beat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache entries")
    parser.add_argument("--n_files", type=int, default=0, help="Process only first N pairs (0 = all)")
    args = parser.parse_args()

    config = get_default_config()
    cache_dir = args.cache_dir or (Path(config.data.cache_dir) if hasattr(config.data, 'cache_dir') else Path("feature_cache/paired_mels"))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedAudioDataset(
        data_dir=args.data_dir,
        config=config,
        cache_dir=str(cache_dir),
        include_reference=True,
        use_augmentation=False,
    )

    pairs = dataset.pairs
    total = len(pairs)
    logger.info(f"Found {total} pairs to process")
    if args.n_files > 0:
        pairs = pairs[: args.n_files]

    for i, (raw_path, edited_path) in enumerate(pairs):
        # Determine canonical cache path used by PairedAudioDataset
        existing_cache_path = dataset._get_cache_path(edited_path, prefix="")
        # If there's an existing cached file, inspect it for per-beat mel
        if existing_cache_path and existing_cache_path.exists():
            try:
                cached = torch.load(existing_cache_path)
                if "per_beat_mel" in cached and not args.force:
                    logger.info(f"Per-beat mel already cached for {edited_path.name}; skipping")
                    continue
            except Exception:
                # If loading fails, we'll recompute and overwrite if --force
                if not args.force:
                    logger.warning(f"Could not read cache {existing_cache_path}; recomputing (use --force to overwrite)")

        try:
            processed = dataset._process_file(edited_path)
            # processed returns torch tensors for mel and beat_times
            mel_t = processed.get("mel")
            beat_times_t = processed.get("beat_times")
            sr = int(processed.get("sample_rate", config.audio.sample_rate))
            hop_length = int(getattr(config.audio, 'hop_length', 512))

            if hasattr(mel_t, "numpy"):
                mel_np = mel_t.numpy()
            else:
                mel_np = np.array(mel_t)

            if hasattr(beat_times_t, "numpy"):
                beat_times = beat_times_t.numpy()
            else:
                beat_times = np.array(beat_times_t)

            per_beat = compute_per_beat_mel(mel_np, beat_times, sr=sr, hop_length=hop_length)

            # Save using PairedAudioDataset cache helper (integrates with existing cache layout)
            cache_data = {
                "mel": mel_np,
                "per_beat_mel": per_beat,
            }
            # Use empty prefix so it augments the existing paired cache file
            dataset._save_to_cache(edited_path, cache_data, prefix="")
            logger.info(f"Cached mel/per-beat for {edited_path.name} ({i+1}/{total})")
        except Exception as e:
            logger.exception(f"Failed to process {edited_path}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()

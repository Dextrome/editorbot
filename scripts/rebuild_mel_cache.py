"""Rebuild missing mel caches for paired dataset.

Usage:
    python scripts/rebuild_mel_cache.py [--data_dir PATH] [--cache_dir PATH] [--limit N] [--force]

This script iterates all paired raw/edited files (and reference tracks) and ensures
there is a cached mel spectrogram for each audio file. It is idempotent and will
skip files that already have cached mel unless `--force` is used.

It uses `PairedAudioDataset._process_file()` to compute features and let the
existing FeatureCache save functions persist results.
"""

import argparse
import logging
import sys
from pathlib import Path

import tqdm

# Ensure project root is on PYTHONPATH so `rl_editor` package imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset


logger = logging.getLogger("rebuild_mel_cache")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Rebuild missing mel cache for paired dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir override (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs processed (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force recompute and overwrite existing mel cache")
    parser.add_argument("--dry_run", action="store_true", help="Don't write anything, just report missing items")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir

    ds = PairedAudioDataset(cfg.data.data_dir, cfg, cache_dir=cfg.data.cache_dir, use_augmentation=False)

    # Collect unique file paths to check (raw + edited + references)
    file_paths = []
    for raw_path, edited_path in ds.pairs:
        file_paths.append(Path(raw_path))
        file_paths.append(Path(edited_path))
    for ref in ds.reference_files:
        file_paths.append(Path(ref))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for p in file_paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(p)

    logger.info(f"Found {len(unique_files)} files to check (pairs+references)")

    fc = ds.feature_cache
    if fc is None:
        logger.error("Feature cache is not enabled. Enable caching in config.data.cache_features.")
        sys.exit(1)

    total = len(unique_files) if args.limit <= 0 else min(len(unique_files), args.limit)

    n_fixed = 0
    n_missing = 0

    for p in tqdm.tqdm(unique_files[:total], desc="Checking files"):
        try:
            has_mel = fc.has_cached(p, cache_type="mel")
            if has_mel and not args.force:
                continue

            # If dry run, just report
            if args.dry_run:
                if not has_mel:
                    logger.info(f"MISSING mel: {p}")
                    n_missing += 1
                continue

            # Compute features (this will save mel via _process_file)
            logger.info(f"Computing features for: {p}")
            # Use dataset's processing (avoids duplicating logic and ensures cache saving)
            ds._process_file(p)

            # Verify mel cached after processing
            if fc.has_cached(p, cache_type="mel"):
                n_fixed += 1
            else:
                logger.warning(f"Processed but mel still missing for: {p}")
        except Exception as e:
            logger.exception(f"Failed processing {p}: {e}")

    logger.info(f"Done. mel computed for {n_fixed} files; missing reported: {n_missing}")


if __name__ == "__main__":
    main()

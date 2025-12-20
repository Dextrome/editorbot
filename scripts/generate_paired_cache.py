"""Generate or update paired .pt caches in-place.

Usage:
    python scripts/generate_paired_cache.py [--data_dir PATH] [--cache_dir PATH] [--limit N] [--force]

This script iterates paired raw/edited files and reference tracks, processes each
file through `PairedAudioDataset._process_file()` and saves the result into the
paired cache using `PairedAudioDataset._save_to_cache()`.

It will skip files that already have a paired cache unless `--force` is specified.
"""
import argparse
import logging
import sys
from pathlib import Path

import tqdm

# Ensure workspace root on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("generate_paired_cache")


def main():
    parser = argparse.ArgumentParser(description="Generate/update paired .pt caches")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir override (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs processed (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force recompute and overwrite existing paired caches")
    parser.add_argument("--dry_run", action="store_true", help="Don't write anything, just report which files would be updated")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir

    ds = PairedAudioDataset(cfg.data.data_dir, cfg, cache_dir=cfg.data.cache_dir, use_augmentation=False)

    # Build list of files: raw + edited + reference
    file_paths = []
    for raw_path, edited_path in ds.pairs:
        file_paths.append(Path(raw_path))
        file_paths.append(Path(edited_path))
    for ref in ds.reference_files:
        file_paths.append(Path(ref))

    # Deduplicate
    seen = set()
    unique_files = []
    for p in file_paths:
        k = str(Path(p).resolve())
        if k not in seen:
            seen.add(k)
            unique_files.append(p)

    logger.info(f"Found {len(unique_files)} files (pairs+references)")

    total = len(unique_files) if args.limit <= 0 else min(len(unique_files), args.limit)
    n_updated = 0
    n_skipped = 0

    for p in tqdm.tqdm(unique_files[:total], desc="Processing files"):
        try:
            # Determine cache path and whether to skip
            # For reference files, cache prefix is 'ref_'
            prefix = "ref_" if str(p) in [str(r) for r in ds.reference_files] else ""
            cache_path = ds._get_cache_path(p, prefix=prefix)
            exists = cache_path.exists() if cache_path is not None else False

            if exists and not args.force:
                n_skipped += 1
                continue

            if args.dry_run:
                logger.info(f"Would update: {p} (prefix='{prefix}')")
                n_updated += 1
                continue

            # Process the file (this will compute mel, beat_features, etc.)
            result = ds._process_file(p)

            # Save full processed dict into paired cache using existing helper
            ds._save_to_cache(p, result, prefix=prefix)
            n_updated += 1
        except Exception as e:
            logger.exception(f"Failed to process {p}: {e}")

    logger.info(f"Done. Updated: {n_updated}; Skipped: {n_skipped}")


if __name__ == '__main__':
    main()

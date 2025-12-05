#!/usr/bin/env python
"""Pre-cache Demucs stems for faster training.

Usage:
    python -m rl_editor.precache_stems --data_dir ./training_data --cache_dir ./feature_cache/rl_editor

This script runs Demucs separation on all audio files and caches the results
to disk, so training with stem features doesn't require on-the-fly separation.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache Demucs stems for faster training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./feature_cache/rl_editor",
        help="Directory to store cached stems",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate for output stems",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        help="Demucs model to use (htdemucs, htdemucs_ft, etc.)",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["input", "desired_output", "reference", "test_input"],
        help="Subdirectories to process",
    )
    
    args = parser.parse_args()
    
    # Check if Demucs is available
    try:
        from rl_editor.features import StemProcessor, precache_stems_for_directory
    except ImportError as e:
        logger.error(f"Failed to import stem processing: {e}")
        logger.error("Make sure you have installed demucs: pip install demucs")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Process each subdirectory
    total_processed = 0
    for subdir in args.subdirs:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            logger.info(f"Subdirectory not found, skipping: {subdir_path}")
            continue
        
        logger.info(f"Processing {subdir_path}...")
        
        try:
            precache_stems_for_directory(
                audio_dir=str(subdir_path),
                cache_dir=args.cache_dir,
                sr=args.sr,
                model=args.model,
            )
        except Exception as e:
            logger.error(f"Failed to process {subdir_path}: {e}")
            continue
    
    logger.info("Stem pre-caching complete!")
    logger.info(f"Cached stems are stored in: {args.cache_dir}/stems/")


if __name__ == "__main__":
    main()

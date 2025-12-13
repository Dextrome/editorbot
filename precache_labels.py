"""Pre-cache edit labels for all training pairs.

The DTW-based label inference is expensive. Run this before training
to cache all labels upfront.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Pre-cache edit labels for training pairs")
    parser.add_argument("--data_dir", type=str, default="./training_data")
    parser.add_argument("--cache_dir", type=str, default="./rl_editor/cache")
    
    args = parser.parse_args()
    
    from rl_editor.config import Config
    from rl_editor.data import PairedAudioDataset
    
    config = Config()
    config.data.data_dir = args.data_dir
    config.data.cache_dir = args.cache_dir
    
    logger.info("Initializing dataset (this will trigger label caching)...")
    
    # Create dataset - this will preload audio
    dataset = PairedAudioDataset(
        data_dir=args.data_dir,
        config=config,
        cache_dir=args.cache_dir,
        include_reference=False,  # Skip reference tracks
        use_augmentation=False,
    )
    
    logger.info(f"Found {len(dataset)} pairs")
    
    # Access each item to trigger label caching
    cached = 0
    for i in range(len(dataset)):
        logger.info(f"[{i+1}/{len(dataset)}] Processing pair {i}...")
        try:
            item = dataset[i]
            cached += 1
            logger.info(f"  ✓ Labels cached for {item.get('pair_id', 'unknown')}")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
    
    logger.info(f"\nDone! Cached labels for {cached}/{len(dataset)} pairs")


if __name__ == "__main__":
    main()

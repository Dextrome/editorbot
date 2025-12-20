"""Inspect mel cache files and report shapes/stats.

Usage:
    python scripts/inspect_mel_cache.py [--cache_dir PATH] [--limit N]

Prints a short summary for each mel cache file found.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure workspace root on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rl_editor.cache import FeatureCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("inspect_mel_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Max files to inspect (0 = all)")
    args = parser.parse_args()

    fc = FeatureCache(cache_dir=args.cache_dir, enabled=True)
    mel_dir = Path(fc.cache_dir) / "mel"
    if not mel_dir.exists():
        logger.error(f"Mel cache dir not found: {mel_dir}")
        return

    files = sorted(mel_dir.glob("*.npz"))
    n = len(files)
    if n == 0:
        logger.info("No mel cache files found.")
        return

    limit = args.limit if args.limit > 0 else n
    logger.info(f"Found {n} mel files, inspecting up to {limit}")

    for i, p in enumerate(files[:limit]):
        try:
            data = np.load(p, allow_pickle=True)
            keys = data.files
            info = {k: data[k] for k in keys}
            # Prefer key 'mel' else first array
            if 'mel' in info:
                mel = info['mel']
            else:
                # take first array-like entry
                first_key = keys[0]
                mel = info[first_key]

            shape = getattr(mel, 'shape', None)
            dtype = getattr(mel, 'dtype', None)
            # Compute basic stats if numeric and not huge
            try:
                s = mel.size
                if s > 0 and s <= 1000000:
                    vmin = float(np.min(mel))
                    vmax = float(np.max(mel))
                    mean = float(np.mean(mel))
                else:
                    vmin = vmax = mean = None
            except Exception:
                vmin = vmax = mean = None

            print(f"{i+1}/{limit}: {p.name} | keys={keys} | shape={shape} | dtype={dtype} | min={vmin} max={vmax} mean={mean}")
        except Exception as e:
            logger.exception(f"Failed to read {p}: {e}")


if __name__ == '__main__':
    main()

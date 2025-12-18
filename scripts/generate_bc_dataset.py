"""Generate a simple behavior-cloning dataset from paired training data.

Produces a `.npz` with per-beat features and KEEP/CUT labels suitable for
supervised pretraining (binary classification). This is intentionally small and
fast: it extracts `beat_features` and `edit_labels` from `PairedAudioDataset`
and writes them to disk.

Usage:
    python -m scripts.generate_bc_dataset --data_dir training_data --out bc_dataset.npz
"""

import argparse
import logging
from pathlib import Path
import numpy as np

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset

logger = logging.getLogger("generate_bc_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--out", type=str, default="bc_dataset.npz")
    parser.add_argument("--max_pairs", type=int, default=0, help="Limit number of pairs (0=all)")
    args = parser.parse_args()

    config = get_default_config()
    dataset = PairedAudioDataset(data_dir=args.data_dir, config=config, cache_dir=None, include_reference=True)

    features = []
    labels = []
    pair_ids = []

    n = len(dataset)
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)

    logger.info(f"Generating BC dataset from {n} items")

    for i in range(n):
        item = dataset[i]
        raw = item.get("raw")
        edit_labels = item.get("edit_labels")
        pair_id = item.get("pair_id", f"pair_{i}")

        if raw is None or edit_labels is None:
            continue

        beat_features = raw.get("beat_features")
        if hasattr(beat_features, 'numpy'):
            bf = beat_features.numpy()
        else:
            bf = np.array(beat_features)

        if hasattr(edit_labels, 'numpy'):
            lab = edit_labels.numpy()
        else:
            lab = np.array(edit_labels)

        # Store per-beat rows
        for j in range(len(lab)):
            features.append(bf[j].astype(np.float32))
            labels.append(float(lab[j]))
            pair_ids.append(pair_id)

    features = np.stack(features) if features else np.zeros((0, 0), dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    pair_ids = np.array(pair_ids, dtype=object)

    out_path = Path(args.out)
    np.savez_compressed(out_path, features=features, labels=labels, pair_ids=pair_ids)
    logger.info(f"Wrote BC dataset to {out_path} (n={len(labels)})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()

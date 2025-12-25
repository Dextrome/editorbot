"""Augment BC dataset with synthetic examples for underrepresented action types.

This script takes an existing BC dataset and adds synthetic examples for action
types that have zero or few samples. This helps the policy learn all action types,
not just the ones that appear in the training data.

Usage:
    python -m scripts.augment_bc_with_synthetic --input bc_rich.npz --output bc_augmented.npz
"""

import argparse
import logging
from pathlib import Path
import numpy as np

from rl_editor.actions import ActionType, ActionSize, ActionAmount

logger = logging.getLogger("augment_bc")


def get_action_type_semantics():
    """Define semantic groups for action types to guide synthetic generation."""
    return {
        # Structural actions (affect which beats are included)
        "structural": [
            ActionType.KEEP, ActionType.CUT, ActionType.LOOP,
            ActionType.REORDER, ActionType.JUMP_BACK, ActionType.SKIP,
            ActionType.REPEAT_PREV, ActionType.SWAP_NEXT,
        ],
        # Effect actions (modify audio without changing structure)
        "effects": [
            ActionType.FADE_IN, ActionType.FADE_OUT, ActionType.GAIN,
            ActionType.SPEED_UP, ActionType.SPEED_DOWN, ActionType.REVERSE,
            ActionType.PITCH_UP, ActionType.PITCH_DOWN,
            ActionType.EQ_LOW, ActionType.EQ_HIGH,
            ActionType.DISTORTION, ActionType.REVERB,
        ],
    }


def generate_synthetic_examples(
    existing_states: np.ndarray,
    existing_good_bad: np.ndarray,
    action_type: ActionType,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple:
    """Generate synthetic examples for a given action type.

    Strategy:
    - For "keep-like" actions (LOOP, REORDER, etc.): sample from good_bad=1 states
    - For "cut-like" actions (SKIP, JUMP_BACK): sample from good_bad=0 states
    - For "effect" actions: sample from good_bad=1 states (effects are applied to kept beats)

    Returns:
        Tuple of (states, type_labels, size_labels, amount_labels, good_bad)
    """
    n_existing = len(existing_states)

    # Determine which states to sample from based on action semantics
    keep_like = [
        ActionType.KEEP, ActionType.LOOP, ActionType.REORDER,
        ActionType.REPEAT_PREV, ActionType.SWAP_NEXT,
        ActionType.FADE_IN, ActionType.FADE_OUT, ActionType.GAIN,
        ActionType.SPEED_UP, ActionType.SPEED_DOWN, ActionType.REVERSE,
        ActionType.PITCH_UP, ActionType.PITCH_DOWN,
        ActionType.EQ_LOW, ActionType.EQ_HIGH,
        ActionType.DISTORTION, ActionType.REVERB,
    ]

    cut_like = [ActionType.CUT, ActionType.SKIP, ActionType.JUMP_BACK]

    if action_type in keep_like:
        # Sample from "good" states
        good_indices = np.where(existing_good_bad == 1)[0]
        if len(good_indices) == 0:
            good_indices = np.arange(n_existing)
        sample_indices = rng.choice(good_indices, size=n_samples, replace=True)
        good_bad_val = 1
    else:
        # Sample from "bad" states
        bad_indices = np.where(existing_good_bad == 0)[0]
        if len(bad_indices) == 0:
            bad_indices = np.arange(n_existing)
        sample_indices = rng.choice(bad_indices, size=n_samples, replace=True)
        good_bad_val = 0

    states = existing_states[sample_indices].copy()

    # Add small noise to make synthetic examples slightly different
    noise = rng.normal(0, 0.01, states.shape).astype(np.float32)
    states = states + noise

    # Generate labels
    type_labels = np.full(n_samples, action_type.value, dtype=np.int64)

    # Size: mostly BEAT, some BAR for variety
    size_probs = [0.7, 0.2, 0.05, 0.04, 0.01]  # BEAT, BAR, PHRASE, TWO_BARS, TWO_PHRASES
    size_labels = rng.choice(
        [s.value for s in ActionSize],
        size=n_samples,
        p=size_probs
    ).astype(np.int64)

    # Amount: depends on action type
    if action_type in [ActionType.KEEP, ActionType.CUT, ActionType.LOOP,
                       ActionType.REORDER, ActionType.SKIP, ActionType.REVERSE,
                       ActionType.REPEAT_PREV, ActionType.SWAP_NEXT, ActionType.JUMP_BACK]:
        # These don't have meaningful amount - use NEUTRAL
        amount_labels = np.full(n_samples, ActionAmount.NEUTRAL.value, dtype=np.int64)
    elif action_type in [ActionType.FADE_IN, ActionType.GAIN, ActionType.PITCH_UP,
                         ActionType.SPEED_UP, ActionType.EQ_HIGH]:
        # These are "positive" effects - use positive amounts
        amount_probs = [0.0, 0.0, 0.3, 0.5, 0.2]  # favor POS_SMALL, some NEUTRAL, some POS_LARGE
        amount_labels = rng.choice(
            [a.value for a in ActionAmount],
            size=n_samples,
            p=amount_probs
        ).astype(np.int64)
    elif action_type in [ActionType.FADE_OUT, ActionType.PITCH_DOWN,
                         ActionType.SPEED_DOWN, ActionType.EQ_LOW]:
        # These are "negative" effects - use negative amounts
        amount_probs = [0.2, 0.5, 0.3, 0.0, 0.0]  # favor NEG_SMALL, some NEUTRAL, some NEG_LARGE
        amount_labels = rng.choice(
            [a.value for a in ActionAmount],
            size=n_samples,
            p=amount_probs
        ).astype(np.int64)
    else:
        # DISTORTION, REVERB - use neutral or positive
        amount_probs = [0.0, 0.1, 0.4, 0.4, 0.1]
        amount_labels = rng.choice(
            [a.value for a in ActionAmount],
            size=n_samples,
            p=amount_probs
        ).astype(np.int64)

    good_bad = np.full(n_samples, good_bad_val, dtype=np.int64)

    return states, type_labels, size_labels, amount_labels, good_bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="bc_rich.npz", help="Input BC dataset")
    parser.add_argument("--output", type=str, default="bc_augmented.npz", help="Output augmented dataset")
    parser.add_argument("--min_samples", type=int, default=1000, help="Minimum samples per action type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load existing dataset
    logger.info(f"Loading BC dataset from {args.input}")
    data = np.load(args.input, allow_pickle=True)

    states = data['states']
    type_labels = data['type_labels']
    size_labels = data['size_labels']
    amount_labels = data['amount_labels']
    good_bad = data['good_bad']
    pair_ids = data['pair_ids']

    logger.info(f"Loaded {len(states)} samples")

    # Count existing samples per action type
    logger.info("\nExisting action type distribution:")
    type_counts = {}
    for at in ActionType:
        count = np.sum(type_labels == at.value)
        type_counts[at] = count
        if count > 0:
            logger.info(f"  {at.name}: {count}")

    # Identify underrepresented types
    types_to_augment = []
    for at in ActionType:
        if type_counts[at] < args.min_samples:
            types_to_augment.append(at)

    logger.info(f"\nAction types below {args.min_samples} samples: {[at.name for at in types_to_augment]}")

    # Generate synthetic examples
    all_new_states = []
    all_new_types = []
    all_new_sizes = []
    all_new_amounts = []
    all_new_good_bad = []
    all_new_pair_ids = []

    for at in types_to_augment:
        n_needed = args.min_samples - type_counts[at]
        if n_needed <= 0:
            continue

        logger.info(f"Generating {n_needed} synthetic examples for {at.name}")

        new_states, new_types, new_sizes, new_amounts, new_gb = generate_synthetic_examples(
            states, good_bad, at, n_needed, rng
        )

        all_new_states.append(new_states)
        all_new_types.append(new_types)
        all_new_sizes.append(new_sizes)
        all_new_amounts.append(new_amounts)
        all_new_good_bad.append(new_gb)
        all_new_pair_ids.extend([f"synthetic_{at.name}_{i}" for i in range(n_needed)])

    # Combine original and synthetic data
    if all_new_states:
        states = np.concatenate([states] + all_new_states, axis=0)
        type_labels = np.concatenate([type_labels] + all_new_types, axis=0)
        size_labels = np.concatenate([size_labels] + all_new_sizes, axis=0)
        amount_labels = np.concatenate([amount_labels] + all_new_amounts, axis=0)
        good_bad = np.concatenate([good_bad] + all_new_good_bad, axis=0)
        pair_ids = np.concatenate([pair_ids, np.array(all_new_pair_ids, dtype=object)], axis=0)

    # Shuffle
    perm = rng.permutation(len(states))
    states = states[perm]
    type_labels = type_labels[perm]
    size_labels = size_labels[perm]
    amount_labels = amount_labels[perm]
    good_bad = good_bad[perm]
    pair_ids = pair_ids[perm]

    # Log final distribution
    logger.info("\nFinal action type distribution:")
    for at in ActionType:
        count = np.sum(type_labels == at.value)
        logger.info(f"  {at.name}: {count}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states,
        type_labels=type_labels,
        size_labels=size_labels,
        amount_labels=amount_labels,
        good_bad=good_bad,
        pair_ids=pair_ids,
    )
    logger.info(f"\nSaved augmented BC dataset to {out_path} (n={len(states)})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()

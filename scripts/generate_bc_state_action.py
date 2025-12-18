"""Generate state-action BC dataset matching agent observation shape.

For each paired track, this script constructs per-beat observations using
`StateRepresentation.construct_observation` and maps edit labels to factored
action labels (KEEP/CUT -> type,size,amount). Saves compressed NPZ with
`states`, `type_labels`, `size_labels`, `amount_labels`, `pair_ids`.

Usage:
    python -m scripts.generate_bc_state_action --data_dir training_data --out bc_state_action.npz
"""

import argparse
import logging
from pathlib import Path
import numpy as np

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset
from rl_editor.state import StateRepresentation, AudioState, EditHistory
from rl_editor.actions import ActionType, ActionSize, ActionAmount

logger = logging.getLogger("generate_bc_state_action")


def map_label_to_action(label: float):
    """Map KEEP/CUT binary label to factored action indices.

    KEEP (1.0) -> (ActionType.KEEP, ActionSize.BEAT, ActionAmount.NEUTRAL)
    CUT  (0.0) -> (ActionType.CUT,  ActionSize.BEAT, ActionAmount.NEUTRAL)
    """
    if label >= 0.5:
        return ActionType.KEEP.value, ActionSize.BEAT.value, ActionAmount.NEUTRAL.value
    else:
        return ActionType.CUT.value, ActionSize.BEAT.value, ActionAmount.NEUTRAL.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--out", type=str, default="bc_state_action.npz")
    parser.add_argument("--max_pairs", type=int, default=0)
    args = parser.parse_args()

    config = get_default_config()
    dataset = PairedAudioDataset(data_dir=args.data_dir, config=config, cache_dir=None, include_reference=True)

    state_repr = StateRepresentation(config)

    states = []
    type_labels = []
    size_labels = []
    amount_labels = []
    good_bad = []
    pair_ids = []

    n = len(dataset)
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)

    logger.info(f"Building BC state-action dataset from {n} items")

    for i in range(n):
        item = dataset[i]
        raw = item.get("raw")
        edit_labels = item.get("edit_labels")
        pair_id = item.get("pair_id", f"pair_{i}")

        if raw is None or edit_labels is None:
            continue

        # Extract beat-level arrays
        beat_times = raw.get("beat_times")
        beat_features = raw.get("beat_features")
        duration = float(raw.get("duration", 0.0))

        if hasattr(beat_times, 'numpy'):
            beat_times = beat_times.numpy()
        if hasattr(beat_features, 'numpy'):
            beat_features = beat_features.numpy()
        if hasattr(edit_labels, 'numpy'):
            labels = edit_labels.numpy()
        else:
            labels = np.array(edit_labels)

        # Set beat_feature_dim for state_repr
        if beat_features is not None and len(beat_features.shape) == 2:
            try:
                state_repr.set_beat_feature_dim(beat_features.shape[1])
            except Exception:
                pass

        # For each beat, build AudioState and observation
        for b_idx in range(len(labels)):
            audio_state = AudioState(
                beat_index=b_idx,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=float(raw.get("tempo", 120.0)),
                raw_audio=None,
                sample_rate=int(raw.get("sample_rate", config.audio.sample_rate)),
                target_mel=raw.get("mel") if raw.get("mel") is not None else None,
                pair_id=pair_id,
            )

            edit_history = EditHistory()
            remaining = max(0.0, duration - (beat_times[b_idx] if len(beat_times) > b_idx else 0.0))
            obs = state_repr.construct_observation(audio_state, edit_history, remaining_duration=remaining, total_duration=duration)

            t, s, a = map_label_to_action(float(labels[b_idx]))
            gb = 1 if float(labels[b_idx]) >= 0.5 else 0

            states.append(obs.astype(np.float32))
            type_labels.append(int(t))
            size_labels.append(int(s))
            amount_labels.append(int(a))
            good_bad.append(int(gb))
            pair_ids.append(pair_id)

    states = np.stack(states) if states else np.zeros((0, state_repr.feature_dim), dtype=np.float32)
    type_labels = np.array(type_labels, dtype=np.int64)
    size_labels = np.array(size_labels, dtype=np.int64)
    amount_labels = np.array(amount_labels, dtype=np.int64)
    pair_ids = np.array(pair_ids, dtype=object)

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        states=states,
        type_labels=type_labels,
        size_labels=size_labels,
        amount_labels=amount_labels,
        good_bad=np.array(good_bad, dtype=np.int64),
        pair_ids=pair_ids,
    )
    logger.info(f"Wrote BC state-action dataset to {out_path} (n={len(type_labels)})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()

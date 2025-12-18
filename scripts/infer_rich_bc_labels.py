"""Infer richer BC action labels (loops, reorders, fades, EQ, gain) from paired audio.

This script applies heuristics to align original -> edited per-beat mel vectors
and infers factored action labels per beat. Output is a compressed NPZ with
`states`, `type_labels`, `size_labels`, `amount_labels`, `pair_ids` like the
other BC generator, but with richer action types where detected.

Note: heuristics are approximate — review outputs before using for full training.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import librosa

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset
from rl_editor.state import StateRepresentation, AudioState, EditHistory
from rl_editor.actions import ActionType, ActionSize, ActionAmount

logger = logging.getLogger("infer_rich_bc_labels")


def detect_fades(mel_segment):
    """Detect fades by checking monotonic energy ramp and slope significance."""
    energy = mel_segment.mean(axis=1)
    if len(energy) < 3:
        return None
    x = np.arange(len(energy))
    coef = np.polyfit(x, energy, 1)
    slope = coef[0]
    # normalize slope by mean energy to be robust across levels
    norm_slope = slope / (np.mean(energy) + 1e-9)
    if norm_slope > 0.03:
        return 'in'
    if norm_slope < -0.03:
        return 'out'
    return None


def detect_eq_change(orig_mel, edited_mel, sr=22050):
    """Detect EQ-like change by comparing low vs high band energy (in dB).

    Returns 'high' if high-band boosted, 'low' if low-band boosted, else None.
    """
    try:
        if orig_mel is None or edited_mel is None:
            return None
        # assume mel vectors shape (n_mels,) or (n_frames, n_mels)
        orig = np.array(orig_mel)
        ed = np.array(edited_mel)
        if orig.ndim == 2:
            orig = orig.mean(axis=0)
        if ed.ndim == 2:
            ed = ed.mean(axis=0)
        n_mels = orig.shape[0]
        # split bands
        low = orig[: n_mels // 3].sum() + 1e-9
        high = orig[-(n_mels // 3) :].sum() + 1e-9
        low2 = ed[: n_mels // 3].sum() + 1e-9
        high2 = ed[-(n_mels // 3) :].sum() + 1e-9
        # dB change in high/low ratio
        before = 10.0 * np.log10(high / low)
        after = 10.0 * np.log10(high2 / low2)
        diff_db = after - before
        if diff_db > 1.5:
            return 'high'
        if diff_db < -1.5:
            return 'low'
    except Exception:
        return None
    return None


def estimate_pitch_shift_semitones(orig_mel, edited_mel, sr=22050):
    """Estimate approximate pitch shift (semitones) by comparing spectral centroid in Hz.

    Returns float semitones (positive = up)."""
    try:
        orig = np.array(orig_mel)
        ed = np.array(edited_mel)
        if orig.ndim == 2:
            orig = orig.mean(axis=0)
        if ed.ndim == 2:
            ed = ed.mean(axis=0)
        n_mels = orig.shape[0]
        # mel bin center freqs
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        cen_o = (freqs * orig).sum() / (orig.sum() + 1e-9)
        cen_e = (freqs * ed).sum() / (ed.sum() + 1e-9)
        if cen_o <= 0 or cen_e <= 0:
            return 0.0
        semitones = 12.0 * np.log2(cen_e / cen_o)
        return float(semitones)
    except Exception:
        return 0.0


def map_to_action_type(orig_keep, extra):
    """Map inferred flags to ActionType and ActionAmount heuristics."""
    if extra.get('loop', False):
        return ActionType.LOOP, ActionSize.BEAT, ActionAmount.NEUTRAL
    if extra.get('reorder', False):
        return ActionType.REORDER, ActionSize.BEAT, ActionAmount.NEUTRAL
    fade = extra.get('fade')
    if fade == 'in':
        return ActionType.FADE_IN, ActionSize.BEAT, ActionAmount.NEUTRAL
    if fade == 'out':
        return ActionType.FADE_OUT, ActionSize.BEAT, ActionAmount.NEUTRAL
    if extra.get('eq') == 'high':
        # choose amount based on magnitude if provided
        amt = ActionAmount.POS_SMALL
        if extra.get('eq_db', 0.0) > 4.0:
            amt = ActionAmount.POS_LARGE
        return ActionType.EQ_HIGH, ActionSize.BEAT, amt
    if extra.get('eq') == 'low':
        amt = ActionAmount.NEG_SMALL
        if extra.get('eq_db', 0.0) < -4.0:
            amt = ActionAmount.NEG_LARGE
        return ActionType.EQ_LOW, ActionSize.BEAT, amt
    if extra.get('gain') is not None:
        db = extra['gain']
        if db > 2.0:
            amt = ActionAmount.POS_LARGE
        elif db > 0.5:
            amt = ActionAmount.POS_SMALL
        elif db < -2.0:
            amt = ActionAmount.NEG_LARGE
        elif db < -0.5:
            amt = ActionAmount.NEG_SMALL
        else:
            amt = ActionAmount.NEUTRAL
        return ActionType.GAIN, ActionSize.BEAT, amt
    # pitch shifts: map to PITCH_UP / PITCH_DOWN if significant
    if 'pitch_semitones' in extra:
        semis = extra['pitch_semitones']
        if semis > 0.5:
            if semis >= 4.0:
                amt = ActionAmount.POS_LARGE
            elif semis >= 2.0:
                amt = ActionAmount.POS_SMALL
            else:
                amt = ActionAmount.NEUTRAL
            return ActionType.PITCH_UP, ActionSize.BEAT, amt
        elif semis < -0.5:
            if semis <= -4.0:
                amt = ActionAmount.POS_LARGE
            elif semis <= -2.0:
                amt = ActionAmount.POS_SMALL
            else:
                amt = ActionAmount.NEUTRAL
            return ActionType.PITCH_DOWN, ActionSize.BEAT, amt

    # Default KEEP/CUT
    if orig_keep:
        return ActionType.KEEP, ActionSize.BEAT, ActionAmount.NEUTRAL
    return ActionType.CUT, ActionSize.BEAT, ActionAmount.NEUTRAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--out", type=str, default="output/bc_rich_state_action.npz")
    parser.add_argument("--max_pairs", type=int, default=0)
    args = parser.parse_args()

    config = get_default_config()
    dataset = PairedAudioDataset(data_dir=args.data_dir, config=config, cache_dir=None, include_reference=True)
    state_repr = StateRepresentation(config)

    states = []
    type_labels = []
    size_labels = []
    amount_labels = []
    pair_ids = []

    n = len(dataset)
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)

    logger.info(f"Inferring rich BC labels for {n} pairs")

    for i in range(n):
        sample = dataset[i]
        raw = sample.get('raw', sample)
        edited = sample.get('edited') or {}
        pair_id = sample.get('pair_id', f'pair_{i}')

        beat_times = raw.get('beat_times')
        beat_features = raw.get('beat_features')
        # Per-beat mel if available
        orig_mel_per_beat = raw.get('per_beat_mel')
        edited_mel_per_beat = edited.get('per_beat_mel') if edited else None

        if hasattr(beat_times, 'numpy'):
            beat_times = beat_times.numpy()
        if hasattr(beat_features, 'numpy'):
            beat_features = beat_features.numpy()

        if beat_features is None:
            continue

        # Set state repr beat dim
        try:
            state_repr.set_beat_feature_dim(beat_features.shape[1])
        except Exception:
            pass

        n_beats = len(beat_times)

        # If we have edited per-beat mel, compute cross-similarity mapping
        mapping = [-1] * n_beats
        if edited_mel_per_beat is not None:
            orig = np.array(orig_mel_per_beat)
            ed = np.array(edited_mel_per_beat)
            # Normalize
            onorm = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-9)
            ednorm = ed / (np.linalg.norm(ed, axis=1, keepdims=True) + 1e-9)
            sim = np.clip(onorm @ ednorm.T, -1.0, 1.0)
            for oi in range(n_beats):
                best = np.argmax(sim[oi])
                if sim[oi, best] > 0.75:
                    mapping[oi] = int(best)
                else:
                    mapping[oi] = -1

        # For each beat, infer extras
        for b in range(n_beats):
            orig_keep = True
            extra = {}
            # If mapping missing, likely cut
            if mapping[b] == -1:
                orig_keep = False

            # detect loop: if mapping[b] appears multiple times in edited mapping (non-consecutive repeated matches)
            if mapping[b] != -1:
                occurrences = [i for i, m in enumerate(mapping) if m == mapping[b]]
                if len(occurrences) > 1:
                    # if occurrences are non-consecutive or repeated later in sequence -> loop
                    if any(occurrences[i+1] - occurrences[i] > 1 for i in range(len(occurrences) - 1)):
                        extra['loop'] = True

            # detect reorder: compute inversion fraction in the mapped edited positions
            ed_positions = [m for m in mapping if m != -1]
            if len(ed_positions) > 4:
                # Count inversions (O(n^2) OK for short sequences)
                inv = 0
                L = len(ed_positions)
                for i in range(L):
                    for j in range(i + 1, L):
                        if ed_positions[i] > ed_positions[j]:
                            inv += 1
                inv_frac = inv / max(1, (L * (L - 1) / 2))
                if inv_frac > 0.03:  # small threshold for reorders
                    extra['reorder'] = True

            # detect fade using per-beat mel energy
            if orig_mel_per_beat is not None and edited_mel_per_beat is not None and mapping[b] != -1:
                oseg = np.array(orig_mel_per_beat[b])
                eseg = np.array(edited_mel_per_beat[mapping[b]])
                try:
                    fade = detect_fades(eseg.reshape(-1, eseg.shape[0]) if eseg.ndim==1 else eseg)
                    if fade:
                        extra['fade'] = fade
                except Exception:
                    pass

            # detect EQ and pitch shift magnitude
            if orig_mel_per_beat is not None and edited_mel_per_beat is not None and mapping[b] != -1:
                oseg = np.array(orig_mel_per_beat[b])
                eseg = np.array(edited_mel_per_beat[mapping[b]])
                try:
                    eq = detect_eq_change(oseg, eseg, sr=int(raw.get('sample_rate', 22050)))
                    if eq:
                        extra['eq'] = eq
                    semis = estimate_pitch_shift_semitones(oseg, eseg, sr=int(raw.get('sample_rate', 22050)))
                    if abs(semis) >= 0.8:
                        extra['pitch_semitones'] = float(semis)
                except Exception:
                    pass

            # Map to action
            a_type, a_size, a_amount = map_to_action_type(orig_keep, extra)

            # Build state observation for this beat
            audio_state = AudioState(
                beat_index=b,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=float(raw.get('tempo', 120.0)),
                raw_audio=None,
                sample_rate=int(raw.get('sample_rate', config.audio.sample_rate)),
                target_mel=edited.get('per_beat_mel') if edited else None,
                pair_id=pair_id,
            )
            edit_hist = EditHistory()
            remaining = max(0.0, (beat_times[-1] - beat_times[b]) if len(beat_times)>b else 0.0)
            obs = state_repr.construct_observation(audio_state, edit_hist, remaining_duration=remaining, total_duration=beat_times[-1] if len(beat_times)>0 else 0.0)

            states.append(obs.astype(np.float32))
            type_labels.append(int(a_type.value))
            size_labels.append(int(a_size.value))
            amount_labels.append(int(a_amount.value))
            pair_ids.append(pair_id)

    states = np.stack(states) if states else np.zeros((0, state_repr.feature_dim), dtype=np.float32)
    type_labels = np.array(type_labels, dtype=np.int64)
    size_labels = np.array(size_labels, dtype=np.int64)
    amount_labels = np.array(amount_labels, dtype=np.int64)
    pair_ids = np.array(pair_ids, dtype=object)

    out_path = Path(args.out)
    np.savez_compressed(out_path, states=states, type_labels=type_labels, size_labels=size_labels, amount_labels=amount_labels, pair_ids=pair_ids)
    logger.info(f"Wrote rich BC state-action dataset to {out_path} (n={len(type_labels)})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()

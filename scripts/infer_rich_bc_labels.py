"""Infer richer BC action labels (loops, reorders, fades, EQ, gain) from paired audio.

Enhanced version that:
1. Loads per-beat mel from cached full mel spectrograms
2. Detects gain changes via RMS energy comparison
3. Infers multi-beat action sizes (BAR, PHRASE) from consecutive similar actions
4. Includes good_bad labels for auxiliary binary classifier

Output: compressed NPZ with `states`, `type_labels`, `size_labels`, `amount_labels`,
`good_bad`, `pair_ids`.

Note: heuristics are approximate — review outputs before using for full training.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import librosa
import hashlib

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset
from rl_editor.state import StateRepresentation, AudioState, EditHistory
from rl_editor.actions import ActionType, ActionSize, ActionAmount

logger = logging.getLogger("infer_rich_bc_labels")


def get_file_hash(filepath: Path) -> str:
    """Get hash for cache key matching - must match FeatureCache._get_file_hash."""
    filepath = Path(filepath)
    if not filepath.exists():
        return hashlib.md5(str(filepath).encode()).hexdigest()[:16]
    # Hash based on: absolute path + file size + modification time (matches cache.py)
    stat = filepath.stat()
    key = f"{filepath.absolute()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_cached_mel(cache_dir: Path, filepath: Path) -> np.ndarray:
    """Load mel spectrogram from cache if available."""
    file_hash = get_file_hash(filepath)
    stem = filepath.stem
    mel_cache_path = cache_dir / "mel" / f"{stem}_{file_hash}_mel.npz"

    if mel_cache_path.exists():
        try:
            data = np.load(mel_cache_path)
            return data['mel']
        except Exception as e:
            logger.warning(f"Failed to load cached mel from {mel_cache_path}: {e}")
    return None


def slice_mel_by_beats(mel: np.ndarray, beat_times: np.ndarray, sr: int = 22050, hop_length: int = 512) -> list:
    """Slice full mel spectrogram into per-beat segments.

    Args:
        mel: Full mel spectrogram (n_mels, n_frames)
        beat_times: Beat times in seconds
        sr: Sample rate
        hop_length: Hop length used for mel computation

    Returns:
        List of mel segments, one per beat
    """
    n_mels, n_frames = mel.shape
    per_beat_mel = []

    for i in range(len(beat_times)):
        start_frame = int(beat_times[i] * sr / hop_length)
        if i < len(beat_times) - 1:
            end_frame = int(beat_times[i + 1] * sr / hop_length)
        else:
            end_frame = n_frames

        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, n_frames))

        segment = mel[:, start_frame:end_frame]
        # Summarize to fixed-size vector (mean across time)
        per_beat_mel.append(segment.mean(axis=1))

    return per_beat_mel


def detect_fades(mel_segment: np.ndarray) -> str:
    """Detect fades by checking monotonic energy ramp and slope significance.

    Args:
        mel_segment: Mel segment (n_mels,) or (n_frames, n_mels)

    Returns:
        'in', 'out', or None
    """
    if mel_segment is None:
        return None

    seg = np.array(mel_segment)
    if seg.ndim == 1:
        # Single frame - can't detect fade
        return None

    # seg shape: (n_frames, n_mels) - compute energy per frame
    energy = seg.mean(axis=1) if seg.ndim == 2 else seg
    if len(energy) < 3:
        return None

    x = np.arange(len(energy))
    try:
        coef = np.polyfit(x, energy, 1)
        slope = coef[0]
        # Normalize slope by mean energy
        norm_slope = slope / (np.mean(energy) + 1e-9)
        if norm_slope > 0.03:
            return 'in'
        if norm_slope < -0.03:
            return 'out'
    except Exception:
        pass
    return None


def detect_gain_change_db(orig_mel: np.ndarray, edited_mel: np.ndarray) -> float:
    """Detect gain change in dB by comparing RMS energy.

    Args:
        orig_mel: Original mel vector (n_mels,) or (n_frames, n_mels)
        edited_mel: Edited mel vector

    Returns:
        Gain change in dB (positive = louder, negative = quieter)
    """
    if orig_mel is None or edited_mel is None:
        return 0.0

    try:
        orig = np.array(orig_mel)
        ed = np.array(edited_mel)

        # Flatten to 1D if needed
        if orig.ndim == 2:
            orig = orig.mean(axis=0)
        if ed.ndim == 2:
            ed = ed.mean(axis=0)

        # RMS energy
        rms_orig = np.sqrt(np.mean(orig ** 2)) + 1e-9
        rms_ed = np.sqrt(np.mean(ed ** 2)) + 1e-9

        # dB difference
        db_diff = 20.0 * np.log10(rms_ed / rms_orig)
        return float(db_diff)
    except Exception:
        return 0.0


def detect_eq_change(orig_mel: np.ndarray, edited_mel: np.ndarray, sr: int = 22050) -> tuple:
    """Detect EQ-like change by comparing low vs high band energy.

    Returns:
        Tuple of (eq_type: 'high'/'low'/None, eq_db: float)
    """
    try:
        if orig_mel is None or edited_mel is None:
            return None, 0.0

        orig = np.array(orig_mel)
        ed = np.array(edited_mel)
        if orig.ndim == 2:
            orig = orig.mean(axis=0)
        if ed.ndim == 2:
            ed = ed.mean(axis=0)

        n_mels = orig.shape[0]
        # Split into low/high bands
        low = orig[: n_mels // 3].sum() + 1e-9
        high = orig[-(n_mels // 3):].sum() + 1e-9
        low2 = ed[: n_mels // 3].sum() + 1e-9
        high2 = ed[-(n_mels // 3):].sum() + 1e-9

        # dB change in high/low ratio
        before = 10.0 * np.log10(high / low)
        after = 10.0 * np.log10(high2 / low2)
        diff_db = after - before

        if diff_db > 1.5:
            return 'high', diff_db
        if diff_db < -1.5:
            return 'low', diff_db
    except Exception:
        pass
    return None, 0.0


def estimate_pitch_shift_semitones(orig_mel: np.ndarray, edited_mel: np.ndarray, sr: int = 22050) -> float:
    """Estimate pitch shift in semitones by comparing spectral centroid."""
    try:
        orig = np.array(orig_mel)
        ed = np.array(edited_mel)
        if orig.ndim == 2:
            orig = orig.mean(axis=0)
        if ed.ndim == 2:
            ed = ed.mean(axis=0)

        n_mels = orig.shape[0]
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)

        cen_o = (freqs * orig).sum() / (orig.sum() + 1e-9)
        cen_e = (freqs * ed).sum() / (ed.sum() + 1e-9)

        if cen_o <= 0 or cen_e <= 0:
            return 0.0

        semitones = 12.0 * np.log2(cen_e / cen_o)
        return float(semitones)
    except Exception:
        return 0.0


def map_to_action(orig_keep: bool, extra: dict) -> tuple:
    """Map inferred flags to ActionType and ActionAmount.

    Returns:
        Tuple of (ActionType, ActionAmount)
    """
    # Priority order: loop > reorder > fade > eq > gain > pitch > keep/cut

    if extra.get('loop', False):
        return ActionType.LOOP, ActionAmount.NEUTRAL

    if extra.get('reorder', False):
        return ActionType.REORDER, ActionAmount.NEUTRAL

    fade = extra.get('fade')
    if fade == 'in':
        return ActionType.FADE_IN, ActionAmount.NEUTRAL
    if fade == 'out':
        return ActionType.FADE_OUT, ActionAmount.NEUTRAL

    eq_type = extra.get('eq')
    if eq_type == 'high':
        eq_db = extra.get('eq_db', 0.0)
        amt = ActionAmount.POS_LARGE if eq_db > 4.0 else ActionAmount.POS_SMALL
        return ActionType.EQ_HIGH, amt
    if eq_type == 'low':
        eq_db = extra.get('eq_db', 0.0)
        amt = ActionAmount.NEG_LARGE if eq_db < -4.0 else ActionAmount.NEG_SMALL
        return ActionType.EQ_LOW, amt

    # Gain detection
    gain_db = extra.get('gain_db', 0.0)
    if abs(gain_db) > 1.5:  # Threshold for significant gain change
        if gain_db > 4.0:
            amt = ActionAmount.POS_LARGE
        elif gain_db > 1.5:
            amt = ActionAmount.POS_SMALL
        elif gain_db < -4.0:
            amt = ActionAmount.NEG_LARGE
        else:
            amt = ActionAmount.NEG_SMALL
        return ActionType.GAIN, amt

    # Pitch shift
    semis = extra.get('pitch_semitones', 0.0)
    if semis > 0.8:
        if semis >= 4.0:
            amt = ActionAmount.POS_LARGE
        elif semis >= 2.0:
            amt = ActionAmount.POS_SMALL
        else:
            amt = ActionAmount.NEUTRAL
        return ActionType.PITCH_UP, amt
    elif semis < -0.8:
        if semis <= -4.0:
            amt = ActionAmount.POS_LARGE
        elif semis <= -2.0:
            amt = ActionAmount.POS_SMALL
        else:
            amt = ActionAmount.NEUTRAL
        return ActionType.PITCH_DOWN, amt

    # Default: KEEP or CUT
    if orig_keep:
        return ActionType.KEEP, ActionAmount.NEUTRAL
    return ActionType.CUT, ActionAmount.NEUTRAL


def infer_action_sizes(type_labels: list, beat_times: np.ndarray, tempo: float) -> list:
    """Infer multi-beat action sizes by detecting consecutive similar actions.

    Args:
        type_labels: List of ActionType values
        beat_times: Beat times array
        tempo: Track tempo in BPM

    Returns:
        List of ActionSize values
    """
    n = len(type_labels)
    sizes = [ActionSize.BEAT.value] * n

    if n < 2:
        return sizes

    # Estimate beats per bar (assume 4/4 time)
    beats_per_bar = 4
    beats_per_phrase = beats_per_bar * 4  # 4 bars = 1 phrase

    i = 0
    while i < n:
        action_type = type_labels[i]

        # Count consecutive beats with same action type
        run_length = 1
        while i + run_length < n and type_labels[i + run_length] == action_type:
            run_length += 1

        # Determine size based on run length
        # ActionSize: BEAT=0, BAR=1, PHRASE=2, TWO_BARS=3, TWO_PHRASES=4
        if run_length >= beats_per_phrase * 2:  # 32+ beats
            size = ActionSize.TWO_PHRASES.value
        elif run_length >= beats_per_phrase:  # 16+ beats
            size = ActionSize.PHRASE.value
        elif run_length >= beats_per_bar * 2:  # 8+ beats
            size = ActionSize.TWO_BARS.value
        elif run_length >= beats_per_bar:  # 4+ beats
            size = ActionSize.BAR.value
        else:
            size = ActionSize.BEAT.value

        # Assign size to first beat of the run (others stay as BEAT)
        sizes[i] = size

        i += run_length

    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--cache_dir", type=str, default="rl_editor/cache")
    parser.add_argument("--out", type=str, default="bc_rich.npz")
    parser.add_argument("--max_pairs", type=int, default=0, help="Limit pairs (0=all)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length")
    args = parser.parse_args()

    config = get_default_config()
    cache_dir = Path(args.cache_dir)

    # Load dataset
    dataset = PairedAudioDataset(
        data_dir=args.data_dir,
        config=config,
        cache_dir=str(cache_dir),
        include_reference=True
    )
    state_repr = StateRepresentation(config)

    # Collect all data
    all_states = []
    all_type_labels = []
    all_amount_labels = []
    all_good_bad = []
    all_pair_ids = []
    all_beat_times_list = []
    all_tempos = []

    n = len(dataset)
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)

    logger.info(f"Inferring rich BC labels for {n} pairs")
    logger.info(f"Cache dir: {cache_dir}")

    for idx in range(n):
        try:
            sample = dataset[idx]
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            continue

        raw = sample.get('raw', sample)
        edited = sample.get('edited') or {}
        pair_id = sample.get('pair_id', f'pair_{idx}')
        raw_path = Path(sample.get('raw_path', ''))
        edited_path = Path(sample.get('edited_path', ''))
        is_reference = sample.get('is_reference', False)

        beat_times = raw.get('beat_times')
        beat_features = raw.get('beat_features')
        tempo = float(raw.get('tempo', 120.0))
        sr = int(raw.get('sample_rate', args.sr))

        # Get ground-truth edit labels from dataset (pre-computed via alignment)
        edit_labels = sample.get('edit_labels')
        if hasattr(edit_labels, 'numpy'):
            edit_labels = edit_labels.numpy()
        elif edit_labels is not None:
            edit_labels = np.array(edit_labels)

        if hasattr(beat_times, 'numpy'):
            beat_times = beat_times.numpy()
        if hasattr(beat_features, 'numpy'):
            beat_features = beat_features.numpy()

        if beat_features is None or beat_times is None:
            logger.warning(f"Skipping {pair_id}: missing features")
            continue

        # Set state repr beat dim
        try:
            state_repr.set_beat_feature_dim(beat_features.shape[1])
        except Exception:
            pass

        n_beats = len(beat_times)
        logger.info(f"Processing {pair_id}: {n_beats} beats (reference={is_reference})")

        # Load mel spectrograms from cache
        orig_mel_full = None
        edited_mel_full = None
        orig_mel_per_beat = None
        edited_mel_per_beat = None

        if raw_path.exists():
            orig_mel_full = load_cached_mel(cache_dir, raw_path)
            if orig_mel_full is not None:
                orig_mel_per_beat = slice_mel_by_beats(
                    orig_mel_full, beat_times, sr=sr, hop_length=args.hop_length
                )
                logger.debug(f"  Loaded orig mel: {orig_mel_full.shape}")

        if edited_path.exists() and not is_reference:
            edited_mel_full = load_cached_mel(cache_dir, edited_path)
            if edited_mel_full is not None:
                # Need edited beat times - use raw beat times as approximation
                edited_mel_per_beat = slice_mel_by_beats(
                    edited_mel_full, beat_times, sr=sr, hop_length=args.hop_length
                )
                logger.debug(f"  Loaded edited mel: {edited_mel_full.shape}")

        # Compute beat-to-beat mapping via mel similarity
        mapping = [-1] * n_beats
        if orig_mel_per_beat is not None and edited_mel_per_beat is not None:
            orig_arr = np.array(orig_mel_per_beat)
            ed_arr = np.array(edited_mel_per_beat)

            # Normalize for cosine similarity
            onorm = orig_arr / (np.linalg.norm(orig_arr, axis=1, keepdims=True) + 1e-9)
            ednorm = ed_arr / (np.linalg.norm(ed_arr, axis=1, keepdims=True) + 1e-9)

            # Compute similarity matrix
            sim = np.clip(onorm @ ednorm.T, -1.0, 1.0)

            for oi in range(min(n_beats, len(orig_arr))):
                if oi < sim.shape[0]:
                    best = np.argmax(sim[oi])
                    if sim[oi, best] > 0.70:  # Slightly lower threshold
                        mapping[oi] = int(best)

        # For reference tracks: all beats map to themselves (all KEEP)
        if is_reference:
            mapping = list(range(n_beats))

        # Compute per-beat action types and extras
        beat_type_labels = []
        beat_amount_labels = []
        beat_good_bad = []
        beat_states = []

        # Pre-compute for detecting per-beat reorders
        # Build a list of (raw_idx, edited_idx) for valid mappings
        valid_mappings = [(i, m) for i, m in enumerate(mapping) if m != -1]

        for b in range(n_beats):
            # Use ground-truth labels if available, else fall back to mel similarity
            if edit_labels is not None and b < len(edit_labels):
                orig_keep = edit_labels[b] >= 0.5
            else:
                orig_keep = mapping[b] != -1
            extra = {}

            # Detect loop: A beat in the EDITED track that came from multiple places in the original
            # This is detected by checking if any edited position is mapped to by multiple raw positions
            # Note: multiple raw beats -> same edited beat = CUT (not loop!)
            # Loop = same raw content appears multiple times at different positions in edited
            # We detect this by looking for edited positions that appear multiple times in mapping
            if mapping[b] != -1:
                # Check if this raw beat's content appears at multiple NON-CONSECUTIVE edited positions
                # This would indicate the same raw beat was looped/repeated in the edit
                raw_beat_content_positions = []
                for i, m in enumerate(mapping):
                    if m != -1 and i != b:
                        # Check if beat i has similar content to beat b (both map to same-ish area)
                        if abs(m - mapping[b]) <= 1:  # Very similar edited position
                            raw_beat_content_positions.append(i)

                # If raw beat b's content shows up at non-adjacent raw positions -> might be a copy
                # Actually, let's simplify: detect loop if we see INCREASING edited positions that repeat
                # For now, disable over-aggressive loop detection
                pass  # Loop detection disabled - too noisy with simple heuristics

            # Only detect effects (reorder, gain, EQ, pitch) on KEEP beats
            if orig_keep:
                # Reorder detection: check if this beat's position violates monotonicity
                # A beat is "reordered" if it maps to an edited position that breaks the expected order
                if mapping[b] != -1 and len(valid_mappings) > 1:
                    # Find this beat's position in the valid_mappings list
                    my_pos = next((i for i, (raw_i, _) in enumerate(valid_mappings) if raw_i == b), -1)
                    if my_pos > 0 and my_pos < len(valid_mappings) - 1:
                        prev_edited = valid_mappings[my_pos - 1][1]
                        curr_edited = mapping[b]
                        next_edited = valid_mappings[my_pos + 1][1]
                        # Reorder if this beat's edited position is not between prev and next
                        if not (prev_edited <= curr_edited <= next_edited or prev_edited >= curr_edited >= next_edited):
                            extra['reorder'] = True

            # Mel-based detections (only for kept beats with valid mapping)
            if orig_keep and orig_mel_per_beat is not None and mapping[b] != -1:
                oseg = np.array(orig_mel_per_beat[b])

                if edited_mel_per_beat is not None and mapping[b] < len(edited_mel_per_beat):
                    eseg = np.array(edited_mel_per_beat[mapping[b]])

                    # Gain detection
                    gain_db = detect_gain_change_db(oseg, eseg)
                    if abs(gain_db) > 1.5:
                        extra['gain_db'] = gain_db

                    # EQ detection
                    eq_type, eq_db = detect_eq_change(oseg, eseg, sr=sr)
                    if eq_type:
                        extra['eq'] = eq_type
                        extra['eq_db'] = eq_db

                    # Pitch detection
                    semis = estimate_pitch_shift_semitones(oseg, eseg, sr=sr)
                    if abs(semis) >= 0.8:
                        extra['pitch_semitones'] = semis

            # Map to action type and amount
            a_type, a_amount = map_to_action(orig_keep, extra)

            # Good/bad label: 1 if kept (or transformed), 0 if cut
            # Use ground-truth labels when available
            if edit_labels is not None and b < len(edit_labels):
                good_bad = 1 if edit_labels[b] >= 0.5 else 0
            else:
                good_bad = 1 if orig_keep else 0

            # Build state observation
            audio_state = AudioState(
                beat_index=b,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=tempo,
                raw_audio=None,
                sample_rate=sr,
                target_mel=None,
                pair_id=pair_id,
            )
            edit_hist = EditHistory()
            remaining = max(0.0, (beat_times[-1] - beat_times[b]) if len(beat_times) > b else 0.0)
            total_dur = beat_times[-1] if len(beat_times) > 0 else 0.0

            try:
                obs = state_repr.construct_observation(
                    audio_state, edit_hist,
                    remaining_duration=remaining,
                    total_duration=total_dur
                )
                beat_states.append(obs.astype(np.float32))
                beat_type_labels.append(a_type.value)
                beat_amount_labels.append(a_amount.value)
                beat_good_bad.append(good_bad)
            except Exception as e:
                logger.warning(f"Failed to construct observation for beat {b}: {e}")
                continue

        if len(beat_states) == 0:
            continue

        # Store for later size inference
        all_states.extend(beat_states)
        all_type_labels.extend(beat_type_labels)
        all_amount_labels.extend(beat_amount_labels)
        all_good_bad.extend(beat_good_bad)
        all_pair_ids.extend([pair_id] * len(beat_states))
        all_beat_times_list.append((len(all_states) - len(beat_states), beat_times, tempo))
        all_tempos.append(tempo)

    # Infer action sizes based on consecutive same-type runs
    all_size_labels = [ActionSize.BEAT.value] * len(all_type_labels)

    # Process each track's worth of labels for size inference
    prev_end = 0
    for start_idx, beat_times, tempo in all_beat_times_list:
        track_end = start_idx + len(beat_times)
        if track_end > len(all_type_labels):
            track_end = len(all_type_labels)

        track_types = all_type_labels[start_idx:track_end]
        track_sizes = infer_action_sizes(track_types, beat_times, tempo)

        for i, size in enumerate(track_sizes):
            if start_idx + i < len(all_size_labels):
                all_size_labels[start_idx + i] = size

        prev_end = track_end

    # Convert to arrays
    states = np.stack(all_states) if all_states else np.zeros((0, state_repr.feature_dim), dtype=np.float32)
    type_labels = np.array(all_type_labels, dtype=np.int64)
    size_labels = np.array(all_size_labels, dtype=np.int64)
    amount_labels = np.array(all_amount_labels, dtype=np.int64)
    good_bad = np.array(all_good_bad, dtype=np.int64)
    pair_ids = np.array(all_pair_ids, dtype=object)

    # Log action distribution
    logger.info(f"\nAction type distribution:")
    for at in ActionType:
        count = np.sum(type_labels == at.value)
        if count > 0:
            logger.info(f"  {at.name}: {count} ({100*count/len(type_labels):.1f}%)")

    logger.info(f"\nAction size distribution:")
    for az in ActionSize:
        count = np.sum(size_labels == az.value)
        if count > 0:
            logger.info(f"  {az.name}: {count} ({100*count/len(size_labels):.1f}%)")

    logger.info(f"\nGood/bad distribution: good={np.sum(good_bad)}, bad={len(good_bad)-np.sum(good_bad)}")

    # Save
    out_path = Path(args.out)
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
    logger.info(f"\nWrote rich BC dataset to {out_path} (n={len(type_labels)})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()

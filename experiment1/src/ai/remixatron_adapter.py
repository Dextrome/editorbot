"""
Adapter for using InfiniteJukebox (Remixatron) for creative rearrangement.
"""
import numpy as np
import logging
from typing import List, Optional, Callable
from scipy.signal import resample as sciresample
import librosa
import sys
import os

# Ensure third_party/Remixatron is on sys.path
remixatron_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/Remixatron'))
if remixatron_path not in sys.path:
    sys.path.insert(0, remixatron_path)

from Remixatron import InfiniteJukebox
def soft_clip_signal(arr: np.ndarray, threshold: float = 0.995) -> np.ndarray:
    """
    Soft-tanh based clipper. Scales signal by threshold and applies tanh for gentle limiting.
    """
    if threshold <= 0 or threshold >= 1.0:
        return arr
    scaled = arr * (1.0 / threshold)
    return threshold * np.tanh(scaled)


from .remixatron_blend import blend_beats

class RemixatronAdapter:
    def __init__(self, clusters: int = 0, use_v1: bool = False, stem_method: str = 'demucs', sample_rate: int = 44100, blend_duration: float = 0.2, demucs_device: str | None = None, max_jump: int | None = None, gap_heal_ms: int | None = None, gap_heal_threshold: float = 1e-4, gap_mode: str = 'heal', truncate_enabled: bool = False, truncate_min_ms: int = 100, truncate_max_ms: int = 300, truncate_threshold: float = 0.01, truncate_crossfade_ms: int = 20, truncate_adaptive_factor: float = 0.0, truncate_mode: str = 'remove', truncate_compress_ms: int = 20, truncate_sample_pct: float = 0.95, phrase_beats: int = 4):
        self.clusters = clusters
        self.use_v1 = use_v1
        self.sample_rate = sample_rate
        self.blend_duration = blend_duration
        # Optional demucs device override: 'cpu' or 'cuda'. If None, Demucs will auto-detect via torch.
        self.demucs_device = demucs_device
        # Maximum allowed beat jump to limit randomness and preserve musical continuity (in beats)
        self.max_jump = max_jump
        # If provided, heal short near-zero gaps shorter than this many milliseconds
        self.gap_heal_ms = gap_heal_ms
        # amplitude threshold to consider 'near-zero' for gap detection
        self.gap_heal_threshold = gap_heal_threshold
        # Modes: preserve_large | trim_small | heal | stem | always_blend
        self.gap_mode = gap_mode
        # Truncate silence post-process (between truncate_min_ms and truncate_max_ms, in ms)
        self.truncate_enabled = truncate_enabled
        self.truncate_min_ms = truncate_min_ms
        self.truncate_max_ms = truncate_max_ms
        self.truncate_threshold = truncate_threshold
        self.truncate_crossfade_ms = truncate_crossfade_ms
        self.truncate_adaptive_factor = truncate_adaptive_factor
        self.truncate_mode = truncate_mode
        self.truncate_compress_ms = truncate_compress_ms
        self.truncate_sample_pct = truncate_sample_pct
        # Number of beats to group into a single phrase/segment for more coherent longer phrases
        self.phrase_beats = int(phrase_beats) if phrase_beats and phrase_beats > 0 else 4

    def rearrange(self, audio_path: str, progress_callback: Optional[Callable[[float, str], None]] = None, debug_return_segments: bool = False) -> np.ndarray:
        """
        Run InfiniteJukebox on the given audio file and return a rearranged audio array.
        Args:
            audio_path: Path to the audio file (wav, mp3, etc).
            progress_callback: Optional callback for progress updates.
        Returns:
            Numpy array of rearranged audio (float32, -1..1).
        """
        logger = logging.getLogger(__name__)
        logger.info("[RemixatronAdapter] Using Remixatron InfiniteJukebox for rearrangement!")
        # Run Demucs once on the input file and cache stems
        from src.audio.demucs_wrapper import DemucsSeparator
        demucs = DemucsSeparator()
        if self.demucs_device:
            logger.info("[RemixatronAdapter] Demucs device override: %s", self.demucs_device)
        logger.info("[RemixatronAdapter] Separating full track with Demucs (this happens only once)...")
        try:
            full_stems = demucs.separate(audio_path, resample_to=self.sample_rate, device=self.demucs_device)
        except RuntimeError as e:
            logger.warning("Demucs failed: %s. Falling back to non-stem blending (no stems will be used).", e)
            full_stems = {}
        # Check returned sample rate from Demucs stems
        if '_sr' in full_stems:
            demucs_sr = full_stems['_sr']
            if demucs_sr != self.sample_rate:
                logger.warning("Demucs output sample rate %d != expected %d. Results may be misaligned.", demucs_sr, self.sample_rate)
        logger.info("[RemixatronAdapter] Got stems: %s", list(full_stems.keys()))

        jukebox = InfiniteJukebox(
            filename=audio_path,
            clusters=self.clusters,
            progress_callback=progress_callback,
            do_async=False,
            use_v1=self.use_v1
        )
        logger.debug("[RemixatronAdapter] First 20 play_vector entries: %s", [entry['beat'] for entry in jukebox.play_vector[:20]])
        logger.info("[RemixatronAdapter] Total beats: %d, play_vector length: %d", len(jukebox.beats), len(jukebox.play_vector))
        # Optionally smooth the play_vector to limit the maximum jump per beat to improve coherence
        if self.max_jump is not None and self.max_jump > 0:
            logger.info("Applying Remixatron max_jump=%d to play vector", self.max_jump)
            prev_beat = None
            for i, entry in enumerate(jukebox.play_vector):
                if prev_beat is None:
                    prev_beat = entry['beat']
                    continue
                next_beat = entry['beat']
                if abs(next_beat - prev_beat) > self.max_jump:
                    sign = 1 if (next_beat - prev_beat) > 0 else -1
                    new_beat = prev_beat + sign * self.max_jump
                    logger.debug("Clamping beat %d from %d to %d due to max_jump", i, next_beat, new_beat)
                    jukebox.play_vector[i]['beat'] = new_beat
                    prev_beat = new_beat
                else:
                    prev_beat = next_beat
        # Build rearranged audio by following the play_vector, with stem-based blending
        # First, optionally group consecutive beats into phrases of `self.phrase_beats` beats
        phrase_list = []
        use_phrase_mode = self.phrase_beats and self.phrase_beats > 1
        if use_phrase_mode:
            logger.debug("Grouping beats into phrases of %d beats", self.phrase_beats)
            accum_bufs = []
            accum_stems = []
            count = 0
            phrase_idx = 0
            # iterate over the ordered play_vector entries
            for entry in jukebox.play_vector[:len(jukebox.beats)]:
                bidx = entry['beat']
                if bidx < 0:
                    continue
                beat = jukebox.beats[bidx]
                buf = beat.get('buffer')
                # normalize dtype to float32 here minimally; exact stereo conversion is done later
                if buf is None:
                    continue
                if buf.dtype == np.int16:
                    cbuf = buf.astype(np.float32) / 32768.0
                else:
                    cbuf = buf.astype(np.float32)
                accum_bufs.append(cbuf)
                # slice stems for this beat if available
                start = int(beat['start']) if 'start' in beat else 0
                end = int(beat['end']) if 'end' in beat else start + len(cbuf)
                beat_stems = {k: v[start:end] for k, v in full_stems.items() if k != '_sr'} if full_stems else {}
                accum_stems.append(beat_stems)
                count += 1
                if count >= self.phrase_beats:
                    # concatenate phrase buffers and stems
                    try:
                        phrase_audio = np.concatenate(accum_bufs, axis=0)
                    except Exception:
                        phrase_audio = np.concatenate([np.atleast_2d(x) for x in accum_bufs], axis=0)
                    combined = {}
                    if any(bool(s) for s in accum_stems):
                        keys = set(k for s in accum_stems for k in s.keys())
                        for k in keys:
                            parts = [s[k] for s in accum_stems if s and k in s]
                            if parts:
                                try:
                                    combined[k] = np.concatenate(parts, axis=0)
                                except Exception:
                                    combined[k] = np.concatenate([np.atleast_2d(p) for p in parts], axis=0)
                    # Use phrase_idx for sequential ordering; start/end are buffer-relative (not used for gap calc in phrase mode)
                    phrase_list.append({'phrase_idx': phrase_idx, 'beat': bidx, 'buffer': phrase_audio, 'stems': combined})
                    phrase_idx += 1
                    accum_bufs = []
                    accum_stems = []
                    count = 0
            # flush remainder
            if accum_bufs:
                try:
                    phrase_audio = np.concatenate(accum_bufs, axis=0)
                except Exception:
                    phrase_audio = np.concatenate([np.atleast_2d(x) for x in accum_bufs], axis=0)
                combined = {}
                if any(bool(s) for s in accum_stems):
                    keys = set(k for s in accum_stems for k in s.keys())
                    for k in keys:
                        parts = [s[k] for s in accum_stems if s and k in s]
                        if parts:
                            try:
                                combined[k] = np.concatenate(parts, axis=0)
                            except Exception:
                                combined[k] = np.concatenate([np.atleast_2d(p) for p in parts], axis=0)
                phrase_list.append({'phrase_idx': phrase_idx, 'beat': bidx, 'buffer': phrase_audio, 'stems': combined})
            logger.info("[RemixatronAdapter] Created %d phrases from %d beats", len(phrase_list), len(jukebox.beats))
        else:
            # No grouping requested; use original play_vector entries
            phrase_list = list(jukebox.play_vector[:len(jukebox.beats)])
        # We'll store appended segments as (array, tag) where tag is 'pad' or 'content'
        beat_audio = []
        prev_buf = None
        prev_stems = None
        def ensure_stereo(arr):
            arr = np.asarray(arr)
            # 1D -> duplicate to stereo
            if arr.ndim == 1:
                return np.stack([arr, arr], axis=-1)
            # 2D: prefer (n_samples, n_channels) with channels last
            if arr.ndim == 2:
                # already channels-last
                if arr.shape[1] == 2:
                    return arr
                # channels-first (2, n_samples) -> transpose
                if arr.shape[0] == 2:
                    return arr.T
                # single-channel stored as (n,1) -> squeeze and duplicate
                if arr.shape[1] == 1:
                    return np.concatenate([arr, arr], axis=1)
                if arr.shape[0] == 1:
                    # shape (1, n) -> treat as mono and duplicate
                    mono = arr.flatten()
                    return np.stack([mono, mono], axis=-1)
                # Fallback: if ambiguous, try to return channels-last with two channels
                if arr.shape[1] < arr.shape[0]:
                    # assume (n_samples, channels)
                    return arr[:, :2] if arr.shape[1] >= 2 else np.concatenate([arr, arr], axis=1)
                else:
                    # assume channels-first
                    return arr.T[:, :2]
            # Unexpected ndim: coerce to zeros stereo
            return np.zeros((0, 2), dtype=arr.dtype)
        def pop_trim_last_n(segments, n_samples, allow_trim_pad=False):
            """Trim the last n_samples of content in segments.
            Segments are (arr, tag). By default, do not trim into pad segments
            unless allow_trim_pad=True.
            """
            remaining = n_samples
            # Work from the end but temporarily store skipped pad segments to restore them after trimming
            skipped_pads = []
            while remaining > 0 and segments:
                last_arr, last_tag = segments.pop()
                last = np.asarray(last_arr)
                last_len = last.shape[0]
                # If this is a large pad and we're not allowed to trim into pads, treat it as a barrier and stop
                if last_tag == 'pad_large' and not allow_trim_pad:
                    # Put it back and break without trimming any further
                    segments.append((last, last_tag))
                    break
                # If this is a small pad and we're not allowed to trim into pads, skip it for now and restore later
                if last_tag in ('pad_small', 'pad') and not allow_trim_pad:
                    skipped_pads.append((last, last_tag))
                    continue
                if last_len > remaining:
                    # Trim the last part of this segment and push the remaining portion back
                    keep_len = last_len - remaining
                    segments.append((last[:keep_len], last_tag))
                    remaining = 0
                else:
                    # drop the entire segment and continue
                    remaining -= last_len
            # restore skipped pads in their original order
            while skipped_pads:
                segments.append(skipped_pads.pop())
            return segments
        def append_segment(arr, tag: str):
            arr = np.asarray(arr)
            beat_audio.append((arr, tag))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Appended segment: tag=%s, len=%d, min=%f, max=%f", tag, arr.shape[0], float(arr.min() if arr.size else 0.0), float(arr.max() if arr.size else 0.0))
        last_end = None
        # track the last inserted gap size (0 none, gap in samples). Used to allow trimming only for small gaps
        last_inserted_gap_size = 0
        n_blend = int(self.sample_rate * self.blend_duration)
        stems_available = bool(full_stems)
        
        for i, entry in enumerate(phrase_list):
            # Support both original per-beat entries and phrase entries produced above (which include 'buffer')
            if isinstance(entry, dict) and 'buffer' in entry:
                # Phrase mode: buffer and stems are already concatenated, no gap calculation needed
                buf = entry['buffer']
                if buf.dtype == np.int16:
                    buf = buf.astype(np.float32) / 32768.0
                else:
                    buf = buf.astype(np.float32)
                buf = ensure_stereo(buf)
                beat_stems = entry.get('stems', {})
                # In phrase mode, we don't use start/end for gap calculation
                # Each phrase is self-contained; just blend sequentially
                start = None  # Signal that we're in phrase mode
                end = None
            else:
                beat_idx = entry['beat']
                beat = jukebox.beats[beat_idx]
                # Preserve original dtype until conversion checks so we can scale if needed
                buf = beat['buffer']
                if buf.dtype == np.int16:
                    buf = buf.astype(np.float32) / 32768.0
                else:
                    buf = buf.astype(np.float32)
                # Ensure buffer is stereo shape (n_samples, 2)
                buf = ensure_stereo(buf)
                start = int(beat['start']) if 'start' in beat else 0
                end = int(beat['end']) if 'end' in beat else start + len(buf)
                # Extract stems for this beat; ignore the internal _sr key if present
                beat_stems = {k: v[start:end] for k, v in full_stems.items() if k != '_sr'}

            # In phrase mode (start is None), skip beat-level alignment and gap handling
            # Phrases are already self-contained concatenated audio
            if start is not None:
                # Per-beat mode: Ensure beat buffer is aligned with stems slice length (resample per-beat if needed)
                beat_len = end - start
                if beat_len <= 0:
                    logger.warning("Invalid beat length %d for beat %d; skipping.", beat_len, i)
                    continue
                if buf.shape[0] != beat_len:
                    logger.debug("Resampling beat buffer from %d to %d samples to match stem slice length.", buf.shape[0], beat_len)
                    if buf.ndim == 1:
                        buf = sciresample(buf, beat_len)
                    else:
                        buf = np.stack([sciresample(buf[:, ch], beat_len) for ch in range(buf.shape[1])], axis=-1)

                # Trim small leading/trailing silence in the beat buffers to avoid introducing gaps
                def trim_leading_trailing_silence(arr: np.ndarray, silence_threshold: float = 1e-4, max_trim_ms: int = 250):
                    """
                    Trim quiet leading/trailing portions based on a mean-abs threshold. 'max_trim_ms' limits amount trimmed on each side.
                    Returns (trimmed_arr, trim_left, trim_right).
                    """
                    if arr.ndim == 2:
                        mono = np.mean(np.abs(arr), axis=1)
                    else:
                        mono = np.abs(arr)
                    max_trim_samples = int(self.sample_rate * float(max_trim_ms) / 1000.0)
                    # Find first index above threshold
                    above = np.where(mono > silence_threshold)[0]
                    if above.size == 0:
                        # all silent; don't trim entirely, just return original (keeps audio shape)
                        return arr, 0, 0
                    left = max(0, above[0] - min(above[0], max_trim_samples))
                    right = max(0, arr.shape[0] - (above[-1] + 1) - min(arr.shape[0] - (above[-1] + 1), max_trim_samples))
                    # The right returned is number of samples to trim from right
                    trimmed_arr = arr[left:arr.shape[0] - right] if right > 0 else arr[left:]
                    return trimmed_arr, left, right

                trimmed, trim_left, trim_right = trim_leading_trailing_silence(buf)
                if trim_left != 0 or trim_right != 0:
                    logger.debug("Trimmed beat %d by left=%d right=%d samples from original len %d to new len %d.", i, trim_left, trim_right, buf.shape[0], trimmed.shape[0])
                    # Update buffer and adjust stems accordingly
                    buf = trimmed
                    start += trim_left
                    end -= trim_right
                    # update beat_stems to match trimmed region
                    new_beat_stems = {}
                    for k, v in beat_stems.items():
                        if v is None:
                            new_beat_stems[k] = v
                            continue
                        s = trim_left
                        e = v.shape[0] - trim_right if trim_right > 0 else v.shape[0]
                        if e <= s:
                            new_beat_stems[k] = np.zeros((0, 2), dtype=v.dtype)
                        else:
                            new_beat_stems[k] = v[s:e]
                    beat_stems = new_beat_stems

                # Debug beat alignment (per-beat mode only)
                gap = None
                if last_end is not None:
                    gap = start - last_end
                    logger.debug("Beat %d: buf shape %s, start %d, end %d, gap from last end: %d", i, buf.shape, start, end, gap)
                else:
                    logger.debug("Beat %d: buf shape %s, start %d, end %d", i, buf.shape, start, end)
            else:
                # Phrase mode: no gap calculation, just log
                gap = None
                logger.debug("Phrase %d: buf shape %s, len %d samples (%.2fs)", i, buf.shape, buf.shape[0], buf.shape[0] / self.sample_rate)
            
            for k, v in beat_stems.items():
                logger.debug("   stem '%s' shape: %s", k, v.shape)

            # Fix stem shapes if needed (pad or trim to (n_blend, 2))
            def fix_stem_shape(arr, target_shape):
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=-1)
                if arr.shape[0] < target_shape[0]:
                    pad = np.zeros((target_shape[0] - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                elif arr.shape[0] > target_shape[0]:
                    arr = arr[:target_shape[0]]
                if arr.shape[1] != target_shape[1]:
                    arr = arr[:, :target_shape[1]]
                return arr

            if prev_buf is not None:
                # Handle gap/overlap between beats (only in per-beat mode, not phrase mode)
                if gap is not None and gap != 0:
                    if gap > 0:
                        logger.warning("Gap of %d samples detected between beats %d and %d.", gap, i-1, i)
                        logger.debug("Gap debug: gap=%d n_blend=%d gap>n_blend=%s", gap, n_blend, gap > n_blend)
                        # Mode-specific behavior
                        if self.gap_mode == 'preserve_large':
                            # Keep current behavior: insert large pad if needed but avoid trimming small zeros
                            if gap > n_blend:
                                pad_len = gap - n_blend
                                pad = np.zeros((pad_len, 2), dtype=np.float32)
                                append_segment(pad, 'pad_large')
                                last_inserted_gap_size = gap
                                logger.debug("   inserted silence pad of length %d, gap=%d, n_blend=%d", pad_len, gap, n_blend)
                        elif self.gap_mode == 'trim_small':
                            # Only trim small-leading/trailing zeros; do not heal later
                            if gap > n_blend:
                                pad_len = gap - n_blend
                                pad = np.zeros((pad_len, 2), dtype=np.float32)
                                append_segment(pad, 'pad_large')
                                last_inserted_gap_size = gap
                                logger.debug("   inserted silence pad of length %d, gap=%d, n_blend=%d", pad_len, gap, n_blend)
                            else:
                                # Trim zeros (same as previous behavior)
                                def count_leading_zeros(arr, threshold=1e-4):
                                    if arr.ndim == 2:
                                        mono = np.mean(np.abs(arr), axis=1)
                                    else:
                                        mono = np.abs(arr)
                                    run = 0
                                    for idx in range(len(mono)):
                                        if mono[idx] <= threshold:
                                            run += 1
                                        else:
                                            break
                                    return run
                                def count_trailing_zeros(arr, threshold=1e-4):
                                    if arr.ndim == 2:
                                        mono = np.mean(np.abs(arr), axis=1)
                                    else:
                                        mono = np.abs(arr)
                                    run = 0
                                    for idx in range(len(mono)-1, -1, -1):
                                        if mono[idx] <= threshold:
                                            run += 1
                                        else:
                                            break
                                    return run
                                leading_zeros = count_leading_zeros(buf, threshold=1e-4)
                                trailing_zeros = count_trailing_zeros(prev_buf, threshold=1e-4)
                                logger.debug("   small gap handling: leading_zeros=%d trailing_zeros=%d, gap=%d", leading_zeros, trailing_zeros, gap)
                                to_remove_from_next = min(leading_zeros, gap)
                                to_remove_from_prev = min(trailing_zeros, gap - to_remove_from_next)
                                if to_remove_from_next > 0:
                                    buf = buf[to_remove_from_next:]
                                    start += to_remove_from_next
                                    if beat_stems:
                                        beat_stems = {k: v[to_remove_from_next:] for k, v in beat_stems.items()}
                                    gap -= to_remove_from_next
                                if to_remove_from_prev > 0:
                                    if prev_buf.shape[0] > to_remove_from_prev:
                                        prev_buf = prev_buf[:-to_remove_from_prev]
                                    else:
                                        prev_buf = np.zeros((0,2), dtype=prev_buf.dtype)
                                    if prev_stems:
                                        new_prev_stems = {}
                                        for k, v in prev_stems.items():
                                            if v.shape[0] > to_remove_from_prev:
                                                new_prev_stems[k] = v[:-to_remove_from_prev]
                                            else:
                                                new_prev_stems[k] = np.zeros((0,2), dtype=v.dtype)
                                        prev_stems = new_prev_stems
                                    gap -= to_remove_from_prev
                                logger.debug("   small gap remaining after trimming: %d", gap)
                        elif self.gap_mode == 'stem':
                            # Try to synthesize per-stem gap filler for small gaps; if not possible, fall back to default behavior
                            if gap > n_blend:
                                pad_len = gap - n_blend
                                pad = np.zeros((pad_len, 2), dtype=np.float32)
                                append_segment(pad, 'pad_large')
                                last_inserted_gap_size = gap
                                logger.debug("   inserted silence pad of length %d, gap=%d, n_blend=%d", pad_len, gap, n_blend)
                            else:
                                try:
                                    # Build gap_fill by motif-tiling each stem's tail and head
                                    motif_len = min(256, gap)
                                    stem_fill = np.zeros((gap, 2), dtype=np.float32)
                                    for k in (prev_stems or {}).keys():
                                        prev_s = prev_stems.get(k)
                                        next_s = beat_stems.get(k)
                                        if prev_s is None or next_s is None:
                                            continue
                                        prev_tail = prev_s[-motif_len:] if prev_s.shape[0] >= motif_len else prev_s
                                        # detect period in prev_tail
                                        try:
                                            prev_mono = np.mean(prev_tail, axis=1)
                                            ac = np.correlate(prev_mono, prev_mono, mode='full')
                                            ac = ac[ac.size//2:]
                                            minlag = 6
                                            maxlag = min(256, prev_mono.shape[0]//2)
                                            period = None
                                            if maxlag > minlag:
                                                lag_range = ac[minlag:maxlag]
                                                peak = np.argmax(lag_range) + minlag
                                                if ac[0] > 0 and ac[peak] / (ac[0] + 1e-12) > 0.2:
                                                    period = int(peak)
                                            motif = prev_tail if period is None else prev_tail[-period:]
                                            repeats = int(np.ceil(gap / motif.shape[0]))
                                            tiled = np.tile(motif, (repeats, 1))[:gap]
                                            # Simple crossfade into next stem (if present)
                                            fade_len = min(motif.shape[0], tiled.shape[0], 64)
                                            if next_s.shape[0] >= fade_len:
                                                fade_in = np.linspace(0, 1, fade_len)[:, None]
                                                tiled[-fade_len:] = tiled[-fade_len:] * (1 - fade_in) + next_s[:fade_len] * fade_in
                                            # Mix into stem_fill
                                            stem_fill += tiled
                                        except Exception:
                                            continue
                                    # normalize stem_fill
                                    peak = np.max(np.abs(stem_fill)) if stem_fill.size else 0.0
                                    if peak > 1e-6:
                                        stem_fill = stem_fill * (0.8 / (peak + 1e-12))
                                    append_segment(stem_fill, 'content')
                                    last_inserted_gap_size = 0
                                    logger.debug("   created stem-based gap_fill for gap=%d", gap)
                                except Exception as e:
                                    logger.debug("   stem gap fill failed: %s", e)
                                    # fallback to default: insert silence
                                    pad_len = gap - n_blend
                                    if pad_len > 0:
                                        pad = np.zeros((pad_len, 2), dtype=np.float32)
                                        append_segment(pad, 'pad_large')
                                        last_inserted_gap_size = gap
                        else:
                            # default: heal (existing behavior)
                            if gap > n_blend:
                                pad_len = gap - n_blend
                                pad = np.zeros((pad_len, 2), dtype=np.float32)
                                append_segment(pad, 'pad_large')
                                last_inserted_gap_size = gap
                            else:
                                # For small gaps, try to remove leading/trailing silence on either side to eliminate the audible gap.
                                def count_leading_zeros(arr, threshold=1e-4):
                                    if arr.ndim == 2:
                                        mono = np.mean(np.abs(arr), axis=1)
                                    else:
                                        mono = np.abs(arr)
                                    run = 0
                                    for idx in range(len(mono)):
                                        if mono[idx] <= threshold:
                                            run += 1
                                        else:
                                            break
                                    return run
                                def count_trailing_zeros(arr, threshold=1e-4):
                                    if arr.ndim == 2:
                                        mono = np.mean(np.abs(arr), axis=1)
                                    else:
                                        mono = np.abs(arr)
                                    run = 0
                                    for idx in range(len(mono)-1, -1, -1):
                                        if mono[idx] <= threshold:
                                            run += 1
                                        else:
                                            break
                                    return run
                                leading_zeros = count_leading_zeros(buf, threshold=1e-4)
                                trailing_zeros = count_trailing_zeros(prev_buf, threshold=1e-4)
                                logger.debug("   small gap handling: leading_zeros=%d trailing_zeros=%d, gap=%d", leading_zeros, trailing_zeros, gap)
                                # Remove as many silent samples as we can up to the gap size
                                to_remove_from_next = min(leading_zeros, gap)
                                to_remove_from_prev = min(trailing_zeros, gap - to_remove_from_next)
                                if to_remove_from_next > 0:
                                    buf = buf[to_remove_from_next:]
                                    start += to_remove_from_next
                                    # Slice stems accordingly
                                    if beat_stems:
                                        beat_stems = {k: v[to_remove_from_next:] for k, v in beat_stems.items()}
                                    gap -= to_remove_from_next
                                if to_remove_from_prev > 0:
                                    if prev_buf.shape[0] > to_remove_from_prev:
                                        prev_buf = prev_buf[:-to_remove_from_prev]
                                    else:
                                        prev_buf = np.zeros((0,2), dtype=prev_buf.dtype)
                                    # Also adjust prev_stems
                                    if prev_stems:
                                        new_prev_stems = {}
                                        for k, v in prev_stems.items():
                                            if v.shape[0] > to_remove_from_prev:
                                                new_prev_stems[k] = v[:-to_remove_from_prev]
                                            else:
                                                new_prev_stems[k] = np.zeros((0,2), dtype=v.dtype)
                                        prev_stems = new_prev_stems
                                    gap -= to_remove_from_prev
                                logger.debug("   small gap remaining after trimming: %d", gap)
                        # No manual crossfade; blending will handle n_blend at the edges
                    elif gap < 0:
                        overlap = -gap
                        logger.debug("Overlap of %d samples detected between beats %d and %d. Using blend instead of trimming to avoid dropping content.", overlap, i-1, i)

                # If the buffer is now empty after trimming (extreme overlap), skip blending
                if buf.size == 0:
                    logger.warning("After trimming overlap, beat %d buffer is empty; skipping this beat.", i)
                    # do not update prev_buf/prev_stems; continue to next beat
                    continue

                # Adaptive blend length — don't require n_blend to be present for blending
                blend_len = min(n_blend, prev_buf.shape[0], buf.shape[0]) if prev_buf is not None else 0

                # If stems are not available, perform a simple raw audio crossfade over blend_len samples
                if not stems_available:
                    fade = np.linspace(0, 1, n_blend)[:, None]
                    # Ensure prev_buf has at least n_blend for fade; otherwise fallback to concatenation
                    if blend_len > 0 and prev_buf.shape[0] >= blend_len and buf.shape[0] >= blend_len:
                        prev_segment = prev_buf[-n_blend:]
                        next_segment = buf[:n_blend]
                        # RMS match the segments to reduce abrupt changes
                        def rms(x):
                            return np.sqrt(np.mean(x**2))
                        prev_r = rms(prev_segment)
                        next_r = rms(next_segment)
                        if prev_r > 1e-9 and next_r > 1e-9:
                            target_r = 0.5 * (prev_r + next_r)
                            scale_prev = min(1.0, target_r / (prev_r + 1e-12))
                            scale_next = min(1.0, target_r / (next_r + 1e-12))
                            prev_segment = prev_segment * scale_prev
                            next_segment = next_segment * scale_next
                        crossfade = prev_segment * (1 - fade) + next_segment * fade
                        # Peak-check the crossfade and scale if needed
                        peak = np.max(np.abs(crossfade))
                        if peak > 0.99:
                            crossfade = crossfade * (0.99 / (peak + 1e-12))
                        # Trim last n_blend from constructed output and append crossfade and remainder
                        # Allow trimming into earlier small pad segments so blends can overlap (avoid leftover zeros between blends)
                        allow_trim_pad = (last_inserted_gap_size > 0 and last_inserted_gap_size <= n_blend)
                        pop_trim_last_n(beat_audio, blend_len, allow_trim_pad=allow_trim_pad)
                        # reset pad length tracker after trimming
                        last_inserted_gap_size = 0
                        append_segment(np.clip(crossfade, -1.0, 1.0), 'content')
                        if buf.shape[0] > blend_len:
                            remainder = ensure_stereo(buf[blend_len:]); append_segment(np.clip(remainder, -1.0, 1.0), 'content')
                    else:
                        # Not enough samples to crossfade, append simple concatenation
                        append_segment(ensure_stereo(buf), 'content')
                    prev_buf = buf
                    prev_stems = beat_stems
                    if end is not None:
                        last_end = end
                    continue

                # Slice stems to blend window only and adapt to the actual blend length
                if blend_len <= 0:
                    # No blending possible: append simple silence for safety and continue
                    zero_pad = np.zeros((0, 2), dtype=np.float32)
                    append_segment(zero_pad, 'pad_large')
                    prev_buf = buf
                    prev_stems = beat_stems
                    if end is not None:
                        last_end = end
                    continue
                prev_stems_blend = {k: fix_stem_shape(v[-blend_len:], (blend_len, 2)) for k, v in prev_stems.items()}
                beat_stems_blend = {k: fix_stem_shape(v[:blend_len], (blend_len, 2)) for k, v in beat_stems.items()}
                # Debug: print actual blend window stem shapes
                for k in prev_stems_blend:
                        logger.debug("   prev_stems_blend['%s'] shape: %s (expected (%d, 2))", k, prev_stems_blend[k].shape, n_blend)
                for k in beat_stems_blend:
                    logger.debug("   beat_stems_blend['%s'] shape: %s (expected (%d, 2))", k, beat_stems_blend[k].shape, n_blend)
                # Run blending for the computed blend length (blend_beats expects blend_duration in seconds)
                blend_duration_seconds = float(blend_len) / float(self.sample_rate)
                try:
                    blended = blend_beats(
                        prev_buf, buf,
                        sample_rate=self.sample_rate,
                        blend_duration=blend_duration_seconds,
                        prev_stems=prev_stems_blend,
                        next_stems=beat_stems_blend,
                        demucs_device=self.demucs_device
                    )
                except Exception as e:
                    logger.warning("blend_beats failed: %s — falling back to RMS crossfade", e)
                    # Fallback: create a simple RMS-matched crossfade of length blend_len
                    fade = np.linspace(0, 1, blend_len)[:, None]
                    prev_segment = prev_buf[-blend_len:]
                    next_segment = buf[:blend_len]
                    def rms(x):
                        return np.sqrt(np.mean(x**2))
                    prev_r = rms(prev_segment) if prev_segment.size else 0.0
                    next_r = rms(next_segment) if next_segment.size else 0.0
                    if prev_r > 1e-9 and next_r > 1e-9:
                        target_r = 0.5 * (prev_r + next_r)
                        scale_prev = min(1.0, target_r / (prev_r + 1e-12))
                        scale_next = min(1.0, target_r / (next_r + 1e-12))
                        prev_segment = prev_segment * scale_prev
                        next_segment = next_segment * scale_next
                    crossfade = prev_segment * (1 - fade) + next_segment * fade
                    # Build merged array: head (prev minus overlap) + crossfade + remainder of next
                    head = prev_buf[:-blend_len] if prev_buf.shape[0] > blend_len else np.zeros((0, 2), dtype=prev_buf.dtype)
                    tail = buf[blend_len:] if buf.shape[0] > blend_len else np.zeros((0, 2), dtype=buf.dtype)
                    blended = np.concatenate([head, crossfade, tail], axis=0)
                logger.debug("   blended shape: %s", blended.shape)
                # To avoid duplicating the tail of previously appended audio, trim the last blend_len samples
                # from the constructed output (excluding pad segments) before appending the blended segment.
                allow_trim_pad = (last_inserted_gap_size > 0 and last_inserted_gap_size <= n_blend)
                pop_trim_last_n(beat_audio, blend_len, allow_trim_pad=allow_trim_pad)
                last_inserted_gap_size = 0
                # Determine the tail of the blended segment to append so we don't duplicate prev_buf content
                tail_start = max(0, prev_buf.shape[0] - blend_len)
                blended_tail = blended[tail_start:]
                arr = ensure_stereo(blended_tail)
                arr = np.clip(arr, -1.0, 1.0)
                append_segment(arr, 'content')
                logger.debug("   appended blend segment, shape: %s", arr.shape)
            else:
                arr = ensure_stereo(buf)
                arr = np.clip(arr, -1.0, 1.0)
                append_segment(arr, 'content')
                logger.debug("   appended first beat segment, shape: %s", arr.shape)
                prev_buf = buf
            prev_stems = beat_stems
            # In phrase mode, last_end is not used for gap calculation, but set it to avoid errors
            if end is not None:
                last_end = end
        # Concatenate arrays stored in (arr, tag) tuples
        arrays = [arr for arr, tag in beat_audio if arr is not None and arr.shape[0] > 0]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("beat_audio segments summary: %s", [(tag, a.shape[0], float(a.min() if a.size else 0), float(a.max() if a.size else 0)) for a, tag in beat_audio if a is not None])
        if arrays:
            out = np.concatenate(arrays)
        else:
            out = np.zeros((0, 2), dtype=np.float32)
        # If requested, heal short near-zero gaps in the assembled output.
        if self.gap_heal_ms and self.gap_heal_ms > 0 and out.shape[0] > 0 and self.gap_mode == 'heal':
            out = self._heal_short_gaps(out, gap_heal_ms=self.gap_heal_ms, sample_rate=self.sample_rate, amplitude_threshold=self.gap_heal_threshold)
        
        # Apply soft clipping first to gently tame overshoots before peak normalization
        # This prevents harsh digital clipping on transients
        out = soft_clip_signal(out, threshold=0.95)
        
        # Final peak normalization if still needed
        maxval = np.max(np.abs(out))
        peak_limit = 0.98
        if maxval > peak_limit:
            logger.debug("Final output peak %f > %f, scaling to %f.", maxval, peak_limit, peak_limit)
            out = out * (peak_limit / (maxval + 1e-12))
        
        logger.debug("Final output shape: %s, min: %f, max: %f", out.shape, out.min(), out.max())
        # Optional inference: truncate silence runs of mid length after all healing
        if self.truncate_enabled and out.shape[0] > 0:
            out = self._truncate_silence_runs(out, min_ms=self.truncate_min_ms, max_ms=self.truncate_max_ms, sample_rate=self.sample_rate, amplitude_threshold=self.truncate_threshold, crossfade_ms=self.truncate_crossfade_ms)

        if debug_return_segments:
            return out, beat_audio
        return out

    def _heal_short_gaps(self, audio: np.ndarray, gap_heal_ms: int = 20, sample_rate: int = 44100, amplitude_threshold: float = 1e-4) -> np.ndarray:
        """
        Replace short near-zero runs (gaps) with a linear interpolation across the gap to prevent tiny audible clicks or silence.
        Only replaces runs shorter than 'gap_heal_ms'.
        """
        logger = logging.getLogger(__name__)
        max_gap_samples = int(sample_rate * float(gap_heal_ms) / 1000.0)
        if max_gap_samples <= 0:
            return audio
        arr = audio.copy()
        # use mean across channels to find gaps
        mono = np.mean(np.abs(arr), axis=1)
        small = mono <= amplitude_threshold
        # find runs of True values
        runs = []
        start = None
        for i, v in enumerate(small):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - start))
                start = None
        if start is not None:
            runs.append((start, len(small) - start))
        if not runs:
            return arr
        logger.debug("Healing short gaps: found %d near-zero runs", len(runs))
        for run_start, run_len in runs:
            if run_len <= max_gap_samples and run_start > 0 and (run_start + run_len) < arr.shape[0] - 1:
                left_idx = run_start - 1
                right_idx = run_start + run_len
                # Use context-aware crossfade: sample short contexts from left and right and resample to run_len.
                ctx = min(256, left_idx + 1, arr.shape[0] - right_idx)
                if ctx <= 0:
                    # Fallback to single-sample linear interpolation (edges)
                    for ch in range(arr.shape[1]):
                        left_val = float(arr[left_idx, ch])
                        right_val = float(arr[right_idx, ch])
                        interp = np.linspace(left_val, right_val, run_len + 2)[1:-1]
                        arr[left_idx+1:right_idx, ch] = interp
                    logger.debug("Healed gap at %d len=%d via scalar interpolation (edge case)", run_start, run_len)
                    continue
                # Extract contexts
                left_ctx = arr[left_idx - ctx + 1:left_idx + 1]
                right_ctx = arr[right_idx:right_idx + ctx]
                # Resample contexts to run_len using linear resampling across the first dimension
                if run_len > 1:
                    # create index positions
                    left_pos = np.linspace(0, left_ctx.shape[0] - 1, run_len)
                    right_pos = np.linspace(0, right_ctx.shape[0] - 1, run_len)
                    # Interpolate per channel by linear interpolation
                    for ch in range(arr.shape[1]):
                        left_vals = np.interp(left_pos, np.arange(left_ctx.shape[0]), left_ctx[:, ch])
                        right_vals = np.interp(right_pos, np.arange(right_ctx.shape[0]), right_ctx[:, ch])
                        weights = np.linspace(0.0, 1.0, run_len + 2)[1:-1]
                        arr[left_idx+1:right_idx, ch] = left_vals * (1.0 - weights) + right_vals * weights
                else:
                    # Single-sample run
                    for ch in range(arr.shape[1]):
                        arr[left_idx+1, ch] = 0.5 * (arr[left_idx, ch] + arr[right_idx, ch])
                logger.debug("Healed gap at %d len=%d (<= %d) via context-aware crossfade.", run_start, run_len, max_gap_samples)

                # Additional improvement: if the left context has a clear periodicity,
                # tile a motif from the left context and use it as the content for the gap,
                # crossfading smoothly into the right context; this preserves texture.
                try:
                    # Mono signal from left context
                    left_mono = np.mean(left_ctx, axis=1)
                    # Autocorrelation to detect period
                    ac = np.correlate(left_mono, left_mono, mode='full')
                    ac = ac[ac.size//2:]
                    # Ignore 0 lag; search for peak between 20 and 1000 samples
                    minlag = 20
                    maxlag = min(1000, left_mono.shape[0]//2 if left_mono.shape[0]//2 > 20 else left_mono.shape[0])
                    if maxlag > minlag:
                        lag_range = ac[minlag:maxlag]
                        peak = np.argmax(lag_range) + minlag
                        peak_val = ac[peak]
                        # heuristic: if peak is significant relative to zero-lag
                        if ac[0] > 0 and peak_val / (ac[0] + 1e-12) > 0.2:
                            period = int(peak)
                            motif = left_ctx[-period:]
                            # Tile motif to fill run_len
                            repeats = int(np.ceil(run_len / motif.shape[0]))
                            tiled = np.tile(motif, (repeats, 1))[:run_len]
                            # Crossfade tiled result into right context over a short window
                            fade_len = min(256, run_len)
                            fade_in = np.linspace(0, 1, fade_len)[:, None]
                            if tiled.shape[0] >= fade_len and right_ctx.shape[0] >= fade_len:
                                tiled[:fade_len] = tiled[:fade_len] * (1 - fade_in) + right_ctx[:fade_len] * fade_in
                            arr[left_idx+1:right_idx] = tiled
                            logger.debug("Healed gap at %d len=%d using tiled motif period=%d", run_start, run_len, period)
                except Exception:
                    # Ignore tuning errors and keep the crossfade
                    pass
        return arr

    def _truncate_silence_runs(self, audio: np.ndarray, min_ms: int = 100, max_ms: int = 300, sample_rate: int = 44100, amplitude_threshold: float = 0.01, crossfade_ms: int = 10) -> np.ndarray:
        """
        Remove silence runs between min_ms (inclusive) and max_ms (inclusive) using the amplitude threshold to classify silence.
        Replaces the removed run by joining the left and right segments and performs a small crossfade to avoid click artifacts.
        """
        logger = logging.getLogger(__name__)
        min_samples = int(sample_rate * float(min_ms) / 1000.0)
        max_samples = int(sample_rate * float(max_ms) / 1000.0)
        if min_samples <= 0 or max_samples < min_samples:
            return audio
        arr = audio.copy()
        mono = np.mean(np.abs(arr), axis=1)
        sil = mono <= amplitude_threshold
        runs = []
        start = None
        for i, v in enumerate(sil):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - start))
                start = None
        if start is not None:
            runs.append((start, len(sil) - start))
        if not runs:
            return arr
        logger.debug("Truncate silence: found %d runs; min=%d samples max=%d samples", len(runs), min_samples, max_samples)
        # Use offset to adjust subsequent indices when removing ranges
        offset = 0
        removed_count = 0
        # If an adaptive factor is set and > 0, we'll use local RMS to make threshold detection adaptive.
        adapt_factor = getattr(self, 'truncate_adaptive_factor', 0.0)
        crossfade_samples = max(1, int(float(crossfade_ms) / 1000.0 * sample_rate))
        # The fraction of samples below threshold required to classify a run as 'silence'
        sample_pct = getattr(self, 'truncate_sample_pct', 0.95)
        for run_start, run_len in runs:
            # apply offset correction
            run_s = run_start - offset
            if run_s < 0:
                continue
            run_e = run_s + run_len
            if run_len >= min_samples and run_len <= max_samples and run_e < arr.shape[0]:
                # Compute adaptive threshold based on local RMS if requested
                if adapt_factor and adapt_factor > 0:
                    # Determine local window for RMS calc
                    ctx_w = max(1, int(0.05 * sample_rate))  # 50ms
                    ctx_left_s = max(0, run_s - ctx_w)
                    ctx_left_e = run_s
                    ctx_right_s = run_e
                    ctx_right_e = min(arr.shape[0], run_e + ctx_w)
                    left_rms = 0.0
                    right_rms = 0.0
                    if ctx_left_e - ctx_left_s > 0:
                        left_rms = float(np.sqrt(np.mean(np.mean(arr[ctx_left_s:ctx_left_e]**2, axis=1))))
                    if ctx_right_e - ctx_right_s > 0:
                        right_rms = float(np.sqrt(np.mean(np.mean(arr[ctx_right_s:ctx_right_e]**2, axis=1))))
                    local_rms = max(left_rms, right_rms, 1e-12)
                    threshold_run = max(amplitude_threshold, local_rms * adapt_factor)
                else:
                    threshold_run = amplitude_threshold
                # Recompute the run as near-silent given local threshold
                run_mono = np.mean(np.abs(arr[run_s:run_e]), axis=1)
                # Robust silence detection: treat as silence if most samples are below threshold OR median is below threshold
                frac_below = float(np.mean(run_mono <= threshold_run))
                median_val = float(np.median(run_mono))
                if frac_below >= sample_pct or median_val <= threshold_run:
                    # Build left and right
                    left = arr[:run_s]
                    right = arr[run_e:]
                    # If compress mode is set, replace the run with a short silence of length truncate_compress_ms
                    if getattr(self, 'truncate_mode', 'remove') == 'compress':
                        compress_ms = getattr(self, 'truncate_compress_ms', 20)
                        compress_samples = max(1, int(float(compress_ms) / 1000.0 * sample_rate))
                        # For crossfade across silence, ensure we don't exceed compress samples
                        left_cf = crossfade_samples
                        right_cf = crossfade_samples
                        # If compress space is too small, reduce crossfade per side
                        if compress_samples < (left_cf + right_cf):
                            left_cf = right_cf = max(1, compress_samples // 2)
                        silence_mid_len = compress_samples - (left_cf + right_cf)
                        if left.shape[0] >= left_cf and right.shape[0] >= right_cf:
                            left_pre = left[:-left_cf]
                            left_fade = left[-left_cf:]
                            right_fade = right[:right_cf]
                            right_post = right[right_cf:]
                            fade_l = np.linspace(0.0, 1.0, left_cf)[:, None]
                            fade_r = np.linspace(0.0, 1.0, right_cf)[:, None]
                            left_to_sil = left_fade * (1.0 - fade_l)
                            sil_mid = np.zeros((max(0, silence_mid_len), arr.shape[1]), dtype=arr.dtype)
                            sil_to_right = right_fade * fade_r
                            arr = np.concatenate([left_pre, left_to_sil, sil_mid, sil_to_right, right_post], axis=0)
                        else:
                            # Not enough context for crossfade; just slot in a compressed silent run
                            silence_small = np.zeros((compress_samples, arr.shape[1]), dtype=arr.dtype)
                            arr = np.concatenate([left, silence_small, right], axis=0)
                        offset += run_len - compress_samples
                        removed_count += 1
                        logger.debug("Compressed silence at %d len=%d -> %d samples", run_start, run_len, compress_samples)
                        continue
                    # If both left and right have enough samples, crossfade
                    if left.shape[0] >= crossfade_samples and right.shape[0] >= crossfade_samples:
                        # preserve unaffected tails
                        left_pre = left[:-crossfade_samples]
                        left_fade = left[-crossfade_samples:]
                        right_fade = right[:crossfade_samples]
                        right_post = right[crossfade_samples:]
                        fade = np.linspace(0.0, 1.0, crossfade_samples)[:, None]
                        cross = left_fade * (1.0 - fade) + right_fade * fade
                        arr = np.concatenate([left_pre, cross, right_post], axis=0)
                    else:
                        # Not enough to crossfade: just concatenate
                        arr = np.concatenate([left, right], axis=0)
                    offset += run_len
                    logger.debug("Truncated silence at %d len=%d -> removed %d samples", run_start, run_len, run_len)
                    removed_count += 1
                    continue
                else:
                    # Not considered a silence under threshold; keep as-is
                    continue
        logger.info("Truncate silence summary: %d runs inspected, %d runs removed/compressed (min=%dms max=%dms threshold=%f adaptive_factor=%f)",
                len(runs), removed_count, min_ms, max_ms, amplitude_threshold, adapt_factor)
        return arr

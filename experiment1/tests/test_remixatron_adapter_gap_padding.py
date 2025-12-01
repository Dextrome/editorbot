import numpy as np
import pytest
from unittest import mock

from src.ai.remixatron_adapter import RemixatronAdapter


class FakeJukebox:
    """Minimal fake InfiniteJukebox for testing.
    Exposes beats (list of dicts) and play_vector (list of dicts with 'beat').
    """
    def __init__(self, beats, play_vector):
        self.beats = beats
        self.play_vector = play_vector


def _make_fake_beats(sample_rate=44100, prev_len=1024, next_len=1024, gap=100):
    # Build sample stereo buffers with deterministic data
    prev_buf = np.ones((prev_len, 2), dtype=np.float32) * 0.1  # simple non-zero tone
    next_buf = np.ones((next_len, 2), dtype=np.float32) * 0.2
    # Beats include start and end that yields a gap between last_end and start
    prev_start = 0
    prev_end = prev_len  # end index
    next_start = prev_end + gap
    next_end = next_start + next_len
    beats = [
        {'buffer': prev_buf.copy(), 'start': prev_start, 'end': prev_end},
        {'buffer': next_buf.copy(), 'start': next_start, 'end': next_end},
    ]
    play_vector = [{'beat': 0}, {'beat': 1}]
    return beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end


def _make_full_stems(beats, total_length):
    # Create a fake full track stems dict where slices will return our beat segments
    drums = np.zeros((total_length, 2), dtype=np.float32)
    vocals = np.zeros((total_length, 2), dtype=np.float32)
    for b in beats:
        s = int(b['start'])
        e = int(b['end'])
        drums[s:e, :] = np.ones((e - s, 2), dtype=np.float32) * 0.1
        vocals[s:e, :] = np.ones((e - s, 2), dtype=np.float32) * 0.2
    return {'drums': drums, 'vocals': vocals, '_sr': 44100}


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_small_gap_filled_by_blend(fake_jukebox_cls, fake_demucs_cls):
    import logging
    logging.getLogger('src.ai.remixatron_adapter').setLevel(logging.DEBUG)
    # Arrange: fake beats with a small positive gap
    sample_rate = 44100
    beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end = _make_fake_beats(sample_rate=sample_rate, prev_len=1024, next_len=1024, gap=100)
    total_length = int(next_end)
    full_stems = _make_full_stems(beats, total_length)

    # Fake DemucsSeparator.separate call returns our stems
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = full_stems

    # Fake InfiniteJukebox that exposes beats and a play_vector
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox

    # Act: rearrange using the adapter; we use a small blend duration
    adapter = RemixatronAdapter(blend_duration=0.2, sample_rate=sample_rate, demucs_device='cpu', max_jump=8)
    out, segs = adapter.rearrange('fake_path.wav', debug_return_segments=True)

    # Assert: For a small positive gap, the adapter should fill the gap via blending so the
    # tiny gap region should not be silent. We look in a small window around the expected gap index.
    prev_len = prev_buf.shape[0]
    # The adapter should preserve inserted pad segments (tagged as 'pad') so we expect
    # the explicit zero pad to remain at the intended gap position in the output.
    pad_len = 100
    # Find runs of zero samples (both channels close to 0) in output
    abs_out = np.max(np.abs(out), axis=1)
    is_zero = abs_out < 1e-6
    # find consecutive zero runs
    runs = []
    start = None
    for idx, val in enumerate(is_zero):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    # Assert there is NOT a zero-run in the expected small gap region
    # Look around prev_len for pad_len zeros (within a small window) - we expect none.
    window_start = max(0, prev_len - 10)
    window_end = min(len(out), prev_len + pad_len + 10)
    window = is_zero[window_start:window_end]
    assert not any(window), f"Gap region around {prev_len} contains silent run after blending; runs: {runs}"


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_large_gap_preserved(fake_jukebox_cls, fake_demucs_cls):
    import logging
    logging.getLogger('src.ai.remixatron_adapter').setLevel(logging.DEBUG)
    # Arrange: create beats with a large gap (larger than blend window) to ensure pad is preserved
    sample_rate = 44100
    n_blend = int(sample_rate * 0.2)
    large_gap = n_blend + 100
    beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end = _make_fake_beats(sample_rate=sample_rate, prev_len=1024, next_len=1024, gap=large_gap)
    total_length = int(next_end)
    full_stems = _make_full_stems(beats, total_length)
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = full_stems
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox

    adapter = RemixatronAdapter(blend_duration=0.2, sample_rate=sample_rate, demucs_device='cpu', max_jump=8)
    out, segs = adapter.rearrange('fake_path.wav', debug_return_segments=True)
    # Now assert there is a zero-run of length >= expected_pad_len around prev_buf boundary
    prev_len = prev_buf.shape[0]
    expected_pad_len = large_gap - n_blend
    abs_out = np.max(np.abs(out), axis=1)
    is_zero = abs_out < 1e-6
    runs = []
    start = None
    for idx, val in enumerate(is_zero):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    # find if any run intersects the expected gap area and is long enough
    found = False
    for run_start, run_len in runs:
        if run_len >= expected_pad_len:
            # ensure run is close to prev_len
            if abs(run_start - prev_len) < 50:
                found = True
                break
    if not found:
        print('DEBUG: beat_audio segments:')
        for a, t in segs:
            print(t, a.shape[0], float(a.min() if a.size else 0.0), float(a.max() if a.size else 0.0))
    assert found, f"No preserved large zero-run found near expected gap; runs={runs}, prev_len={prev_len}, expected_pad_len={expected_pad_len}, large_gap={large_gap}"
    # For debugging: print a summary of appended segments
    for a, tag in segs:
        print(tag, a.shape[0], np.min(a) if a.size else 0.0, np.max(a) if a.size else 0.0)
    # DEBUG: show a sample region around prev_len for inspection
    print('DEBUG: out samples at prev_len..prev_len+40:', out[prev_len:prev_len+40])
    print('DEBUG: min/max of out:', out.min(), out.max())


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_stem_mode_creates_nonzero_gap_fill(fake_jukebox_cls, fake_demucs_cls):
    import logging
    logging.getLogger('src.ai.remixatron_adapter').setLevel(logging.DEBUG)
    # Use a small gap; in stem mode the adapter should attempt to fill the gap using per-stem motif fills
    sample_rate = 44100
    gap = 100
    beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end = _make_fake_beats(sample_rate=sample_rate, prev_len=1024, next_len=1024, gap=gap)
    total_length = int(next_end)
    # Make multi-stem full track: drums and vocals
    drums = np.zeros((total_length, 2), dtype=np.float32)
    vocals = np.zeros((total_length, 2), dtype=np.float32)
    for b in beats:
        s = int(b['start'])
        e = int(b['end'])
        drums[s:e, :] = np.ones((e - s, 2), dtype=np.float32) * 0.05
        vocals[s:e, :] = np.ones((e - s, 2), dtype=np.float32) * 0.15
    full_stems = {'drums': drums, 'vocals': vocals, '_sr': 44100}

    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = full_stems

    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox

    adapter = RemixatronAdapter(blend_duration=0.2, sample_rate=sample_rate, demucs_device='cpu', max_jump=8, gap_mode='stem')
    out, segs = adapter.rearrange('fake_path.wav', debug_return_segments=True)

    # Verify that the gap region does not contain a zero-run of the previous gap length
    prev_len = prev_buf.shape[0]
    abs_out = np.max(np.abs(out), axis=1)
    is_zero = abs_out < 1e-6
    window_start = max(0, prev_len - 10)
    window_end = min(len(out), prev_len + gap + 10)
    window = is_zero[window_start:window_end]
    assert not any(window), "Gap region unexpectedly contains silent run when using stem gap-mode"


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_trim_small_leaves_gap_when_heal_disabled(fake_jukebox_cls, fake_demucs_cls):
    import logging
    logging.getLogger('src.ai.remixatron_adapter').setLevel(logging.DEBUG)
    sample_rate = 44100
    gap = 100
    beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end = _make_fake_beats(sample_rate=sample_rate, prev_len=1024, next_len=1024, gap=gap)
    total_length = int(next_end)
    full_stems = _make_full_stems(beats, total_length)
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = full_stems
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox

    # gap_heal_ms is set, but gap_mode='trim_small' should avoid final healing pass
    # Set blend_duration to 0 to disable blending so small gaps remain as explicit zero pads
    adapter = RemixatronAdapter(blend_duration=0.0, sample_rate=sample_rate, demucs_device='cpu', max_jump=8, gap_heal_ms=250, gap_mode='trim_small')
    out, segs = adapter.rearrange('fake_path.wav', debug_return_segments=True)
    prev_len = prev_buf.shape[0]
    abs_out = np.max(np.abs(out), axis=1)
    is_zero = abs_out < 1e-6
    window_start = max(0, prev_len - 10)
    window_end = min(len(out), prev_len + gap + 10)
    window = is_zero[window_start:window_end]
    # We should see some zeros in the gap area as trimming only removes leading/trailing silence and does not heal it
    assert any(window), "Expected zero pad around gap when gap_mode='trim_small' but found none"


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_rearrange_truncates_mid_length_silence_when_enabled(fake_jukebox_cls, fake_demucs_cls):
    import logging
    logging.getLogger('src.ai.remixatron_adapter').setLevel(logging.DEBUG)
    sample_rate = 44100
    # Set gap in beats to be roughly 200ms (8820 samples)
    gap = int(0.2 * sample_rate)
    beats, play_vector, prev_buf, next_buf, prev_start, prev_end, next_start, next_end = _make_fake_beats(sample_rate=sample_rate, prev_len=1024, next_len=1024, gap=gap)
    total_length = int(next_end)
    full_stems = _make_full_stems(beats, total_length)
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = full_stems
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox

    # Use a small blend duration so the pad gets inserted for mid-length gaps
    adapter = RemixatronAdapter(blend_duration=0.01, sample_rate=sample_rate, demucs_device='cpu', max_jump=8, gap_mode='preserve_large', truncate_enabled=True, truncate_min_ms=100, truncate_max_ms=300, truncate_threshold=0.01)
    out, segs = adapter.rearrange('fake_path.wav', debug_return_segments=True)
    # verify there is NOT a long zero-run (~200ms)
    prev_len = prev_buf.shape[0]
    abs_out = np.max(np.abs(out), axis=1)
    is_zero = abs_out < 1e-6
    runs = []
    start = None
    for idx, val in enumerate(is_zero):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    assert not any(l >= int(0.2 * sample_rate) for s, l in runs), f"Found redundant ~200ms silent run after truncation: {runs}"

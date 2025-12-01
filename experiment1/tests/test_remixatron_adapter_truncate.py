import numpy as np
from src.ai.remixatron_adapter import RemixatronAdapter


def make_truncate_signal(sr=44100, gap_len_ms=200):
    # Make a signal with a mid-length gap between two sections
    pre = np.ones((100, 2), dtype=np.float32) * 0.1
    gap_len = int(sr * gap_len_ms / 1000.0)
    gap = np.zeros((gap_len, 2), dtype=np.float32)
    tail = np.ones((100, 2), dtype=np.float32) * 0.2
    return np.concatenate([pre, gap, tail], axis=0)


def test_truncate_removes_200ms_gap():
    sr = 44100
    arr = make_truncate_signal(sr=sr, gap_len_ms=200)
    adapter = RemixatronAdapter(truncate_enabled=True, truncate_min_ms=100, truncate_max_ms=300, truncate_threshold=0.01, sample_rate=sr)
    out = adapter._truncate_silence_runs(arr, min_ms=100, max_ms=300, sample_rate=sr, amplitude_threshold=0.01)
    # Ensure the big ~200ms silent run no longer exists
    mono = np.mean(np.abs(out), axis=1)
    is_zero = mono <= 0.01
    # should not have a run >= 200ms
    runs = []
    start = None
    for i, v in enumerate(is_zero):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    assert not any(l >= int(0.200 * sr) for s, l in runs), f"Found a run >= 200ms after truncation: {runs}"


def test_truncate_does_not_remove_small_50ms_gap():
    sr = 44100
    # A 50ms gap should not be removed if min_ms=100
    arr = make_truncate_signal(sr=sr, gap_len_ms=50)
    adapter = RemixatronAdapter(truncate_enabled=True, truncate_min_ms=100, truncate_max_ms=300, truncate_threshold=0.01, sample_rate=sr)
    out = adapter._truncate_silence_runs(arr, min_ms=100, max_ms=300, sample_rate=sr, amplitude_threshold=0.01)
    mono = np.mean(np.abs(out), axis=1)
    is_zero = mono <= 0.01
    # There should still be at least one small silent run (50ms)
    runs = []
    start = None
    for i, v in enumerate(is_zero):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    assert any(l >= int(0.050 * sr) for s, l in runs), f"Expected a 50ms run to remain after truncation but none found: {runs}"


def test_truncate_adaptive_detects_low_amplitude_gap():
    sr = 44100
    # 200ms gap at amplitude 0.005, left/right contexts at 0.1 -> with adaptive factor 0.05, threshold=0.005
    left = np.ones((100, 2), dtype=np.float32) * 0.1
    gap_len = int(0.200 * sr)
    gap = np.ones((gap_len, 2), dtype=np.float32) * 0.005
    right = np.ones((100, 2), dtype=np.float32) * 0.1
    arr = np.concatenate([left, gap, right], axis=0)
    adapter = RemixatronAdapter(truncate_enabled=True, truncate_min_ms=100, truncate_max_ms=300, truncate_threshold=0.001, sample_rate=sr, truncate_adaptive_factor=0.05)
    out = adapter._truncate_silence_runs(arr, min_ms=100, max_ms=300, sample_rate=sr, amplitude_threshold=0.001, crossfade_ms=10)
    mono = np.mean(np.abs(out), axis=1)
    is_zero = mono <= 0.001
    runs = []
    start = None
    for i, v in enumerate(is_zero):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_zero) - start))
    assert not any(l >= int(0.200 * sr) for s, l in runs), "Adaptive truncation should remove 200ms low-amplitude gap"


def test_truncate_compress_reduces_200ms_gap():
    sr = 44100
    # create signal with 200ms of silence
    left = np.ones((100, 2), dtype=np.float32) * 0.1
    gap_len = int(0.200 * sr)
    gap = np.zeros((gap_len, 2), dtype=np.float32)
    right = np.ones((100, 2), dtype=np.float32) * 0.2
    arr = np.concatenate([left, gap, right], axis=0)
    # compress to 20ms
    adapter = RemixatronAdapter(truncate_enabled=True, truncate_min_ms=100, truncate_max_ms=300, truncate_threshold=0.01, sample_rate=sr, truncate_adaptive_factor=0.0, truncate_mode='compress', truncate_compress_ms=20)
    out = adapter._truncate_silence_runs(arr, min_ms=100, max_ms=300, sample_rate=sr, amplitude_threshold=0.01, crossfade_ms=10)
    # expected reduced length (original length - (gap_len - compress_samples))
    compress_samples = int(0.020 * sr)
    expected_len = arr.shape[0] - (gap_len - compress_samples)
    assert abs(out.shape[0] - expected_len) <= 2, f"Compressed gap did not produce expected length: {out.shape[0]} != {expected_len}"

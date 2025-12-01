import numpy as np
from src.ai.remixatron_adapter import RemixatronAdapter


def make_test_signal(short_gap_len=10, long_gap_len=4000, sr=44100):
    left = np.ones((100, 1), dtype=np.float32) * 0.1
    right = np.ones((100, 1), dtype=np.float32) * 0.1
    lead = np.concatenate([left, right], axis=1)
    gap_short = np.zeros((short_gap_len, 2), dtype=np.float32)
    mid = np.ones((100, 2), dtype=np.float32) * 0.2
    gap_long = np.zeros((long_gap_len, 2), dtype=np.float32)
    tail = np.ones((100, 2), dtype=np.float32) * 0.3
    arr = np.concatenate([lead, gap_short, mid, gap_long, tail], axis=0)
    return arr


def test_heal_short_gap_and_preserve_long_gap():
    adapter = RemixatronAdapter(gap_heal_ms=20)
    sr = 44100
    arr = make_test_signal(short_gap_len=10, long_gap_len=4000, sr=sr)
    # Ensure we have a few small-gap runs
    healed = adapter._heal_short_gaps(arr, gap_heal_ms=20, sample_rate=sr, amplitude_threshold=1e-6)
    # Short gap (10 samples) should be healed — mean magnitude in that area should be > 0
    gap_start = 100
    gap_end = gap_start + 10
    mid_val = np.mean(np.abs(healed[gap_start:gap_end]))
    assert mid_val > 1e-6, "Short gap should be healed"
    # Long gap (400 samples) should remain near-zero
    long_gap_start = 100 + 10 + 100
    long_gap_end = long_gap_start + 400
    long_gap_val = np.mean(np.abs(healed[long_gap_start:long_gap_end]))
    assert long_gap_val < 1e-6, "Long gap should be preserved"


def test_heal_nonzero_low_amplitude_200ms_gap():
    # Create nearly-silent gap (not exact zeros) of 200ms amplitude 0.005
    sr = 44100
    gap_len = int(0.200 * sr)
    arr = np.concatenate([
        np.ones((100, 2), dtype=np.float32) * 0.1,
        np.ones((gap_len, 2), dtype=np.float32) * 0.005,
        np.ones((100, 2), dtype=np.float32) * 0.2
    ], axis=0)
    adapter = RemixatronAdapter(gap_heal_ms=250, gap_heal_threshold=0.01)
    healed = adapter._heal_short_gaps(arr, gap_heal_ms=250, sample_rate=sr, amplitude_threshold=0.01)
    # The gap area should no longer be a flat 0.005 region (healed to interpolation)
    gap_mid_val = np.mean(np.abs(healed[100:100+gap_len]))
    assert gap_mid_val > 0.005, "Gap should be interpolated/healed and amplitude should deviate from 0.005"


def test_heal_texture_preserved_with_periodic_left_ctx():
    # Generate a 440Hz sine wave as left context, gap of 200ms, then right context
    sr = 44100
    t = np.linspace(0, 0.01, int(0.01 * sr), endpoint=False)
    sine = (0.1 * np.sin(2*np.pi*440*t)).reshape(-1,1)
    left = np.concatenate([sine, sine], axis=1)
    gap_len = int(0.200 * sr)
    gap = np.zeros((gap_len, 2), dtype=np.float32)
    right = np.ones((100,2), dtype=np.float32) * 0.2
    arr = np.concatenate([left, gap, right], axis=0)
    adapter = RemixatronAdapter(gap_heal_ms=250, gap_heal_threshold=1e-6)
    healed = adapter._heal_short_gaps(arr, gap_heal_ms=250, sample_rate=sr, amplitude_threshold=1e-6)
    # The healed gap should have significantly more variance than silence
    gap_std = np.std(healed[left.shape[0]:left.shape[0]+gap_len])
    assert gap_std > 1e-3, "Healed gap should contain waveform texture (std > threshold)"

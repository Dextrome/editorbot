import numpy as np
import os, sys
# Ensure workspace project root is on sys.path so imports like src.* work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai.remixatron_blend import blend_beats
from src.audio.ai_blend import AITransitionBlender
from src.ai.remixatron_adapter import soft_clip_signal


def sine(duration_s, sr=44100, freq=440.0, amp=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def stereoify(arr):
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=-1)
    return arr


def test_blend_beats_basic():
    sr = 44100
    blend_dur = 0.2
    n_blend = int(sr * blend_dur)
    prev = sine(1.0, sr)  # 1s
    nextb = sine(1.0, sr, freq=660.0)
    prev_stems = {
        'drums': stereoify(np.tile(prev[:, None], (1, 2)) * 0.8),
        'bass': stereoify(np.tile(prev[:, None], (1, 2)) * 0.6),
        'other': stereoify(np.tile(prev[:, None], (1, 2)) * 0.4),
    }
    next_stems = {
        'drums': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.7),
        'bass': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.5),
        'other': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.3),
    }

    out = blend_beats(prev, nextb, sample_rate=sr, blend_duration=blend_dur, prev_stems=prev_stems, next_stems=next_stems)
    # Expect out length = len(prev) + len(next) - n_blend
    assert out.shape[0] == prev.shape[0] + nextb.shape[0] - n_blend
    # Stereo
    assert out.ndim == 2 and out.shape[1] == 2
    # Values normalized to [-1,1]
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_blend_sections_small_prev():
    # prev shorter than blend window -> concat fallback
    sr = 44100
    blend_dur = 0.2
    n_blend = int(sr * blend_dur)
    prev = sine(0.05, sr)  # shorter than blend
    nextb = sine(1.0, sr)
    prev_stems = {
        'drums': stereoify(np.tile(prev[:, None], (1, 2)) * 0.8),
        'bass': stereoify(np.tile(prev[:, None], (1, 2)) * 0.6),
        'other': stereoify(np.tile(prev[:, None], (1, 2)) * 0.4),
    }
    next_stems = {
        'drums': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.7),
        'bass': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.5),
        'other': stereoify(np.tile(nextb[:, None], (1, 2)) * 0.3),
    }
    out = blend_beats(prev, nextb, sample_rate=sr, blend_duration=blend_dur, prev_stems=prev_stems, next_stems=next_stems)
    # Expect simple concatenation
    assert out.shape[0] == prev.shape[0] + nextb.shape[0]


def test_blend_stem_padding_short_stems():
    sr = 44100
    blend_dur = 0.2
    # prev/next both 1s, but stems shorter than n_blend
    prev = sine(1.0, sr)
    nextb = sine(1.0, sr)
    n_blend = int(sr * blend_dur)
    short = int(n_blend // 2)
    prev_stems = {
        'drums': stereoify(np.tile(prev[:short, None], (1, 2)) * 0.8),
        'bass': stereoify(np.tile(prev[:short, None], (1, 2)) * 0.6),
        'other': stereoify(np.tile(prev[:short, None], (1, 2)) * 0.4),
    }
    next_stems = {
        'drums': stereoify(np.tile(nextb[:short, None], (1, 2)) * 0.7),
        'bass': stereoify(np.tile(nextb[:short, None], (1, 2)) * 0.5),
        'other': stereoify(np.tile(nextb[:short, None], (1, 2)) * 0.3),
    }

    out = blend_beats(prev, nextb, sample_rate=sr, blend_duration=blend_dur, prev_stems=prev_stems, next_stems=next_stems)
    assert out.shape[0] == prev.shape[0] + nextb.shape[0] - n_blend
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_blend_sections_high_amplitude_avoid_clipping():
    sr = 44100
    blend_dur = 0.2
    n_blend = int(sr * blend_dur)
    # high amplitude signals that would clip if added without scaling
    prev = sine(1.0, sr, amp=0.95)
    nextb = -sine(1.0, sr, amp=0.95)  # inverted phase to create large transients when combined
    prev_stems = {
        'drums': np.tile(prev[:, None] * 0.95, (1, 2)),
        'bass': np.tile(prev[:, None] * 0.95, (1, 2)),
        'other': np.tile(prev[:, None] * 0.95, (1, 2)),
    }
    next_stems = {
        'drums': np.tile(nextb[:, None] * 0.95, (1, 2)),
        'bass': np.tile(nextb[:, None] * 0.95, (1, 2)),
        'other': np.tile(nextb[:, None] * 0.95, (1, 2)),
    }
    out = blend_beats(prev, nextb, sample_rate=sr, blend_duration=blend_dur, prev_stems=prev_stems, next_stems=next_stems)
    # The blended region should not exceed our safety threshold (0.99) per blend implementation
    assert np.max(np.abs(out)) <= 0.999 + 1e-6


def test_soft_clip_signal_respects_threshold():
    arr = np.linspace(-2.0, 2.0, 1000)
    clipped = soft_clip_signal(arr, threshold=0.98)
    assert np.max(np.abs(clipped)) <= 0.98 + 1e-6


if __name__ == '__main__':
    test_blend_beats_basic()
    test_blend_sections_small_prev()
    test_blend_stem_padding_short_stems()
    print('All blend simulation tests passed')

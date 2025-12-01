import numpy as np
from unittest import mock

from src.ai.remixatron_adapter import RemixatronAdapter


class FakeJukebox:
    def __init__(self, beats, play_vector):
        self.beats = beats
        self.play_vector = play_vector


def _make_beats_and_play_vector(count=4, beat_length=2048, gap=0):
    # Create several beats and a play_vector that contains large jumps
    beats = []
    play_vector = []
    start = 0
    for i in range(count):
        end = start + beat_length
        buf = np.ones((beat_length, 2), dtype=np.float32) * (i + 1) * 0.1
        beats.append({'buffer': buf, 'start': start, 'end': end})
        start = end + gap
    # Create a play_vector with a large jump between beats to exercise max_jump clamping
    play_vector = [{'beat': 0}, {'beat': 3}, {'beat': 0}, {'beat': 3}]
    return beats, play_vector


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_demucs_separator_called_once_per_run(fake_jukebox_cls, fake_demucs_cls):
    beats, play_vector = _make_beats_and_play_vector(count=4)
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox
    # Fake demucs separate returns empty stems (we don't need stems for this assertion)
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = {'_sr': 44100}

    adapter = RemixatronAdapter(blend_duration=0.1, sample_rate=44100, demucs_device='cpu')
    # Run rearrange once: the separate method should be called exactly once for the track
    adapter.rearrange('fake_path.wav')
    assert fake_demucs.separate.call_count == 1


@mock.patch('src.audio.demucs_wrapper.DemucsSeparator')
@mock.patch('src.ai.remixatron_adapter.InfiniteJukebox')
def test_max_jump_clamping_modifies_play_vector(fake_jukebox_cls, fake_demucs_cls):
    beats, play_vector = _make_beats_and_play_vector(count=4)
    # Create a play vector with a large jump between beat 0 and beat 3
    play_vector = [{'beat': 0}, {'beat': 3}, {'beat': 0}, {'beat': 3}]
    fake_jukebox = FakeJukebox(beats, play_vector)
    fake_jukebox_cls.return_value = fake_jukebox
    fake_demucs = fake_demucs_cls.return_value
    fake_demucs.separate.return_value = {'_sr': 44100}

    # Setup adapter with a max_jump of 1, so jumps should be clamped
    adapter = RemixatronAdapter(blend_duration=0.1, sample_rate=44100, demucs_device='cpu', max_jump=1)
    adapter.rearrange('fake_path.wav')
    # Now the play_vector in fake_jukebox should be modified such that consecutive beat diffs <= max_jump
    diffs = [abs(fake_jukebox.play_vector[i]['beat'] - fake_jukebox.play_vector[i-1]['beat']) for i in range(1, len(fake_jukebox.play_vector))]
    assert all(d <= 1 for d in diffs)

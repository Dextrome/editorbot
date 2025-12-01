import os,sys
sys.path.insert(0, os.path.abspath('.'))
from src.ai.remixatron_blend import blend_beats
from src.audio.ai_blend import AITransitionBlender
import numpy as np

def sine(duration_s, sr=44100, freq=440.0, amp=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

prev = sine(0.05)
nextb = sine(1.0)
print('prev_len', prev.shape[0])
print('next_len', nextb.shape[0])
print('n_blend', int(44100 * 0.2))
prev_stems = {'drums': np.stack([prev,prev],axis=-1), 'bass': np.stack([prev,prev],axis=-1), 'other': np.stack([prev,prev],axis=-1)}
next_stems = {'drums': np.stack([nextb,nextb],axis=-1), 'bass': np.stack([nextb,nextb],axis=-1), 'other': np.stack([nextb,nextb],axis=-1)}
out = blend_beats(prev, nextb, sample_rate=44100, blend_duration=0.2, prev_stems=prev_stems, next_stems=next_stems)
print('out_len', out.shape[0])

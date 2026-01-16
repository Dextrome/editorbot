"""Debug audio slicer scores."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from audio_slicer import AudioSlicer

torch.cuda.empty_cache()

# Load model
print("Loading model...")
slicer = AudioSlicer('F:/editorbot/models/audio_slicer_fast/best.pt')

# Test on a raw audio file
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'

print(f"\nExtracting mel from {input_file}")
mel = slicer.extract_mel(input_file)
print(f"Mel shape: {mel.shape}")

# Score segments and show distribution
print("\nScoring segments...")
scored = slicer.score_segments(mel, segment_frames=128, hop_frames=64)

scores = [s[2] for s in scored]
print(f"\nScore statistics:")
print(f"  Min: {min(scores):.4f}")
print(f"  Max: {max(scores):.4f}")
print(f"  Mean: {np.mean(scores):.4f}")
print(f"  Std: {np.std(scores):.4f}")

# Histogram
bins = np.linspace(0, 1, 11)
hist, _ = np.histogram(scores, bins=bins)
print(f"\nScore distribution:")
for i in range(len(bins)-1):
    bar = '#' * (hist[i] // 10)
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} {bar}")

# Show some examples
print(f"\nLowest scoring segments:")
sorted_by_score = sorted(scored, key=lambda x: x[2])[:10]
for start, end, score in sorted_by_score:
    time_start = start / 86.1  # frames to seconds
    time_end = end / 86.1
    print(f"  {time_start:.1f}-{time_end:.1f}s: score={score:.4f}")

print(f"\nHighest scoring segments:")
sorted_by_score = sorted(scored, key=lambda x: x[2], reverse=True)[:10]
for start, end, score in sorted_by_score:
    time_start = start / 86.1
    time_end = end / 86.1
    print(f"  {time_start:.1f}-{time_end:.1f}s: score={score:.4f}")

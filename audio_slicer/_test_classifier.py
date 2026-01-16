"""Test quality classifier for segment selection."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from audio_slicer.models.scorer import QualityScorer

torch.cuda.empty_cache()

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'F:/editorbot/models/quality_classifier/best.pt'
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_file = 'F:/editorbot/test_output_classifier.wav'

# Mel extraction params (must match training)
sr = 22050
n_fft = 2048
hop_length = 256
n_mels = 128

# Segment params
segment_frames = 64  # ~0.75 seconds
hop_frames = 32  # 50% overlap
threshold = 0.4  # Keep segments with score > threshold

print(f"Loading model from {model_path}")
model = QualityScorer.from_checkpoint(model_path).to(device)
model.eval()

print(f"\nLoading audio from {input_file}")
audio, _ = librosa.load(input_file, sr=sr)
print(f"Audio length: {len(audio) / sr:.1f} seconds")

print("Extracting mel spectrogram...")
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = (mel_db + 80) / 80  # Normalize to 0-1
mel_db = np.clip(mel_db, 0, 1)
mel = mel_db.T  # (T, n_mels)
print(f"Mel shape: {mel.shape}")

# Score all segments
print(f"\nScoring segments (segment_frames={segment_frames}, hop_frames={hop_frames})...")
scores = []
positions = []

with torch.no_grad():
    for start in tqdm(range(0, len(mel) - segment_frames + 1, hop_frames)):
        end = start + segment_frames
        segment = torch.from_numpy(mel[start:end]).float().unsqueeze(0).to(device)
        score = model.score(segment).item()
        scores.append(score)
        positions.append((start, end))

scores = np.array(scores)
print(f"\nScore statistics:")
print(f"  Min: {scores.min():.4f}")
print(f"  Max: {scores.max():.4f}")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")

# Histogram
bins = np.linspace(0, 1, 11)
hist, _ = np.histogram(scores, bins=bins)
print(f"\nScore distribution:")
for i in range(len(bins)-1):
    bar = '#' * (hist[i] // 10)
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} {bar}")

# Select segments above threshold
good_indices = np.where(scores >= threshold)[0]
print(f"\nSegments above threshold {threshold}: {len(good_indices)} / {len(scores)}")

if len(good_indices) == 0:
    print("No segments above threshold! Lowering threshold to select top 30%...")
    threshold = np.percentile(scores, 70)
    good_indices = np.where(scores >= threshold)[0]
    print(f"New threshold: {threshold:.4f}, segments: {len(good_indices)}")

# Merge overlapping/adjacent segments
good_segments = [positions[i] for i in good_indices]
good_segments.sort()

merged = []
for start, end in good_segments:
    if merged and start <= merged[-1][1] + hop_frames:
        # Extend previous segment
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"After merging: {len(merged)} segments")

# Calculate total kept
total_kept_frames = sum(e - s for s, e in merged)
print(f"Total kept: {total_kept_frames} frames ({total_kept_frames / len(mel) * 100:.1f}%)")

# Show segment times
print("\nKept segments:")
for i, (start, end) in enumerate(merged[:10]):
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {i+1}. {t_start:.1f}s - {t_end:.1f}s ({t_end - t_start:.1f}s)")
if len(merged) > 10:
    print(f"  ... and {len(merged) - 10} more")

# Reconstruct audio by keeping only good segments
print("\nReconstructing audio...")
output_audio = []
for start, end in merged:
    audio_start = start * hop_length
    audio_end = min(end * hop_length, len(audio))
    segment_audio = audio[audio_start:audio_end]

    # Apply short fade in/out to avoid clicks
    fade_samples = int(0.01 * sr)  # 10ms fade
    if len(segment_audio) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        segment_audio[:fade_samples] *= fade_in
        segment_audio[-fade_samples:] *= fade_out

    output_audio.append(segment_audio)

output_audio = np.concatenate(output_audio)
print(f"Output length: {len(output_audio) / sr:.1f} seconds")
print(f"Compression: {len(output_audio) / len(audio) * 100:.1f}%")

# Save
sf.write(output_file, output_audio, sr)
print(f"\nSaved to {output_file}")

# Also show which parts were CUT
print("\n--- Lowest scoring segments (would be cut) ---")
sorted_indices = np.argsort(scores)[:10]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: score={scores[idx]:.4f}")

print("\n--- Highest scoring segments (would be kept) ---")
sorted_indices = np.argsort(scores)[-10:][::-1]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: score={scores[idx]:.4f}")

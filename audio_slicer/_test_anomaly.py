"""Test anomaly detector for segment selection.

Key insight:
- Model trained ONLY on edited (good) audio
- Good segments = low reconstruction error
- Bad segments = high reconstruction error (anomalies)
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from audio_slicer.trainers.anomaly_trainer import SimpleAutoencoder

torch.cuda.empty_cache()

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'F:/editorbot/models/anomaly_detector/best.pt'
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_file = 'F:/editorbot/test_output_anomaly.wav'

# Mel params
sr = 22050
n_fft = 2048
hop_length = 256
n_mels = 128

# Segment params
segment_frames = 64
hop_frames = 32

print(f"Loading model from {model_path}")
model = SimpleAutoencoder.from_checkpoint(model_path).to(device)
model.eval()

print(f"\nLoading audio from {input_file}")
audio, _ = librosa.load(input_file, sr=sr)
print(f"Audio length: {len(audio) / sr:.1f} seconds")

print("Extracting mel spectrogram...")
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = (mel_db + 80) / 80
mel_db = np.clip(mel_db, 0, 1)
mel = mel_db.T  # (T, n_mels)
print(f"Mel shape: {mel.shape}")

# Score all segments by reconstruction error
print(f"\nScoring segments by reconstruction error...")
errors = []
positions = []

with torch.no_grad():
    for start in tqdm(range(0, len(mel) - segment_frames + 1, hop_frames)):
        end = start + segment_frames
        segment = torch.from_numpy(mel[start:end]).float().unsqueeze(0).to(device)

        # Reconstruct
        out = model(segment)
        recon = out['reconstruction']

        # MSE error
        error = ((recon - segment) ** 2).mean().item()
        errors.append(error)
        positions.append((start, end))

errors = np.array(errors)
print(f"\nReconstruction error statistics:")
print(f"  Min: {errors.min():.6f}")
print(f"  Max: {errors.max():.6f}")
print(f"  Mean: {errors.mean():.6f}")
print(f"  Std: {errors.std():.6f}")

# Convert errors to scores (low error = high score)
# Use exponential decay: score = exp(-error * scale)
scale = 100  # Adjust based on error distribution
scores = np.exp(-errors * scale)

print(f"\nScore statistics (exp(-error * {scale})):")
print(f"  Min: {scores.min():.4f}")
print(f"  Max: {scores.max():.4f}")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")

# Histogram of errors
print(f"\nError distribution:")
percentiles = [0, 10, 25, 50, 75, 90, 100]
for p in percentiles:
    val = np.percentile(errors, p)
    print(f"  {p}th percentile: {val:.6f}")

# Use percentile threshold
# Keep segments below 75th percentile error (lower error = better)
error_threshold = np.percentile(errors, 40)  # Keep ~40% lowest error (stricter)
print(f"\nError threshold (40th percentile): {error_threshold:.6f}")

good_indices = np.where(errors <= error_threshold)[0]
print(f"Segments below threshold: {len(good_indices)} / {len(errors)}")

# Merge overlapping/adjacent segments
good_segments = [positions[i] for i in good_indices]
good_segments.sort()

merged = []
for start, end in good_segments:
    if merged and start <= merged[-1][1] + hop_frames:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"After merging: {len(merged)} segments")

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

# Reconstruct audio
print("\nReconstructing audio...")
output_audio = []
for start, end in merged:
    audio_start = start * hop_length
    audio_end = min(end * hop_length, len(audio))
    segment_audio = audio[audio_start:audio_end]

    # Short fade
    fade_samples = int(0.01 * sr)
    if len(segment_audio) > 2 * fade_samples:
        segment_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        segment_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    output_audio.append(segment_audio)

output_audio = np.concatenate(output_audio)
print(f"Output length: {len(output_audio) / sr:.1f} seconds")
print(f"Compression: {len(output_audio) / len(audio) * 100:.1f}%")

sf.write(output_file, output_audio, sr)
print(f"\nSaved to {output_file}")

# Show anomalies (high error = bad segments)
print("\n--- ANOMALIES (high reconstruction error - likely bad) ---")
sorted_indices = np.argsort(errors)[-15:][::-1]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: error={errors[idx]:.6f}")

print("\n--- GOOD SEGMENTS (low reconstruction error) ---")
sorted_indices = np.argsort(errors)[:15]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: error={errors[idx]:.6f}")

"""Test large anomaly detector with beat-aligned cutting.

Uses the overnight-trained large model (8000 epochs).
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
model_path = 'F:/editorbot/models/anomaly_detector_large/best.pt'
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_file = 'F:/editorbot/test_output_large.wav'

# Mel params
sr = 22050
n_fft = 2048
hop_length = 256
n_mels = 128

# Segment params - must match model training
segment_frames = 128  # Large model uses 128 frames (~1.5s)
hop_frames = 64  # 50% overlap
error_percentile = 40  # Keep lowest 40% error

# Beat-alignment params
min_segment_seconds = 2.0
crossfade_ms = 50

print(f"Loading model from {model_path}")
model = SimpleAutoencoder.from_checkpoint(model_path).to(device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

print(f"\nLoading audio from {input_file}")
audio, _ = librosa.load(input_file, sr=sr)
print(f"Audio length: {len(audio) / sr:.1f} seconds")

# Detect beats
print("Detecting beats...")
tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units='frames')
tempo = float(np.asarray(tempo).flat[0]) if hasattr(tempo, '__len__') else tempo
beat_samples = librosa.frames_to_samples(beat_frames)
beat_mel_frames = beat_samples // hop_length
print(f"Detected {len(beat_frames)} beats (tempo: {tempo:.1f} BPM)")

print("Extracting mel spectrogram...")
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = (mel_db + 80) / 80
mel_db = np.clip(mel_db, 0, 1)
mel = mel_db.T
print(f"Mel shape: {mel.shape}")

# Score all segments
print(f"\nScoring segments (segment_frames={segment_frames})...")
errors = []
positions = []

with torch.no_grad():
    for start in tqdm(range(0, len(mel) - segment_frames + 1, hop_frames)):
        end = start + segment_frames
        segment = torch.from_numpy(mel[start:end]).float().unsqueeze(0).to(device)
        out = model(segment)
        recon = out['reconstruction']
        error = ((recon - segment) ** 2).mean().item()
        errors.append(error)
        positions.append((start, end))

errors = np.array(errors)
print(f"\nError stats: min={errors.min():.6f}, max={errors.max():.6f}, mean={errors.mean():.6f}, std={errors.std():.6f}")

# Select segments
error_threshold = np.percentile(errors, error_percentile)
print(f"Error threshold ({error_percentile}th percentile): {error_threshold:.6f}")

good_indices = np.where(errors <= error_threshold)[0]
print(f"Segments below threshold: {len(good_indices)} / {len(errors)}")

# Merge overlapping
good_segments = [positions[i] for i in good_indices]
good_segments.sort()

merged = []
for start, end in good_segments:
    if merged and start <= merged[-1][1] + hop_frames:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"After merging: {len(merged)} segments")


def snap_to_beat(frame_idx, beat_mel_frames, direction='nearest'):
    if len(beat_mel_frames) == 0:
        return frame_idx
    diffs = np.abs(beat_mel_frames - frame_idx)
    nearest_idx = np.argmin(diffs)
    if direction == 'nearest':
        return beat_mel_frames[nearest_idx]
    elif direction == 'before':
        candidates = beat_mel_frames[beat_mel_frames <= frame_idx]
        return candidates[-1] if len(candidates) > 0 else beat_mel_frames[0]
    elif direction == 'after':
        candidates = beat_mel_frames[beat_mel_frames >= frame_idx]
        return candidates[0] if len(candidates) > 0 else beat_mel_frames[-1]
    return frame_idx


# Beat-align
print("\nSnapping to beat boundaries...")
beat_aligned = []
min_segment_frames = int(min_segment_seconds * sr / hop_length)

for start, end in merged:
    start_snapped = snap_to_beat(start, beat_mel_frames, 'before')
    end_snapped = snap_to_beat(end, beat_mel_frames, 'after')
    if end_snapped - start_snapped >= min_segment_frames:
        beat_aligned.append((start_snapped, end_snapped))
    elif end - start >= min_segment_frames:
        beat_aligned.append((start, end))

# Re-merge
beat_aligned.sort()
final_segments = []
for start, end in beat_aligned:
    if final_segments and start <= final_segments[-1][1]:
        final_segments[-1] = (final_segments[-1][0], max(final_segments[-1][1], end))
    else:
        final_segments.append((start, end))

final_segments = [(s, e) for s, e in final_segments if e - s >= min_segment_frames]
print(f"After beat alignment + min length: {len(final_segments)} segments")

total_kept_frames = sum(e - s for s, e in final_segments)
print(f"Total kept: {total_kept_frames} frames ({total_kept_frames / len(mel) * 100:.1f}%)")

# Show segments
print("\nKept segments:")
for i, (start, end) in enumerate(final_segments[:15]):
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {i+1}. {t_start:.1f}s - {t_end:.1f}s ({t_end - t_start:.1f}s)")
if len(final_segments) > 15:
    print(f"  ... and {len(final_segments) - 15} more")

# Reconstruct with crossfades
print("\nReconstructing audio...")
crossfade_samples = int(crossfade_ms / 1000 * sr)
output_audio = []

for i, (start, end) in enumerate(final_segments):
    audio_start = start * hop_length
    audio_end = min(end * hop_length, len(audio))
    segment_audio = audio[audio_start:audio_end].copy()

    if i > 0 and len(segment_audio) > crossfade_samples:
        segment_audio[:crossfade_samples] *= np.linspace(0, 1, crossfade_samples)
    if i < len(final_segments) - 1 and len(segment_audio) > crossfade_samples:
        segment_audio[-crossfade_samples:] *= np.linspace(1, 0, crossfade_samples)

    output_audio.append(segment_audio)

output_audio = np.concatenate(output_audio)
print(f"Output length: {len(output_audio) / sr:.1f} seconds")
print(f"Compression: {len(output_audio) / len(audio) * 100:.1f}%")

sf.write(output_file, output_audio, sr)
print(f"\nSaved to {output_file}")

# Show cut segments
print("\n--- ANOMALIES (cut segments with highest error) ---")
sorted_indices = np.argsort(errors)[-10:][::-1]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: error={errors[idx]:.6f}")

print("\n--- BEST SEGMENTS (lowest error) ---")
sorted_indices = np.argsort(errors)[:10]
for idx in sorted_indices:
    start, end = positions[idx]
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    print(f"  {t_start:.1f}s - {t_end:.1f}s: error={errors[idx]:.6f}")

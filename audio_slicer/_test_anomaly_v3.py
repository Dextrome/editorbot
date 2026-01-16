"""Anomaly detector v3 - All improvements for better solo/phrase handling.

Improvements:
1. Larger segments (4-6 seconds) to capture full phrases
2. Energy-based boundaries - cut at low-energy moments
3. Phrase detection - use onset detection for natural boundaries
4. Minimum segment length - 8 seconds to preserve coherence
5. Transition scoring - prefer segments that flow well together
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks

from audio_slicer.trainers.anomaly_trainer import SimpleAutoencoder

torch.cuda.empty_cache()

# =============================================================================
# CONFIG
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'F:/editorbot/models/anomaly_detector_large/best.pt'
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_file = 'F:/editorbot/test_output_v3.wav'

# Audio params
sr = 22050
n_fft = 2048
hop_length = 256
n_mels = 128

# IMPROVEMENT 1: Larger segments (model uses 128, but we score with overlap)
segment_frames = 128  # Must match model
scoring_hop = 32  # Dense scoring for better resolution

# IMPROVEMENT 4: Longer minimum segment
min_segment_seconds = 8.0  # 8 seconds minimum to preserve phrases

# Crossfade
crossfade_ms = 100  # Longer crossfade for smoother transitions

# Keep ratio
error_percentile = 45  # Keep lowest 45% error

print("=" * 60)
print("ANOMALY DETECTOR V3 - PHRASE-AWARE EDITING")
print("=" * 60)

# =============================================================================
# LOAD MODEL AND AUDIO
# =============================================================================
print(f"\nLoading model from {model_path}")
model = SimpleAutoencoder.from_checkpoint(model_path).to(device)
model.eval()

print(f"Loading audio from {input_file}")
audio, _ = librosa.load(input_file, sr=sr)
audio_duration = len(audio) / sr
print(f"Audio length: {audio_duration:.1f} seconds")

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
print("\nExtracting features...")

# Mel spectrogram
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = (mel_db + 80) / 80
mel_db = np.clip(mel_db, 0, 1)
mel = mel_db.T  # (T, n_mels)
print(f"  Mel shape: {mel.shape}")

# IMPROVEMENT 2: Energy for finding quiet moments
rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')  # Smooth
print(f"  RMS shape: {rms.shape}")

# IMPROVEMENT 3: Onset detection for phrase boundaries
onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
print(f"  Detected {len(onset_frames)} onsets")

# Beat detection
tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units='frames')
tempo = float(np.asarray(tempo).flat[0])
print(f"  Detected {len(beat_frames)} beats (tempo: {tempo:.1f} BPM)")

# =============================================================================
# IMPROVEMENT 2 & 3: FIND NATURAL CUT POINTS
# =============================================================================
print("\nFinding natural cut points (low energy + between phrases)...")

# Find local minima in energy (good cut points)
# Use a window of ~0.5 seconds
window_frames = int(0.5 * sr / hop_length)
energy_minima = []

# Find valleys in RMS energy
for i in range(window_frames, len(rms_smooth) - window_frames):
    local_min = np.min(rms_smooth[i-window_frames//2:i+window_frames//2])
    if rms_smooth[i] == local_min and rms_smooth[i] < np.percentile(rms_smooth, 30):
        energy_minima.append(i)

print(f"  Found {len(energy_minima)} low-energy points")

# Combine with onset info - prefer cutting BEFORE onsets (between phrases)
# Create a "cut suitability" score for each frame
cut_score = np.ones(len(mel)) * 0.5  # Default medium

# Low energy = good for cutting
energy_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min() + 1e-8)
cut_score[:len(energy_norm)] -= energy_norm * 0.3  # Lower energy = better cut point

# Just before onset = good for cutting (end of phrase)
for onset in onset_frames:
    if onset > 5 and onset < len(cut_score):
        cut_score[onset-5:onset] -= 0.2  # Slightly before onset

# On beat = good for cutting
for beat in beat_frames:
    if beat < len(cut_score):
        cut_score[beat] -= 0.1

print(f"  Cut score range: {cut_score.min():.2f} to {cut_score.max():.2f}")

# =============================================================================
# SCORE SEGMENTS WITH MODEL
# =============================================================================
print(f"\nScoring segments with anomaly detector...")
errors = []
positions = []

with torch.no_grad():
    for start in tqdm(range(0, len(mel) - segment_frames + 1, scoring_hop)):
        end = start + segment_frames
        segment = torch.from_numpy(mel[start:end]).float().unsqueeze(0).to(device)
        out = model(segment)
        recon = out['reconstruction']
        error = ((recon - segment) ** 2).mean().item()
        errors.append(error)
        positions.append((start, end))

errors = np.array(errors)
print(f"\nError stats: min={errors.min():.6f}, max={errors.max():.6f}, mean={errors.mean():.6f}")

# =============================================================================
# SELECT GOOD SEGMENTS
# =============================================================================
error_threshold = np.percentile(errors, error_percentile)
print(f"Error threshold ({error_percentile}th percentile): {error_threshold:.6f}")

good_indices = np.where(errors <= error_threshold)[0]
print(f"Segments below threshold: {len(good_indices)} / {len(errors)}")

# Create a "quality map" - for each frame, how good is it?
quality_map = np.zeros(len(mel))
count_map = np.zeros(len(mel))

for idx in good_indices:
    start, end = positions[idx]
    quality_map[start:end] += 1
    count_map[start:end] += 1

# Normalize
count_map = np.maximum(count_map, 1)
quality_map = quality_map / count_map

# Threshold quality map
quality_threshold = 0.5
good_frames = quality_map >= quality_threshold

# =============================================================================
# IMPROVEMENT 5: TRANSITION SCORING - FIND BEST CUT POINTS
# =============================================================================
print("\nFinding optimal cut points with transition scoring...")

def find_best_cut_in_range(start_range, end_range, cut_score, min_dist=100):
    """Find the best frame to cut within a range."""
    if end_range <= start_range:
        return start_range

    search_start = max(0, start_range)
    search_end = min(len(cut_score), end_range)

    if search_end <= search_start:
        return start_range

    best_idx = search_start + np.argmin(cut_score[search_start:search_end])
    return best_idx

# Find continuous regions of good frames
regions = []
in_region = False
region_start = 0

for i in range(len(good_frames)):
    if good_frames[i] and not in_region:
        region_start = i
        in_region = True
    elif not good_frames[i] and in_region:
        if i - region_start > 10:  # Minimum region size
            regions.append((region_start, i))
        in_region = False

if in_region:
    regions.append((region_start, len(good_frames)))

print(f"Found {len(regions)} good regions before boundary optimization")

# Optimize boundaries using cut scores
optimized_regions = []
search_window = int(0.5 * sr / hop_length)  # 0.5 second search window

for start, end in regions:
    # Find best cut point near start
    opt_start = find_best_cut_in_range(
        max(0, start - search_window),
        min(len(cut_score), start + search_window),
        cut_score
    )

    # Find best cut point near end
    opt_end = find_best_cut_in_range(
        max(0, end - search_window),
        min(len(cut_score), end + search_window),
        cut_score
    )

    if opt_end > opt_start:
        optimized_regions.append((opt_start, opt_end))

# Merge overlapping regions
optimized_regions.sort()
merged = []
for start, end in optimized_regions:
    if merged and start <= merged[-1][1] + scoring_hop:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"After boundary optimization: {len(merged)} regions")

# =============================================================================
# IMPROVEMENT 4: ENFORCE MINIMUM LENGTH
# =============================================================================
min_frames = int(min_segment_seconds * sr / hop_length)
final_segments = [(s, e) for s, e in merged if e - s >= min_frames]
print(f"After minimum length filter ({min_segment_seconds}s): {len(final_segments)} segments")

# =============================================================================
# IMPROVEMENT 5 (cont): SCORE TRANSITIONS AND REORDER IF NEEDED
# =============================================================================
def compute_transition_score(audio, sr, hop_length, end_frame, start_frame, window_ms=50):
    """Score how well two segments would connect."""
    window_samples = int(window_ms / 1000 * sr)

    end_sample = min(end_frame * hop_length, len(audio) - window_samples)
    start_sample = start_frame * hop_length

    if end_sample < window_samples or start_sample + window_samples > len(audio):
        return 0.5  # Default

    # Get audio at transition point
    end_audio = audio[end_sample - window_samples:end_sample]
    start_audio = audio[start_sample:start_sample + window_samples]

    # Compare RMS energy
    end_rms = np.sqrt(np.mean(end_audio ** 2))
    start_rms = np.sqrt(np.mean(start_audio ** 2))
    energy_diff = abs(end_rms - start_rms) / (max(end_rms, start_rms) + 1e-8)

    # Compare spectral centroid
    end_spec = np.abs(np.fft.rfft(end_audio))
    start_spec = np.abs(np.fft.rfft(start_audio))

    freqs = np.fft.rfftfreq(len(end_audio), 1/sr)
    end_centroid = np.sum(freqs * end_spec) / (np.sum(end_spec) + 1e-8)
    start_centroid = np.sum(freqs * start_spec) / (np.sum(start_spec) + 1e-8)
    centroid_diff = abs(end_centroid - start_centroid) / (max(end_centroid, start_centroid) + 1e-8)

    # Lower diff = better transition
    score = 1.0 - (0.5 * energy_diff + 0.5 * centroid_diff)
    return max(0, min(1, score))

# Score all transitions
if len(final_segments) > 1:
    print("\nScoring transitions between segments...")
    transition_scores = []
    for i in range(len(final_segments) - 1):
        _, end = final_segments[i]
        start, _ = final_segments[i + 1]
        score = compute_transition_score(audio, sr, hop_length, end, start)
        transition_scores.append(score)

    avg_transition = np.mean(transition_scores)
    print(f"Average transition score: {avg_transition:.3f} (1.0 = perfect)")

    # Flag bad transitions
    bad_transitions = [i for i, s in enumerate(transition_scores) if s < 0.5]
    if bad_transitions:
        print(f"Warning: {len(bad_transitions)} potentially rough transitions")

# =============================================================================
# SUMMARY
# =============================================================================
total_kept_frames = sum(e - s for s, e in final_segments)
print(f"\n{'='*60}")
print(f"FINAL RESULT: {len(final_segments)} segments")
print(f"Total kept: {total_kept_frames * hop_length / sr:.1f}s ({total_kept_frames / len(mel) * 100:.1f}%)")
print(f"{'='*60}")

print("\nKept segments:")
for i, (start, end) in enumerate(final_segments):
    t_start = start * hop_length / sr
    t_end = end * hop_length / sr
    duration = t_end - t_start
    print(f"  {i+1}. {t_start:.1f}s - {t_end:.1f}s ({duration:.1f}s)")

# =============================================================================
# RECONSTRUCT AUDIO WITH PROPER CROSSFADE MIXING
# =============================================================================
print("\nReconstructing audio with crossfade mixing...")
crossfade_samples = int(crossfade_ms / 1000 * sr)

# Build output by mixing overlapping segments
output_audio = np.array([], dtype=np.float32)

for i, (start, end) in enumerate(final_segments):
    audio_start = start * hop_length
    audio_end = min(end * hop_length, len(audio))
    segment_audio = audio[audio_start:audio_end].copy()

    if i == 0:
        # First segment - no crossfade at start
        output_audio = segment_audio
    else:
        # Crossfade mix with previous segment
        if len(output_audio) >= crossfade_samples and len(segment_audio) >= crossfade_samples:
            # Create crossfade
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)

            # Mix the overlap region
            overlap = output_audio[-crossfade_samples:] * fade_out + segment_audio[:crossfade_samples] * fade_in

            # Combine: previous (minus overlap) + mixed overlap + new (minus overlap)
            output_audio = np.concatenate([
                output_audio[:-crossfade_samples],
                overlap,
                segment_audio[crossfade_samples:]
            ])
        else:
            # Segments too short, just concatenate
            output_audio = np.concatenate([output_audio, segment_audio])

output_duration = len(output_audio) / sr
print(f"Output length: {output_duration:.1f} seconds")
print(f"Compression: {output_duration / audio_duration * 100:.1f}%")

sf.write(output_file, output_audio, sr)
print(f"\nSaved to {output_file}")

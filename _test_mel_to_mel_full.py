"""Full song mel-to-mel inference with chunked vocoder."""
import sys
sys.path.insert(0, '.')

import os
import torch
import numpy as np
import soundfile as sf
from mel_to_mel_editor import MelToMelPipeline
from shared.audio_utils import denormalize_mel_for_vocoder

torch.cuda.empty_cache()

# Input file
test_input = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_path = 'F:/editorbot/test_output_mel2mel_full.wav'

print(f"Input: {test_input}")
print(f"Output: {output_path}")

# Load pipeline
print("Loading pipeline...")
pipeline = MelToMelPipeline('F:/editorbot/models/mel_to_mel/best.pt')

# Extract and process mel
print("Extracting mel...")
raw_mel = pipeline.extract_mel(test_input)
print(f"Mel shape: {raw_mel.shape} ({raw_mel.shape[0] / 86:.1f} seconds)")

print("Processing through model...")
edited_mel = pipeline.process(raw_mel, chunk_size=512)

# Show delta stats
delta = edited_mel - raw_mel
print(f"\nDelta stats:")
print(f"  Mean: {delta.mean():.4f}, Std: {delta.std():.4f}")
print(f"  Range: [{delta.min():.4f}, {delta.max():.4f}]")

# Chunked vocoder to avoid OOM
print("\nConverting to audio (chunked vocoder)...")

# Load vocoder once
vocoder = pipeline._load_vocoder()

# Process in chunks
chunk_size = 500  # frames per chunk
overlap = 50  # overlap for crossfade
sample_rate = 44100
hop_length = 512  # BigVGAN hop length

# Prepare mel for vocoder
edited_mel = np.clip(edited_mel, 0, 1)
mel_tensor = torch.from_numpy(edited_mel).float()
mel_log = denormalize_mel_for_vocoder(mel_tensor).numpy()

# Process chunks
all_audio = []
total_frames = len(mel_log)

for start in range(0, total_frames, chunk_size - overlap):
    end = min(start + chunk_size, total_frames)
    chunk_mel = mel_log[start:end]

    # Convert chunk
    mel_input = torch.FloatTensor(chunk_mel.T).unsqueeze(0).to(pipeline.device)

    with torch.no_grad():
        chunk_audio = vocoder(mel_input).squeeze().cpu().numpy()

    all_audio.append((start, end, chunk_audio))

    # Clear cache periodically
    if len(all_audio) % 10 == 0:
        torch.cuda.empty_cache()
        print(f"  Processed {end}/{total_frames} frames ({100*end/total_frames:.1f}%)")

print(f"  Processed {total_frames}/{total_frames} frames (100.0%)")

# Stitch audio with crossfade
print("Stitching audio...")
samples_per_frame = hop_length
total_samples = total_frames * samples_per_frame
output_audio = np.zeros(total_samples + sample_rate)  # extra buffer
weights = np.zeros(total_samples + sample_rate)

for start_frame, end_frame, chunk_audio in all_audio:
    start_sample = start_frame * samples_per_frame
    chunk_len = len(chunk_audio)

    # Create weight for crossfade
    w = np.ones(chunk_len)
    fade_samples = overlap * samples_per_frame // 2

    if start_frame > 0:
        fade_len = min(fade_samples, chunk_len // 4)
        w[:fade_len] = np.linspace(0, 1, fade_len)
    if end_frame < total_frames:
        fade_len = min(fade_samples, chunk_len // 4)
        w[-fade_len:] = np.linspace(1, 0, fade_len)

    end_sample = start_sample + chunk_len
    if end_sample > len(output_audio):
        end_sample = len(output_audio)
        chunk_audio = chunk_audio[:end_sample - start_sample]
        w = w[:end_sample - start_sample]

    output_audio[start_sample:end_sample] += chunk_audio * w
    weights[start_sample:end_sample] += w

# Normalize by weights
weights = np.maximum(weights, 1e-8)
output_audio = output_audio / weights

# Trim silence and normalize
output_audio = output_audio[:total_samples]
if np.abs(output_audio).max() > 0:
    output_audio = output_audio / np.abs(output_audio).max() * 0.9

# Save
sf.write(output_path, output_audio, sample_rate)
print(f"\nSaved: {output_path}")
print(f"Duration: {len(output_audio)/sample_rate:.1f} seconds")
print("Done!")

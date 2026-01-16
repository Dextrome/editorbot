"""Test mel-to-mel inference on a sample file."""
import sys
sys.path.insert(0, '.')

import os
import torch
import numpy as np
from mel_to_mel_editor import MelToMelPipeline

# Clear GPU memory
torch.cuda.empty_cache()

# Find a test audio file
test_files = [
    'F:/editorbot/training_data/input',
]

# Find first available raw audio
test_input = None
for folder in test_files:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.wav') or f.endswith('.mp3'):
                test_input = os.path.join(folder, f)
                break
    if test_input:
        break

if not test_input:
    print("No test audio found!")
    sys.exit(1)

print(f"Test input: {test_input}")

# Load model
model_path = 'F:/editorbot/models/mel_to_mel/best.pt'
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

print("Loading pipeline...")
pipeline = MelToMelPipeline(model_path)

# Process only first 30 seconds to avoid OOM
print(f"Extracting mel...")
raw_mel = pipeline.extract_mel(test_input)
print(f"Full mel shape: {raw_mel.shape}")

# Limit to 30 seconds (~1300 frames at 44.1kHz)
max_frames = 1300
if len(raw_mel) > max_frames:
    print(f"Truncating to {max_frames} frames for OOM safety")
    raw_mel = raw_mel[:max_frames]

print(f"Processing {len(raw_mel)} frames through model...")
edited_mel = pipeline.process(raw_mel)

print(f"Converting to audio...")
torch.cuda.empty_cache()  # Clear before vocoder
audio = pipeline.mel_to_audio(edited_mel)

output_path = 'F:/editorbot/test_output_mel2mel.wav'
pipeline.save_audio(audio, output_path)

print(f"Output saved to: {output_path}")
print(f"Input shape: {raw_mel.shape}")
print(f"Output shape: {edited_mel.shape}")

# Show delta stats
delta = edited_mel - raw_mel
print(f"\nModel output stats:")
print(f"  Delta mean: {delta.mean():.4f}")
print(f"  Delta std: {delta.std():.4f}")
print(f"  Delta min: {delta.min():.4f}")
print(f"  Delta max: {delta.max():.4f}")
print("Done!")

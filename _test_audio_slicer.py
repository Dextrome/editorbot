"""Test audio slicer inference."""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer import AudioSlicer

torch.cuda.empty_cache()

# Load model
print("Loading model...")
slicer = AudioSlicer('F:/editorbot/models/audio_slicer_fast/best.pt')

# Test on a raw audio file
input_file = 'F:/editorbot/training_data/input/20250715wartsnall6_raw.wav'
output_file = 'F:/editorbot/test_output_sliced.wav'

print(f"\nInput: {input_file}")
print(f"Output: {output_file}")

# Slice with different thresholds
result = slicer.slice_audio(
    input_file,
    output_file,
    threshold=0.3,  # Lower threshold = keep more
    transform=True,  # Apply raw→edited transformation
)

print(f"\nResults:")
print(f"  Input frames: {result['input_frames']:,}")
print(f"  Output frames: {result['output_frames']:,}")
print(f"  Keep ratio: {result['keep_ratio']*100:.1f}%")
print(f"  Segments kept: {result['n_segments']}")
print("\nDone!")

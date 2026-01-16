"""Debug what the effects are actually doing."""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

# Test the segment finding and effect application directly
from super_editor.models.decoder import (
    find_contiguous_segments,
    apply_fade_in,
    apply_fade_out,
    apply_loop,
    apply_effect,
)

# Create a simple test mel (100 frames, 128 mels)
# Fill with constant value 0.5 for easy debugging
T, M = 100, 128
test_mel = torch.ones(T, M) * 0.5

# Test segment finding
labels = torch.tensor([1]*20 + [3]*30 + [1]*50)  # KEEP, FADE_IN, KEEP
segments = find_contiguous_segments(labels, 3)
print(f"FADE_IN segments in [KEEP*20, FADE_IN*30, KEEP*50]: {segments}")
print(f"Expected: [(20, 50)]")

# Test fade_in on the segment
result = apply_fade_in(test_mel.clone(), 20, 50)
print(f"\nFADE_IN result:")
print(f"  Frame 20 (start): {result[20, 0].item():.4f} (should be ~0, ramp starts)")
print(f"  Frame 35 (middle): {result[35, 0].item():.4f} (should be ~0.25)")
print(f"  Frame 49 (end-1): {result[49, 0].item():.4f} (should be ~0.5)")
print(f"  Frame 50 (after): {result[50, 0].item():.4f} (should be 0.5, unchanged)")

# Test fade_out
result = apply_fade_out(test_mel.clone(), 20, 50)
print(f"\nFADE_OUT result:")
print(f"  Frame 20 (start): {result[20, 0].item():.4f} (should be ~0.5, ramp starts at 1)")
print(f"  Frame 35 (middle): {result[35, 0].item():.4f} (should be ~0.25)")
print(f"  Frame 49 (end-1): {result[49, 0].item():.4f} (should be ~0)")

# Test loop
result = apply_loop(test_mel.clone(), 20, 50)
print(f"\nLOOP result:")
print(f"  Frame 19 (before): {result[19, 0].item():.4f}")
print(f"  Frame 20 (start): {result[20, 0].item():.4f}")
print(f"  Frame 35 (middle): {result[35, 0].item():.4f}")
print(f"  All loop frames same? {torch.allclose(result[20:50, 0], result[20, 0])}")

# Now test with actual audio
print("\n" + "="*60)
print("Testing with actual audio mel spectrogram...")
print("="*60)

from super_editor.inference.full_pipeline import SuperEditorPipeline

pipeline = SuperEditorPipeline(
    recon_model_path='F:/editorbot/models/phase1_fixed/best.pt'
)

test_audio = 'F:/editorbot/training_data/input/20250805wartsnall10_raw.mp3'
raw_mel = pipeline.extract_mel(test_audio).numpy()[:200]  # First 200 frames
T = len(raw_mel)

print(f"\nRaw mel stats: min={raw_mel.min():.4f}, max={raw_mel.max():.4f}, mean={raw_mel.mean():.4f}")

# Test FADE_OUT specifically - should be very obvious
labels = np.ones(T, dtype=np.int64)
labels[50:150] = 4  # FADE_OUT in the middle

result = pipeline.process(raw_mel, edit_labels=labels)
pred_mel = result['pred_mel']

print(f"\nFADE_OUT test (frames 50-150):")
print(f"  Raw mel frame 50 mean: {raw_mel[50].mean():.4f}")
print(f"  Pred mel frame 50 mean: {pred_mel[50].mean():.4f} (should be close to raw, start of fade)")
print(f"  Raw mel frame 100 mean: {raw_mel[100].mean():.4f}")
print(f"  Pred mel frame 100 mean: {pred_mel[100].mean():.4f} (should be ~half of raw)")
print(f"  Raw mel frame 149 mean: {raw_mel[149].mean():.4f}")
print(f"  Pred mel frame 149 mean: {pred_mel[149].mean():.4f} (should be near 0)")

# Check if KEEP regions are preserved
print(f"\nKEEP region check (frame 0):")
print(f"  Raw: {raw_mel[0].mean():.4f}, Pred: {pred_mel[0].mean():.4f}, Same: {np.allclose(raw_mel[0], pred_mel[0])}")

"""Test all hard-coded effects in the updated decoder."""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from super_editor.inference.full_pipeline import SuperEditorPipeline

print('Loading model...', flush=True)
pipeline = SuperEditorPipeline(
    recon_model_path='F:/editorbot/models/phase1_fixed/best.pt'
)

# Use a test file
test_audio = 'F:/editorbot/training_data/input/20250805wartsnall10_raw.mp3'
print(f'Extracting mel from {test_audio}...', flush=True)
raw_mel = pipeline.extract_mel(test_audio).numpy()

# Use first 500 frames (~11 seconds)
T = min(500, len(raw_mel))
raw_mel = raw_mel[:T]
print(f'Using {T} frames (~{T * 512 / 22050:.1f} seconds)', flush=True)

# Test each effect
effects = {
    'CUT': 0,
    'KEEP': 1,
    'LOOP': 2,
    'FADE_IN': 3,
    'FADE_OUT': 4,
    'EFFECT': 5,
    'TRANSITION': 6,
}

for name, label_val in effects.items():
    print(f'\n=== Testing {name} (label={label_val}) ===', flush=True)

    if name == 'KEEP':
        # All KEEP - should be identical to input
        labels = np.ones(T, dtype=np.int64)
    elif name == 'CUT':
        # KEEP with CUT in middle
        labels = np.ones(T, dtype=np.int64)
        labels[100:200] = 0  # 100 frames of silence in middle
    else:
        # KEEP with effect in middle
        labels = np.ones(T, dtype=np.int64)
        labels[100:200] = label_val  # 100 frames of effect in middle

    # Process
    result = pipeline.process(raw_mel, edit_labels=labels)
    pred_mel = result['pred_mel']

    # Stats for the effect region
    effect_region = pred_mel[100:200]
    raw_region = raw_mel[100:200]

    print(f'  Effect region mean: {effect_region.mean():.4f}')
    print(f'  Raw region mean: {raw_region.mean():.4f}')
    print(f'  Difference: {np.abs(effect_region - raw_region).mean():.4f}')

    # Convert to audio
    audio = pipeline.mel_to_audio(pred_mel)
    output_path = f'F:/editorbot/test_effect_{name}.wav'
    pipeline.save_audio(audio, output_path)
    print(f'  Saved: {output_path}', flush=True)

print('\n=== All effects tested! ===')
print('Listen to each file to verify:')
print('  - test_effect_KEEP.wav: Should sound like original')
print('  - test_effect_CUT.wav: Should have silence in middle (~4.6s to ~9.3s)')
print('  - test_effect_LOOP.wav: Should have repeated sound in middle')
print('  - test_effect_FADE_IN.wav: Should fade in during middle section')
print('  - test_effect_FADE_OUT.wav: Should fade out during middle section')
print('  - test_effect_EFFECT.wav: Should sound muffled/filtered in middle')
print('  - test_effect_TRANSITION.wav: Should have partial fade in middle')

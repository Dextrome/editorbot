"""Test full song inference with and without edit labels."""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from super_editor.inference.full_pipeline import SuperEditorPipeline

print('Loading model...', flush=True)
pipeline = SuperEditorPipeline(
    recon_model_path='F:/editorbot/models/phase1_fixed/best.pt'
)

# Use a shorter test file
test_audio = 'F:/editorbot/training_data/input/20250805wartsnall10_raw.mp3'
print(f'Extracting mel from {test_audio}...', flush=True)
raw_mel = pipeline.extract_mel(test_audio).numpy()

# Use full song (no limit)
max_frames = len(raw_mel)  # Full song
T = len(raw_mel)
print(f'Using {T} frames (~{T * 512 / 22050:.1f} seconds)', flush=True)

# Process in chunks
chunk_size = 1800
overlap = 100

def process_full(raw_mel, labels, pipeline):
    """Process full mel in chunks."""
    T = len(raw_mel)
    chunks = []
    for start in range(0, T, chunk_size - overlap):
        end = min(start + chunk_size, T)
        chunks.append((start, end))
        if end >= T:
            break

    output_chunks = []
    for i, (start, end) in enumerate(chunks):
        result = pipeline.process(raw_mel[start:end], edit_labels=labels[start:end])
        output_chunks.append((start, end, result['pred_mel']))
        print(f'  Chunk {i+1}/{len(chunks)}', flush=True)

    # Stitch with crossfade
    full_output = np.zeros_like(raw_mel)
    weight_sum = np.zeros(T)

    for start, end, chunk_output in output_chunks:
        chunk_len = end - start
        weights = np.ones(chunk_len)
        fade_len = min(overlap // 2, chunk_len // 4)
        if start > 0:
            weights[:fade_len] = np.linspace(0, 1, fade_len)
        if end < T:
            weights[-fade_len:] = np.linspace(1, 0, fade_len)
        weights = weights.reshape(-1, 1)
        full_output[start:end] += chunk_output * weights
        weight_sum[start:end] += weights.squeeze()

    weight_sum = np.maximum(weight_sum, 1e-8).reshape(-1, 1)
    return full_output / weight_sum

# === VERSION 1: All KEEP (baseline) ===
print('\n=== Processing all-KEEP version ===', flush=True)
labels_keep = np.ones(T, dtype=np.int64)
output_keep = process_full(raw_mel, labels_keep, pipeline)

print('Converting to audio...', flush=True)
audio_keep = pipeline.mel_to_audio(output_keep)
pipeline.save_audio(audio_keep, 'F:/editorbot/full_song_KEEP.wav')
print(f'Saved: full_song_KEEP.wav ({len(audio_keep)/22050:.1f}s)', flush=True)

# === VERSION 2: With edits ===
print('\n=== Processing edited version ===', flush=True)
labels_edit = np.ones(T, dtype=np.int64)

# Add periodic edits
frames_per_10sec = int(10 * 22050 / 512)
for i in range(0, T, frames_per_10sec):
    edit_type = (i // frames_per_10sec) % 4
    if edit_type == 0:
        labels_edit[i:min(i+50, T)] = 0  # CUT
    elif edit_type == 1:
        labels_edit[i:min(i+80, T)] = 3  # FADE_IN
    elif edit_type == 2:
        labels_edit[i:min(i+80, T)] = 4  # FADE_OUT
    elif edit_type == 3:
        labels_edit[i:min(i+100, T)] = 5  # EFFECT

print(f'Edits: CUT={sum(labels_edit==0)}, KEEP={sum(labels_edit==1)}, FADE_IN={sum(labels_edit==3)}, FADE_OUT={sum(labels_edit==4)}, EFFECT={sum(labels_edit==5)}', flush=True)

output_edit = process_full(raw_mel, labels_edit, pipeline)

print('Converting to audio...', flush=True)
audio_edit = pipeline.mel_to_audio(output_edit)
pipeline.save_audio(audio_edit, 'F:/editorbot/full_song_EDITED.wav')
print(f'Saved: full_song_EDITED.wav ({len(audio_edit)/22050:.1f}s)', flush=True)

print('\n=== Done! ===')
print('Compare:')
print('  - full_song_KEEP.wav (reconstruction, should match original)')
print('  - full_song_EDITED.wav (with cuts, fades, effects)')

"""Test Phase 2 inference - predict edit labels and apply DSP effects."""
import sys
sys.path.insert(0, '.')
import os
import json
import numpy as np
import torch

from super_editor.config import Phase2Config, AudioConfig
from super_editor.models import ActorCritic, DSPEditor
from shared.audio_utils import (
    compute_mel_spectrogram_bigvgan_from_file,
    normalize_mel_for_model,
    denormalize_mel_for_vocoder,
)

print("Loading models...", flush=True)

# Load trained Phase 2 model
checkpoint = torch.load('F:/editorbot/models/phase2_fixed/final.pt', map_location='cuda', weights_only=False)
config = checkpoint.get('config', Phase2Config())

actor_critic = ActorCritic(config).cuda()
actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
actor_critic.eval()

# DSP Editor for applying effects
dsp_editor = DSPEditor().cuda()

# Audio config
audio_config = AudioConfig()

# Test audio
test_audio = 'F:/editorbot/training_data/input/20250805wartsnall10_raw.mp3'
print(f"Extracting mel from {test_audio}...", flush=True)

# Extract mel using BigVGAN-compatible method
mel_log, _ = compute_mel_spectrogram_bigvgan_from_file(
    test_audio, config=audio_config, normalize_volume=True, device='cpu'
)
mel_log = mel_log.T  # (n_mels, T) -> (T, n_mels)
raw_mel = normalize_mel_for_model(mel_log).numpy()  # Normalize to [0, 1]

# Use first 1000 frames (~23 seconds) for testing
T = min(1000, len(raw_mel))
raw_mel = raw_mel[:T]
print(f"Using {T} frames (~{T * 512 / 22050:.1f} seconds)", flush=True)

# Convert to tensor
raw_mel_tensor = torch.from_numpy(raw_mel).float().unsqueeze(0).cuda()  # (1, T, 128)
mask = torch.ones(1, T, dtype=torch.bool, device='cuda')

# Predict edit labels
print("\nPredicting edit labels...", flush=True)
with torch.no_grad():
    actions, log_probs, entropy, values = actor_critic.get_action_and_value(raw_mel_tensor, mask)

pred_labels_raw = actions[0].cpu().numpy()  # (T,)

# Post-process: enforce minimum segment length to avoid rapid switching
def smooth_labels(labels, min_segment_len=20):
    """Smooth labels by enforcing minimum segment length."""
    smoothed = labels.copy()
    T = len(labels)
    i = 0
    while i < T:
        # Find current segment
        current_label = smoothed[i]
        j = i + 1
        while j < T and smoothed[j] == current_label:
            j += 1
        seg_len = j - i

        # If segment too short, extend it or merge with neighbors
        if seg_len < min_segment_len and j < T:
            # Look ahead - what's the next segment?
            next_label = smoothed[j] if j < T else current_label
            # Extend current segment to minimum length
            end = min(i + min_segment_len, T)
            smoothed[i:end] = current_label
            i = end
        else:
            i = j
    return smoothed

pred_labels = smooth_labels(pred_labels_raw, min_segment_len=30)  # ~0.7 sec minimum
print(f"Smoothed labels: {len(np.unique(pred_labels_raw))} unique -> segments of 30+ frames")

# Count label distribution
label_names = ['CUT', 'KEEP', 'LOOP', 'FADE_IN', 'FADE_OUT', 'EFFECT', 'TRANSITION', 'PAD']
print("\nPredicted label distribution:")
for i, name in enumerate(label_names):
    count = (pred_labels == i).sum()
    pct = count / len(pred_labels) * 100
    if count > 0:
        print(f"  {name}: {count} ({pct:.1f}%)")

# Apply DSP effects
print("\nApplying DSP effects...", flush=True)
with torch.no_grad():
    edited_mel = dsp_editor(raw_mel_tensor, actions, mask)

edited_mel = edited_mel[0].cpu().numpy()  # (T, 128)

# Load BigVGAN vocoder
print("Loading BigVGAN vocoder...", flush=True)
bigvgan_path = 'F:/editorbot/vocoder/BigVGAN'
sys.path.insert(0, bigvgan_path)
from bigvgan import BigVGAN
from env import AttrDict

pretrained_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/"
    "snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
)
with open(os.path.join(pretrained_dir, 'config.json')) as f:
    vocoder_config = AttrDict(json.load(f))

vocoder = BigVGAN(vocoder_config)
state_dict = torch.load(os.path.join(pretrained_dir, 'bigvgan_generator.pt'), map_location='cpu')
vocoder.load_state_dict(state_dict.get('generator', state_dict))
vocoder = vocoder.cuda().eval()
vocoder.remove_weight_norm()
print("BigVGAN loaded!")

def mel_to_audio(mel_norm, vocoder):
    """Convert normalized mel [0,1] to audio."""
    mel_tensor = torch.from_numpy(mel_norm).float()
    mel_log = denormalize_mel_for_vocoder(mel_tensor).numpy()  # [0,1] -> [-11.5, 2.5]
    mel_input = torch.FloatTensor(mel_log.T).unsqueeze(0).cuda()  # (1, n_mels, T)
    with torch.no_grad():
        audio = vocoder(mel_input).squeeze().cpu().numpy()
    audio = audio / np.abs(audio).max() * 0.9  # Normalize volume
    return audio

# Convert to audio
print("Converting to audio...", flush=True)
audio_edited = mel_to_audio(edited_mel, vocoder)
import soundfile as sf
sf.write('F:/editorbot/phase2_inference_output.wav', audio_edited, audio_config.sample_rate)
print(f"Saved: phase2_inference_output.wav ({len(audio_edited)/audio_config.sample_rate:.1f}s)", flush=True)

# Also save the original for comparison
audio_original = mel_to_audio(raw_mel, vocoder)
sf.write('F:/editorbot/phase2_inference_original.wav', audio_original, audio_config.sample_rate)
print(f"Saved: phase2_inference_original.wav ({len(audio_original)/audio_config.sample_rate:.1f}s)", flush=True)

# Visualize edit labels as text
print("\nEdit label timeline (first 100 frames):")
timeline = ""
for i in range(min(100, T)):
    label = pred_labels[i]
    if label == 0:
        timeline += "X"  # CUT
    elif label == 1:
        timeline += "."  # KEEP
    elif label == 2:
        timeline += "L"  # LOOP
    elif label == 3:
        timeline += "/"  # FADE_IN
    elif label == 4:
        timeline += "\\"  # FADE_OUT
    elif label == 5:
        timeline += "E"  # EFFECT
    elif label == 6:
        timeline += "T"  # TRANSITION
    else:
        timeline += "?"
print(timeline)
print("Legend: . = KEEP, X = CUT, L = LOOP, / = FADE_IN, \\ = FADE_OUT, E = EFFECT, T = TRANSITION")

print("\n=== Done! ===")
print("Compare:")
print("  - phase2_inference_original.wav (raw audio)")
print("  - phase2_inference_output.wav (with predicted edits)")

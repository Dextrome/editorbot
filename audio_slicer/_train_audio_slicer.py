"""Train FaceSwap-style audio slicer.

Like FaceSwap:
- Raw audio segments → Encoder_shared + Decoder_raw
- Edited audio segments → Encoder_shared + Decoder_edited
- Shared encoder learns content, decoders learn style
"""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer import TrainConfig, Trainer

print("=" * 60)
print("Audio Slicer Training (FaceSwap-style)")
print("=" * 60)

config = TrainConfig()

# Model
config.model.hidden_dims = [64, 128, 256]
config.model.latent_dim = 128
config.model.segment_frames = 128  # ~1.5 seconds

# Training
config.batch_size = 32
config.epochs = 100
config.learning_rate = 1e-3

# Augmentation (like FaceSwap warping)
config.noise_std = 0.05
config.time_mask_ratio = 0.15
config.freq_mask_ratio = 0.15

# Data
config.segments_per_track = 100  # Random segments per track
config.num_workers = 0

print(f"Segment frames: {config.model.segment_frames}")
print(f"Hidden dims: {config.model.hidden_dims}")
print(f"Latent dim: {config.model.latent_dim}")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.epochs}")
print("=" * 60)

torch.backends.cudnn.benchmark = True

trainer = Trainer(
    config=config,
    cache_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/audio_slicer',
)

trainer.train()
print("Training complete!")

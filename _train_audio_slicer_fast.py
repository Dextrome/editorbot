"""Fast training of audio slicer for testing."""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer import TrainConfig, Trainer

print("=" * 60)
print("Audio Slicer Training (FAST VERSION)")
print("=" * 60)

config = TrainConfig()

# Smaller/faster model
config.model.hidden_dims = [32, 64, 128]  # Smaller
config.model.latent_dim = 64  # Smaller
config.model.segment_frames = 64  # Shorter segments (~0.75 seconds)

# Training
config.batch_size = 64  # Larger batches
config.epochs = 30  # Fewer epochs
config.learning_rate = 2e-3  # Higher LR

# Less augmentation for speed
config.noise_std = 0.02
config.time_mask_ratio = 0.1
config.freq_mask_ratio = 0.1

# Fewer segments per track
config.segments_per_track = 50
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
    cache_dir='F:/editorbot/training_data/super_editor_cache',
    save_dir='F:/editorbot/models/audio_slicer_fast',
)

trainer.train()
print("Training complete!")

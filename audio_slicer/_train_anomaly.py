"""Train anomaly detector (autoencoder on edited audio only).

The idea:
- Train autoencoder only on EDITED (good) audio
- Good segments will have low reconstruction error
- Bad segments (that were cut) will have high reconstruction error
"""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer.config import TrainConfig, ModelConfig
from audio_slicer.trainers.anomaly_trainer import AnomalyTrainer

torch.cuda.empty_cache()

# Configure
config = TrainConfig()

# Model
config.model.hidden_dims = [32, 64, 128]
config.model.latent_dim = 64
config.model.segment_frames = 64

# Training
config.batch_size = 64
config.epochs = 100  # More epochs since simpler task
config.learning_rate = 1e-3
config.segments_per_track = 200  # More samples per track

print("=" * 60)
print("Anomaly Detector Training (Edited Only)")
print("=" * 60)
print(f"Segment frames: {config.model.segment_frames}")
print(f"Hidden dims: {config.model.hidden_dims}")
print(f"Latent dim: {config.model.latent_dim}")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.epochs}")
print("=" * 60)

trainer = AnomalyTrainer(
    config=config,
    cache_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/anomaly_detector',
)

trainer.train()

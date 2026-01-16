"""Train larger anomaly detector for longer.

Improvements:
- Larger model: [64, 128, 256] with latent_dim=128
- Longer segments: 128 frames (~1.5s) for better phrase context
- More epochs: 300
- More samples per track
"""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer.config import TrainConfig, ModelConfig
from audio_slicer.trainers.anomaly_trainer import AnomalyTrainer

torch.cuda.empty_cache()

# Configure larger model
config = TrainConfig()

# Larger model
config.model.hidden_dims = [64, 128, 256]
config.model.latent_dim = 128
config.model.segment_frames = 128  # ~1.5 seconds for better phrase context

# Training - overnight (~8 hours)
config.batch_size = 32  # Smaller batch for larger model
config.epochs = 8000  # ~8 hours at ~3.5s/epoch
config.learning_rate = 5e-4  # Slightly lower LR for stability
config.segments_per_track = 300  # More samples

# Save checkpoints
config.save_every = 500  # Every ~30 min

print("=" * 60)
print("Anomaly Detector Training (LARGE MODEL - OVERNIGHT)")
print("=" * 60)
print(f"Segment frames: {config.model.segment_frames} (~{config.model.segment_frames * 256 / 22050:.1f}s)")
print(f"Hidden dims: {config.model.hidden_dims}")
print(f"Latent dim: {config.model.latent_dim}")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.epochs}")
print(f"Learning rate: {config.learning_rate}")
print("=" * 60)

trainer = AnomalyTrainer(
    config=config,
    cache_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/anomaly_detector_large',
)

trainer.train()

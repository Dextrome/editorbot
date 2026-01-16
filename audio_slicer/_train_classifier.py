"""Train binary quality classifier.

Simple approach:
- edited audio segments = good (label 1)
- raw audio segments = bad (label 0)

Then use classifier to score segments and keep high-scoring ones.
"""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer.config import TrainConfig, ModelConfig
from audio_slicer.trainers import ClassifierTrainer

torch.cuda.empty_cache()

# Configure
config = TrainConfig()

# Smaller model for faster training
config.model.hidden_dims = [32, 64, 128]
config.model.latent_dim = 64
config.model.segment_frames = 64  # ~0.75 seconds
config.model.projection_dim = 32

# Training params
config.batch_size = 64
config.epochs = 50
config.learning_rate = 1e-3

# Data params
config.segments_per_track = 100

print("=" * 60)
print("Quality Classifier Training")
print("=" * 60)
print(f"Segment frames: {config.model.segment_frames}")
print(f"Hidden dims: {config.model.hidden_dims}")
print(f"Latent dim: {config.model.latent_dim}")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.epochs}")
print("=" * 60)

# Train
trainer = ClassifierTrainer(
    config=config,
    cache_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/quality_classifier',
)

trainer.train()

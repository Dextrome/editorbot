"""Train anomaly detector - 500 epochs (quick iteration)."""
import sys
sys.path.insert(0, '.')

import torch
from audio_slicer.config import TrainConfig, ModelConfig
from audio_slicer.trainers.anomaly_trainer import AnomalyTrainer

torch.cuda.empty_cache()

config = TrainConfig()

# Same large model architecture
config.model.hidden_dims = [64, 128, 256]
config.model.latent_dim = 128
config.model.segment_frames = 128  # ~1.5 seconds

# Training - 500 epochs
config.batch_size = 32
config.epochs = 500
config.learning_rate = 5e-4
config.segments_per_track = 300

config.save_every = 100

print("=" * 60)
print("Anomaly Detector Training (500 epochs)")
print("=" * 60)
print(f"Segment frames: {config.model.segment_frames}")
print(f"Hidden dims: {config.model.hidden_dims}")
print(f"Latent dim: {config.model.latent_dim}")
print(f"Epochs: {config.epochs}")
print("=" * 60)

trainer = AnomalyTrainer(
    config=config,
    cache_dir='F:/editorbot/training_data/super_editor_cache',
    save_dir='F:/editorbot/models/anomaly_detector_500',
)

trainer.train()

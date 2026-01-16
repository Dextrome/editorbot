"""Train mel-to-mel editor.

Direct transformation learning - no labels, no DSP, just learns from data.
"""
import sys
sys.path.insert(0, '.')

import torch
from mel_to_mel_editor import TrainConfig, Trainer

print("=" * 60)
print("Mel-to-Mel Editor Training")
print("Direct audio transformation - learns from paired data")
print("=" * 60)

config = TrainConfig()

# Training settings
config.num_workers = 0  # Windows compatibility
config.batch_size = 4  # Small due to large model
config.epochs = 50
config.max_seq_len = 512  # ~12 seconds per sample

# Model - use smaller for initial test
config.model.encoder_channels = [32, 64, 128, 256]
config.model.bottleneck_channels = 256
config.model.decoder_channels = [256, 128, 64, 32]
config.model.use_residual = True  # Learn delta, not full output
config.model.use_attention = True

# Losses
config.loss.l1_weight = 1.0
config.loss.stft_weight = 0.5
config.loss.preservation_weight = 0.5

# Learning rate
config.learning_rate = 1e-4
config.warmup_epochs = 3

print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Max seq len: {config.max_seq_len}")
print(f"Learning rate: {config.learning_rate}")
print(f"Model: U-Net with {config.model.encoder_channels}")
print(f"Residual learning: {config.model.use_residual}")
print("=" * 60)

# Enable optimizations
torch.backends.cudnn.benchmark = True

trainer = Trainer(
    config=config,
    cache_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/mel_to_mel',
)

trainer.train()
print("Training complete!")

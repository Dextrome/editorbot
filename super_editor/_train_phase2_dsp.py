"""Train Phase 2: Edit Label Prediction using DSP Editor.

Quick test training with speed optimizations.
"""
import sys
sys.path.insert(0, '.')

import torch
from super_editor.config import Phase2Config
from super_editor.trainers import Phase2Trainer

print("=" * 60)
print("Phase 2: Edit Label Prediction (PPO)")
print("Using DSP Editor (no neural network, instant effects)")
print("=" * 60)

config = Phase2Config()

# Speed optimizations
config.num_workers = 0  # Windows compatibility
config.batch_size = 32  # Larger batches for GPU efficiency
config.n_epochs_per_update = 2  # Fewer PPO epochs (faster)
config.rollout_steps = 4  # Smaller rollout buffer (faster updates)

# Quick test settings
config.total_epochs = 20  # Quick test
config.save_interval = 10

# Shorter sequences for speed
config.use_curriculum = True
config.curriculum_initial_seq_len = 256  # Start very short
config.curriculum_final_seq_len = 512  # End shorter too

# PPO settings
config.learning_rate = 1e-3  # Higher LR for faster learning
config.clip_ratio = 0.2
config.entropy_coeff = 0.1  # Higher entropy for exploration
config.value_coeff = 0.5
config.target_kl = 0.03

# Reward weights
config.reconstruction_reward_weight = 1.0
config.label_accuracy_reward_weight = 5.0
config.duration_match_reward_weight = 2.0
config.smoothness_penalty_weight = 0.3

print(f"Total epochs: {config.total_epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"Entropy coeff: {config.entropy_coeff}")
print(f"Sequence length: {config.curriculum_initial_seq_len} -> {config.curriculum_final_seq_len}")
print("=" * 60)

# Enable torch optimizations
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

trainer = Phase2Trainer(
    config=config,
    data_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/phase2_dsp',
)

trainer.train()
print("Phase 2 training complete!")

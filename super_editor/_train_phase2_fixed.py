"""Train Phase 2 with fixes to prevent policy collapse.

Key fixes:
1. Higher entropy coefficient that doesn't decay
2. Diversity bonus - reward for using multiple label types
3. Penalize uniform predictions (all same label)
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from super_editor.config import Phase2Config
from super_editor.trainers.phase2_trainer import Phase2Trainer, RewardComputer

print("=" * 60)
print("Phase 2: Edit Label Prediction (PPO)")
print("WITH FIXES: Anti-collapse measures")
print("=" * 60)

config = Phase2Config()

# Basic settings
config.num_workers = 0  # Windows compatibility
config.batch_size = 16  # Smaller for more updates
config.n_epochs_per_update = 3
config.rollout_steps = 4

# Training length
config.total_epochs = 50
config.save_interval = 25

# Sequence length
config.use_curriculum = True
config.curriculum_initial_seq_len = 256
config.curriculum_final_seq_len = 512

# PPO settings - ANTI-COLLAPSE
config.learning_rate = 3e-4
config.clip_ratio = 0.2
config.value_coeff = 0.5
config.target_kl = 0.05  # Allow more divergence

# CRITICAL: High entropy that DOESN'T decay
config.entropy_coeff = 0.2  # Much higher - force exploration
config.entropy_coeff_decay = False  # DON'T decay entropy
config.entropy_coeff_min = 0.2  # Keep high

# Reward weights - reduce label accuracy to allow exploration
config.reconstruction_reward_weight = 1.0
config.label_accuracy_reward_weight = 2.0  # Lower - don't overfit to GT labels
config.duration_match_reward_weight = 1.0
config.smoothness_penalty_weight = 0.1  # Lower - allow more variety

print(f"Total epochs: {config.total_epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"Entropy coeff: {config.entropy_coeff} (NO DECAY)")
print(f"Sequence length: {config.curriculum_initial_seq_len} -> {config.curriculum_final_seq_len}")
print("=" * 60)

# Monkey-patch the reward computer to add diversity bonus
original_compute_reward = RewardComputer.compute_reward

def compute_reward_with_diversity(self, raw_mel, pred_labels, target_mel, target_labels, mask=None):
    """Compute reward with added diversity bonus."""
    total_reward, components = original_compute_reward(
        self, raw_mel, pred_labels, target_mel, target_labels, mask
    )

    B, T = pred_labels.shape

    # Diversity bonus: reward using multiple label types
    # Count unique labels per batch item
    diversity_bonus = torch.zeros(B, device=pred_labels.device)
    uniformity_penalty = torch.zeros(B, device=pred_labels.device)

    for b in range(B):
        labels = pred_labels[b]
        unique_labels = torch.unique(labels)
        n_unique = len(unique_labels)

        # Bonus for using more labels (max ~+2 for using 4+ labels)
        diversity_bonus[b] = min(n_unique - 1, 3) * 0.5

        # Penalty for using same label >90% of time
        label_counts = torch.bincount(labels, minlength=8).float()
        max_ratio = label_counts.max() / T
        if max_ratio > 0.9:
            uniformity_penalty[b] = -(max_ratio - 0.5) * 5.0  # Strong penalty

    total_reward = total_reward + diversity_bonus + uniformity_penalty
    components['diversity_bonus'] = diversity_bonus
    components['uniformity_penalty'] = uniformity_penalty

    return total_reward, components

RewardComputer.compute_reward = compute_reward_with_diversity
print("Patched RewardComputer with diversity bonus")

# Enable torch optimizations
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

trainer = Phase2Trainer(
    config=config,
    data_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/phase2_fixed',
)

trainer.train()
print("Phase 2 training complete!")

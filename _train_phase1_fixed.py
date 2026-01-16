"""Train Phase 1 model with critical fixes applied:
1. Decoder output clamped to [0, 1]
2. STFT loss disabled (was incorrectly applied to mel values)
3. Gating disabled (use simple residual instead)
"""

import torch
import sys
sys.path.insert(0, '.')

from super_editor.config import Phase1Config
from super_editor.trainers import Phase1Trainer

# Configure training with optimized parameters
config = Phase1Config()

# Windows compatibility
config.num_workers = 0

# Training schedule
config.epochs = 100
config.batch_size = 4  # Conservative for stability
config.learning_rate = 1e-4
config.warmup_steps = 500
config.gradient_clip = 1.0

# Loss weights (STFT is now 0.0 by default after fix)
config.l1_weight = 1.5  # Increased since STFT is gone
config.mse_weight = 0.0
config.stft_weight = 0.0  # Disabled - was computing STFT on mel (wrong!)
config.consistency_weight = 0.5

# Label-enforcing losses (critical for model to use edit labels)
config.label_conditioned_weight = 3.0  # Stronger supervision
config.label_contrastive_weight = 0.5

print("=" * 60)
print("Phase 1 Training - FIXED VERSION")
print("=" * 60)
print("Fixes applied:")
print("  - Decoder output clamped to [0, 1]")
print("  - STFT loss disabled (was incorrectly applied to mel)")
print("  - Gating disabled (simple residual instead)")
print("=" * 60)
print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"L1 weight: {config.l1_weight}")
print(f"STFT weight: {config.stft_weight}")
print(f"Label conditioned weight: {config.label_conditioned_weight}")
print(f"Label contrastive weight: {config.label_contrastive_weight}")
print("=" * 60)

trainer = Phase1Trainer(
    config=config,
    data_dir='F:/editorbot/training_data/super_editor_cache',
    save_dir='F:/editorbot/models/phase1_fixed',
    resume_from=None,  # Fresh start with fixes
)

trainer.train()
print("Training complete!")

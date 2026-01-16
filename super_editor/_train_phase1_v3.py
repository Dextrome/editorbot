
import torch
import sys
sys.path.insert(0, '.')

from super_editor.config import Phase1Config
from super_editor.trainers import Phase1Trainer

# Configure training
config = Phase1Config()
config.num_workers = 0  # Avoid Windows multiprocessing issues
config.epochs = 50
config.batch_size = 4  # Smaller batch for stability

print("=" * 60)
print("Phase 1 Training with Label-Aware Decoder")
print("=" * 60)
print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Label conditioned weight: {config.label_conditioned_weight}")
print(f"Label contrastive weight: {config.label_contrastive_weight}")
print("=" * 60)

trainer = Phase1Trainer(
    config=config,
    data_dir='F:/editorbot/cache',
    save_dir='F:/editorbot/models/phase1_v3_label_aware',
    resume_from=None,  # Fresh start with new architecture
)

trainer.train()
print("Training complete!")

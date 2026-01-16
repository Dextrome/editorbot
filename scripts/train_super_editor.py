"""Train super_editor model: Phase 1 until loss <= 0.1, then Phase 2 for ~30k steps."""

import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from super_editor.config import Phase1Config, Phase2Config
from super_editor.trainers.phase1_trainer import Phase1Trainer
from super_editor.trainers.phase2_trainer import train_phase2


def train_phase1_until_target(
    data_dir: str = "training_data/super_editor_cache",
    save_dir: str = "models/super_editor_v2",
    target_loss: float = 0.1,
    max_epochs: int = 500,
    resume_from: str = None,
):
    """Train Phase 1 until target loss is reached."""

    print("=" * 60)
    print("PHASE 1: Supervised Reconstruction")
    print(f"Target loss: <= {target_loss}")
    print(f"Max epochs: {max_epochs}")
    print("=" * 60)

    # Config with more epochs for longer training
    config = Phase1Config(
        encoder_dim=512,
        decoder_dim=512,
        n_encoder_layers=6,
        n_decoder_layers=3,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.2,  # Higher dropout for regularization
        learning_rate=1e-4,
        weight_decay=0.05,  # Stronger weight decay
        batch_size=8,
        epochs=max_epochs,
        warmup_steps=50,  # Quick warmup for small dataset
        lr_decay_steps=10000,  # Slower decay for longer training
        l1_weight=1.0,
        stft_weight=1.0,
        consistency_weight=0.5,
        log_interval=50,
        save_interval=500,
        num_workers=0,  # Windows compatibility
    )

    phase1_save_dir = Path(save_dir) / "phase1"
    phase1_save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Phase1Trainer(
        config=config,
        data_dir=data_dir,
        save_dir=str(phase1_save_dir),
        resume_from=resume_from,
    )

    # Custom training loop with early stopping on target loss
    from super_editor.data import create_dataloader

    train_loader = create_dataloader(
        str(data_dir), config, split='train', shuffle=True
    )
    val_loader = create_dataloader(
        str(data_dir), config, split='val', shuffle=False
    )

    print(f"Training: {len(train_loader.dataset)} samples")
    print(f"Validation: {len(val_loader.dataset)} samples")

    target_reached = False

    for epoch in range(trainer.epoch, max_epochs):
        trainer.epoch = epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"{'='*60}")

        # Train epoch
        train_metrics = trainer._train_epoch(train_loader)

        # Validation
        val_metrics = trainer._validate(val_loader)

        # Log epoch metrics
        trainer._log_epoch(train_metrics, val_metrics)

        # Check if target reached
        current_loss = val_metrics['l1']
        print(f"\nCurrent L1 loss: {current_loss:.4f} (target: {target_loss})")

        # Save checkpoint
        if val_metrics['total'] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics['total']
            trainer._save_checkpoint('best.pt')

        if (epoch + 1) % 10 == 0:
            trainer._save_checkpoint(f'epoch_{epoch + 1}.pt')

        # Check for early stopping on target
        if current_loss <= target_loss:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED! L1 loss = {current_loss:.4f} <= {target_loss}")
            print(f"{'='*60}")
            trainer._save_checkpoint('target_reached.pt')
            target_reached = True
            break

    trainer._save_checkpoint('final.pt')
    trainer.writer.close()

    best_checkpoint = phase1_save_dir / "best.pt"
    print(f"\nPhase 1 complete. Best model: {best_checkpoint}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    return str(best_checkpoint), target_reached


def train_phase2_steps(
    phase1_checkpoint: str,
    data_dir: str = "training_data/super_editor_cache",
    save_dir: str = "models/super_editor_v2",
    total_steps: int = 30000,
):
    """Train Phase 2 for specified number of steps."""

    print("\n" + "=" * 60)
    print("PHASE 2: RL Edit Prediction")
    print(f"Target steps: {total_steps}")
    print(f"Phase 1 model: {phase1_checkpoint}")
    print("=" * 60)

    # Calculate epochs needed for target steps
    # Assuming ~100 steps per epoch with batch_size=32
    # Adjust based on actual dataset size
    estimated_epochs = total_steps // 100 + 100  # Add buffer

    config = Phase2Config(
        predictor_dim=256,
        n_predictor_layers=4,
        n_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.02,
        entropy_coeff=0.02,
        entropy_coeff_decay=True,
        entropy_coeff_min=0.005,
        value_coeff=0.5,
        max_grad_norm=1.0,
        batch_size=32,
        n_epochs_per_update=4,
        rollout_steps=128,
        total_epochs=estimated_epochs,
        use_curriculum=True,
        curriculum_initial_seq_len=512,
        curriculum_final_seq_len=2048,
        curriculum_warmup_epochs=100,
        use_observation_normalization=True,
        use_auxiliary_tasks=True,
        save_interval=100,
        log_interval=10,
        num_workers=0,  # Windows compatibility
    )

    phase2_save_dir = Path(save_dir) / "phase2"

    # Custom training with step limit
    from super_editor.trainers.phase2_trainer import Phase2Trainer

    trainer = Phase2Trainer(
        config=config,
        recon_model_path=phase1_checkpoint,
        data_dir=data_dir,
        save_dir=str(phase2_save_dir),
    )

    # Track actual steps
    print(f"Training until {total_steps} steps...")

    # Use the built-in train method but we'll monitor steps
    # The trainer.global_step tracks progress
    trainer.train()

    print(f"\nPhase 2 complete. Final step: {trainer.global_step}")
    return str(phase2_save_dir / "final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training_data/super_editor_cache")
    parser.add_argument("--save-dir", default="models/super_editor_v2")
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--phase2-steps", type=int, default=30000)
    parser.add_argument("--resume-phase1", type=str, default=None)
    parser.add_argument("--skip-phase1", action="store_true", help="Skip to Phase 2")
    parser.add_argument("--phase1-checkpoint", type=str, default=None,
                        help="Phase 1 checkpoint for Phase 2 (required if --skip-phase1)")
    args = parser.parse_args()

    start_time = time.time()

    if args.skip_phase1:
        if not args.phase1_checkpoint:
            raise ValueError("--phase1-checkpoint required when --skip-phase1")
        phase1_checkpoint = args.phase1_checkpoint
    else:
        # Phase 1
        phase1_checkpoint, target_reached = train_phase1_until_target(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            target_loss=args.target_loss,
            max_epochs=args.max_epochs,
            resume_from=args.resume_phase1,
        )

    # Phase 2
    phase2_checkpoint = train_phase2_steps(
        phase1_checkpoint=phase1_checkpoint,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        total_steps=args.phase2_steps,
    )

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Total time: {hours}h {minutes}m")
    print(f"Phase 1 model: {phase1_checkpoint}")
    print(f"Phase 2 model: {phase2_checkpoint}")
    print("=" * 60)

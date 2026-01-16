"""Phase 1 Trainer: Supervised reconstruction training.

Trains the autoencoder to reconstruct edited audio from (raw_mel, edit_labels).
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Phase1Config
from ..models import ReconstructionModel
from ..losses import Phase1Loss
from ..data import PairedMelDataset, collate_fn, create_dataloader


class Phase1Trainer:
    """Trainer for Phase 1 supervised reconstruction.

    Trains ReconstructionModel to produce edited mel spectrograms
    from (raw_mel, edit_labels) inputs.
    """

    def __init__(
        self,
        config: Phase1Config,
        data_dir: str,
        save_dir: str,
        resume_from: Optional[str] = None,
    ):
        """
        Args:
            config: Phase1Config with training hyperparameters
            data_dir: Directory containing preprocessed data
            save_dir: Directory to save checkpoints and logs
            resume_from: Optional checkpoint path to resume from
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Enable cudnn benchmark for consistent input sizes
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Model
        self.model = ReconstructionModel(config).to(self.device)
        print(f"Model parameters: {self.model.get_num_params():,}")

        # Loss function
        self.loss_fn = Phase1Loss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        self.use_amp = config.use_mixed_precision

        # Logging
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.config.warmup_steps) / (
                    self.config.lr_decay_steps - self.config.warmup_steps
                )
                return max(
                    self.config.min_lr / self.config.learning_rate,
                    0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
                )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self, num_epochs: Optional[int] = None):
        """Run training loop.

        Args:
            num_epochs: Number of epochs (uses config.epochs if None)
        """
        if num_epochs is None:
            num_epochs = self.config.epochs

        # Create data loaders
        train_loader = create_dataloader(
            str(self.data_dir), self.config, split='train', shuffle=True
        )
        val_loader = create_dataloader(
            str(self.data_dir), self.config, split='val', shuffle=False
        )

        print(f"Training: {len(train_loader.dataset)} samples")
        print(f"Validation: {len(val_loader.dataset)} samples")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train epoch
            train_metrics = self._train_epoch(train_loader)

            # Validation
            val_metrics = self._validate(val_loader)

            # Log epoch metrics
            self._log_epoch(train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self._save_checkpoint('best.pt')

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt')

        self._save_checkpoint('final.pt')
        self.writer.close()

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = {
            'total': 0, 'l1': 0, 'mse': 0, 'stft': 0, 'consistency': 0,
            'cut_loss': 0, 'keep_loss': 0, 'contrastive': 0
        }
        n_batches = 0

        pbar = tqdm(loader, desc='Training')
        for batch in pbar:
            # Move to device
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            edit_labels = batch['edit_labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                pred_mel = self.model(raw_mel, edit_labels, mask)
                losses = self.loss_fn(pred_mel, edit_mel, raw_mel, edit_labels, mask)

            # Backward pass
            self.scaler.scale(losses['total']).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] += v.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'l1': f"{losses['l1'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Step logging
            if self.global_step % self.config.log_interval == 0:
                self._log_step(losses)

            self.global_step += 1

        # Average losses
        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_losses = {
            'total': 0, 'l1': 0, 'mse': 0, 'stft': 0, 'consistency': 0,
            'cut_loss': 0, 'keep_loss': 0, 'contrastive': 0
        }
        n_batches = 0

        for batch in tqdm(loader, desc='Validation'):
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            edit_labels = batch['edit_labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            with autocast(enabled=self.use_amp):
                pred_mel = self.model(raw_mel, edit_labels, mask)
                losses = self.loss_fn(pred_mel, edit_mel, raw_mel, edit_labels, mask)

            for k, v in losses.items():
                total_losses[k] += v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def _log_step(self, losses: Dict[str, torch.Tensor]):
        """Log metrics for current step."""
        for k, v in losses.items():
            self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        print(f"\nTrain - Loss: {train_metrics['total']:.4f}, "
              f"L1: {train_metrics['l1']:.4f}, STFT: {train_metrics['stft']:.4f}")
        print(f"Val   - Loss: {val_metrics['total']:.4f}, "
              f"L1: {val_metrics['l1']:.4f}, STFT: {val_metrics['stft']:.4f}")

        for k, v in train_metrics.items():
            self.writer.add_scalar(f'epoch/train_{k}', v, self.epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'epoch/val_{k}', v, self.epoch)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.save_dir / filename
        self.model.save_checkpoint(
            str(path),
            optimizer=self.optimizer,
            epoch=self.epoch,
            loss=self.best_val_loss,
            global_step=self.global_step,
            scheduler_state_dict=self.scheduler.state_dict(),
            scaler_state_dict=self.scaler.state_dict(),
        )
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('loss', float('inf'))

        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    @torch.no_grad()
    def visualize_samples(self, loader: DataLoader, n_samples: int = 4):
        """Generate and log sample reconstructions."""
        self.model.eval()

        batch = next(iter(loader))
        raw_mel = batch['raw_mel'][:n_samples].to(self.device)
        edit_mel = batch['edit_mel'][:n_samples].to(self.device)
        edit_labels = batch['edit_labels'][:n_samples].to(self.device)
        mask = batch['mask'][:n_samples].to(self.device)

        with autocast(enabled=self.use_amp):
            pred_mel = self.model(raw_mel, edit_labels, mask)

        # Log mel spectrograms as images
        for i in range(n_samples):
            length = batch['lengths'][i].item()

            # Create comparison figure
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))

            axes[0].imshow(raw_mel[i, :length].cpu().T, aspect='auto', origin='lower')
            axes[0].set_title('Raw Input')

            axes[1].imshow(edit_mel[i, :length].cpu().T, aspect='auto', origin='lower')
            axes[1].set_title('Target (Edited)')

            axes[2].imshow(pred_mel[i, :length].cpu().T, aspect='auto', origin='lower')
            axes[2].set_title('Predicted')

            plt.tight_layout()
            self.writer.add_figure(f'samples/sample_{i}', fig, self.global_step)
            plt.close()


def train_phase1(
    data_dir: str,
    save_dir: str,
    config: Optional[Phase1Config] = None,
    resume_from: Optional[str] = None,
):
    """Convenience function to train Phase 1 model.

    Args:
        data_dir: Directory with preprocessed data
        save_dir: Directory for checkpoints
        config: Optional config (uses defaults if None)
        resume_from: Optional checkpoint to resume from
    """
    if config is None:
        config = Phase1Config()

    trainer = Phase1Trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        resume_from=resume_from,
    )

    trainer.train()

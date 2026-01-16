"""Trainer for mel-to-mel model."""

import os
from pathlib import Path
from typing import Optional, Dict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..config import TrainConfig
from ..models import MelUNet
from ..losses import CombinedLoss
from ..data import PairedMelDataset, collate_fn


class Trainer:
    """Trainer for mel-to-mel model."""

    def __init__(
        self,
        config: TrainConfig,
        cache_dir: str,
        save_dir: str,
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model
        self.model = MelUNet(config.model).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Loss
        self.criterion = CombinedLoss(config.loss)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')

        # Resume
        if resume_from:
            self.load_checkpoint(resume_from)

        # TensorBoard
        self.writer = None
        if config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config.log_dir)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5,
            )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )

    def train(self, num_epochs: Optional[int] = None):
        """Run training loop."""
        if num_epochs is None:
            num_epochs = self.config.epochs

        # Data loaders
        train_dataset = PairedMelDataset(str(self.cache_dir), self.config, split='train')
        val_dataset = PairedMelDataset(str(self.cache_dir), self.config, split='val')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

        print(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"Validation: {len(val_dataset)} samples")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate(val_loader)

            # Update scheduler
            if self.config.lr_scheduler == "plateau":
                self.scheduler.step(val_metrics['total'])
            else:
                self.scheduler.step()

            # Logging
            self._log_epoch(train_metrics, val_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

            # Save best
            if val_metrics['total'] < self.best_loss:
                self.best_loss = val_metrics['total']
                self.save_checkpoint('best.pt')

        self.save_checkpoint('final.pt')
        if self.writer:
            self.writer.close()

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_metrics = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f'Epoch {self.epoch + 1}')
        for batch in pbar:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            if self.scaler:
                with autocast():
                    pred_mel = self.model(raw_mel)
                    losses = self.criterion(pred_mel, edit_mel, raw_mel)
                    loss = losses['total']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_mel = self.model(raw_mel)
                losses = self.criterion(pred_mel, edit_mel, raw_mel)
                loss = losses['total']

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            # Accumulate metrics
            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item()
            n_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average
        return {k: v / n_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        total_metrics = {}
        n_batches = 0

        for batch in loader:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)

            pred_mel = self.model(raw_mel)
            losses = self.criterion(pred_mel, edit_mel, raw_mel)

            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        lr = self.optimizer.param_groups[0]['lr']

        print(f"\nEpoch {self.epoch + 1}: "
              f"Train Loss={train_metrics['total']:.4f}, "
              f"Val Loss={val_metrics['total']:.4f}, "
              f"LR={lr:.6f}")

        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, self.epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.epoch)
            self.writer.add_scalar('lr', lr, self.epoch)

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'model_config': self.config.model,
            'train_config': self.config,
        }, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from epoch {self.epoch}")

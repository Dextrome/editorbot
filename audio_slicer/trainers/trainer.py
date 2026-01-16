"""Trainer for FaceSwap-style dual autoencoder.

Key insights from FaceSwap:
- Warp/augment inputs to make encoder robust
- Multi-loss training (reconstruction + perceptual)
- Shared encoder learns identity-agnostic features
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from ..config import TrainConfig, ModelConfig
from ..models import DualAutoencoder
from ..data import AudioSegmentDatasetFast


def warp_mel(mel: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    """Apply FaceSwap-style warping/augmentation to mel.

    This makes the encoder more robust by learning to ignore distortions.
    """
    B, T, F = mel.shape

    # Time masking (like SpecAugment)
    if random.random() < 0.5:
        mask_len = int(T * config.time_mask_ratio * random.random())
        mask_start = random.randint(0, max(1, T - mask_len))
        mel = mel.clone()
        mel[:, mask_start:mask_start + mask_len, :] = 0

    # Frequency masking
    if random.random() < 0.5:
        mask_len = int(F * config.freq_mask_ratio * random.random())
        mask_start = random.randint(0, max(1, F - mask_len))
        mel = mel.clone()
        mel[:, :, mask_start:mask_start + mask_len] = 0

    # Add noise
    if config.noise_std > 0 and random.random() < 0.5:
        noise = torch.randn_like(mel) * config.noise_std * random.random()
        mel = torch.clamp(mel + noise, 0, 1)

    return mel


class Trainer:
    """Train dual autoencoder FaceSwap-style."""

    def __init__(
        self,
        config: TrainConfig,
        cache_dir: str,
        save_dir: str,
    ):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model
        self.model = DualAutoencoder(config.model).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Data (use fast dataset that pre-loads to memory)
        self.train_dataset = AudioSegmentDatasetFast(cache_dir, config, split='train')
        self.val_dataset = AudioSegmentDatasetFast(cache_dir, config, split='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        # Tracking
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with FaceSwap-style warping."""
        self.model.train()
        total_raw_loss = 0
        total_edit_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)

            # FaceSwap-style: warp inputs, reconstruct clean targets
            raw_mel_warped = warp_mel(raw_mel, self.config)
            edit_mel_warped = warp_mel(edit_mel, self.config)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    # Forward with warped input, loss against clean target
                    raw_out = self.model.forward_raw(raw_mel_warped)
                    edit_out = self.model.forward_edited(edit_mel_warped)

                    # Reconstruction loss against CLEAN targets
                    raw_loss = F.mse_loss(raw_out['reconstruction'], raw_mel)
                    edit_loss = F.mse_loss(edit_out['reconstruction'], edit_mel)

                    # Combined loss
                    loss = raw_loss + edit_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                raw_out = self.model.forward_raw(raw_mel_warped)
                edit_out = self.model.forward_edited(edit_mel_warped)
                raw_loss = F.mse_loss(raw_out['reconstruction'], raw_mel)
                edit_loss = F.mse_loss(edit_out['reconstruction'], edit_mel)
                loss = raw_loss + edit_loss
                loss.backward()
                self.optimizer.step()

            total_raw_loss += raw_loss.item()
            total_edit_loss += edit_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'raw': f"{raw_loss.item():.4f}",
                'edit': f"{edit_loss.item():.4f}",
            })

        return {
            'raw_loss': total_raw_loss / n_batches,
            'edit_loss': total_edit_loss / n_batches,
            'total_loss': (total_raw_loss + total_edit_loss) / n_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        total_raw_loss = 0
        total_edit_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)

            raw_out = self.model.forward_raw(raw_mel)
            edit_out = self.model.forward_edited(edit_mel)

            total_raw_loss += raw_out['loss'].item()
            total_edit_loss += edit_out['loss'].item()
            n_batches += 1

        return {
            'raw_loss': total_raw_loss / n_batches,
            'edit_loss': total_edit_loss / n_batches,
            'total_loss': (total_raw_loss + total_edit_loss) / n_batches,
        }

    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # LR step
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Raw: {train_metrics['raw_loss']:.4f}, Edit: {train_metrics['edit_loss']:.4f}")
            print(f"  Val   - Raw: {val_metrics['raw_loss']:.4f}, Edit: {val_metrics['edit_loss']:.4f}")
            print(f"  LR: {lr:.6f}")

            # Save best
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.model.save(self.save_dir / 'best.pt')
                print(f"  Saved best model!")

            # Periodic save
            if epoch % self.config.save_every == 0:
                self.model.save(self.save_dir / f'epoch_{epoch}.pt')

        print("\nTraining complete!")
        self.model.save(self.save_dir / 'final.pt')

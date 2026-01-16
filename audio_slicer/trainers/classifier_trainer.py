"""Trainer for binary quality classifier.

Simple approach: edited audio = good (1), raw audio = bad (0).
Train classifier to distinguish, then use for segment selection.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import numpy as np

from ..config import TrainConfig, ModelConfig
from ..models.scorer import QualityScorer
from ..data import AudioSegmentDatasetFast


class ClassifierTrainer:
    """Train binary quality classifier."""

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
        self.model = QualityScorer(config.model).to(self.device)
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
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None

        # Data (use fast dataset that pre-loads to memory)
        self.train_dataset = AudioSegmentDatasetFast(cache_dir, config, split='train')
        self.val_dataset = AudioSegmentDatasetFast(cache_dir, config, split='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Faster for in-memory
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Tracking
        self.best_val_acc = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with binary classification."""
        self.model.train()
        total_loss = 0
        n_correct = 0
        n_total = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            B = raw_mel.size(0)

            # Create labels: raw=0 (bad), edit=1 (good)
            raw_labels = torch.zeros(B, device=self.device)
            edit_labels = torch.ones(B, device=self.device)

            # Combine into single batch
            mels = torch.cat([raw_mel, edit_mel], dim=0)
            labels = torch.cat([raw_labels, edit_labels], dim=0)

            # Shuffle
            perm = torch.randperm(2 * B)
            mels = mels[perm]
            labels = labels[perm]

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast('cuda'):
                    logits = self.model.score_logits(mels)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model.score_logits(mels)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            n_correct += (preds == labels).sum().item()
            n_total += 2 * B

            total_loss += loss.item()
            n_batches += 1

            acc = n_correct / n_total
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.1%}",
            })

        return {
            'loss': total_loss / n_batches,
            'accuracy': n_correct / n_total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        total_loss = 0
        n_correct = 0
        n_total = 0
        n_batches = 0

        # Track scores for analysis
        raw_scores = []
        edit_scores = []

        for batch in self.val_loader:
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            B = raw_mel.size(0)

            raw_labels = torch.zeros(B, device=self.device)
            edit_labels = torch.ones(B, device=self.device)

            # Score separately for analysis
            raw_s = self.model.score(raw_mel)
            edit_s = self.model.score(edit_mel)

            raw_scores.extend(raw_s.cpu().numpy())
            edit_scores.extend(edit_s.cpu().numpy())

            # Combined loss
            mels = torch.cat([raw_mel, edit_mel], dim=0)
            labels = torch.cat([raw_labels, edit_labels], dim=0)
            scores = torch.cat([raw_s, edit_s], dim=0)

            loss = F.binary_cross_entropy(scores, labels)
            preds = (scores > 0.5).float()
            n_correct += (preds == labels).sum().item()
            n_total += 2 * B
            total_loss += loss.item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'accuracy': n_correct / n_total,
            'raw_score_mean': np.mean(raw_scores),
            'edit_score_mean': np.mean(edit_scores),
            'score_separation': np.mean(edit_scores) - np.mean(raw_scores),
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
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.1%}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.1%}")
            print(f"  Val   - Raw score: {val_metrics['raw_score_mean']:.3f}, Edit score: {val_metrics['edit_score_mean']:.3f}")
            print(f"  Val   - Score separation: {val_metrics['score_separation']:.3f}")
            print(f"  LR: {lr:.6f}")

            # Save best
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.model.save(self.save_dir / 'best.pt')
                print(f"  Saved best model!")

            # Periodic save
            if epoch % self.config.save_every == 0:
                self.model.save(self.save_dir / f'epoch_{epoch}.pt')

        print("\nTraining complete!")
        self.model.save(self.save_dir / 'final.pt')

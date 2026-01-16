"""Trainer for pointer network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from ..models import PointerNetwork, STOP_TOKEN, PAD_TOKEN
from ..data.dataset import PointerDataset, collate_fn
from ..config import TrainConfig


class PointerNetworkTrainer:
    """Trainer for pointer network with all losses."""

    def __init__(
        self,
        config: TrainConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device

        # Create model
        self.model = PointerNetwork(
            n_mels=config.model.n_mels,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            dropout=config.model.dropout,
        ).to(device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
        )

        # Loss weights
        self.pointer_loss_weight = 1.0
        self.length_loss_weight = 0.1
        self.vae_kl_weight = 0.01
        self.stop_loss_weight = 0.5
        self.structure_loss_weight = 0.1

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders."""
        # Create full dataset
        full_dataset = PointerDataset(
            cache_dir=self.config.cache_dir,
            pointer_dir=self.config.pointer_dir,
            chunk_size=self.config.model.chunk_size,
            chunk_overlap=self.config.model.chunk_overlap,
        )

        # Split 90/10
        n_samples = len(full_dataset)
        n_train = int(0.9 * n_samples)
        n_val = n_samples - n_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        return train_loader, val_loader

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_pointers: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all training losses.

        Args:
            outputs: Model outputs containing logits, length_pred, kl_loss, etc.
            target_pointers: Ground truth pointer indices (B, T)
            padding_mask: True for padded positions (B, T)
        """
        losses = {}
        batch_size, seq_len = target_pointers.shape

        # Pointer prediction loss (cross-entropy)
        logits = outputs['logits']  # (B, T, vocab_size)
        vocab_size = logits.shape[-1]

        # Create targets with STOP token at end of valid sequence
        targets = target_pointers.clone()
        # Replace padding with ignore index
        targets[padding_mask] = -100  # PyTorch ignore index

        # Add STOP token at end of each sequence
        seq_lengths = (~padding_mask).sum(dim=1)  # Actual length of each sequence
        for i in range(batch_size):
            if seq_lengths[i] < seq_len:
                # Convert STOP_TOKEN to positive index (vocab_size - 1)
                targets[i, seq_lengths[i]] = vocab_size - 1  # STOP token at position vocab_size-1

        # Flatten for cross entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        pointer_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=-100,
            label_smoothing=0.1,
        )
        losses['pointer_loss'] = pointer_loss

        # Length prediction loss (MSE)
        if 'length_pred' in outputs:
            true_lengths = seq_lengths.float()
            length_loss = F.mse_loss(outputs['length_pred'].squeeze(), true_lengths)
            losses['length_loss'] = length_loss

        # VAE KL loss
        if 'kl_loss' in outputs:
            losses['kl_loss'] = outputs['kl_loss']

        # Stop prediction loss (binary cross entropy)
        if 'stop_logits' in outputs:
            stop_logits = outputs['stop_logits']  # (B, T, 1)
            # Create stop targets: 1 at sequence end, 0 elsewhere
            stop_targets = torch.zeros_like(stop_logits.squeeze(-1))
            for i in range(batch_size):
                if seq_lengths[i] < seq_len:
                    stop_targets[i, seq_lengths[i]] = 1.0

            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits.squeeze(-1)[~padding_mask],
                stop_targets[~padding_mask],
            )
            losses['stop_loss'] = stop_loss

        # Structure prediction loss (if model has structure head)
        if 'structure_logits' in outputs:
            # Placeholder - would need structure labels
            pass

        # Total weighted loss
        total_loss = (
            self.pointer_loss_weight * losses['pointer_loss']
            + self.length_loss_weight * losses.get('length_loss', 0)
            + self.vae_kl_weight * losses.get('kl_loss', 0)
            + self.stop_loss_weight * losses.get('stop_loss', 0)
            + self.structure_loss_weight * losses.get('structure_loss', 0)
        )
        losses['total_loss'] = total_loss

        return losses

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        target_pointers: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        with torch.no_grad():
            logits = outputs['logits']
            predictions = logits.argmax(dim=-1)

            # Accuracy (ignoring padding)
            correct = (predictions == target_pointers) & (~padding_mask)
            accuracy = correct.sum().float() / (~padding_mask).sum().float()

            # Within-N accuracy (pointer within N frames of target)
            for n in [1, 5, 10]:
                within_n = (torch.abs(predictions - target_pointers) <= n) & (~padding_mask)
                within_n_acc = within_n.sum().float() / (~padding_mask).sum().float()

            metrics = {
                'accuracy': accuracy.item(),
                'within_1_acc': within_n_acc.item() if n == 1 else 0,
                'within_5_acc': within_n_acc.item() if n == 5 else 0,
                'within_10_acc': within_n_acc.item() if n == 10 else 0,
            }

            # Length prediction error
            if 'length_pred' in outputs:
                true_lengths = (~padding_mask).sum(dim=1).float()
                length_error = torch.abs(outputs['length_pred'].squeeze() - true_lengths).mean()
                metrics['length_mae'] = length_error.item()

            return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        raw_mel = batch['raw_mel'].to(self.device)
        target_pointers = batch['target_pointers'].to(self.device)
        raw_padding_mask = batch['raw_padding_mask'].to(self.device)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device)

        # Forward pass
        outputs = self.model(
            raw_mel=raw_mel,
            target_pointers=target_pointers,
            raw_padding_mask=raw_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Compute losses
        losses = self.compute_losses(outputs, target_pointers, tgt_padding_mask)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.model.gradient_clip,
        )

        self.optimizer.step()
        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_losses = {}
        total_metrics = {}
        n_batches = 0

        for batch in val_loader:
            raw_mel = batch['raw_mel'].to(self.device)
            target_pointers = batch['target_pointers'].to(self.device)
            raw_padding_mask = batch['raw_padding_mask'].to(self.device)
            tgt_padding_mask = batch['tgt_padding_mask'].to(self.device)

            outputs = self.model(
                raw_mel=raw_mel,
                target_pointers=target_pointers,
                raw_padding_mask=raw_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            losses = self.compute_losses(outputs, target_pointers, tgt_padding_mask)
            metrics = self.compute_metrics(outputs, target_pointers, tgt_padding_mask)

            # Accumulate
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

            n_batches += 1

        # Average
        avg_losses = {f'val_{k}': v / n_batches for k, v in total_losses.items()}
        avg_metrics = {f'val_{k}': v / n_batches for k, v in total_metrics.items()}

        return {**avg_losses, **avg_metrics}

    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.save_dir / f"checkpoint_epoch{self.epoch}.pt"

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def train(self, resume_from: Optional[Path] = None):
        """Full training loop."""
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training loop
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            epoch_losses = {}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)

                # Accumulate epoch losses
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'ptr': f"{losses['pointer_loss']:.4f}",
                })

                # Log periodically
                if self.global_step % self.config.log_every == 0:
                    print(f"Step {self.global_step}: loss={losses['total_loss']:.4f}")

            # End of epoch
            self.scheduler.step()

            # Average epoch losses
            n_batches = len(train_loader)
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch} average losses: {avg_losses}")

            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                val_results = self.validate(val_loader)
                print(f"Validation: {val_results}")

                # Check if best
                val_loss = val_results['val_total_loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

            # Periodic save
            elif (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()

        print("Training complete!")


def train_pointer_network(config: Optional[TrainConfig] = None):
    """Main training function."""
    if config is None:
        config = TrainConfig()

    trainer = PointerNetworkTrainer(config)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="F:/editorbot/training_data/super_editor_cache")
    parser.add_argument("--pointer-dir", type=str, default="F:/editorbot/training_data/pointer_sequences")
    parser.add_argument("--save-dir", type=str, default="F:/editorbot/models/pointer_network")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        cache_dir=args.cache_dir,
        pointer_dir=args.pointer_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    trainer = PointerNetworkTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)

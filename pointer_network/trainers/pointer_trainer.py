"""Trainer for pointer network with performance optimizations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
import sys

from ..models import PointerNetwork, STOP_TOKEN, PAD_TOKEN
from ..data.dataset import PointerDataset, collate_fn, make_fixed_collate_fn
from ..data.augmentation import AugmentationConfig
from ..config import TrainConfig


def build_augmentation_config(config: TrainConfig) -> AugmentationConfig:
    """Build AugmentationConfig from TrainConfig fields."""
    return AugmentationConfig(
        enabled=getattr(config, 'augmentation_enabled', False),
        noise_enabled=True,
        noise_probability=getattr(config, 'augmentation_noise_prob', 0.5),
        noise_level_min=getattr(config, 'augmentation_noise_level_min', 0.05),
        noise_level_max=getattr(config, 'augmentation_noise_level_max', 0.15),
        spec_augment_enabled=True,
        spec_augment_probability=getattr(config, 'augmentation_spec_augment_prob', 0.5),
        freq_masks=getattr(config, 'augmentation_freq_masks', 2),
        freq_mask_width=getattr(config, 'augmentation_freq_mask_width', 20),
        time_masks=getattr(config, 'augmentation_time_masks', 2),
        time_mask_width=getattr(config, 'augmentation_time_mask_width', 50),
        gain_enabled=True,
        gain_probability=getattr(config, 'augmentation_gain_prob', 0.5),
        gain_scale_min=getattr(config, 'augmentation_gain_scale_min', 0.7),
        gain_scale_max=getattr(config, 'augmentation_gain_scale_max', 1.3),
        channel_dropout_enabled=True,
        channel_dropout_probability=getattr(config, 'augmentation_channel_dropout_prob', 0.3),
        channel_dropout_rate=getattr(config, 'augmentation_channel_dropout_rate', 0.1),
        chunk_shuffle_enabled=True,
        chunk_shuffle_probability=getattr(config, 'augmentation_chunk_shuffle_prob', 0.3),
        chunk_size=getattr(config, 'augmentation_chunk_size', 1000),
        crop_enabled=True,
        crop_probability=getattr(config, 'augmentation_crop_prob', 0.5),
        crop_min_len=getattr(config, 'augmentation_crop_min_len', 500),
        crop_max_len=getattr(config, 'augmentation_crop_max_len', 5000),
    )


class PointerNetworkTrainer:
    """Trainer for pointer network with all losses and performance optimizations."""

    def __init__(
        self,
        config: TrainConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device

        # Create model
        use_checkpoint = getattr(config, 'use_gradient_checkpoint', False)
        self.model = PointerNetwork(
            n_mels=config.model.n_mels,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            dropout=config.model.dropout,
            use_checkpoint=use_checkpoint,
        ).to(device)
        if use_checkpoint:
            print("Using gradient checkpointing (saves VRAM, slightly slower)")

        # Optionally compile model for PyTorch 2.0+
        if config.use_compile and hasattr(torch, 'compile'):
            compile_backend = getattr(config, 'compile_backend', 'inductor')
            try:
                # Only inductor supports 'mode' argument
                if compile_backend == 'inductor':
                    self.model = torch.compile(self.model, mode='reduce-overhead', backend=compile_backend)
                else:
                    self.model = torch.compile(self.model, backend=compile_backend)
                print(f"Model compiled with torch.compile(backend='{compile_backend}')")
            except Exception as e:
                print(f"torch.compile failed (will use eager mode): {e}")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler - OneCycleLR often converges faster
        self.use_onecycle = getattr(config, 'use_onecycle', True)
        self.scheduler = None  # Created after dataloaders for OneCycleLR
        self.scheduler_type = 'onecycle' if self.use_onecycle else 'cosine'

        # Warmup settings (only for cosine)
        self.warmup_steps = getattr(config, 'warmup_steps', 500)
        self.base_lr = config.learning_rate

        # Mixed precision scaler
        self.use_amp = config.use_amp and device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Using automatic mixed precision (AMP)")

        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

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

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")

    def create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders with optimized settings.

        Training uses all data (real + synthetic).
        Validation uses only real data (no _synth samples) for honest evaluation.
        """
        # Check if we should use real-only validation
        use_real_only_val = getattr(self.config, 'real_only_validation', True)

        # Build augmentation config if enabled
        use_augmentation = getattr(self.config, 'augmentation_enabled', False)
        aug_config = build_augmentation_config(self.config) if use_augmentation else None

        if use_real_only_val:
            # Create separate train dataset (all data) and val dataset (real only)
            train_dataset = PointerDataset(
                cache_dir=self.config.cache_dir,
                pointer_dir=self.config.pointer_dir,
                chunk_size=self.config.model.chunk_size,
                chunk_overlap=self.config.model.chunk_overlap,
                use_mmap=True,
                preload_pointers=True,
                exclude_samples=getattr(self.config, 'exclude_samples', []),
                real_only=False,  # Train on everything
                augment=use_augmentation,
                augmentation_config=aug_config,
            )

            val_dataset = PointerDataset(
                cache_dir=self.config.cache_dir,
                pointer_dir=self.config.pointer_dir,
                chunk_size=self.config.model.chunk_size,
                chunk_overlap=self.config.model.chunk_overlap,
                use_mmap=True,
                preload_pointers=True,
                exclude_samples=getattr(self.config, 'exclude_samples', []),
                real_only=True,  # Validate only on real data
                augment=False,  # No augmentation for validation
            )

            print(f"Train: {len(train_dataset)} samples (real + synthetic)")
            print(f"Val: {len(val_dataset)} samples (real only)")
            if use_augmentation:
                print("Augmentation enabled for training")
        else:
            # Original behavior: random split
            full_dataset = PointerDataset(
                cache_dir=self.config.cache_dir,
                pointer_dir=self.config.pointer_dir,
                chunk_size=self.config.model.chunk_size,
                chunk_overlap=self.config.model.chunk_overlap,
                use_mmap=True,
                preload_pointers=True,
                exclude_samples=getattr(self.config, 'exclude_samples', []),
                augment=use_augmentation,
                augmentation_config=aug_config,
            )

            # Split 90/10
            n_samples = len(full_dataset)
            n_train = int(0.9 * n_samples)
            n_val = n_samples - n_train

            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [n_train, n_val]
            )
            if use_augmentation:
                print("Augmentation enabled for training")

        # DataLoader settings for performance
        num_workers = self.config.num_workers
        # On Windows, multiprocessing can be tricky
        if sys.platform == 'win32' and num_workers > 0:
            # Windows requires spawn method, which is slower but works
            print(f"Using {num_workers} workers on Windows")

        # Choose collate function (fixed sizes for torch.compile optimization)
        use_fixed_length = getattr(self.config, 'use_fixed_length', False)
        if use_fixed_length:
            fixed_raw_len = getattr(self.config, 'fixed_raw_len', 2048)
            fixed_tgt_len = getattr(self.config, 'fixed_tgt_len', 512)
            the_collate_fn = make_fixed_collate_fn(fixed_raw_len, fixed_tgt_len)
            print(f"Using fixed-length collation: raw={fixed_raw_len}, tgt={fixed_tgt_len}")
        else:
            the_collate_fn = collate_fn

        # Common dataloader kwargs
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'collate_fn': the_collate_fn,
            'pin_memory': self.config.pin_memory and self.device == "cuda",
        }

        # Add worker settings if using multiple workers
        if num_workers > 0:
            loader_kwargs.update({
                'num_workers': num_workers,
                'prefetch_factor': self.config.prefetch_factor,
                'persistent_workers': self.config.persistent_workers,
            })
        else:
            loader_kwargs['num_workers'] = 0

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
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

    def _apply_warmup(self):
        """Apply linear warmup to learning rate (only for CosineAnnealing)."""
        # OneCycleLR has built-in warmup, so skip
        if self.use_onecycle:
            return
        if self.global_step < self.warmup_steps:
            warmup_factor = (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor

    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int) -> Dict[str, float]:
        """Single training step with AMP and gradient accumulation."""
        self.model.train()

        # Apply warmup
        self._apply_warmup()

        # Move to device (non-blocking for better overlap)
        raw_mel = batch['raw_mel'].to(self.device, non_blocking=True)
        target_pointers = batch['target_pointers'].to(self.device, non_blocking=True)

        # Forward pass with automatic mixed precision
        with autocast(enabled=self.use_amp):
            outputs = self.model(
                raw_mel=raw_mel,
                target_pointers=target_pointers,
            )
            total_loss = outputs['loss']

            # Check for NaN and skip if detected
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected at step {self.global_step}, skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                return {
                    'total_loss': 0.0, 'pointer_loss': 0.0, 'bar_pointer_loss': 0.0,
                    'beat_pointer_loss': 0.0, 'stop_loss': 0.0, 'kl_loss': 0.0,
                    'length_loss': 0.0, 'structure_loss': 0.0,
                }

            # Scale loss by accumulation steps for correct averaging
            total_loss = total_loss / self.gradient_accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()

        # Only update weights after accumulation steps
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Check gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.model.gradient_clip,
            )

            # Skip update only if gradient norm is NaN/Inf (not just large - clipping handles that)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: Skipping update at step {self.global_step}, grad_norm={grad_norm}")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Log gradient norm periodically
                if self.global_step % self.config.log_every == 0:
                    self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)

            self.global_step += 1

        # Return all losses for logging (including hierarchical)
        loss_dict = {
            'total_loss': outputs['loss'].item(),
            'pointer_loss': outputs['pointer_loss'].item(),
            'bar_pointer_loss': outputs.get('bar_pointer_loss', torch.tensor(0.0)).item(),
            'beat_pointer_loss': outputs.get('beat_pointer_loss', torch.tensor(0.0)).item(),
            'stop_loss': outputs['stop_loss'].item(),
            'kl_loss': outputs['kl_loss'].item(),
            'length_loss': outputs['length_loss'].item(),
            'structure_loss': outputs['structure_loss'].item(),
        }
        return loss_dict

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation with AMP."""
        self.model.eval()

        total_losses = {
            'loss': 0.0, 'pointer_loss': 0.0, 'bar_pointer_loss': 0.0,
            'beat_pointer_loss': 0.0, 'stop_loss': 0.0,
        }
        total_correct = 0
        total_frames = 0
        n_batches = 0

        for batch in val_loader:
            raw_mel = batch['raw_mel'].to(self.device, non_blocking=True)
            target_pointers = batch['target_pointers'].to(self.device, non_blocking=True)

            # Use AMP for validation too
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    raw_mel=raw_mel,
                    target_pointers=target_pointers,
                    use_vae=False,  # Deterministic for validation
                )

            # Accumulate losses
            total_losses['loss'] += outputs['loss'].item()
            total_losses['pointer_loss'] += outputs['pointer_loss'].item()
            total_losses['bar_pointer_loss'] += outputs.get('bar_pointer_loss', torch.tensor(0.0)).item()
            total_losses['beat_pointer_loss'] += outputs.get('beat_pointer_loss', torch.tensor(0.0)).item()
            total_losses['stop_loss'] += outputs['stop_loss'].item()

            # Compute accuracy (frame-level)
            predictions = outputs['pointer_logits'].argmax(dim=-1)
            stop_mask = target_pointers == STOP_TOKEN
            correct = (predictions == target_pointers) & (~stop_mask)
            total_correct += correct.sum().item()
            total_frames += (~stop_mask).sum().item()

            n_batches += 1

        # Average
        avg_losses = {f'val_{k}': v / n_batches for k, v in total_losses.items()}
        avg_losses['val_accuracy'] = total_correct / max(total_frames, 1)

        return avg_losses

    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.save_dir / f"checkpoint_epoch{self.epoch}.pt"

        # Get state dict - handle compiled models
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod  # Get uncompiled model

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Handle compiled models
        model_to_load = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_load = self.model._orig_mod

        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except RuntimeError as e:
                print(f"Warning: Could not load scaler state (AMP setting changed): {e}")

        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def train(self, resume_from: Optional[Path] = None):
        """Full training loop with all optimizations."""
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        print(f"Effective batch size: {self.config.batch_size * self.gradient_accumulation_steps}")

        # Create scheduler (needs total steps for OneCycleLR)
        total_steps = len(train_loader) * self.config.epochs // self.gradient_accumulation_steps
        if self.use_onecycle:
            # max_lr multiplier - configurable, default 3x (conservative)
            max_lr_mult = getattr(self.config, 'max_lr_mult', 3)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * max_lr_mult,
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=max_lr_mult,  # Start at base_lr
                final_div_factor=100,  # End at base_lr/100
            )
            print(f"Using OneCycleLR scheduler (total_steps={total_steps}, max_lr={self.config.learning_rate * max_lr_mult:.2e})")
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
            print(f"Using CosineAnnealingLR scheduler")

        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training loop
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            epoch_losses = {}

            # Zero gradients at start of epoch
            self.optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch, batch_idx)

                # Accumulate epoch losses
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

                # Update progress bar with hierarchical losses
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'bar': f"{losses['bar_pointer_loss']:.4f}",
                    'beat': f"{losses['beat_pointer_loss']:.4f}",
                    'frame': f"{losses['pointer_loss']:.4f}",
                })

                # Log periodically
                if self.global_step % self.config.log_every == 0 and self.global_step > 0:
                    print(f"Step {self.global_step}: loss={losses['total_loss']:.4f}")
                    # TensorBoard logging
                    self.writer.add_scalar('train/loss', losses['total_loss'], self.global_step)
                    self.writer.add_scalar('train/pointer_loss', losses['pointer_loss'], self.global_step)
                    self.writer.add_scalar('train/bar_pointer_loss', losses['bar_pointer_loss'], self.global_step)
                    self.writer.add_scalar('train/beat_pointer_loss', losses['beat_pointer_loss'], self.global_step)
                    self.writer.add_scalar('train/stop_loss', losses['stop_loss'], self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                # OneCycleLR steps per batch, CosineAnnealing steps per epoch
                if self.use_onecycle and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scheduler.step()

            # End of epoch - step for CosineAnnealing (per-epoch scheduler)
            if not self.use_onecycle:
                self.scheduler.step()

            # Average epoch losses
            n_batches = len(train_loader)
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch} average losses: {avg_losses}")

            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                val_results = self.validate(val_loader)
                print(f"Validation: {val_results}")
                # TensorBoard validation logging
                for k, v in val_results.items():
                    self.writer.add_scalar(k, v, self.global_step)

                # Check if best
                val_loss = val_results['val_loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

            # Periodic save
            elif (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()

        print("Training complete!")
        self.writer.close()


def train_pointer_network(config: Optional[TrainConfig] = None):
    """Main training function."""
    if config is None:
        config = TrainConfig()

    trainer = PointerNetworkTrainer(config)
    trainer.train()


def load_config(config_path: str) -> TrainConfig:
    """Load config from JSON file."""
    from ..config import PointerNetworkConfig

    with open(config_path, 'r') as f:
        data = json.load(f)

    # Build model config if present
    model_data = data.pop('model', {})
    model_config = PointerNetworkConfig(**model_data)

    # Build training config
    return TrainConfig(model=model_config, **data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default config")
        config = TrainConfig()

    # Print model config summary
    print(f"Model: d_model={config.model.d_model}, n_heads={config.model.n_heads}, "
          f"enc_layers={config.model.n_encoder_layers}, dec_layers={config.model.n_decoder_layers}")

    trainer = PointerNetworkTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)

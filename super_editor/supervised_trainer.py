"""Supervised Mel Reconstruction Trainer.

Inspired by FaceSwap's autoencoder approach - directly learns to reconstruct
edited audio from raw audio + edit labels using multi-scale perceptual losses.

Architecture:
    Raw Mel + Edit Labels → Encoder → Latent → Decoder → Edited Mel

Losses:
    - L1/L2 reconstruction on mel spectrogram
    - Multi-scale STFT loss for phase-aware training
    - Feature loss (perceptual) using pretrained audio embeddings
    - Edit consistency loss (penalize edits where labels say keep)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from rl_editor.config import Config, AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class SupervisedConfig:
    """Configuration for supervised mel reconstruction."""

    # Model architecture
    encoder_dim: int = 512
    decoder_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    # Input/output
    n_mels: int = 128
    max_beats: int = 2048

    # Training
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    batch_size: int = 8
    gradient_clip: float = 1.0

    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 0.0
    multiscale_stft_weight: float = 1.0
    feature_loss_weight: float = 0.1
    edit_consistency_weight: float = 0.5

    # Multi-scale STFT params
    stft_fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    stft_hop_sizes: List[int] = field(default_factory=lambda: [128, 256, 512])
    stft_win_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EditEncoder(nn.Module):
    """Encodes raw mel + edit labels into latent representation."""

    def __init__(self, config: SupervisedConfig):
        super().__init__()
        self.config = config

        # Project mel to encoder dim
        self.mel_proj = nn.Linear(config.n_mels, config.encoder_dim)

        # Edit label embedding (0=cut, 1=keep, 2=loop, etc.)
        self.edit_embed = nn.Embedding(8, config.encoder_dim // 4)

        # Combine mel + edit
        self.combine = nn.Linear(config.encoder_dim + config.encoder_dim // 4, config.encoder_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.encoder_dim, config.max_beats)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.encoder_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Layer norm
        self.norm = nn.LayerNorm(config.encoder_dim)

    def forward(
        self,
        mel: torch.Tensor,  # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T) int
        mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = valid
    ) -> torch.Tensor:
        # Project inputs
        mel_feat = self.mel_proj(mel)  # (B, T, D)
        edit_feat = self.edit_embed(edit_labels)  # (B, T, D//4)

        # Combine
        x = torch.cat([mel_feat, edit_feat], dim=-1)
        x = self.combine(x)  # (B, T, D)

        # Add positional encoding
        x = self.pos_enc(x)

        # Create attention mask (True = ignore)
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)

        return x


class MelDecoder(nn.Module):
    """Decodes latent to edited mel spectrogram."""

    def __init__(self, config: SupervisedConfig):
        super().__init__()
        self.config = config

        # Transformer decoder (self-attention only, no cross-attention needed)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.n_layers // 2)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(config.decoder_dim, config.decoder_dim),
            nn.GELU(),
            nn.Linear(config.decoder_dim, config.n_mels),
        )

        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        latent: torch.Tensor,  # (B, T, D)
        raw_mel: torch.Tensor,  # (B, T, n_mels) for residual
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Create attention mask
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        # Decode
        x = self.transformer(latent, src_key_padding_mask=attn_mask)

        # Project to mel
        out = self.out_proj(x)  # (B, T, n_mels)

        # Add residual from raw (helps preserve unedited regions)
        out = out + self.residual_weight * raw_mel

        return out


class MelReconstructionModel(nn.Module):
    """Full model: Raw Mel + Edit Labels → Edited Mel."""

    def __init__(self, config: SupervisedConfig):
        super().__init__()
        self.config = config
        self.encoder = EditEncoder(config)
        self.decoder = MelDecoder(config)

    def forward(
        self,
        raw_mel: torch.Tensor,  # (B, T, n_mels)
        edit_labels: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.encoder(raw_mel, edit_labels, mask)
        pred_mel = self.decoder(latent, raw_mel, mask)
        return pred_mel


class MultiScaleSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for better phase/frequency reconstruction."""

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_sizes: List[int] = [512, 1024, 2048],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def stft_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> torch.Tensor:
        # Compute STFT magnitude
        window = torch.hann_window(win_size, device=pred.device)

        pred_stft = torch.stft(
            pred, fft_size, hop_size, win_size, window,
            return_complex=True
        )
        target_stft = torch.stft(
            target, fft_size, hop_size, win_size, window,
            return_complex=True
        )

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        # Spectral convergence + log magnitude loss
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        mag_loss = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

        return sc_loss + mag_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, T) predicted waveform or (B, T, M) mel (will flatten)
            target: (B, T) target waveform or (B, T, M) mel
        """
        # If mel spectrogram, flatten to pseudo-waveform for STFT
        if pred.dim() == 3:
            pred = pred.reshape(pred.size(0), -1)
            target = target.reshape(target.size(0), -1)

        total_loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            if pred.size(1) >= win:
                total_loss = total_loss + self.stft_loss(pred, target, fft, hop, win)

        return total_loss / len(self.fft_sizes)


class SupervisedMelTrainer:
    """Trainer for supervised mel reconstruction."""

    def __init__(
        self,
        config: SupervisedConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model
        self.model = MelReconstructionModel(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # LR scheduler with warmup
        def lr_lambda(step):
            if step < config.lr_warmup_steps:
                return step / config.lr_warmup_steps
            else:
                progress = (step - config.lr_warmup_steps) / config.lr_decay_steps
                return max(0.1, 1.0 - 0.9 * min(1.0, progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Losses
        self.msstft_loss = MultiScaleSTFTLoss(
            config.stft_fft_sizes,
            config.stft_hop_sizes,
            config.stft_win_sizes,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')

    def compute_loss(
        self,
        pred_mel: torch.Tensor,  # (B, T, M)
        target_mel: torch.Tensor,  # (B, T, M)
        edit_labels: torch.Tensor,  # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all losses."""

        if mask is not None:
            # Apply mask to predictions and targets
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            pred_mel = pred_mel * mask_expanded
            target_mel = target_mel * mask_expanded

        losses = {}
        total_loss = 0.0

        # L1 reconstruction loss
        if self.config.l1_weight > 0:
            l1 = F.l1_loss(pred_mel, target_mel)
            losses['l1'] = l1.item()
            total_loss = total_loss + self.config.l1_weight * l1

        # MSE reconstruction loss
        if self.config.mse_weight > 0:
            mse = F.mse_loss(pred_mel, target_mel)
            losses['mse'] = mse.item()
            total_loss = total_loss + self.config.mse_weight * mse

        # Multi-scale STFT loss
        if self.config.multiscale_stft_weight > 0:
            msstft = self.msstft_loss(pred_mel, target_mel)
            losses['msstft'] = msstft.item()
            total_loss = total_loss + self.config.multiscale_stft_weight * msstft

        # Edit consistency loss: penalize changes where edit_labels say KEEP (1)
        if self.config.edit_consistency_weight > 0:
            keep_mask = (edit_labels == 1).unsqueeze(-1).float()  # (B, T, 1)
            keep_diff = (pred_mel - target_mel).abs() * keep_mask
            consistency = keep_diff.mean()
            losses['consistency'] = consistency.item()
            total_loss = total_loss + self.config.edit_consistency_weight * consistency

        losses['total'] = total_loss.item()
        return total_loss, losses

    def train_step(
        self,
        raw_mel: torch.Tensor,
        target_mel: torch.Tensor,
        edit_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""

        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        raw_mel = raw_mel.to(self.device)
        target_mel = target_mel.to(self.device)
        edit_labels = edit_labels.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
            pred_mel = self.model(raw_mel, edit_labels, mask)
            loss, loss_dict = self.compute_loss(pred_mel, target_mel, edit_labels, mask)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )
        loss_dict['grad_norm'] = grad_norm.item()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1
        loss_dict['lr'] = self.scheduler.get_last_lr()[0]

        return loss_dict

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Validation pass."""

        self.model.eval()
        all_losses = []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            raw_mel = batch['raw_mel'].to(self.device)
            target_mel = batch['target_mel'].to(self.device)
            edit_labels = batch['edit_labels'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                pred_mel = self.model(raw_mel, edit_labels, mask)
                _, loss_dict = self.compute_loss(pred_mel, target_mel, edit_labels, mask)

            all_losses.append(loss_dict)

        # Average losses
        avg_losses = {}
        for key in all_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in all_losses])

        return avg_losses

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


class PairedMelDataset(Dataset):
    """Dataset for paired raw/edited mel spectrograms with edit labels."""

    def __init__(
        self,
        data_dir: str,
        config: SupervisedConfig,
        max_len: int = 2048,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.max_len = max_len

        # Find all paired samples from cache
        self.samples = []
        cache_dir = self.data_dir / "rl_editor" / "cache" / "features"
        if cache_dir.exists():
            for f in cache_dir.glob("*_raw.npz"):
                edit_f = f.parent / f.name.replace("_raw.npz", "_edit.npz")
                if edit_f.exists():
                    self.samples.append((f, edit_f))

        logger.info(f"Found {len(self.samples)} paired samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_path, edit_path = self.samples[idx]

        # Load cached features
        raw_data = np.load(raw_path)
        edit_data = np.load(edit_path)

        raw_mel = raw_data.get('mel', raw_data.get('beat_features'))
        edit_mel = edit_data.get('mel', edit_data.get('beat_features'))

        # Get or generate edit labels
        if 'edit_labels' in raw_data:
            edit_labels = raw_data['edit_labels']
        else:
            # Default: 1 (keep) for matching regions, 0 (cut) otherwise
            edit_labels = np.ones(len(raw_mel), dtype=np.int64)

        # Truncate/pad to max_len
        T = min(len(raw_mel), len(edit_mel), self.max_len)

        raw_mel = raw_mel[:T]
        edit_mel = edit_mel[:T]
        edit_labels = edit_labels[:T]

        # Create mask (all valid)
        mask = np.ones(T, dtype=bool)

        # Pad if needed
        if T < self.max_len:
            pad_len = self.max_len - T
            raw_mel = np.pad(raw_mel, ((0, pad_len), (0, 0)), mode='constant')
            edit_mel = np.pad(edit_mel, ((0, pad_len), (0, 0)), mode='constant')
            edit_labels = np.pad(edit_labels, (0, pad_len), mode='constant', constant_values=0)
            mask = np.pad(mask, (0, pad_len), mode='constant', constant_values=False)

        return {
            'raw_mel': torch.from_numpy(raw_mel).float(),
            'target_mel': torch.from_numpy(edit_mel).float(),
            'edit_labels': torch.from_numpy(edit_labels).long(),
            'mask': torch.from_numpy(mask).bool(),
        }


def train_supervised(
    config: SupervisedConfig,
    data_dir: str,
    save_dir: str,
    n_epochs: int = 100,
    val_split: float = 0.1,
    log_interval: int = 100,
    save_interval: int = 1000,
):
    """Main training loop for supervised mel reconstruction."""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = PairedMelDataset(data_dir, config)

    # Train/val split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Trainer
    trainer = SupervisedMelTrainer(config)

    logger.info(f"Training supervised mel reconstruction")
    logger.info(f"  Train samples: {n_train}, Val samples: {n_val}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Device: {trainer.device}")

    # Training loop
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_losses = []

        for batch in train_loader:
            loss_dict = trainer.train_step(
                batch['raw_mel'],
                batch['target_mel'],
                batch['edit_labels'],
                batch.get('mask'),
            )
            epoch_losses.append(loss_dict)

            # Logging
            if trainer.global_step % log_interval == 0:
                avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                logger.info(
                    f"Step {trainer.global_step} | Loss: {avg_loss:.4f} | "
                    f"LR: {loss_dict['lr']:.2e}"
                )

            # Save checkpoint
            if trainer.global_step % save_interval == 0:
                ckpt_path = save_path / f"checkpoint_step_{trainer.global_step}.pt"
                trainer.save_checkpoint(str(ckpt_path))

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_train_loss = np.mean([l['total'] for l in epoch_losses])

        # Validation
        val_losses = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_losses['total'] < trainer.best_loss:
            trainer.best_loss = val_losses['total']
            best_path = save_path / "best.pt"
            trainer.save_checkpoint(str(best_path))
            logger.info(f"New best model saved (loss: {trainer.best_loss:.4f})")

    logger.info("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train supervised mel reconstruction")
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--save-dir", type=str, default="./models/supervised")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    config = SupervisedConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    train_supervised(
        config=config,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        n_epochs=args.epochs,
    )

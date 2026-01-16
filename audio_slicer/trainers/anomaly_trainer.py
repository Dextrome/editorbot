"""Trainer for anomaly-based quality detection.

Key insight: Train autoencoder ONLY on edited (good) audio.
Then raw audio segments that can't be reconstructed well = bad segments.

This works because:
- Edited audio represents "good" content
- Raw audio contains both good and bad content
- Bad content should have higher reconstruction error
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np

from ..config import TrainConfig, ModelConfig


class EditedOnlyDataset(Dataset):
    """Dataset that only loads edited audio segments."""

    def __init__(
        self,
        cache_dir: str,
        config: TrainConfig,
        split: str = 'train',
    ):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.segment_frames = config.model.segment_frames

        # Find all edited files
        features_dir = self.cache_dir / 'features'
        edit_files = list(features_dir.glob('*_edit.npz'))

        # Train/val split
        n_val = max(1, int(len(edit_files) * config.val_split))
        if split == 'train':
            edit_files = edit_files[n_val:]
        else:
            edit_files = edit_files[:n_val]

        # Pre-load all edited mels
        print(f"Pre-loading {len(edit_files)} edited files into memory...")
        self.mels = []
        for f in edit_files:
            data = np.load(f)
            self.mels.append(data['mel'].astype(np.float32))

        self.total_frames = sum(len(m) for m in self.mels)
        self.n_samples = config.segments_per_track * len(edit_files)

        print(f"EditedOnlyDataset ({split}):")
        print(f"  Files: {len(edit_files)}")
        print(f"  Total frames: {self.total_frames:,}")
        print(f"  Samples per epoch: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random file
        file_idx = random.randint(0, len(self.mels) - 1)
        mel = self.mels[file_idx]

        # Random segment
        if len(mel) > self.segment_frames:
            start = random.randint(0, len(mel) - self.segment_frames)
            segment = mel[start:start + self.segment_frames]
        else:
            segment = mel

        return {'mel': torch.from_numpy(segment)}


class SimpleAutoencoder(nn.Module):
    """Simple convolutional autoencoder for mel reconstruction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.input_proj = nn.Linear(config.n_mels, config.hidden_dims[0])

        encoder_layers = []
        in_dim = config.hidden_dims[0]
        for out_dim in config.hidden_dims:
            encoder_layers.append(nn.Conv1d(in_dim, out_dim, 3, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm1d(out_dim))
            encoder_layers.append(nn.GELU())
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.latent_proj = nn.Linear(config.hidden_dims[-1], config.latent_dim)

        # Decoder
        self.latent_unproj = nn.Linear(config.latent_dim, config.hidden_dims[-1])

        decoder_layers = []
        hidden_dims_rev = list(reversed(config.hidden_dims))
        for i, out_dim in enumerate(hidden_dims_rev[1:] + [config.hidden_dims[0]]):
            in_dim = hidden_dims_rev[i]
            decoder_layers.append(nn.ConvTranspose1d(in_dim, out_dim, 4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm1d(out_dim))
            decoder_layers.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_proj = nn.Linear(config.hidden_dims[0], config.n_mels)

    def encode(self, mel):
        # mel: (B, T, n_mels)
        x = self.input_proj(mel)  # (B, T, hidden)
        x = x.transpose(1, 2)  # (B, hidden, T)
        x = self.encoder(x)  # (B, hidden[-1], T')
        x = x.mean(dim=-1)  # (B, hidden[-1])
        z = self.latent_proj(x)  # (B, latent_dim)
        return z

    def decode(self, z, target_len):
        # z: (B, latent_dim)
        x = self.latent_unproj(z)  # (B, hidden[-1])
        x = x.unsqueeze(-1)  # (B, hidden[-1], 1)

        # Calculate needed upsampling
        n_upsample = len(self.config.hidden_dims)
        init_len = max(1, target_len // (2 ** n_upsample))
        x = x.expand(-1, -1, init_len)  # (B, hidden[-1], init_len)

        x = self.decoder(x)  # (B, hidden[0], T')

        # Adjust to target length
        if x.size(-1) != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        x = x.transpose(1, 2)  # (B, T, hidden[0])
        mel = self.output_proj(x)  # (B, T, n_mels)
        return mel

    def forward(self, mel):
        z = self.encode(mel)
        reconstruction = self.decode(z, mel.size(1))
        return {'z': z, 'reconstruction': reconstruction}

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': self.config.__dict__,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str) -> 'SimpleAutoencoder':
        checkpoint = torch.load(path, map_location='cpu')
        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model


class AnomalyTrainer:
    """Train autoencoder on edited audio only."""

    def __init__(
        self,
        config: TrainConfig,
        cache_dir: str,
        save_dir: str,
    ):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = SimpleAutoencoder(config.model).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )

        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None

        self.train_dataset = EditedOnlyDataset(cache_dir, config, split='train')
        self.val_dataset = EditedOnlyDataset(cache_dir, config, split='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            mel = batch['mel'].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast('cuda'):
                    out = self.model(mel)
                    loss = F.mse_loss(out['reconstruction'], mel)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(mel)
                loss = F.mse_loss(out['reconstruction'], mel)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return {'loss': total_loss / n_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            out = self.model(mel)
            loss = F.mse_loss(out['reconstruction'], mel)
            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / n_batches}

    def train(self):
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            print(f"\nEpoch {epoch}:")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val loss: {val_metrics['loss']:.4f}")
            print(f"  LR: {lr:.6f}")

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.model.save(self.save_dir / 'best.pt')
                print(f"  Saved best model!")

            if epoch % self.config.save_every == 0:
                self.model.save(self.save_dir / f'epoch_{epoch}.pt')

        print("\nTraining complete!")
        self.model.save(self.save_dir / 'final.pt')

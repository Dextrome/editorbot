"""
Style Transfer Trainer - Trains the style transfer system.

Training approach:
1. Contrastive learning for style encoder (same song = similar embedding)
2. Reinforcement learning for policy (reward = discriminator score)
3. Adversarial training for discriminator (real vs generated)

The iterative refinement loop:
1. Policy generates remix plan
2. Remix is executed and encoded
3. Discriminator scores similarity to target
4. Policy is updated based on score (REINFORCE or PPO)
5. Discriminator is updated on real/fake pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .style_encoder import StyleEncoder, StyleEncoderNet, FeatureExtractor
from .remix_policy import RemixPolicy, RemixPolicyNet
from .discriminator import StyleDiscriminator, MultiScaleDiscriminator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Paths
    data_dir: str
    output_dir: str
    
    # Training params
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Architecture
    style_dim: int = 256
    hidden_dim: int = 512
    
    # RL params
    gamma: float = 0.99           # Discount factor
    entropy_coef: float = 0.01    # Entropy bonus for exploration
    value_coef: float = 0.5       # Value loss coefficient
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # Data augmentation
    augment_tempo: bool = True
    augment_pitch: bool = True
    augment_noise: bool = True
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 2


class SongDataset(Dataset):
    """Dataset of songs for style learning."""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        max_duration: float = 300.0,  # 5 minutes max
        cache_features: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.cache_features = cache_features
        
        self.feature_extractor = FeatureExtractor(sample_rate)
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.audio_files.extend(self.data_dir.rglob(ext))
        
        logger.info(f"Found {len(self.audio_files)} audio files in {data_dir}")
        
        # Feature cache
        self.cache_dir = self.data_dir / '.feature_cache'
        if cache_features:
            self.cache_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict:
        audio_path = self.audio_files[idx]
        
        # Check cache
        cache_path = self.cache_dir / f"{audio_path.stem}.npz"
        if self.cache_features and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {
                'time_features': cached['time_features'],
                'chroma_features': cached['chroma_features'],
                'global_features': cached['global_features'],
                'path': str(audio_path)
            }
        
        # Load and extract features
        try:
            audio, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate, 
                mono=True,
                duration=self.max_duration
            )
            
            features = self.feature_extractor.extract(audio, sr)
            
            # Convert to tensors (same as StyleEncoder._features_to_tensors)
            time_features, chroma_features, global_features = self._convert_features(features)
            
            # Cache
            if self.cache_features:
                np.savez(
                    cache_path,
                    time_features=time_features,
                    chroma_features=chroma_features,
                    global_features=global_features
                )
            
            return {
                'time_features': time_features,
                'chroma_features': chroma_features,
                'global_features': global_features,
                'path': str(audio_path)
            }
            
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            # Return zeros as fallback
            return {
                'time_features': np.zeros((3, 2048), dtype=np.float32),
                'chroma_features': np.zeros((12, 2048), dtype=np.float32),
                'global_features': np.zeros(16, dtype=np.float32),
                'path': str(audio_path)
            }
    
    def _convert_features(self, features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert StyleFeatures to numpy arrays."""
        target_length = 2048
        
        def resample_1d(arr, target_len):
            if len(arr) == target_len:
                return arr
            indices = np.linspace(0, len(arr) - 1, target_len)
            return np.interp(indices, np.arange(len(arr)), arr)
        
        # Time features: (3, T)
        energy = resample_1d(features.energy_curve, target_length)
        brightness = resample_1d(features.brightness_curve, target_length)
        onset = resample_1d(features.onset_density_curve, target_length)
        time_features = np.stack([energy, brightness, onset], axis=0).astype(np.float32)
        
        # Chroma features: (12, T)
        chroma_resampled = np.zeros((12, target_length), dtype=np.float32)
        for i in range(12):
            chroma_resampled[i] = resample_1d(features.chroma_sequence[i], target_length)
        
        # Global features
        global_features = np.zeros(16, dtype=np.float32)
        global_features[0] = features.tempo / 200.0
        global_features[1] = features.key / 12.0
        global_features[2] = features.avg_phrase_length / 16.0
        global_features[3] = features.energy_variance
        global_features[4] = len(features.phrase_lengths) / 50.0
        global_features[5] = features.duration / 600.0
        
        if features.phrase_lengths:
            global_features[6] = np.mean(features.phrase_lengths) / 8.0
            global_features[7] = np.std(features.phrase_lengths) / 4.0
        
        if features.phrase_energies:
            global_features[8] = np.mean(features.phrase_energies)
            global_features[9] = np.std(features.phrase_energies)
            global_features[10] = np.max(features.phrase_energies)
            global_features[11] = np.min(features.phrase_energies)
        
        if features.transition_types:
            total = len(features.transition_types)
            global_features[12] = features.transition_types.count("buildup") / total
            global_features[13] = features.transition_types.count("drop") / total
            global_features[14] = features.transition_types.count("sustain") / total
        
        return time_features, chroma_resampled, global_features


class StyleTransferTrainer:
    """
    Trains the complete style transfer system.
    
    Training phases:
    1. Pre-train style encoder with contrastive learning
    2. Pre-train discriminator on real songs
    3. Joint training of policy with RL
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize models
        self.style_encoder = StyleEncoderNet(
            embedding_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        self.discriminator = MultiScaleDiscriminator(
            style_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        self.policy = RemixPolicyNet(
            style_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.encoder_optim = optim.AdamW(
            self.style_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.disc_optim = optim.AdamW(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.policy_optim = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate * 0.1,  # Lower LR for RL
            weight_decay=config.weight_decay
        )
        
        # Dataset
        self.dataset = SongDataset(config.data_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_encoder_epoch(self) -> float:
        """
        Train style encoder for one epoch using contrastive learning.
        
        Objective: Same song (with augmentation) should have similar embedding.
        Different songs should have different embeddings.
        """
        self.style_encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.dataloader, desc="Training encoder"):
            time_feat = batch['time_features'].to(self.device)
            chroma_feat = batch['chroma_features'].to(self.device)
            global_feat = batch['global_features'].to(self.device)
            
            B = time_feat.size(0)
            
            # Get embeddings
            embeddings = self.style_encoder(time_feat, chroma_feat, global_feat)
            
            # Create augmented versions (simple: add noise)
            time_aug = time_feat + torch.randn_like(time_feat) * 0.1
            chroma_aug = chroma_feat + torch.randn_like(chroma_feat) * 0.1
            global_aug = global_feat + torch.randn_like(global_feat) * 0.05
            
            embeddings_aug = self.style_encoder(time_aug, chroma_aug, global_aug)
            
            # Contrastive loss (InfoNCE)
            # Positive pairs: original and augmented of same song
            # Negative pairs: different songs
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, dim=-1)
            embeddings_aug = F.normalize(embeddings_aug, dim=-1)
            
            # Similarity matrix
            similarity = torch.mm(embeddings, embeddings_aug.t()) / 0.07  # Temperature
            
            # Labels: diagonal should be positive pairs
            labels = torch.arange(B, device=self.device)
            
            # Cross-entropy loss
            loss = F.cross_entropy(similarity, labels)
            
            # Backward
            self.encoder_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.style_encoder.parameters(), self.config.max_grad_norm)
            self.encoder_optim.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train_discriminator_epoch(self) -> float:
        """
        Train discriminator to distinguish real style matches.
        
        For now, trains on pairs where:
        - Positive: different segments of same song (should be similar)
        - Negative: segments from different songs (should be different)
        """
        self.style_encoder.eval()
        self.discriminator.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.dataloader, desc="Training discriminator"):
            time_feat = batch['time_features'].to(self.device)
            chroma_feat = batch['chroma_features'].to(self.device)
            global_feat = batch['global_features'].to(self.device)
            
            B = time_feat.size(0)
            if B < 2:
                continue
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.style_encoder(time_feat, chroma_feat, global_feat)
            
            # Create positive pairs (augmented versions of same)
            time_aug = time_feat + torch.randn_like(time_feat) * 0.1
            chroma_aug = chroma_feat + torch.randn_like(chroma_feat) * 0.1
            global_aug = global_feat + torch.randn_like(global_feat) * 0.05
            
            with torch.no_grad():
                embeddings_aug = self.style_encoder(time_aug, chroma_aug, global_aug)
            
            # Positive pairs: original and augmented
            pos_score, _ = self.discriminator(embeddings, embeddings_aug)
            pos_labels = torch.ones(B, 1, device=self.device)
            
            # Negative pairs: shuffle one set
            perm = torch.randperm(B)
            neg_score, _ = self.discriminator(embeddings, embeddings_aug[perm])
            neg_labels = torch.zeros(B, 1, device=self.device)
            
            # Binary cross-entropy
            loss = F.binary_cross_entropy(pos_score, pos_labels) + \
                   F.binary_cross_entropy(neg_score, neg_labels)
            
            # Backward
            self.disc_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
            self.disc_optim.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self):
        """Full training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Dataset size: {len(self.dataset)} songs")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train encoder
            enc_loss = self.train_encoder_epoch()
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Encoder loss: {enc_loss:.4f}")
            
            # Train discriminator
            disc_loss = self.train_discriminator_epoch()
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Discriminator loss: {disc_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()
            
            # Track best
            total_loss = enc_loss + disc_loss
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.save_checkpoint(is_best=True)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'policy': self.policy.state_dict(),
            'encoder_optim': self.encoder_optim.state_dict(),
            'disc_optim': self.disc_optim.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'best_loss': self.best_loss,
            'config': vars(self.config)
        }
        
        path = os.path.join(
            self.config.output_dir,
            'best.pt' if is_best else f'checkpoint_{self.epoch+1}.pt'
        )
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.epoch = checkpoint['epoch']
        self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.encoder_optim.load_state_dict(checkpoint['encoder_optim'])
        self.disc_optim.load_state_dict(checkpoint['disc_optim'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


def train_style_transfer(
    data_dir: str,
    output_dir: str,
    **kwargs
) -> StyleTransferTrainer:
    """
    Convenience function to train style transfer.
    
    Args:
        data_dir: Directory containing training songs
        output_dir: Directory for checkpoints
        **kwargs: Additional TrainingConfig options
        
    Returns:
        Trained StyleTransferTrainer
    """
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        **kwargs
    )
    
    trainer = StyleTransferTrainer(config)
    trainer.train()
    
    return trainer

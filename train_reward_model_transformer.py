#!/usr/bin/env python3
"""
Train Transformer-based reward model from human feedback preferences.
Matches the architecture expected by learned_reward_integration.py

Usage:
    python train_reward_model_transformer.py --feedback feedback/preferences.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearnedRewardModel(nn.Module):
    """Transformer-based reward model for RLHF training.
    
    Matches the architecture from rl_editor/learned_reward_integration.py exactly.
    Trained on human preference pairs, predicts scalar reward for music edit sequences.
    """

    def __init__(
        self,
        input_dim: int = 125,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_beats: int = 500,
        device: str = "cuda"
    ):
        """Initialize learned reward model.
        
        Args:
            input_dim: Input feature dimension (125)
            hidden_dim: Hidden dimension (256)
            n_layers: Number of transformer layers (3)
            n_heads: Number of attention heads (4)
            dropout: Dropout rate (0.1)
            max_beats: Maximum sequence length (500)
            device: Compute device
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # Input projection with LayerNorm for stability
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding (smaller initialization to avoid explosions)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_beats, hidden_dim) * 0.01
        )
        
        # Transformer encoder for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Edit action embedding (maps action taken to embedding)
        self.action_embed = nn.Embedding(10, hidden_dim // 4)  # 10 action types max
        
        # Aggregation and output with normalization
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        beat_features: torch.Tensor,
        action_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reward for edit sequence.
        
        Args:
            beat_features: (batch, n_beats, input_dim) - per-beat audio features
            action_ids: (batch, n_beats) - action taken at each beat (0=KEEP, 1=CUT, etc.)
            mask: (batch, n_beats) - True for valid beats, False for padding
            
        Returns:
            (batch,) - scalar reward for each edit
        """
        batch_size, n_beats, _ = beat_features.shape
        
        # Project features and normalize
        x = self.input_proj(beat_features)  # (batch, n_beats, hidden)
        x = self.input_norm(x)  # Normalize for stability
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :n_beats, :]
        
        # Add action embeddings if provided
        if action_ids is not None:
            action_emb = self.action_embed(action_ids)  # (batch, n_beats, hidden//4)
            x = x + nn.functional.pad(action_emb, (0, self.hidden_dim - self.hidden_dim // 4))
        
        # Apply transformer (mask for padding)
        if mask is not None:
            # PyTorch transformer uses inverted mask (True = ignore)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
            
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Aggregate: mean pool over valid positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        
        # Final reward prediction
        x = self.aggregator(x)
        reward = self.reward_head(x).squeeze(-1)
        
        # Clip reward to prevent explosion
        reward = torch.clamp(reward, min=-10.0, max=10.0)
        
        return reward


def generate_synthetic_features(n_beats: int, keep_ratio: float, input_dim: int = 125) -> np.ndarray:
    """Generate synthetic beat features that encode preference info.
    
    Args:
        n_beats: Number of beats
        keep_ratio: Keep ratio (encoded in features)
        input_dim: Feature dimension
        
    Returns:
        (n_beats, input_dim) feature array
    """
    features = np.random.randn(n_beats, input_dim).astype(np.float32) * 0.1
    
    # Encode keep ratio in first few dimensions
    features[:, 0] = keep_ratio  # Primary keep ratio feature
    features[:, 1] = 1.0 - keep_ratio  # Complement
    features[:, 2] = (keep_ratio - 0.5) * 2  # Centered version
    
    # Energy profile (simulate varying energy across beats)
    energy_profile = np.sin(np.linspace(0, 2 * np.pi, n_beats)) * 0.5 + 0.5
    features[:, 3] = energy_profile
    
    # Beat position encoding
    features[:, 4] = np.linspace(0, 1, n_beats)  # Normalized position
    
    return features


def generate_action_ids(n_beats: int, keep_ratio: float) -> np.ndarray:
    """Generate action IDs based on keep ratio.
    
    Args:
        n_beats: Number of beats
        keep_ratio: Target keep ratio
        
    Returns:
        (n_beats,) action ID array (0=KEEP, 1=CUT)
    """
    n_keep = int(n_beats * keep_ratio)
    actions = np.ones(n_beats, dtype=np.int64)  # Default: CUT
    
    # Randomly select beats to keep
    keep_indices = np.random.choice(n_beats, min(n_keep, n_beats), replace=False)
    actions[keep_indices] = 0  # KEEP
    
    return actions


def load_preferences(feedback_path: str) -> List[Dict]:
    """Load preference data from JSON file."""
    with open(feedback_path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        return data.get('preferences', [])
    elif isinstance(data, list):
        # Filter out non-dict items (like "preferences" strings)
        return [item for item in data if isinstance(item, dict)]
    else:
        return []


def prepare_training_data(
    preferences: List[Dict],
    n_beats: int = 50,
    input_dim: int = 125
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from preferences.
    
    For each preference pair (A preferred over B or vice versa):
    - Generate features for both candidates
    - Create paired training examples
    
    Returns:
        features_a: (N, n_beats, input_dim)
        features_b: (N, n_beats, input_dim)
        actions_a: (N, n_beats)
        actions_b: (N, n_beats)
        labels: (N,) - 1 if A better, 0 if B better, 0.5 if tie
    """
    features_a_list = []
    features_b_list = []
    actions_a_list = []
    actions_b_list = []
    labels = []
    
    for pref in preferences:
        # Extract keep ratios from candidate IDs
        try:
            ratio_a = float(str(pref['candidate_a_id']).replace('temp_', ''))
            ratio_b = float(str(pref['candidate_b_id']).replace('temp_', ''))
        except:
            ratio_a = 0.5
            ratio_b = 0.5
        
        # Generate synthetic features
        feat_a = generate_synthetic_features(n_beats, ratio_a, input_dim)
        feat_b = generate_synthetic_features(n_beats, ratio_b, input_dim)
        
        # Generate action IDs
        act_a = generate_action_ids(n_beats, ratio_a)
        act_b = generate_action_ids(n_beats, ratio_b)
        
        features_a_list.append(feat_a)
        features_b_list.append(feat_b)
        actions_a_list.append(act_a)
        actions_b_list.append(act_b)
        
        # Convert preference to label
        # Handle both string ('a', 'b') and numeric (0.0-1.0) formats
        pref_value = pref['preference']
        if isinstance(pref_value, str):
            if pref_value == 'a':
                labels.append(1.0)
            elif pref_value == 'b':
                labels.append(0.0)
            else:
                labels.append(0.5)
        else:
            # Numeric format: already 0-1 range
            labels.append(float(pref_value))
    
    return (
        np.array(features_a_list, dtype=np.float32),
        np.array(features_b_list, dtype=np.float32),
        np.array(actions_a_list, dtype=np.int64),
        np.array(actions_b_list, dtype=np.int64),
        np.array(labels, dtype=np.float32)
    )


class TransformerRewardTrainer:
    """Train transformer reward model from preferences using Bradley-Terry model."""
    
    def __init__(
        self,
        input_dim: int = 125,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        self.model = LearnedRewardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            device=device
        )
        
        self.optimizer = None
        self.scheduler = None
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🔧 Using device: {device}")
        logger.info(f"📊 Model parameters: {n_params:,}")
        logger.info(f"📐 Architecture: input={input_dim}, hidden={hidden_dim}, layers={n_layers}, heads={n_heads}")
    
    def train(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        actions_a: np.ndarray,
        actions_b: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 8,
        val_split: float = 0.2,
        lr: float = 1e-4
    ) -> Dict:
        """Train using Bradley-Terry preference model.
        
        P(A > B) = sigmoid(reward(A) - reward(B))
        """
        # Split into train/val
        n = len(labels)
        n_val = max(1, int(n * val_split))
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        logger.info(f"\n📈 Training on {len(train_indices)} pairs, validating on {len(val_indices)} pairs")
        
        # Create tensors
        train_data = {
            'feat_a': torch.from_numpy(features_a[train_indices]).to(self.device),
            'feat_b': torch.from_numpy(features_b[train_indices]).to(self.device),
            'act_a': torch.from_numpy(actions_a[train_indices]).to(self.device),
            'act_b': torch.from_numpy(actions_b[train_indices]).to(self.device),
            'labels': torch.from_numpy(labels[train_indices]).to(self.device),
        }
        
        val_data = {
            'feat_a': torch.from_numpy(features_a[val_indices]).to(self.device),
            'feat_b': torch.from_numpy(features_b[val_indices]).to(self.device),
            'act_a': torch.from_numpy(actions_a[val_indices]).to(self.device),
            'act_b': torch.from_numpy(actions_b[val_indices]).to(self.device),
            'labels': torch.from_numpy(labels[val_indices]).to(self.device),
        }
        
        # Setup optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        logger.info("\n🚀 Starting training...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_data, batch_size)
            
            # Validate
            val_loss, val_acc = self._validate(val_data)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Save best
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                self._save_checkpoint("models/reward_model_v9_feedback_best.pt")
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} ({train_acc:.1%}) | "
                    f"Val: {val_loss:.4f} ({val_acc:.1%}) | "
                    f"Best: {history['best_val_loss']:.4f}"
                )
            
            self.scheduler.step()
        
        # Save final
        self._save_checkpoint("models/reward_model_v9_feedback_final.pt")
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best validation loss: {history['best_val_loss']:.4f} @ epoch {history['best_epoch']+1}")
        logger.info(f"   Final checkpoint: models/reward_model_v9_feedback_final.pt")
        logger.info(f"   Best checkpoint: models/reward_model_v9_feedback_best.pt")
        
        return history
    
    def _train_epoch(self, data: Dict, batch_size: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        n = len(data['labels'])
        indices = torch.randperm(n)
        
        total_loss = 0
        total_correct = 0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            feat_a = data['feat_a'][batch_idx]
            feat_b = data['feat_b'][batch_idx]
            act_a = data['act_a'][batch_idx]
            act_b = data['act_b'][batch_idx]
            labels = data['labels'][batch_idx]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reward_a = self.model(feat_a, act_a)
            reward_b = self.model(feat_b, act_b)
            
            # Bradley-Terry: P(A > B) = sigmoid(r_A - r_B)
            logits = reward_a - reward_b
            probs = torch.sigmoid(logits)
            
            # Binary cross-entropy loss
            loss = nn.functional.binary_cross_entropy(probs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            predictions = (probs > 0.5).float()
            targets = (labels > 0.5).float()
            total_correct += (predictions == targets).sum().item()
            
            n_batches += 1
        
        return total_loss / max(n_batches, 1), total_correct / n
    
    @torch.no_grad()
    def _validate(self, data: Dict) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        
        reward_a = self.model(data['feat_a'], data['act_a'])
        reward_b = self.model(data['feat_b'], data['act_b'])
        
        logits = reward_a - reward_b
        probs = torch.sigmoid(logits)
        
        loss = nn.functional.binary_cross_entropy(probs, data['labels'])
        
        predictions = (probs > 0.5).float()
        targets = (data['labels'] > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()
        
        return loss.item(), accuracy
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint with config."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
        }
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': config,
            'architecture': 'LearnedRewardModel',
        }
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer reward model from human feedback")
    parser.add_argument("--feedback", type=str, default="feedback/preferences.json",
                        help="Path to preferences JSON file")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_beats", type=int, default=50, help="Number of beats per sequence")
    
    args = parser.parse_args()
    
    # Load preferences
    logger.info(f"📂 Loading preferences from: {args.feedback}")
    preferences = load_preferences(args.feedback)
    logger.info(f"   Found {len(preferences)} preference pairs")
    
    # Prepare data
    logger.info("🔄 Preparing training data...")
    feat_a, feat_b, act_a, act_b, labels = prepare_training_data(
        preferences, n_beats=args.n_beats, input_dim=125
    )
    
    # Count preference distribution
    n_prefer_a = np.sum(labels > 0.5)
    n_prefer_b = np.sum(labels < 0.5)
    n_tie = np.sum(labels == 0.5)
    logger.info(f"   Prefer A: {n_prefer_a}, Prefer B: {n_prefer_b}, Ties: {n_tie}")
    
    # Train model
    trainer = TransformerRewardTrainer(
        input_dim=125,
        hidden_dim=256,
        n_layers=3,
        n_heads=4
    )
    
    history = trainer.train(
        feat_a, feat_b, act_a, act_b, labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Run RLHF training with learned reward:")
    logger.info("   python -m rl_editor.train_parallel --use_learned_rewards --epochs 500")
    logger.info("2. Test on real songs:")
    logger.info("   python deploy_policy_on_real_songs.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train reward model from human feedback preferences - simplified version.
Works with small feedback datasets.

Usage:
    python train_reward_model_simple.py --feedback feedback/preferences.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import random


class SimpleRewardModel(nn.Module):
    """Simple feed-forward reward model"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single reward score output
        )
    
    def forward(self, x):
        return self.net(x)


def features_from_candidate(candidate_info: Dict) -> np.ndarray:
    """Extract feature vector from candidate info"""
    # Simple features: temperature, keep_ratio, duration
    temp = float(str(candidate_info.get('temperature', 0.5)).replace('temp_', ''))
    keep_ratio = candidate_info.get('keep_ratio', 0.5)
    
    # Normalize features
    features = np.array([
        temp / 1.0,  # temperature (0-1 range)
        keep_ratio,  # keep ratio (0-1 range)
        1.0 if keep_ratio > 0.7 else (0.5 if keep_ratio > 0.3 else 0.0),  # conservativeness
    ], dtype=np.float32)
    
    # Pad to 10 dimensions
    features = np.pad(features, (0, 7), mode='constant', constant_values=0)
    
    return features


def prepare_preference_pairs(preferences: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert preference list to training pairs.
    
    Returns:
        - X: [N, 20] features (A_features + B_features)
        - y: [N] preference targets (1 if prefer A, 0 if prefer B, 0.5 if tie)
    """
    X_list = []
    y_list = []
    
    for pref in preferences:
        candidate_a_info = {'temperature': pref['candidate_a_id'], 
                           'keep_ratio': float(str(pref['candidate_a_id']).split('_')[1])}
        candidate_b_info = {'temperature': pref['candidate_b_id'],
                           'keep_ratio': float(str(pref['candidate_b_id']).split('_')[1])}
        
        # Extract features
        feat_a = features_from_candidate(candidate_a_info)
        feat_b = features_from_candidate(candidate_b_info)
        
        # Concatenate
        X = np.concatenate([feat_a, feat_b])
        
        # Convert preference to target
        if pref['preference'] == 'a':
            y = 1.0
        elif pref['preference'] == 'b':
            y = 0.0
        else:  # tie
            y = 0.5
        
        X_list.append(X)
        y_list.append(y)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class RewardModelTrainer:
    """Train reward model from preferences"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SimpleRewardModel(input_dim, hidden_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        
        print(f"🔧 Using device: {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, batch_size: int = 16, val_split: float = 0.2) -> Dict:
        """Train the reward model"""
        
        # Split into train/val
        n = len(X)
        n_val = int(n * val_split)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print(f"\n📈 Training on {len(X_train)} pairs, validating on {len(X_val)} pairs")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        print("\n🚀 Starting training...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                self._save_checkpoint(f"models/reward_model_v9_feedback_best.pt")
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train loss: {train_loss:.6f} | "
                      f"Val loss: {val_loss:.6f} | "
                      f"Best: {history['best_val_loss']:.6f} @ epoch {history['best_epoch']+1}")
            
            self.scheduler.step()
        
        # Save final
        self._save_checkpoint(f"models/reward_model_v9_feedback_final.pt")
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {history['best_val_loss']:.6f} @ epoch {history['best_epoch']+1}")
        print(f"   Final checkpoint: models/reward_model_v9_feedback_final.pt")
        print(f"   Best checkpoint: models/reward_model_v9_feedback_best.pt")
        
        return history
    
    def _train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch).squeeze()
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
        
        return total_loss / len(dataloader.dataset)
    
    def _validate_epoch(self, dataloader) -> float:
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item() * len(X_batch)
        
        return total_loss / len(dataloader.dataset)
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': 20,
                'hidden_dim': 128
            }
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Train reward model from human feedback")
    parser.add_argument("--feedback", type=str, default="feedback/preferences.json",
                        help="Path to preferences JSON file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Load feedback
    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        print(f"❌ Feedback file not found: {feedback_path}")
        return
    
    with open(feedback_path) as f:
        data = json.load(f)
    
    preferences = data.get('preferences', [])
    print(f"📊 Loaded {len(preferences)} preference ratings")
    
    # Prepare data
    print("\n🔄 Preparing training data...")
    X, y = prepare_preference_pairs(preferences)
    print(f"   Features shape: {X.shape}")
    print(f"   Targets shape: {y.shape}")
    
    # Train
    trainer = RewardModelTrainer()
    history = trainer.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
    
    print(f"\n🎯 Summary:")
    print(f"   {len(preferences)} preference pairs trained")
    print(f"   Final model saved to: models/reward_model_v9_feedback_final.pt")
    print(f"\n🚀 Next step:")
    print(f"   python train_rlhf_stable.py --episodes 500 \\")
    print(f"     --reward_model models/reward_model_v9_feedback_final.pt")


if __name__ == "__main__":
    main()

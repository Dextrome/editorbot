"""Learned reward model from human preferences (for RLHF).

Placeholder for full implementation in step 7.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from .config import Config


class RewardModel(nn.Module):
    """Learned reward model trained on human preference pairs.

    Maps (edit_features) -> scalar reward
    Trained via contrastive learning on preference pairs.
    """

    def __init__(self, config: Config, input_dim: int) -> None:
        """Initialize reward model.

        Args:
            config: Configuration object
            input_dim: Input feature dimension
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # Simple MLP for reward prediction
        hidden_dim = config.model.value_hidden_dim # Reuse value net config
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self.device = torch.device(config.training.device)
        self.to(self.device)

    def forward(self, edit_features: torch.Tensor) -> torch.Tensor:
        """Compute reward for edit.

        Args:
            edit_features: Edit feature tensor

        Returns:
            Reward scalar tensor
        """
        return self.net(edit_features)

    def train_on_preferences(
        self,
        edit_a_features: np.ndarray,
        edit_b_features: np.ndarray,
        preferences: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4
    ) -> float:
        """Train on human preference pairs.

        Args:
            edit_a_features: Features of first edits (N, D)
            edit_b_features: Features of second edits (N, D)
            preferences: 1 if a is better, 0 if b is better (N,)
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Final training loss
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        features_a = torch.from_numpy(edit_a_features).float().to(self.device)
        features_b = torch.from_numpy(edit_b_features).float().to(self.device)
        labels = torch.from_numpy(preferences).float().to(self.device)
        
        n_samples = len(features_a)
        final_loss = 0.0
        
        self.train()
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                batch_a = features_a[batch_idx]
                batch_b = features_b[batch_idx]
                batch_labels = labels[batch_idx]
                
                # Predict rewards
                reward_a = self(batch_a).squeeze(-1)
                reward_b = self(batch_b).squeeze(-1)
                
                # Bradley-Terry model: P(a > b) = sigmoid(r_a - r_b)
                logits = reward_a - reward_b
                
                loss = criterion(logits, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            final_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            
        return final_loss


def compute_dense_reward(
    edited_audio: np.ndarray,
    original_audio: np.ndarray,
    beat_times: np.ndarray,
    sr: int = 22050,
    config: Optional[Config] = None,
) -> float:
    """Compute dense reward from automatic metrics.

    Combines:
    - Tempo consistency
    - Energy flow
    - Phrase completeness
    - Transition quality

    Args:
        edited_audio: Edited audio array
        original_audio: Original audio array
        beat_times: Beat time positions in original
        sr: Sample rate
        config: Configuration object

    Returns:
        Dense reward score

    TODO: Implement in step 4
    """
    pass

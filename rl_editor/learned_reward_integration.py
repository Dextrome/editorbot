"""Integration of learned reward model with RL training pipeline.

This module connects the trained LearnedRewardModel (from train_reward_model.py)
to the PPO/DPO training loop for RLHF-based music editing.

Features:
- Load trained reward model checkpoint
- Convert beat trajectories to preference features
- Compute learned rewards during training
- Support for both PPO and DPO algorithms
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json

import numpy as np
import torch
import torch.nn as nn
import logging

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class LearnedRewardConfig:
    """Configuration for learned reward model integration."""
    
    # Model loading - default to feedback-trained model
    checkpoint_path: str = "models/reward_model_v9_feedback_final.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Feature processing
    max_beats: int = 200  # Max beats for feature extraction
    normalize_features: bool = True
    use_cached_features: bool = True
    
    # Reward computation
    reward_scale: float = 1.0  # Scale factor for learned reward
    reward_offset: float = 0.0  # Offset (for centering)
    clamp_reward: Tuple[float, float] = (-10.0, 10.0)  # Clamp range
    
    # RLHF settings
    use_learned_reward: bool = True
    learned_reward_weight: float = 0.8  # Weight vs dense rewards
    dense_reward_weight: float = 0.2
    
    # DPO (Direct Preference Optimization) settings
    use_dpo: bool = False  # If True, use DPO instead of PPO
    dpo_beta: float = 0.1  # Temperature parameter for DPO
    dpo_reference_model_path: Optional[str] = None


class LearnedRewardModel(nn.Module):
    """Transformer-based reward model for RLHF training.
    
    Matches the architecture from train_reward_model.py exactly.
    Trained on human preference pairs, predicts scalar reward
    for music edit sequences.
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


class LearnedRewardIntegration:
    """Integrates learned reward model with RL training pipeline."""

    def __init__(
        self,
        config: Config,
        reward_config: Optional[LearnedRewardConfig] = None,
        checkpoint_path: Optional[str] = None
    ):
        """Initialize learned reward integration.
        
        Args:
            config: Main training configuration
            reward_config: Learned reward configuration
            checkpoint_path: Optional override for checkpoint path
        """
        self.config = config
        self.reward_config = reward_config or LearnedRewardConfig()
        self.device = torch.device(self.reward_config.device)
        
        # Override checkpoint if provided
        if checkpoint_path:
            self.reward_config.checkpoint_path = checkpoint_path
        
        # Model will be loaded on demand
        self.model: Optional[LearnedRewardModel] = None
        self.model_config: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized LearnedRewardIntegration (device: {self.device})")

    def load_model(self) -> bool:
        """Load learned reward model from checkpoint.
        
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = Path(self.reward_config.checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract configuration
            if "config" not in checkpoint:
                logger.error("No config found in checkpoint")
                return False
            
            self.model_config = checkpoint["config"]
            
            # Create model with default architecture (matches training)
            # Architecture params are fixed by design
            self.model = LearnedRewardModel(
                input_dim=125,
                hidden_dim=256,
                n_layers=3,
                n_heads=4,
                dropout=0.1,
                max_beats=500,
                device=str(self.device)
            )
            
            # Load weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.error("No model_state_dict found in checkpoint")
                return False
            
            self.model.eval()
            logger.info(f"Loaded reward model from {checkpoint_path}")
            logger.info(f"Model config: {self.model_config}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compute_learned_reward(
        self,
        beat_features: np.ndarray,
        action_ids: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> float:
        """Compute learned reward for beat sequence.
        
        Args:
            beat_features: Beat features (n_beats, feature_dim)
            action_ids: Action IDs (n_beats,) - defaults to 0 (KEEP) if not provided
            action_mask: Optional mask for valid actions (n_beats,) - True for valid
            batch_size: Batch size for processing
            
        Returns:
            Scalar reward value
        """
        if self.model is None:
            if not self.load_model():
                logger.warning("Could not load reward model, returning 0.0")
                return 0.0
        
        # Pad/truncate to max beats
        n_beats = beat_features.shape[0]
        if n_beats > self.reward_config.max_beats:
            beat_features = beat_features[:self.reward_config.max_beats]
        elif n_beats < self.reward_config.max_beats:
            pad_size = self.reward_config.max_beats - n_beats
            beat_features = np.pad(beat_features, ((0, pad_size), (0, 0)))
        
        # Default action IDs if not provided
        if action_ids is None:
            action_ids = np.zeros(beat_features.shape[0], dtype=np.int64)
        else:
            if len(action_ids) > self.reward_config.max_beats:
                action_ids = action_ids[:self.reward_config.max_beats]
            elif len(action_ids) < self.reward_config.max_beats:
                pad_size = self.reward_config.max_beats - len(action_ids)
                action_ids = np.pad(action_ids, ((0, pad_size),))
        
        # Normalize features if needed
        if self.reward_config.normalize_features:
            beat_features = (beat_features - beat_features.mean(axis=0)) / (beat_features.std(axis=0) + 1e-8)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(beat_features).float().unsqueeze(0).to(self.device)
        action_ids_tensor = torch.from_numpy(action_ids).long().unsqueeze(0).to(self.device)
        
        # Handle action mask (True = valid, False = padding)
        mask_tensor = None
        if action_mask is not None:
            if len(action_mask) > self.reward_config.max_beats:
                action_mask = action_mask[:self.reward_config.max_beats]
            elif len(action_mask) < self.reward_config.max_beats:
                pad_size = self.reward_config.max_beats - len(action_mask)
                action_mask = np.pad(action_mask, ((0, pad_size),), constant_values=False)
            
            mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)
        
        # Compute reward
        with torch.no_grad():
            reward = self.model(features_tensor, action_ids_tensor, mask_tensor)
        
        # Apply scaling and clamping
        reward_value = float(reward[0].cpu())
        reward_value = reward_value * self.reward_config.reward_scale + self.reward_config.reward_offset
        reward_value = np.clip(
            reward_value,
            self.reward_config.clamp_reward[0],
            self.reward_config.clamp_reward[1]
        )
        
        return reward_value

    def compute_trajectory_reward(
        self,
        trajectory: Dict[str, Any],
        dense_reward: float
    ) -> float:
        """Compute combined reward for trajectory.
        
        Combines learned reward with dense reward using configured weights.
        
        Args:
            trajectory: Dictionary with trajectory data
            dense_reward: Dense reward from environment
            
        Returns:
            Combined reward value
        """
        if not self.reward_config.use_learned_reward:
            return dense_reward
        
        # Extract beat features from trajectory
        if "beat_features" not in trajectory:
            logger.warning("No beat_features in trajectory, using dense reward only")
            return dense_reward
        
        beat_features = trajectory["beat_features"]
        action_mask = trajectory.get("action_mask", None)
        
        # Compute learned reward
        learned_reward = self.compute_learned_reward(beat_features, action_mask)
        
        # Combine with weights
        combined_reward = (
            self.reward_config.learned_reward_weight * learned_reward +
            self.reward_config.dense_reward_weight * dense_reward
        )
        
        return combined_reward

    def get_model_state(self) -> Optional[Dict[str, Any]]:
        """Get model state for checkpointing.
        
        Returns:
            Dictionary with model state or None if not loaded
        """
        if self.model is None:
            return None
        
        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.model_config,
            "reward_config": {
                "learned_reward_weight": self.reward_config.learned_reward_weight,
                "dense_reward_weight": self.reward_config.dense_reward_weight,
                "reward_scale": self.reward_config.reward_scale,
                "clamp_reward": self.reward_config.clamp_reward,
            }
        }

    def set_model_state(self, state: Dict[str, Any]) -> bool:
        """Load model state from checkpoint.
        
        Args:
            state: Dictionary with model state
            
        Returns:
            True if successful
        """
        if "model_state_dict" not in state:
            return False
        
        try:
            if self.model is None:
                self.model_config = state.get("config")
                self.model = LearnedRewardModel(
                    input_dim=125,
                    hidden_dim=256,
                    n_layers=3,
                    n_heads=4,
                    dropout=0.1,
                    max_beats=500,
                    device=str(self.device)
                )
            
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            
            # Update reward config
            if "reward_config" in state:
                rc = state["reward_config"]
                self.reward_config.learned_reward_weight = rc.get("learned_reward_weight", 0.8)
                self.reward_config.dense_reward_weight = rc.get("dense_reward_weight", 0.2)
                self.reward_config.reward_scale = rc.get("reward_scale", 1.0)
                self.reward_config.clamp_reward = tuple(rc.get("clamp_reward", (-10.0, 10.0)))
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            return False

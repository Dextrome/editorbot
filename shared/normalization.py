"""Observation normalization utilities for RL training.

Provides running mean/std normalization for observations and rewards,
which is critical for stable RL training.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union


class RunningMeanStd:
    """Tracks the running mean and standard deviation of a data stream.

    Uses Welford's online algorithm for numerically stable computation.
    This is the standard implementation used in RL libraries like Stable Baselines.
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        epsilon: float = 1e-8,
    ):
        """Initialize running statistics tracker.

        Args:
            shape: Shape of the data (excluding batch dimension)
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Start with epsilon to avoid division by zero
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of data.

        Args:
            x: Batch of observations (batch_size, *shape) or single observation (*shape)
        """
        if x.ndim == len(self.mean.shape):
            # Single observation
            x = x[np.newaxis, ...]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update from pre-computed batch statistics.

        Uses parallel algorithm for combining means and variances.

        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        """Get standard deviation."""
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics.

        Args:
            x: Data to normalize

        Returns:
            Normalized data with zero mean and unit variance
        """
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale.

        Args:
            x: Normalized data

        Returns:
            Data in original scale
        """
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class TorchRunningMeanStd:
    """PyTorch version of RunningMeanStd for GPU-accelerated normalization."""

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        epsilon: float = 1e-8,
        device: str = "cpu",
    ):
        """Initialize running statistics tracker.

        Args:
            shape: Shape of the data (excluding batch dimension)
            epsilon: Small constant for numerical stability
            device: Device for tensors
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a batch of data.

        Args:
            x: Batch of observations (batch_size, *shape)
        """
        if x.ndim == len(self.mean.shape):
            x = x.unsqueeze(0)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> None:
        """Update from pre-computed batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> torch.Tensor:
        """Get standard deviation."""
        return torch.sqrt(self.var + self.epsilon)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize data using running statistics."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        return x * self.std + self.mean

    def to(self, device: str) -> 'TorchRunningMeanStd':
        """Move to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.device = device
        return self

    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "mean": self.mean.cpu().numpy(),
            "var": self.var.cpu().numpy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.mean = torch.tensor(state["mean"], device=self.device)
        self.var = torch.tensor(state["var"], device=self.device)
        self.count = state["count"]


class ObservationNormalizer:
    """Normalizer for RL observations with optional clipping.

    Wraps RunningMeanStd with additional features:
    - Optional training mode (only update during training)
    - Configurable clipping range
    - Support for both numpy and torch tensors
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        clip_range: float = 10.0,
        epsilon: float = 1e-8,
        device: str = "cpu",
        use_torch: bool = True,
    ):
        """Initialize observation normalizer.

        Args:
            shape: Shape of observations (excluding batch dimension)
            clip_range: Clip normalized values to [-clip_range, clip_range]
            epsilon: Small constant for numerical stability
            device: Device for torch tensors
            use_torch: Use PyTorch implementation
        """
        self.clip_range = clip_range
        self.training = True
        self.use_torch = use_torch

        if use_torch:
            self.running_stats = TorchRunningMeanStd(shape, epsilon, device)
        else:
            self.running_stats = RunningMeanStd(shape, epsilon)

    def normalize(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        update: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Normalize observations.

        Args:
            obs: Observations to normalize
            update: Whether to update running statistics

        Returns:
            Normalized and clipped observations
        """
        if update and self.training:
            if isinstance(obs, np.ndarray) and self.use_torch:
                self.running_stats.update(torch.from_numpy(obs).to(self.running_stats.device))
            else:
                self.running_stats.update(obs)

        normalized = self.running_stats.normalize(obs)

        # Clip to range
        if isinstance(normalized, torch.Tensor):
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        else:
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def train(self) -> 'ObservationNormalizer':
        """Set to training mode (updates statistics)."""
        self.training = True
        return self

    def eval(self) -> 'ObservationNormalizer':
        """Set to evaluation mode (frozen statistics)."""
        self.training = False
        return self

    def to(self, device: str) -> 'ObservationNormalizer':
        """Move to device."""
        if self.use_torch:
            self.running_stats.to(device)
        return self

    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "running_stats": self.running_stats.state_dict(),
            "clip_range": self.clip_range,
            "training": self.training,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.running_stats.load_state_dict(state["running_stats"])
        self.clip_range = state.get("clip_range", self.clip_range)
        self.training = state.get("training", True)


class RewardNormalizer:
    """Normalizer for RL rewards.

    Normalizes rewards to have approximately unit variance,
    which helps with stable training across different reward scales.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip_range: float = 10.0,
    ):
        """Initialize reward normalizer.

        Args:
            gamma: Discount factor for return estimation
            epsilon: Small constant for numerical stability
            clip_range: Clip normalized rewards to this range
        """
        self.running_stats = RunningMeanStd(shape=())
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.returns = 0.0
        self.training = True

    def normalize(self, reward: float, done: bool = False, update: bool = True) -> float:
        """Normalize a single reward.

        Args:
            reward: Raw reward
            done: Whether episode is done
            update: Whether to update running statistics

        Returns:
            Normalized reward
        """
        if update and self.training:
            # Track discounted returns
            self.returns = self.returns * self.gamma + reward
            self.running_stats.update(np.array([self.returns]))
            if done:
                self.returns = 0.0

        # Normalize by return std (not by reward std)
        normalized = reward / (self.running_stats.std + self.epsilon)
        return float(np.clip(normalized, -self.clip_range, self.clip_range))

    def normalize_batch(self, rewards: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize a batch of rewards.

        Args:
            rewards: Batch of rewards
            update: Whether to update running statistics

        Returns:
            Normalized rewards
        """
        if update and self.training:
            self.running_stats.update(rewards)

        normalized = rewards / (self.running_stats.std + self.epsilon)
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def train(self) -> 'RewardNormalizer':
        """Set to training mode."""
        self.training = True
        return self

    def eval(self) -> 'RewardNormalizer':
        """Set to evaluation mode."""
        self.training = False
        return self

    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "running_stats": self.running_stats.state_dict(),
            "returns": self.returns,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.running_stats.load_state_dict(state["running_stats"])
        self.returns = state.get("returns", 0.0)

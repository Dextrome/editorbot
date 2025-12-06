"""Optimized parallel training for RL audio editor.

Uses:
- Multiple parallel environments (vectorized)
- Multi-worker DataLoader for audio preprocessing
- Batch processing on GPU
- Mixed precision training
- Efficient memory management
- Learning rate scheduling with warmup
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from .config import Config, get_default_config
from .data import PairedAudioDataset, AudioDataset
from .environment import AudioEditingEnv
from .agent import Agent
from .state import AudioState, EditHistory
from .reward import compute_trajectory_return
from .logging_utils import TrainingLogger, create_logger
from .learned_reward_integration import LearnedRewardIntegration

logger = logging.getLogger(__name__)

# Set multiprocessing start method for Windows compatibility
if sys.platform == 'win32':
    mp.set_start_method('spawn', force=True)


def get_lr_scheduler(optimizer: optim.Optimizer, config: Config, total_epochs: int) -> Optional[LambdaLR]:
    """Create learning rate scheduler based on config.
    
    Args:
        optimizer: The optimizer to schedule
        config: Configuration with LR decay settings
        total_epochs: Total number of training epochs
        
    Returns:
        LambdaLR scheduler or None if lr_decay is disabled
    """
    ppo_config = config.ppo
    
    if not ppo_config.lr_decay:
        return None
    
    warmup_epochs = ppo_config.lr_warmup_epochs
    min_ratio = ppo_config.lr_min_ratio
    decay_type = ppo_config.lr_decay_type
    
    def lr_lambda(epoch: int) -> float:
        """Calculate LR multiplier based on epoch."""
        # Warmup phase: linear increase from min_ratio to 1.0
        if epoch < warmup_epochs:
            return min_ratio + (1.0 - min_ratio) * (epoch / warmup_epochs)
        
        # After warmup, apply decay
        epochs_after_warmup = epoch - warmup_epochs
        decay_epochs = total_epochs - warmup_epochs
        
        if decay_type == "cosine":
            # Cosine annealing: smooth decay to min_ratio
            progress = epochs_after_warmup / max(decay_epochs, 1)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        elif decay_type == "linear":
            # Linear decay to min_ratio
            progress = epochs_after_warmup / max(decay_epochs, 1)
            return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)
        
        elif decay_type == "exponential":
            # Exponential decay
            decay_rate = -math.log(min_ratio) / max(ppo_config.lr_decay_epochs, 1)
            return max(min_ratio, math.exp(-decay_rate * epochs_after_warmup))
        
        elif decay_type == "step":
            # Step decay: multiply by step_factor every step_interval epochs
            n_steps = epochs_after_warmup // ppo_config.lr_step_interval
            return max(min_ratio, ppo_config.lr_step_factor ** n_steps)
        
        else:
            # Unknown type, no decay
            logger.warning(f"Unknown lr_decay_type: {decay_type}, using constant LR")
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data across multiple environments."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    masks: List[np.ndarray]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.masks = []
    
    def add(self, state, action, reward, value, log_prob, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
    
    def __len__(self):
        return len(self.states)


class VectorizedEnvWrapper:
    """Wrapper for running multiple environments in parallel."""
    
    def __init__(self, config: Config, n_envs: int = 4, learned_reward_model: Optional[Any] = None):
        self.config = config
        self.n_envs = n_envs
        self.envs: List[Optional[AudioEditingEnv]] = [None] * n_envs
        self.audio_states: List[Optional[AudioState]] = [None] * n_envs
        self.learned_reward_model = learned_reward_model
        
    def set_learned_reward_model(self, model: Any) -> None:
        """Set the learned reward model for all environments."""
        self.learned_reward_model = model
        for env in self.envs:
            if env is not None:
                env.set_learned_reward_model(model)
        
    def set_audio_states(self, audio_states: List[AudioState]):
        """Set audio states for all environments."""
        for i, state in enumerate(audio_states[:self.n_envs]):
            self.audio_states[i] = state
            self.envs[i] = AudioEditingEnv(
                self.config, audio_state=state, 
                learned_reward_model=self.learned_reward_model
            )
    
    def reset_all(self) -> Tuple[List[np.ndarray], List[dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []
        for env in self.envs:
            if env is not None:
                obs, info = env.reset()
                obs_list.append(obs)
                info_list.append(info)
        return obs_list, info_list
    
    def step_all(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """Step all environments with given actions.
        
        When trajectory rewards are enabled, adds end-of-episode quality reward.
        When learned rewards are enabled, also adds learned reward model output.
        """
        next_obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for env, action in zip(self.envs, actions):
            if env is not None:
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Add trajectory reward at episode end if enabled
                if (terminated or truncated) and self.config.reward.use_trajectory_rewards:
                    trajectory_reward = env.compute_trajectory_reward()
                    reward += trajectory_reward
                    info['trajectory_reward'] = trajectory_reward
                
                # Add learned reward at episode end if enabled
                if (terminated or truncated) and self.config.reward.use_learned_rewards:
                    learned_reward = env.compute_learned_episode_reward()
                    # Scale learned reward to be in similar range as trajectory
                    learned_reward_scale = self.config.reward.trajectory_reward_scale * 0.5
                    scaled_learned_reward = learned_reward * learned_reward_scale
                    reward += scaled_learned_reward
                    info['learned_reward'] = scaled_learned_reward
                
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                terminated_list.append(terminated)
                truncated_list.append(truncated)
                info_list.append(info)
        
        return next_obs_list, reward_list, terminated_list, truncated_list, info_list
    
    def get_action_masks(self, keep_cut_only: bool = False) -> List[np.ndarray]:
        """Get action masks for all environments.
        
        Args:
            keep_cut_only: If True, only allow KEEP and CUT actions
        """
        masks = []
        for env in self.envs:
            if env is not None:
                if keep_cut_only:
                    mask = env.action_space.get_keep_cut_only_mask(
                        current_beat_index=env.audio_state.beat_index,
                        edited_beats=env.edit_history.get_edited_beats(),
                        total_beats=len(env.audio_state.beat_times),
                    )
                else:
                    mask = env.action_space.get_action_mask(
                        current_beat_index=env.audio_state.beat_index,
                        remaining_duration=env._get_remaining_duration(),
                        edited_beats=env.edit_history.get_edited_beats(),
                        total_beats=len(env.audio_state.beat_times),
                    )
                masks.append(mask)
        return masks


class ParallelPPOTrainer:
    """Optimized PPO trainer with parallel environments and batch processing."""
    
    def __init__(
        self,
        config: Config,
        n_envs: int = 4,
        prefetch_factor: int = 2,
        total_epochs: int = 1000,
        keep_cut_only: bool = True,  # Default to KEEP/CUT only for stability
        learned_reward_model: Optional[Any] = None,
    ):
        self.config = config
        self.n_envs = n_envs
        self.prefetch_factor = prefetch_factor
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.keep_cut_only = keep_cut_only  # Restrict to KEEP/CUT actions
        self.learned_reward_model = learned_reward_model
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        # Entropy coefficient (can decay over training)
        self.initial_entropy_coeff = config.ppo.entropy_coeff
        self.entropy_coeff_min = getattr(config.ppo, 'entropy_coeff_min', 0.01)
        self.entropy_coeff_decay = getattr(config.ppo, 'entropy_coeff_decay', True)
        
        # Vectorized environments (pass learned reward model if provided)
        self.vec_env = VectorizedEnvWrapper(config, n_envs, learned_reward_model=learned_reward_model)
        
        # Agent (initialized later when we know input dimensions)
        self.agent: Optional[Agent] = None
        self.policy_optimizer: Optional[optim.Adam] = None
        self.value_optimizer: Optional[optim.Adam] = None
        self.policy_scheduler: Optional[LambdaLR] = None
        self.value_scheduler: Optional[LambdaLR] = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.ppo.use_mixed_precision and torch.cuda.is_available())
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Tracking
        self.global_step = 0
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Reward normalization (running statistics)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
        # Target KL for early stopping
        self.target_kl = getattr(config.ppo, 'target_kl', 0.01)
        
        # Logging
        self.training_logger: Optional[TrainingLogger] = None
        if config.training.use_tensorboard or config.training.use_wandb:
            self.training_logger = create_logger(config)
        
        # Create directories
        Path(config.training.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ParallelPPOTrainer with {n_envs} environments on {self.device}")
        logger.info(f"Action mode: {'KEEP/CUT only' if keep_cut_only else 'Full action space'}")
        logger.info(f"Target KL for early stopping: {self.target_kl}")
        if config.ppo.lr_decay:
            logger.info(f"LR decay enabled: {config.ppo.lr_decay_type}, warmup={config.ppo.lr_warmup_epochs}, min_ratio={config.ppo.lr_min_ratio}")
    
    def _init_agent(self, input_dim: int, n_actions: int):
        """Initialize agent and optimizers."""
        self.agent = Agent(self.config, input_dim, n_actions)
        self.policy_optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=self.config.ppo.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.agent.value_net.parameters(),
            lr=self.config.ppo.learning_rate
        )
        
        # Initialize LR schedulers
        self.policy_scheduler = get_lr_scheduler(
            self.policy_optimizer, self.config, self.total_epochs
        )
        self.value_scheduler = get_lr_scheduler(
            self.value_optimizer, self.config, self.total_epochs
        )
        
        if self.policy_scheduler:
            logger.info(f"LR schedulers initialized for {self.total_epochs} epochs")
    
    def step_schedulers(self):
        """Step the learning rate schedulers after each epoch."""
        if self.policy_scheduler:
            self.policy_scheduler.step()
        if self.value_scheduler:
            self.value_scheduler.step()
        self.current_epoch += 1
        
        # Log current learning rate and entropy coefficient
        if self.policy_optimizer:
            current_lr = self.policy_optimizer.param_groups[0]['lr']
            if self.training_logger:
                self.training_logger.log_scalar("learning_rate", current_lr, self.current_epoch)
                self.training_logger.log_scalar("entropy_coeff", self.get_current_entropy_coeff(), self.current_epoch)
    
    def get_current_entropy_coeff(self) -> float:
        """Get current entropy coefficient with decay."""
        if not self.entropy_coeff_decay:
            return self.initial_entropy_coeff
        
        # Linear decay from initial to min
        progress = self.current_epoch / max(self.total_epochs, 1)
        decay_range = self.initial_entropy_coeff - self.entropy_coeff_min
        current = self.initial_entropy_coeff - progress * decay_range
        return max(current, self.entropy_coeff_min)
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self.policy_optimizer:
            return self.policy_optimizer.param_groups[0]['lr']
        return self.config.ppo.learning_rate
    
    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics (Welford's algorithm).
        
        This reduces reward variance and helps stabilize training.
        """
        # Update running statistics
        for r in rewards.flatten():
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_var += delta * delta2
        
        # Compute standard deviation
        if self.reward_count > 1:
            std = np.sqrt(self.reward_var / (self.reward_count - 1))
        else:
            std = 1.0
        
        # Normalize with small epsilon for stability
        std = max(std, 1e-8)
        normalized = (rewards - self.reward_mean) / std
        
        # Clip to prevent extreme values
        return np.clip(normalized, -10.0, 10.0)
    
    def collect_rollouts_parallel(
        self,
        audio_states: List[AudioState],
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect rollouts from multiple environments in parallel.
        
        Args:
            audio_states: List of audio states for each environment
            n_steps: Number of steps to collect per environment
            
        Returns:
            Dictionary with batched trajectory data
        """
        # Setup environments
        self.vec_env.set_audio_states(audio_states)
        obs_list, _ = self.vec_env.reset_all()
        
        # Initialize agent if needed
        if self.agent is None:
            input_dim = obs_list[0].shape[0]
            n_actions = self.vec_env.envs[0].action_space.n_discrete_actions
            self._init_agent(input_dim, n_actions)
        
        self.buffer.clear()
        episode_rewards = [0.0] * len(obs_list)
        episode_lengths = [0] * len(obs_list)
        
        # Exploration rate decays over training
        exploration_rate = max(0.1, 0.5 - self.global_step / 500000)
        
        for step in range(n_steps):
            # Batch observations
            obs_batch = np.stack(obs_list)
            obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
            
            # Get action masks
            masks = self.vec_env.get_action_masks(keep_cut_only=self.keep_cut_only)
            mask_batch = np.stack(masks)
            mask_tensor = torch.from_numpy(mask_batch).bool().to(self.device)
            
            # Batch inference
            with torch.no_grad():
                actions, log_probs = self.agent.select_action_batch(obs_tensor, mask_tensor)
                values = self.agent.compute_value_batch(obs_tensor)
            
            # Epsilon-greedy exploration: randomly select valid action with probability exploration_rate
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            for i in range(len(obs_list)):
                if np.random.random() < exploration_rate:
                    # Select random valid action
                    valid_actions = np.where(masks[i])[0]
                    if len(valid_actions) > 0:
                        random_action = np.random.choice(valid_actions)
                        actions_np[i] = random_action
                        # Approximate log prob for random action (uniform over valid)
                        log_probs_np[i] = -np.log(len(valid_actions))
            
            # Convert to lists
            actions_list = actions_np.tolist()
            log_probs_list = log_probs_np.tolist()
            values_list = values.cpu().numpy().tolist()
            
            # Step all environments
            next_obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step_all(actions_list)
            
            # Store in buffer (per-environment)
            for i in range(len(obs_list)):
                self.buffer.add(
                    state=obs_list[i],
                    action=actions_list[i],
                    reward=rewards[i],
                    value=values_list[i],
                    log_prob=log_probs_list[i],
                    done=terminateds[i] or truncateds[i],
                    mask=masks[i],
                )
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
            
            # Handle episode ends
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    # Log episode completion with learned reward if applicable
                    if self.config.reward.use_learned_rewards:
                        learned_r = infos[i].get('learned_reward', 0.0) if i < len(infos) else 0.0
                        logger.debug(f"Episode complete: len={episode_lengths[i]}, total_r={episode_rewards[i]:.2f}, learned_r={learned_r:.2f}")
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    # Reset this environment
                    obs, _ = self.vec_env.envs[i].reset()
                    next_obs_list[i] = obs
            
            obs_list = next_obs_list
            self.global_step += len(obs_list)
        
        # Compute returns and advantages
        returns, _ = compute_trajectory_return(
            self.buffer.rewards, 
            gamma=self.config.ppo.gamma
        )
        
        # Convert to numpy arrays with NaN handling at the source
        states_arr = np.array(self.buffer.states)
        values_arr = np.array(self.buffer.values)
        returns_arr = np.array(returns)
        log_probs_arr = np.array(self.buffer.log_probs)
        
        # Handle any NaN/Inf values in buffer data
        states_arr = np.nan_to_num(states_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        values_arr = np.nan_to_num(values_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        returns_arr = np.nan_to_num(returns_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        log_probs_arr = np.nan_to_num(log_probs_arr, nan=-5.0, posinf=0.0, neginf=-10.0)
        
        advantages = returns_arr - values_arr
        adv_std = advantages.std()
        if adv_std < 1e-8 or np.isnan(adv_std):
            adv_std = 1.0  # Prevent division by zero
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return {
            "states": states_arr,
            "actions": np.array(self.buffer.actions),
            "rewards": np.array(self.buffer.rewards),
            "returns": returns_arr,
            "values": values_arr,
            "advantages": advantages,
            "log_probs": log_probs_arr,
            "masks": self.buffer.masks,
            "dones": np.array(self.buffer.dones),
        }
    
    def update(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update policy and value networks using PPO with batched operations."""
        states = torch.from_numpy(rollout_data["states"]).float().to(self.device)
        actions = torch.from_numpy(rollout_data["actions"]).long().to(self.device)
        returns = torch.from_numpy(rollout_data["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout_data["advantages"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout_data["log_probs"]).float().to(self.device)
        
        ppo_config = self.config.ppo
        batch_size = ppo_config.batch_size
        n_samples = len(states)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []  # Track KL divergence
        nan_batches_skipped = 0
        early_stop_epoch = False
        
        for epoch in range(ppo_config.n_epochs):
            if early_stop_epoch:
                break
                
            indices = np.random.permutation(n_samples)
            epoch_kl_divs = []
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Handle NaN in ALL inputs including old_log_probs
                batch_states = torch.nan_to_num(batch_states, nan=0.0)
                batch_returns = torch.nan_to_num(batch_returns, nan=0.0)
                batch_advantages = torch.nan_to_num(batch_advantages, nan=0.0)
                batch_old_log_probs = torch.nan_to_num(batch_old_log_probs, nan=-5.0)  # reasonable default
                
                # Forward pass (mixed precision disabled for stability)
                # Policy loss
                new_log_probs, entropy = self.agent.evaluate_actions(batch_states, batch_actions)
                
                # Handle NaN in new log probs
                if torch.isnan(new_log_probs).any():
                    new_log_probs = torch.nan_to_num(new_log_probs, nan=-3.0)
                if torch.isnan(entropy).any():
                    entropy = torch.nan_to_num(entropy, nan=0.0)
                
                # Compute ratio with clamping for stability
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # Prevent extreme ratios
                ratio = torch.exp(log_ratio)
                
                # Compute approximate KL divergence for early stopping
                # KL(old || new) ≈ (ratio - 1) - log(ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    epoch_kl_divs.append(approx_kl)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - ppo_config.clip_ratio, 1 + ppo_config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping for stability
                values = self.agent.compute_value_batch(batch_states)
                if torch.isnan(values).any():
                    values = torch.nan_to_num(values, nan=0.0)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus (encourage exploration) - use dynamic entropy coeff
                entropy_loss = -entropy.mean()
                current_entropy_coeff = self.get_current_entropy_coeff()
                
                # Combined loss
                loss = (
                    policy_loss 
                    + ppo_config.value_loss_coeff * value_loss 
                    + current_entropy_coeff * entropy_loss
                )
                
                # Final NaN check - skip only if still NaN after all handling
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_skipped += 1
                    continue
                
                # Backward pass (standard, no mixed precision)
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), ppo_config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.agent.value_net.parameters(), ppo_config.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
            
            # Check KL divergence after each epoch for early stopping
            if epoch_kl_divs:
                mean_kl = np.mean(epoch_kl_divs)
                approx_kl_divs.append(mean_kl)
                if mean_kl > 1.5 * self.target_kl:
                    logger.debug(f"Early stopping at PPO epoch {epoch+1} due to KL divergence: {mean_kl:.4f}")
                    early_stop_epoch = True
        
        # Log warning if many batches were skipped
        if nan_batches_skipped > 0:
            logger.warning(f"Skipped {nan_batches_skipped} batches due to NaN/Inf loss")
        
        # Handle empty loss arrays - this should NOT happen if training is working
        if not policy_losses:
            logger.error("All batches skipped! Training is not functioning properly.")
            # Return high loss to indicate problem
            return {
                "policy_loss": 999.0,
                "value_loss": 999.0, 
                "entropy_loss": 0.0,
                "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
                "total_loss": 999.0,
                "approx_kl": 0.0,
            }
        
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "total_loss": np.mean(policy_losses) + ppo_config.value_loss_coeff * np.mean(value_losses),
            "approx_kl": np.mean(approx_kl_divs) if approx_kl_divs else 0.0,
            "n_episodes": len(self.episode_rewards),
        }
        
        # Log metrics
        if self.training_logger:
            self.training_logger.log_training_step(
                step=self.global_step,
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                entropy=np.mean(entropy_losses),
                episode_reward=metrics["episode_reward"],
                episode_length=int(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0,
            )
            # Log KL divergence
            self.training_logger.log_scalar("approx_kl", metrics["approx_kl"], self.global_step)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "current_epoch": self.current_epoch,
            "policy_state_dict": self.agent.policy_net.state_dict() if self.agent else None,
            "value_state_dict": self.agent.value_net.state_dict() if self.agent else None,
            "policy_optimizer": self.policy_optimizer.state_dict() if self.policy_optimizer else None,
            "value_optimizer": self.value_optimizer.state_dict() if self.value_optimizer else None,
            "policy_scheduler": self.policy_scheduler.state_dict() if self.policy_scheduler else None,
            "value_scheduler": self.value_scheduler.state_dict() if self.value_scheduler else None,
            "scaler": self.scaler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.current_epoch = checkpoint.get("current_epoch", 0)
        if self.agent and checkpoint["policy_state_dict"]:
            self.agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.agent.value_net.load_state_dict(checkpoint["value_state_dict"])
        if self.policy_optimizer and checkpoint["policy_optimizer"]:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        if self.policy_scheduler and checkpoint.get("policy_scheduler"):
            self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler"])
        if self.value_scheduler and checkpoint.get("value_scheduler"):
            self.value_scheduler.load_state_dict(checkpoint["value_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch}, best_reward={self.best_reward:.2f})")


def train_parallel(
    config: Config,
    data_dir: str,
    n_epochs: int = 1000,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    checkpoint_path: Optional[str] = None,
    target_loss: float = 0.01,
):
    """Main training function with parallel environments.
    
    Args:
        config: Configuration object
        data_dir: Path to training data
        n_epochs: Maximum number of epochs
        n_envs: Number of parallel environments
        steps_per_epoch: Steps to collect per epoch
        checkpoint_path: Optional checkpoint to resume from
        target_loss: Stop training when loss reaches this value
    """
    logger.info(f"Starting parallel training with {n_envs} environments")
    logger.info(f"Target loss: {target_loss}")
    
    # Load dataset with multi-worker DataLoader
    dataset = PairedAudioDataset(
        data_dir, 
        config, 
        cache_dir=config.data.cache_dir,
        include_reference=True,
    )
    
    if len(dataset) == 0:
        logger.error(f"No training data found in {data_dir}")
        return
    
    logger.info(f"Loaded {len(dataset)} training samples ({len(dataset.pairs)} pairs, {len(dataset.reference_files)} reference)")
    
    # Load learned reward model if enabled
    learned_reward_model = None
    if config.reward.use_learned_rewards:
        reward_integration = LearnedRewardIntegration(config)
        if reward_integration.load_model():
            learned_reward_model = reward_integration.model
            logger.info("✓ Learned reward model loaded for RLHF training")
        else:
            logger.warning("Could not load learned reward model - using dense rewards only")
            config.reward.use_learned_rewards = False
    
    # Create trainer with total_epochs for LR scheduling
    trainer = ParallelPPOTrainer(config, n_envs=n_envs, total_epochs=n_epochs, learned_reward_model=learned_reward_model)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    
    # Determine starting epoch
    start_epoch = trainer.current_epoch
    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        
        # Sample audio states for parallel environments
        # Limit beats to max 500 to keep action space manageable
        MAX_BEATS = 500
        audio_states = []
        for _ in range(n_envs):
            idx = np.random.randint(len(dataset))
            item = dataset[idx]
            
            # Use raw audio state for training
            raw_data = item["raw"]
            beat_times = raw_data["beat_times"].numpy()
            beat_features = raw_data["beat_features"].numpy()
            
            # Get ground truth edit labels for supervised reward
            edit_labels = item.get("edit_labels")
            if edit_labels is not None:
                edit_labels = edit_labels.numpy() if hasattr(edit_labels, 'numpy') else np.array(edit_labels)
            
            # Limit to MAX_BEATS for tractable action space
            if len(beat_times) > MAX_BEATS:
                # Sample a random window
                start_idx = np.random.randint(0, len(beat_times) - MAX_BEATS)
                beat_times = beat_times[start_idx:start_idx + MAX_BEATS]
                beat_features = beat_features[start_idx:start_idx + MAX_BEATS]
                # Also slice the edit labels if they exist
                if edit_labels is not None and len(edit_labels) > MAX_BEATS:
                    edit_labels = edit_labels[start_idx:start_idx + MAX_BEATS]
                # Re-zero beat times
                beat_times = beat_times - beat_times[0]
            
            audio_state = AudioState(
                beat_index=0,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=raw_data["tempo"].item(),
                target_labels=edit_labels,  # Pass ground truth for supervised reward
            )
            audio_states.append(audio_state)
        
        # Collect rollouts in parallel
        rollout_data = trainer.collect_rollouts_parallel(audio_states, steps_per_epoch // n_envs)
        
        # Update networks
        metrics = trainer.update(rollout_data)
        
        # Step learning rate schedulers
        trainer.step_schedulers()
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = trainer.get_current_lr()
        
        # Log progress
        n_eps = metrics.get('n_episodes', 0)
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {metrics['total_loss']:.4f} (P: {metrics['policy_loss']:.4f}, V: {metrics['value_loss']:.4f}) | "
            f"Reward: {metrics['episode_reward']:.2f} (eps: {n_eps}) | "
            f"LR: {current_lr:.2e} | "
            f"Steps: {trainer.global_step:,} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Check for target loss
        if metrics['total_loss'] < target_loss:
            logger.info(f"🎉 Reached target loss {target_loss}! Stopping training.")
            break
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_file = Path(config.training.save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_file))
        
        # Save best model
        if metrics['episode_reward'] > trainer.best_reward:
            trainer.best_reward = metrics['episode_reward']
            best_file = Path(config.training.save_dir) / "checkpoint_best.pt"
            trainer.save_checkpoint(str(best_file))
    
    # Save final checkpoint
    final_file = Path(config.training.save_dir) / "checkpoint_final.pt"
    trainer.save_checkpoint(str(final_file))
    
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time/60:.1f} minutes")
    logger.info(f"Final loss: {metrics['total_loss']:.4f}")
    logger.info(f"Best reward: {trainer.best_reward:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel RL Audio Editor Training")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=2048, help="Steps per epoch")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--target_loss", type=float, default=0.01, help="Target loss to stop training")
    parser.add_argument("--use_learned_rewards", action="store_true", help="Enable learned reward model (RLHF)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_default_config()
    
    # Enable learned rewards if requested
    if args.use_learned_rewards:
        config.reward.use_learned_rewards = True
        logger.info("Learned rewards ENABLED (RLHF mode)")
    
    train_parallel(
        config=config,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        checkpoint_path=args.checkpoint,
        target_loss=args.target_loss,
    )

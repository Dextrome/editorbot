"""Factored Action Space Training Script.

Uses 3-head policy network: (action_type, action_size, action_amount)
Combined log prob = log P(type) + log P(size|type) + log P(amount|type)

Key differences from single-head training:
1. Actions are tuples (type, size, amount) not single integers
2. Three separate masks for type/size/amount validity
3. Log probs are summed across heads
4. Rollout buffer stores 3-component actions
"""

import time
import sys
import multiprocessing as mp
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast

from .config import Config, get_default_config
from .data import PairedAudioDataset, AudioDataset
from .environment_factored import AudioEditingEnvFactored
from .actions_factored import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
)
from .agent_factored import FactoredAgent
from .state import AudioState, EditHistory
from .reward import compute_trajectory_return
from .logging_utils import TrainingLogger, create_logger
from .auxiliary_tasks import AuxiliaryConfig, AuxiliaryTargetComputer, compute_auxiliary_targets

logger = logging.getLogger(__name__)


def get_lr_scheduler(optimizer: optim.Optimizer, config: Config, total_epochs: int) -> Optional[LambdaLR]:
    """Create learning rate scheduler based on config."""
    ppo_config = config.ppo
    
    if not ppo_config.lr_decay:
        return None
    
    warmup_epochs = ppo_config.lr_warmup_epochs
    min_ratio = ppo_config.lr_min_ratio
    decay_type = ppo_config.lr_decay_type
    
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return min_ratio + (1.0 - min_ratio) * (epoch / warmup_epochs)
        
        epochs_after_warmup = epoch - warmup_epochs
        decay_epochs = total_epochs - warmup_epochs
        
        if decay_type == "cosine":
            progress = epochs_after_warmup / max(decay_epochs, 1)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        elif decay_type == "linear":
            progress = epochs_after_warmup / max(decay_epochs, 1)
            return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)
        else:
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


@dataclass
class FactoredRolloutBuffer:
    """Buffer for storing rollout data with factored actions."""
    states: List[np.ndarray] = field(default_factory=list)
    action_types: List[int] = field(default_factory=list)
    action_sizes: List[int] = field(default_factory=list)
    action_amounts: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)  # Combined log prob
    dones: List[bool] = field(default_factory=list)
    type_masks: List[np.ndarray] = field(default_factory=list)
    size_masks: List[np.ndarray] = field(default_factory=list)
    amount_masks: List[np.ndarray] = field(default_factory=list)
    # Episode tracking
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    beat_indices: List[int] = field(default_factory=list)
    # Auxiliary task targets
    auxiliary_targets: Dict[str, List[np.ndarray]] = field(default_factory=lambda: {
        "tempo": [],
        "energy": [],
        "phrase_boundary": [],
        "beat_reconstruction": [],
    })
    episode_lengths: List[int] = field(default_factory=list)
    beat_indices: List[int] = field(default_factory=list)
    
    def clear(self):
        self.states = []
        self.action_types = []
        self.action_sizes = []
        self.action_amounts = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.type_masks = []
        self.size_masks = []
        self.amount_masks = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.beat_indices = []
        self.auxiliary_targets = {
            "tempo": [],
            "energy": [],
            "phrase_boundary": [],
            "beat_reconstruction": [],
        }
    
    def add(
        self,
        state: np.ndarray,
        action_type: int,
        action_size: int,
        action_amount: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        type_mask: np.ndarray,
        size_mask: np.ndarray,
        amount_mask: np.ndarray,
        beat_index: int = 0,
        aux_targets: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.states.append(state)
        self.action_types.append(action_type)
        self.action_sizes.append(action_size)
        self.action_amounts.append(action_amount)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.type_masks.append(type_mask)
        self.size_masks.append(size_mask)
        self.amount_masks.append(amount_mask)
        self.beat_indices.append(beat_index)
        
        # Store auxiliary targets
        if aux_targets is not None:
            for key in self.auxiliary_targets:
                if key in aux_targets:
                    self.auxiliary_targets[key].append(aux_targets[key])
    
    def get_auxiliary_targets_batch(self) -> Dict[str, np.ndarray]:
        """Get batched auxiliary targets for training."""
        result = {}
        for key, values in self.auxiliary_targets.items():
            if values:
                result[key] = np.array(values)
        return result
    
    def __len__(self):
        return len(self.states)


class VectorizedEnvFactoredWrapper:
    """Wrapper for running multiple factored environments in parallel."""
    
    def __init__(self, config: Config, n_envs: int = 4, learned_reward_model: Optional[Any] = None):
        self.config = config
        self.n_envs = n_envs
        self._executor = ThreadPoolExecutor(max_workers=n_envs)
        self.envs: List[Optional[AudioEditingEnvFactored]] = [None] * n_envs
        self.audio_states: List[Optional[AudioState]] = [None] * n_envs
        self.learned_reward_model = learned_reward_model
    
    def set_audio_states(self, audio_states: List[AudioState]):
        for i, state in enumerate(audio_states[:self.n_envs]):
            self.audio_states[i] = state
            self.envs[i] = AudioEditingEnvFactored(
                self.config, 
                audio_state=state,
                learned_reward_model=self.learned_reward_model,
            )
    
    def set_learned_reward_model(self, model: Any):
        """Set learned reward model for all environments."""
        self.learned_reward_model = model
        for env in self.envs:
            if env is not None:
                env.set_learned_reward_model(model)
    
    def reset_all(self) -> Tuple[List[np.ndarray], List[dict]]:
        obs_list = []
        info_list = []
        for env in self.envs:
            if env is not None:
                obs, info = env.reset()
                obs_list.append(obs)
                info_list.append(info)
        return obs_list, info_list
    
    def step_all(
        self, actions: List[Tuple[int, int, int]]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """Step all envs with factored actions."""
        def step_env(args):
            env, action = args
            if env is not None:
                action_array = np.array(action, dtype=np.int64)
                return env.step(action_array)
            return None, 0.0, False, False, {}
        
        results = list(self._executor.map(step_env, zip(self.envs, actions)))
        
        next_obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]
        
        return next_obs, rewards, terminateds, truncateds, infos
    
    def get_action_masks(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Get factored action masks for all envs."""
        def get_masks(env):
            if env is not None:
                return env.get_action_masks()
            return (
                np.ones(N_ACTION_TYPES, dtype=bool),
                np.ones(N_ACTION_SIZES, dtype=bool),
                np.ones(N_ACTION_AMOUNTS, dtype=bool),
            )
        
        results = list(self._executor.map(get_masks, self.envs))
        
        type_masks = [r[0] for r in results]
        size_masks = [r[1] for r in results]
        amount_masks = [r[2] for r in results]
        
        return type_masks, size_masks, amount_masks
    
    def reset_env(self, i: int) -> Tuple[np.ndarray, dict]:
        if self.envs[i] is not None:
            return self.envs[i].reset()
        return None, {}


class FactoredPPOTrainer:
    """PPO trainer for factored action space."""
    
    def __init__(
        self,
        config: Config,
        n_envs: int = 4,
        total_epochs: int = 1000,
        use_subprocess: bool = False,
        learned_reward_model: Optional[Any] = None,
    ):
        self.config = config
        self.n_envs = n_envs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.use_subprocess = use_subprocess
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        # Entropy coefficient with decay
        self.initial_entropy_coeff = config.ppo.entropy_coeff
        self.entropy_coeff_min = getattr(config.ppo, 'entropy_coeff_min', 0.01)
        self.entropy_coeff_decay = getattr(config.ppo, 'entropy_coeff_decay', True)
        
        # Vectorized environments
        if use_subprocess:
            # For now, fall back to thread-based since subprocess requires more work
            # TODO: Implement subprocess-based factored environment
            logger.warning("Subprocess parallelism not yet implemented for factored actions, using threads")
        self.vec_env = VectorizedEnvFactoredWrapper(config, n_envs)
        logger.info(f"Using thread-based parallel environments ({n_envs} threads)")
        
        # Set learned reward model on environments
        self.learned_reward_model = learned_reward_model
        
        # Agent (initialized later)
        self.agent: Optional[FactoredAgent] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[LambdaLR] = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.ppo.use_mixed_precision and torch.cuda.is_available())
        
        # Buffer
        self.buffer = FactoredRolloutBuffer()
        
        # Tracking
        self.global_step = 0
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Target KL
        self.target_kl = getattr(config.ppo, 'target_kl', 0.01)
        
        # Pending checkpoint
        self._pending_checkpoint: Optional[Dict] = None
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1
        
        # Auxiliary tasks
        self.aux_config = AuxiliaryConfig()
        self.aux_target_computer = AuxiliaryTargetComputer(self.aux_config)
        self.auxiliary_optimizer: Optional[optim.Adam] = None
        
        # Logger
        self.training_logger: Optional[TrainingLogger] = None
        if config.training.use_tensorboard or config.training.use_wandb:
            self.training_logger = create_logger(config)
        
        # Directories
        Path(config.training.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FactoredPPOTrainer with {n_envs} envs on {self.device}")
        logger.info(f"Factored action space: {N_ACTION_TYPES} types × {N_ACTION_SIZES} sizes × {N_ACTION_AMOUNTS} amounts")
    
    def _init_agent(self, input_dim: int, beat_feature_dim: int = 121):
        """Initialize factored agent and optimizer."""
        self.agent = FactoredAgent(
            config=self.config,
            input_dim=input_dim,
            beat_feature_dim=beat_feature_dim,
            use_auxiliary_tasks=True,
        )
        
        # Single optimizer for all networks (policy + value)
        lr = self.config.ppo.learning_rate
        self.optimizer = optim.Adam(
            list(self.agent.policy_net.parameters()) + 
            list(self.agent.value_net.parameters()),
            lr=lr,
        )
        
        # Auxiliary task optimizer (separate to allow different learning rates)
        if self.agent.auxiliary_module is not None:
            self.auxiliary_optimizer = optim.Adam(
                self.agent.auxiliary_module.parameters(),
                lr=self.config.ppo.learning_rate * 2.0,  # Slightly higher LR for auxiliary tasks
            )
            logger.info("Auxiliary task optimizer initialized")
        logger.info(f"Learning rate: {lr:.2e}")
        
        # Scheduler
        self.scheduler = get_lr_scheduler(self.optimizer, self.config, self.total_epochs)
        
        if self.scheduler:
            logger.info(f"LR scheduler initialized for {self.total_epochs} epochs")
        
        # Load pending checkpoint
        if self._pending_checkpoint is not None:
            logger.info("Loading deferred checkpoint weights...")
            self._load_checkpoint_weights(self._pending_checkpoint)
            self._pending_checkpoint = None
            logger.info("✓ Checkpoint loaded")
    
    def get_current_entropy_coeff(self) -> float:
        if not self.entropy_coeff_decay:
            return self.initial_entropy_coeff
        
        progress = self.current_epoch / max(self.total_epochs, 1)
        decay_range = self.initial_entropy_coeff - self.entropy_coeff_min
        current = self.initial_entropy_coeff - progress * decay_range
        return max(current, self.entropy_coeff_min)
    
    def step_scheduler(self):
        if self.scheduler:
            self.scheduler.step()
        self.current_epoch += 1
        
        if self.optimizer and self.training_logger:
            lr = self.optimizer.param_groups[0]['lr']
            self.training_logger.log_scalar("learning_rate", lr, self.current_epoch)
            self.training_logger.log_scalar("entropy_coeff", self.get_current_entropy_coeff(), self.current_epoch)
    
    def collect_rollouts(
        self,
        audio_states: List[AudioState],
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect rollouts with factored actions."""
        # Setup environments
        self.vec_env.set_audio_states(audio_states)
        obs_list, _ = self.vec_env.reset_all()
        
        # Initialize agent if needed
        if self.agent is None:
            input_dim = obs_list[0].shape[0]
            self._init_agent(input_dim)
        
        self.buffer.clear()
        episode_rewards = [0.0] * len(obs_list)
        episode_lengths = [0] * len(obs_list)
        
        # Exploration rate
        exploration_rate = max(0.15, 0.5 - self.global_step / 2000000)
        
        for step in range(n_steps):
            # Batch observations
            obs_batch = np.stack(obs_list)
            
            # Get factored masks
            type_masks, size_masks, amount_masks = self.vec_env.get_action_masks()
            type_mask_batch = np.stack(type_masks)
            size_mask_batch = np.stack(size_masks)
            amount_mask_batch = np.stack(amount_masks)
            
            # To GPU
            obs_tensor = torch.from_numpy(obs_batch).float().to(self.device, non_blocking=True)
            type_mask_tensor = torch.from_numpy(type_mask_batch).bool().to(self.device, non_blocking=True)
            size_mask_tensor = torch.from_numpy(size_mask_batch).bool().to(self.device, non_blocking=True)
            amount_mask_tensor = torch.from_numpy(amount_mask_batch).bool().to(self.device, non_blocking=True)
            
            # Batch inference - returns (types, sizes, amounts, log_probs)
            with torch.no_grad():
                types, sizes, amounts, log_probs = self.agent.select_action_batch(
                    obs_tensor,
                    type_mask_tensor,
                    size_mask_tensor,
                    amount_mask_tensor,
                )
                values = self.agent.compute_value_batch(obs_tensor)
                
                # To CPU
                types_np = types.cpu().numpy()
                sizes_np = sizes.cpu().numpy()
                amounts_np = amounts.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()
            
            # Epsilon-greedy exploration
            explore_mask = np.random.random(len(obs_list)) < exploration_rate
            for i in np.where(explore_mask)[0]:
                # Random valid type
                valid_types = np.where(type_masks[i])[0]
                if len(valid_types) > 0:
                    types_np[i] = np.random.choice(valid_types)
                # Random valid size
                valid_sizes = np.where(size_masks[i])[0]
                if len(valid_sizes) > 0:
                    sizes_np[i] = np.random.choice(valid_sizes)
                # Random valid amount
                valid_amounts = np.where(amount_masks[i])[0]
                if len(valid_amounts) > 0:
                    amounts_np[i] = np.random.choice(valid_amounts)
                # Approximate log prob
                log_probs_np[i] = -(
                    np.log(len(valid_types) + 1) +
                    np.log(len(valid_sizes) + 1) +
                    np.log(len(valid_amounts) + 1)
                )
            
            # Create action tuples
            actions = [(int(types_np[i]), int(sizes_np[i]), int(amounts_np[i])) 
                      for i in range(len(obs_list))]
            
            # Step environments
            next_obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step_all(actions)
            
            # Skip per-step aux targets for speed - they're computed in batch during update
            # aux_targets_list is empty for now
            
            # Store in buffer
            for i in range(len(obs_list)):
                beat_idx = infos[i].get("beat", 0) if infos[i] else 0
                
                self.buffer.add(
                    state=obs_list[i],
                    action_type=types_np[i],
                    action_size=sizes_np[i],
                    action_amount=amounts_np[i],
                    reward=rewards[i],
                    value=values_np[i],
                    log_prob=log_probs_np[i],
                    done=terminateds[i] or truncateds[i],
                    type_mask=type_masks[i],
                    size_mask=size_masks[i],
                    amount_mask=amount_masks[i],
                    beat_index=beat_idx,
                    aux_targets=None,  # Skip aux targets for speed
                )
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
            
            # Handle episode ends
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    self.buffer.episode_rewards.append(episode_rewards[i])
                    self.buffer.episode_lengths.append(episode_lengths[i])
                    
                    logger.debug(f"Episode complete: len={episode_lengths[i]}, reward={episode_rewards[i]:.2f}")
                    
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    
                    obs, _ = self.vec_env.reset_env(i)
                    next_obs_list[i] = obs
            
            obs_list = next_obs_list
            self.global_step += len(obs_list)
        
        # Compute returns (Monte Carlo)
        returns, _ = compute_trajectory_return(
            self.buffer.rewards,
            gamma=self.config.ppo.gamma,
            normalize=False,
        )
        
        # Convert to arrays
        states_arr = np.array(self.buffer.states)
        values_arr = np.array(self.buffer.values)
        returns_arr = np.array(returns)
        log_probs_arr = np.array(self.buffer.log_probs)
        
        # Handle NaN
        states_arr = np.nan_to_num(states_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        values_arr = np.nan_to_num(values_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        returns_arr = np.nan_to_num(returns_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        log_probs_arr = np.nan_to_num(log_probs_arr, nan=-5.0, posinf=0.0, neginf=-10.0)
        
        # Normalize returns for value targets
        ret_mean = returns_arr.mean()
        ret_std = returns_arr.std()
        if ret_std < 1e-8 or np.isnan(ret_std):
            ret_std = 1.0
        value_targets = (returns_arr - ret_mean) / (ret_std + 1e-8)
        value_targets = np.clip(value_targets, -10.0, 10.0)
        
        # Advantages
        advantages = value_targets - values_arr
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std < 1e-8 or np.isnan(adv_std):
            adv_std = 1.0
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return {
            "states": states_arr,
            "action_types": np.array(self.buffer.action_types),
            "action_sizes": np.array(self.buffer.action_sizes),
            "action_amounts": np.array(self.buffer.action_amounts),
            "rewards": np.array(self.buffer.rewards),
            "returns": value_targets,
            "values": values_arr,
            "advantages": advantages,
            "log_probs": log_probs_arr,
            "type_masks": self.buffer.type_masks,
            "size_masks": self.buffer.size_masks,
            "amount_masks": self.buffer.amount_masks,
            "dones": np.array(self.buffer.dones),
        }
    
    def update(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update policy and value networks with factored actions."""
        states = torch.from_numpy(rollout_data["states"]).float().to(self.device)
        action_types = torch.from_numpy(rollout_data["action_types"]).long().to(self.device)
        action_sizes = torch.from_numpy(rollout_data["action_sizes"]).long().to(self.device)
        action_amounts = torch.from_numpy(rollout_data["action_amounts"]).long().to(self.device)
        returns = torch.from_numpy(rollout_data["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout_data["advantages"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout_data["log_probs"]).float().to(self.device)
        
        # Get auxiliary targets from buffer
        aux_targets_np = self.buffer.get_auxiliary_targets_batch()
        aux_targets = {}
        for key, arr in aux_targets_np.items():
            if len(arr) > 0:
                if key in ["tempo", "energy"]:
                    aux_targets[key] = torch.from_numpy(arr).long().to(self.device)
                else:
                    aux_targets[key] = torch.from_numpy(arr).float().to(self.device)
        
        ppo_config = self.config.ppo
        batch_size = ppo_config.batch_size
        n_samples = len(states)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        auxiliary_losses = []
        approx_kl_divs = []
        nan_batches_skipped = 0
        early_stop = False
        
        for epoch in range(ppo_config.n_epochs):
            if early_stop:
                break
            
            indices = np.random.permutation(n_samples)
            epoch_kl_divs = []
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_action_types = action_types[batch_indices]
                batch_action_sizes = action_sizes[batch_indices]
                batch_action_amounts = action_amounts[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Clean inputs
                batch_states = torch.nan_to_num(batch_states, nan=0.0)
                batch_returns = torch.nan_to_num(batch_returns, nan=0.0)
                batch_advantages = torch.nan_to_num(batch_advantages, nan=0.0)
                batch_old_log_probs = torch.nan_to_num(batch_old_log_probs, nan=-5.0)
                
                # Forward pass
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    new_log_probs, entropy = self.agent.evaluate_actions(
                        batch_states,
                        batch_action_types,
                        batch_action_sizes,
                        batch_action_amounts,
                    )
                
                # FP32 for loss
                new_log_probs = new_log_probs.float()
                entropy = entropy.float()
                batch_advantages = batch_advantages.float()
                batch_returns = batch_returns.float()
                batch_old_log_probs = batch_old_log_probs.float()
                
                # Handle NaN outputs
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    new_log_probs = torch.nan_to_num(new_log_probs, nan=-3.0, posinf=-1.0, neginf=-10.0)
                if torch.isnan(entropy).any() or torch.isinf(entropy).any():
                    entropy = torch.nan_to_num(entropy, nan=0.5, posinf=3.0, neginf=0.0)
                
                # Ratio
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                ratio = torch.clamp(ratio, 1e-6, 1e6)
                
                # KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if not np.isnan(approx_kl):
                        epoch_kl_divs.append(approx_kl)
                
                # Policy loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - ppo_config.clip_ratio, 1 + ppo_config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    values = self.agent.compute_value_batch(batch_states)
                values = values.float()
                values = torch.clamp(values, -100.0, 100.0)
                if torch.isnan(values).any():
                    values = torch.nan_to_num(values, nan=0.0)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                value_loss = torch.clamp(value_loss, 0.0, 100.0)
                
                # Entropy
                entropy_loss = -entropy.mean()
                current_entropy_coeff = self.get_current_entropy_coeff()
                
                # Clamp policy loss
                policy_loss = torch.clamp(policy_loss, -10.0, 10.0)
                
                # Auxiliary loss (if available)
                aux_loss = torch.tensor(0.0, device=self.device)
                if self.agent.auxiliary_module is not None and aux_targets:
                    batch_aux_targets = {k: v[batch_indices] for k, v in aux_targets.items() if len(v) > 0}
                    if batch_aux_targets:
                        aux_loss, _ = self.agent.compute_auxiliary_loss(
                            batch_states, batch_aux_targets, self.current_epoch
                        )
                        aux_loss = torch.clamp(aux_loss, 0.0, 10.0)
                
                # Total loss
                loss = (
                    policy_loss +
                    ppo_config.value_loss_coeff * value_loss +
                    current_entropy_coeff * entropy_loss +
                    aux_loss
                )
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_skipped += 1
                    continue
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clip
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.policy_net.parameters()) + list(self.agent.value_net.parameters()),
                    ppo_config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                if aux_loss.item() > 0:
                    auxiliary_losses.append(aux_loss.item())
            
            # Early stop check
            if epoch_kl_divs:
                mean_kl = np.mean(epoch_kl_divs)
                approx_kl_divs.append(mean_kl)
                if mean_kl > 1.5 * self.target_kl:
                    logger.debug(f"Early stop at PPO epoch {epoch+1} due to KL: {mean_kl:.4f}")
                    early_stop = True
        
        if nan_batches_skipped > 0:
            logger.warning(f"Skipped {nan_batches_skipped} batches due to NaN/Inf")
        
        if not policy_losses:
            logger.error("All batches skipped!")
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
            "value_loss": ppo_config.value_loss_coeff * np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "auxiliary_loss": np.mean(auxiliary_losses) if auxiliary_losses else 0.0,
            "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "total_loss": np.mean(policy_losses) + ppo_config.value_loss_coeff * np.mean(value_losses),
            "approx_kl": np.mean(approx_kl_divs) if approx_kl_divs else 0.0,
            "n_episodes": len(self.episode_rewards),
        }
        
        # Log
        if self.training_logger:
            self.training_logger.log_training_step(
                step=self.global_step,
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                entropy=np.mean(entropy_losses),
                episode_reward=metrics["episode_reward"],
                episode_length=int(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0,
            )
            self.training_logger.log_scalar("approx_kl", metrics["approx_kl"], self.global_step)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "version": "factored",
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "current_epoch": self.current_epoch,
            "policy_state_dict": self.agent.policy_net.state_dict() if self.agent else None,
            "value_state_dict": self.agent.value_net.state_dict() if self.agent else None,
            "auxiliary_state_dict": self.agent.auxiliary_module.state_dict() if self.agent and self.agent.auxiliary_module else None,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "auxiliary_optimizer": self.auxiliary_optimizer.state_dict() if self.auxiliary_optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved factored checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        version = checkpoint.get("version", "v1")
        if version not in ("factored",):
            logger.warning(f"Loading non-factored checkpoint (version={version})")
        
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.current_epoch = checkpoint.get("current_epoch", 0)
        
        if self.agent is None:
            logger.info("Agent not initialized - storing checkpoint for deferred loading")
            self._pending_checkpoint = checkpoint
            return
        
        self._load_checkpoint_weights(checkpoint)
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
    
    def _load_checkpoint_weights(self, checkpoint: Dict):
        """Load weights from checkpoint."""
        if checkpoint.get("policy_state_dict"):
            self.agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            logger.info("Loaded policy network")
        if checkpoint.get("value_state_dict"):
            self.agent.value_net.load_state_dict(checkpoint["value_state_dict"])
            logger.info("Loaded value network")
        if self.agent.auxiliary_module and checkpoint.get("auxiliary_state_dict"):
            self.agent.auxiliary_module.load_state_dict(checkpoint["auxiliary_state_dict"])
            logger.info("Loaded auxiliary module")
        if self.optimizer and checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer")
        if self.auxiliary_optimizer and checkpoint.get("auxiliary_optimizer"):
            self.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
            logger.info("Loaded auxiliary optimizer")
        if self.scheduler and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])


def train_factored(
    config: Config,
    data_dir: str,
    n_epochs: int = 1000,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    checkpoint_path: Optional[str] = None,
    use_subprocess: bool = False,
    learned_reward_model: Optional[Any] = None,
):
    """Main training function for factored action space.
    
    Args:
        config: Configuration
        data_dir: Path to training data
        n_epochs: Max epochs
        n_envs: Parallel environments
        steps_per_epoch: Steps per epoch
        checkpoint_path: Resume from checkpoint
        use_subprocess: Use subprocess-based parallelism
        learned_reward_model: Optional learned reward model for RLHF
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("FACTORED ACTION SPACE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Epochs: {n_epochs}, Envs: {n_envs}, Steps/epoch: {steps_per_epoch}")
    logger.info(f"Action space: {N_ACTION_TYPES} types × {N_ACTION_SIZES} sizes × {N_ACTION_AMOUNTS} amounts")
    logger.info(f"Total combinations: {N_ACTION_TYPES * N_ACTION_SIZES * N_ACTION_AMOUNTS}")
    if use_subprocess:
        logger.info("Parallelism: subprocess (true multiprocessing)")
    else:
        logger.info("Parallelism: threading")
    
    # Load dataset with caching (use PairedAudioDataset for fast loading)
    cache_dir = getattr(config.data, 'cache_dir', None) or str(Path(data_dir) / "feature_cache")
    dataset = PairedAudioDataset(
        data_dir, 
        config, 
        cache_dir=cache_dir,
        include_reference=True,
        use_augmentation=False,  # Disable augmentation for faster loading
    )
    if len(dataset) == 0:
        logger.error(f"No data found in {data_dir}")
        return
    logger.info(f"Loaded {len(dataset)} samples (cache: {cache_dir})")
    
    # Trainer
    trainer = FactoredPPOTrainer(
        config=config,
        n_envs=n_envs,
        total_epochs=n_epochs,
        use_subprocess=use_subprocess,
        learned_reward_model=learned_reward_model,
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    # Training loop
    save_dir = Path(config.training.save_dir)
    
    for epoch in range(trainer.current_epoch, n_epochs):
        epoch_start = time.time()
        
        # Sample audio states
        data_start = time.time()
        audio_states = []
        for _ in range(n_envs):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            
            # PairedAudioDataset returns {'raw': {...}, 'edited': {...}, ...}
            raw_data = sample.get('raw', sample)  # Fallback for AudioDataset
            
            audio_state = AudioState(
                beat_index=0,
                beat_times=raw_data["beat_times"],
                beat_features=raw_data["beat_features"],
                tempo=raw_data.get("tempo"),
                raw_audio=raw_data.get("audio"),
                sample_rate=raw_data.get("sample_rate", 22050),
            )
            audio_states.append(audio_state)
        data_time = time.time() - data_start
        
        # Collect rollouts
        rollout_start = time.time()
        rollout_data = trainer.collect_rollouts(audio_states, steps_per_epoch)
        rollout_time = time.time() - rollout_start
        
        # Update
        update_start = time.time()
        metrics = trainer.update(rollout_data)
        update_time = time.time() - update_start
        
        # Step scheduler
        trainer.step_scheduler()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:5d} | "
                f"Reward: {metrics['episode_reward']:7.2f} | "
                f"Policy: {metrics['policy_loss']:.4f} | "
                f"Value: {metrics['value_loss']:.4f} | "
                f"KL: {metrics['approx_kl']:.4f} | "
                f"Time: {epoch_time:.1f}s (D:{data_time:.2f}s R:{rollout_time:.1f}s U:{update_time:.1f}s)"
            )
        
        # Save checkpoints
        checkpoint_interval = getattr(config.training, 'checkpoint_interval', 100)
        if epoch % checkpoint_interval == 0 and epoch > 0:
            ckpt_path = save_dir / f"checkpoint_factored_epoch_{epoch}.pt"
            trainer.save_checkpoint(str(ckpt_path))
        
        # Best model
        if metrics["episode_reward"] > trainer.best_reward:
            trainer.best_reward = metrics["episode_reward"]
            best_path = save_dir / "best_factored.pt"
            trainer.save_checkpoint(str(best_path))
            logger.info(f"New best reward: {trainer.best_reward:.2f}")
    
    # Final save
    final_path = save_dir / "final_factored.pt"
    trainer.save_checkpoint(str(final_path))
    logger.info(f"Training complete. Final checkpoint: {final_path}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train factored action space RL editor")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Training data directory")
    parser.add_argument("--save_dir", type=str, default="models_factored", help="Save directory")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--n_envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=512, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--subprocess", action="store_true", help="Use subprocess parallelism")
    
    args = parser.parse_args()
    
    # Config
    config = get_default_config()
    config.training.save_dir = args.save_dir
    config.ppo.learning_rate = args.lr
    config.ppo.batch_size = args.batch_size
    
    train_factored(
        config=config,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        checkpoint_path=args.checkpoint,
        use_subprocess=args.subprocess,
    )


if __name__ == "__main__":
    main()

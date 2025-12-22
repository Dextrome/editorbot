"""Training Script - Factored Action Space with Episode Rewards.

Uses 3-head policy network: (action_type, action_size, action_amount)
Combined log prob = log P(type) + log P(size|type) + log P(amount|type)

Features:
- Factored action space: 20 types × 5 sizes × 5 amounts = 500 combinations
- Subprocess parallelism for true multiprocessing
- Episode-level rewards (Monte Carlo returns)
- Curriculum learning (short segments → long segments)
- Auxiliary tasks for better representation learning
- RLHF support via learned reward model
"""

import time
import sys
import multiprocessing as mp

from rl_editor.features import BeatFeatureExtractor
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import logging
import math
import os
import random
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
from .environment import AudioEditingEnvFactored
from .actions import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
)
from .agent import Agent
from .state import AudioState, EditHistory
from .reward import compute_trajectory_return
from .logging_utils import TrainingLogger, create_logger
from .learned_reward_integration import LearnedRewardIntegration
from .auxiliary_tasks import AuxiliaryConfig, AuxiliaryTargetComputer, compute_auxiliary_targets
from .infer import load_and_process_audio, run_inference, create_edited_audio

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
        elif decay_type == "exponential":
            decay_rate = -math.log(min_ratio) / max(ppo_config.lr_decay_epochs, 1)
            return max(min_ratio, math.exp(-decay_rate * epochs_after_warmup))
        elif decay_type == "step":
            n_steps = epochs_after_warmup // ppo_config.lr_step_interval
            return max(min_ratio, ppo_config.lr_step_factor ** n_steps)
        else:
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


@dataclass
class RolloutBuffer:
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
    section_decisions: List[int] = field(default_factory=list)
    beat_indices: List[int] = field(default_factory=list)
    # Auxiliary task targets
    auxiliary_targets: Dict[str, List[np.ndarray]] = field(default_factory=lambda: {
        "tempo": [],
        "energy": [],
        "phrase": [],
        "reconstruction": [],
        "reconstruction_mask": [],
        "mel_reconstruction": [],
        "mel_reconstruction_mask": [],
    })
    
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
        self.section_decisions = []
        self.beat_indices = []
        self.auxiliary_targets = {
            "tempo": [],
            "energy": [],
            "phrase": [],
            "reconstruction": [],
            "reconstruction_mask": [],
            "mel_reconstruction": [],
            "mel_reconstruction_mask": [],
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


class VectorizedEnvWrapper:
    """Wrapper for running multiple environments in parallel via threading."""
    
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


class PPOTrainer:
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
            from .subprocess_vec_env import make_subprocess_vec_env
            self.vec_env = make_subprocess_vec_env(config, n_envs, learned_reward_model=learned_reward_model)
            logger.info(f"Using subprocess-based parallel environments ({n_envs} processes)")
        else:
            self.vec_env = VectorizedEnvWrapper(config, n_envs, learned_reward_model=learned_reward_model)
            logger.info(f"Using thread-based parallel environments ({n_envs} threads)")
        
        # Set learned reward model on environments
        self.learned_reward_model = learned_reward_model
        
        # Agent (initialized later)
        self.agent: Optional[Agent] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[LambdaLR] = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.ppo.use_mixed_precision and torch.cuda.is_available())
        
        # Buffer
        self.buffer = RolloutBuffer()
        
        # Tracking
        self.global_step = 0
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Episode reward breakdowns for logging
        self._episode_reward_breakdowns: List[Dict[str, float]] = []
        
        # Target KL
        self.target_kl = getattr(config.ppo, 'target_kl', 0.01)
        
        # Pending checkpoint
        self._pending_checkpoint: Optional[Dict] = None
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1

        # BC (behavior cloning) dataset for mixed supervised loss
        self.bc_states = None
        self.bc_type_labels = None
        self.bc_size_labels = None
        self.bc_amount_labels = None
        self.bc_good_bad_labels = None
        self.bc_loss_weight = 0.0
        self.bc_batch_size = 64
        
        # Auxiliary tasks
        self.aux_config = AuxiliaryConfig()
        self.aux_target_computer = AuxiliaryTargetComputer(self.aux_config)
        self.auxiliary_optimizer: Optional[optim.Adam] = None
        # Training logger (may be created later)
        self.training_logger: Optional[TrainingLogger] = None

    def set_auxiliary_config(self, new_config: AuxiliaryConfig) -> None:
        """Update auxiliary task configuration and clear/recompute any cached targets.

        Call this when `mel_dim` or `use_chroma_continuity` (or other aux settings) change.
        """
        try:
            self.aux_target_computer.update_config(new_config)
            self.aux_config = new_config
            logger.info("Auxiliary config updated and cache invalidated where needed")
            # Reinitialize auxiliary optimizer if agent has auxiliary module
            if getattr(self, 'agent', None) is not None and getattr(self.agent, 'auxiliary_module', None) is not None:
                self.auxiliary_optimizer = optim.Adam(
                    self.agent.auxiliary_module.parameters(),
                    lr=self.config.ppo.learning_rate * 2.0,
                )
                logger.info("Auxiliary optimizer reinitialized after aux config change")
        except Exception:
            logger.exception("Failed to update auxiliary config; clearing aux cache as fallback")
            try:
                self.aux_target_computer.clear_cache()
            except Exception:
                pass
        
        # Ensure save directory exists
        try:
            Path(self.config.training.save_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("Failed to ensure save_dir exists")
        logger.info("Auxiliary config updated and aux cache handled")
    
    def _init_agent(self, input_dim: int, beat_feature_dim: int = 121):
        """Initialize agent and optimizer."""
        self.agent = Agent(
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
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self.optimizer:
            return self.optimizer.param_groups[0]['lr']
        return self.config.ppo.learning_rate
    
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
        # Track episode count before this rollout so we can report per-epoch deltas
        prev_episode_count = len(self.episode_rewards)

        # Clear auxiliary target cache
        self.aux_target_computer._cache.clear()
        
        # Setup environments
        self.vec_env.set_audio_states(audio_states)
        obs_list, _ = self.vec_env.reset_all()
                                       
        # Initialize agent if needed
        if self.agent is None:
            input_dim = obs_list[0].shape[0]
            self._init_agent(input_dim)
        
        self.buffer.clear()
        self._episode_reward_breakdowns = []
        episode_rewards = [0.0] * len(obs_list)
        episode_lengths = [0] * len(obs_list)
        section_decisions = [0] * len(obs_list)
        temporal_penalty_values: List[float] = []
        
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
            
            # Batch auxiliary target computation
            beat_indices = []
            aux_targets_list = []
            
            if self.use_subprocess:
                # Subprocess mode - aux targets computed in subprocess workers
                for i in range(len(obs_list)):
                    info = infos[i] if infos[i] else {}
                    beat_idx = info.get("beat", 0)
                    beat_indices.append(beat_idx)
                    aux_targets = info.get("aux_targets")
                    # Convert any list-like aux targets (serialized arrays) back to numpy
                    if aux_targets is not None:
                        for k, v in list(aux_targets.items()):
                            # skip None or scalar
                            if v is None:
                                continue
                            if isinstance(v, list):
                                try:
                                    aux_targets[k] = np.asarray(v, dtype=np.float32)
                                except Exception:
                                    # leave as-is if conversion fails
                                    pass
                    aux_targets_list.append(aux_targets)
            else:
                # Threading mode - access env data directly
                for i in range(len(obs_list)):
                    env = self.vec_env.envs[i]
                    if env is not None and env.audio_state is not None:
                        beat_idx = int(env.current_beat)
                        beat_indices.append(beat_idx)
                        # Compute per-step auxiliary targets using available edited mel if present
                        try:
                            edited_mel = getattr(env.audio_state, 'target_mel', None)
                            aux = self.aux_target_computer.get_targets(
                                audio_id=env.audio_state.pair_id or f"idx_{i}",
                                beat_times=env.audio_state.beat_times,
                                beat_features=env.audio_state.beat_features,
                                beat_indices=np.array([beat_idx]),
                                edited_mel=edited_mel,
                            )
                            # Squeeze batch dim for single-sample aux targets
                            if aux is not None:
                                aux = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1 else v) for k, v in aux.items()}
                        except Exception:
                            aux = None
                        aux_targets_list.append(aux)
                    else:
                        beat_indices.append(0)
                        aux_targets_list.append(None)
            
            # Store in buffer
            for i in range(len(obs_list)):
                # Track section-level decisions
                if sizes_np[i] >= ActionSize.BAR.value:  # BAR or larger
                    section_decisions[i] += 1
                
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
                    beat_index=beat_indices[i],
                    aux_targets=aux_targets_list[i],
                )
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                # Collect per-step temporal penalty if present in info
                info = infos[i] if infos[i] else {}
                try:
                    tp = float(info.get('temporal_penalty', 0.0))
                except Exception:
                    tp = 0.0
                temporal_penalty_values.append(tp)
            
            # Handle episode ends
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    self.buffer.episode_rewards.append(episode_rewards[i])
                    self.buffer.episode_lengths.append(episode_lengths[i])
                    self.buffer.section_decisions.append(section_decisions[i])
                    
                    # Aggregate reward breakdown
                    info = infos[i] if infos[i] else {}
                    if "reward_breakdown" in info:
                        self._episode_reward_breakdowns.append(info["reward_breakdown"])
                    
                    logger.debug(f"Episode complete: len={episode_lengths[i]}, reward={episode_rewards[i]:.2f}")
                    
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    section_decisions[i] = 0
                    
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

        # Debug: report collected auxiliary targets summary
        try:
            aux_summary = self.buffer.get_auxiliary_targets_batch()
            if aux_summary:
                shapes = {k: v.shape for k, v in aux_summary.items()}
                #logger.info("Collected auxiliary targets summary: %s", shapes)
            else:
                logger.info("No auxiliary targets collected in buffer for this rollout")
        except Exception:
            logger.exception("Failed to summarize auxiliary targets in buffer")
        
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
            "episode_reward_breakdowns": self._episode_reward_breakdowns,
            # Per-rollout new episodes (useful for per-epoch diagnostics)
            "episode_rewards_epoch": np.array(self.episode_rewards[prev_episode_count:]) if len(self.episode_rewards) > prev_episode_count else np.array([]),
            # Per-rollout temporal penalty summary (per-step values aggregated)
            "temporal_penalty_mean_rollout": float(np.mean(temporal_penalty_values)) if temporal_penalty_values else 0.0,
            "temporal_penalty_max_rollout": float(np.max(temporal_penalty_values)) if temporal_penalty_values else 0.0,
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
        # Diagnostic: log which auxiliary target keys are present in the buffer
        try:
            aux_keys = list(aux_targets_np.keys())
            logger.debug("Auxiliary targets in buffer: %s", aux_keys)
            if self.training_logger is not None:
                # log count of aux target keys (helps detect empty aux batches)
                try:
                    self.training_logger.log_scalar("train/aux_targets_count", len(aux_keys), self.global_step)
                except Exception:
                    logger.debug("Failed to log aux_targets_count to training logger")
        except Exception:
            logger.exception("Failed to inspect auxiliary targets in buffer")
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
        
        # Track auxiliary loss breakdown
        aux_loss_breakdown = {}
        
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
                    # Safely index auxiliary targets: only use targets that match rollout sample count
                    batch_aux_targets = {}
                    for k, v in aux_targets.items():
                        try:
                            # v is a tensor on device; ensure first dimension equals n_samples
                            if getattr(v, 'shape', None) and v.shape[0] == n_samples:
                                # batch_indices is numpy array of indices into the rollout (0..n_samples-1)
                                idx_tensor = torch.from_numpy(batch_indices).long().to(self.device)
                                batch_aux_targets[k] = v[idx_tensor]
                        except Exception:
                            # If indexing fails, skip this auxiliary target
                            logger.debug(f"Skipping aux target {k} due to indexing mismatch")

                    if batch_aux_targets:
                        aux_loss, breakdown = self.agent.compute_auxiliary_loss(
                            batch_states, batch_aux_targets, self.current_epoch
                        )
                        aux_loss = torch.clamp(aux_loss, 0.0, 10.0)
                        # Track breakdown
                        for k, v in breakdown.items():
                            if k not in aux_loss_breakdown:
                                aux_loss_breakdown[k] = []
                            aux_loss_breakdown[k].append(v)
                        # Debug: log presence and sizes of batch_aux_targets
                        try:
                            sizes = {k: getattr(v, 'shape', None) for k, v in batch_aux_targets.items()}
                            logger.debug("Aux targets present for batch: %s", sizes)
                        except Exception:
                            logger.debug("Aux targets present (could not stringify shapes)")
                
                # Total loss
                loss = (
                    policy_loss +
                    ppo_config.value_loss_coeff * value_loss +
                    current_entropy_coeff * entropy_loss +
                    aux_loss
                )

                # Mixed BC loss: sample a mini-batch from bc dataset and add supervision
                if self.bc_states is not None and self.bc_loss_weight > 0.0:
                    try:
                        # Random sample indices
                        bc_n = self.bc_states.shape[0]
                        bc_bs = min(self.bc_batch_size, bc_n)
                        idx = np.random.randint(0, bc_n, size=bc_bs)
                        bc_states_batch = torch.from_numpy(self.bc_states[idx]).float().to(self.device)
                        bc_type = torch.from_numpy(self.bc_type_labels[idx]).long().to(self.device)
                        bc_size = torch.from_numpy(self.bc_size_labels[idx]).long().to(self.device)
                        bc_amount = torch.from_numpy(self.bc_amount_labels[idx]).long().to(self.device)

                        encoded = self.agent.policy_net.encoder(bc_states_batch)
                        type_logits = self.agent.policy_net.type_head(encoded)
                        # Use ground-truth type embedding to compute size/amount logits
                        type_embed = self.agent.policy_net.type_embedding(bc_type)
                        size_input = torch.cat([encoded, type_embed], dim=-1)
                        size_logits = self.agent.policy_net.size_head(size_input)
                        amount_input = torch.cat([encoded, type_embed], dim=-1)
                        amount_logits = self.agent.policy_net.amount_head(amount_input)

                        ce = nn.CrossEntropyLoss()
                        bc_loss_type = ce(type_logits, bc_type)
                        bc_loss_size = ce(size_logits, bc_size)
                        bc_loss_amount = ce(amount_logits, bc_amount)
                        bc_loss = bc_loss_type + bc_loss_size + bc_loss_amount
                        # Optional binary classifier BC loss
                        if self.bc_good_bad_labels is not None and self.agent.auxiliary_module is not None and getattr(self.agent.auxiliary_module, 'good_bad_classifier', None) is not None:
                            try:
                                bc_gb = torch.from_numpy(self.bc_good_bad_labels[idx]).float().to(self.device)
                                gb_preds = self.agent.auxiliary_module.good_bad_classifier(encoded).squeeze(-1)
                                bce = nn.BCEWithLogitsLoss()
                                bc_gb_loss = bce(gb_preds, bc_gb)
                                bc_loss = bc_loss + bc_gb_loss
                            except Exception:
                                logger.exception("BC mixed good/bad loss failed")
                        loss = loss + self.bc_loss_weight * bc_loss
                    except Exception:
                        logger.exception("BC mixed loss failed")
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_skipped += 1
                    continue
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clip (capture norm for logging)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.agent.policy_net.parameters()) + list(self.agent.value_net.parameters()),
                    ppo_config.max_grad_norm
                )
                # Log gradient norm to training logger (per update)
                try:
                    if self.training_logger is not None:
                        self.training_logger.log_scalar("grad_norm", float(grad_norm), self.global_step)
                except Exception:
                    logger.exception('Failed to log grad_norm')
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                # Always record auxiliary loss value (may be zero)
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
        
        # Aggregate episode reward breakdowns
        breakdowns = rollout_data.get("episode_reward_breakdowns", [])
        all_keys = set()
        if breakdowns:
            for bd in breakdowns:
                all_keys.update(bd.keys())
        
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": ppo_config.value_loss_coeff * np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "auxiliary_loss": np.mean(auxiliary_losses) if auxiliary_losses else 0.0,
            "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "total_loss": np.mean(policy_losses) + ppo_config.value_loss_coeff * np.mean(value_losses),
            "approx_kl": np.mean(approx_kl_divs) if approx_kl_divs else 0.0,
            "n_episodes": len(self.episode_rewards),
            "section_decisions_per_ep": np.mean(self.buffer.section_decisions) if self.buffer.section_decisions else 0.0,
        }
        
        # Add breakdowns to metrics
        if breakdowns:
            for k in all_keys:
                vals = [bd[k] for bd in breakdowns if k in bd and isinstance(bd[k], (int, float, np.floating, np.integer))]
                if vals:
                    metrics[f"breakdown_{k}"] = float(np.mean(vals))
        
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
            self.training_logger.log_scalar("section_decisions_per_ep", metrics["section_decisions_per_ep"], self.global_step)
            # Log auxiliary loss scalar
            try:
                self.training_logger.log_scalar("train/auxiliary_loss", metrics.get("auxiliary_loss", 0.0), self.global_step)
            except Exception:
                logger.exception("Failed to log auxiliary_loss to training logger")

            # If we collected auxiliary loss breakdowns per-batch, log their means under auxiliary/breakdown/
            try:
                if aux_loss_breakdown:
                    aux_break_means = {k: float(np.mean(v)) for k, v in aux_loss_breakdown.items() if v}
                    if aux_break_means:
                        self.training_logger.log_scalars("auxiliary/breakdown", aux_break_means, self.global_step)
                        # Also log selected auxiliary losses as top-level scalars for easy viewing
                        try:
                            for key in ("chroma_loss", "mel_reconstruction_loss", "reconstruction_loss", "good_bad_loss", "similarity_loss", "energy_loss", "phrase_loss"):
                                if key in aux_break_means:
                                    try:
                                        self.training_logger.log_scalar(f"train/{key}", aux_break_means[key], self.global_step)
                                    except Exception:
                                        logger.debug("Failed to log auxiliary scalar %s", key)
                        except Exception:
                            logger.exception("Failed to log top-level auxiliary scalars")
                        # Also log an aggregate unweighted auxiliary loss (sum of per-task losses)
                        try:
                            per_task_keys = [k for k in aux_break_means.keys() if k.endswith("_loss") and k != "total_auxiliary_loss"]
                            if per_task_keys:
                                aux_unweighted = float(sum(aux_break_means[k] for k in per_task_keys))
                                self.training_logger.log_scalar("train/auxiliary_unweighted", aux_unweighted, self.global_step)
                        except Exception:
                            logger.exception("Failed to compute auxiliary_unweighted")
                else:
                    # No breakdowns collected this update — helpful diagnostic for debugging missing aux logs
                    logger.debug("No auxiliary loss breakdowns collected in this update")
                    try:
                        if self.training_logger is not None:
                            self.training_logger.log_scalar("train/aux_loss_breakdown_present", 0.0, self.global_step)
                    except Exception:
                        logger.debug("Failed to log aux_loss_breakdown_present scalar")
            except Exception:
                logger.exception("Failed to log auxiliary breakdowns")

            # Per-epoch episode reward diagnostics (from collect_rollouts)
            try:
                ep_rewards_epoch = rollout_data.get("episode_rewards_epoch")
                if ep_rewards_epoch is not None:
                    n_ep = int(len(ep_rewards_epoch))
                    mean_ep = float(ep_rewards_epoch.mean()) if n_ep > 0 else 0.0
                    std_ep = float(ep_rewards_epoch.std()) if n_ep > 0 else 0.0
                    # Log counts and statistics to TensorBoard for debugging
                    try:
                        self.training_logger.log_scalar("train/episodes_this_rollout", n_ep, self.global_step)
                        self.training_logger.log_scalar("train/episode_reward_epoch_mean", mean_ep, self.global_step)
                        self.training_logger.log_scalar("train/episode_reward_epoch_std", std_ep, self.global_step)
                    except Exception:
                        logger.debug("Failed to log per-epoch episode reward diagnostics to training logger")
                    logger.info(f"Epoch diagnostic: new_episodes={n_ep} mean_ep_reward={mean_ep:.2f} std={std_ep:.2f}")
                    # Also log per-rollout temporal penalty summaries if present
                    try:
                        tp_mean = float(rollout_data.get('temporal_penalty_mean_rollout', 0.0))
                        tp_max = float(rollout_data.get('temporal_penalty_max_rollout', 0.0))
                        self.training_logger.log_scalar("train/temporal_penalty_mean_rollout", tp_mean, self.global_step)
                        self.training_logger.log_scalar("train/temporal_penalty_max_rollout", tp_max, self.global_step)
                    except Exception:
                        logger.debug("Failed to log temporal penalty rollout summaries")
            except Exception:
                logger.exception("Failed to compute per-epoch episode reward diagnostics")
            
            # Log reward breakdowns and counters
            for k in all_keys:
                if f"breakdown_{k}" not in metrics:
                    continue
                val = metrics[f"breakdown_{k}"]
                # Route action counters to a separate counters/ section
                if k in ("n_actions", "n_creative", "n_keep_ratio"):
                    self.training_logger.log_scalar(f"counters/{k}", val, self.global_step)
                else:
                    # Log all other reward/penalty components under rewards/
                    self.training_logger.log_scalar(f"rewards/{k}", val, self.global_step)

            # Temporal penalty specific logging: total per-episode, events per-episode, and per-event avg
            try:
                if breakdowns:
                    tp_totals = [bd.get('temporal_penalty_total') for bd in breakdowns if 'temporal_penalty_total' in bd]
                    tp_counts = [bd.get('temporal_penalty_count') for bd in breakdowns if 'temporal_penalty_count' in bd]
                    if tp_totals:
                        self.training_logger.log_scalar("rewards/temporal_penalty_total", float(np.mean(tp_totals)), self.global_step)
                    if tp_counts:
                        self.training_logger.log_scalar("train/temporal_penalty_events_per_ep", float(np.mean(tp_counts)), self.global_step)
                        # Compute per-event average across episodes that had events
                        per_event = [ (bd.get('temporal_penalty_total', 0.0) / bd.get('temporal_penalty_count'))
                                      for bd in breakdowns if bd.get('temporal_penalty_count', 0) > 0 ]
                        if per_event:
                            self.training_logger.log_scalar("rewards/temporal_penalty_per_event", float(np.mean(per_event)), self.global_step)
            except Exception:
                logger.exception("Failed to log temporal penalty metrics")
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        # Prefer live agent weights, but fall back to any pending checkpoint saved earlier
        policy_sd = None
        value_sd = None
        aux_sd = None
        if self.agent:
            policy_sd = self.agent.policy_net.state_dict()
            value_sd = self.agent.value_net.state_dict()
            aux_sd = self.agent.auxiliary_module.state_dict() if self.agent.auxiliary_module else None
        else:
            # If agent not initialized, try to reuse weights from deferred checkpoint
            if self._pending_checkpoint is not None:
                policy_sd = self._pending_checkpoint.get('policy_state_dict')
                value_sd = self._pending_checkpoint.get('value_state_dict')
                aux_sd = self._pending_checkpoint.get('auxiliary_state_dict')

        checkpoint = {
            "version": "factored",
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "current_epoch": self.current_epoch,
            "policy_state_dict": policy_sd,
            "value_state_dict": value_sd,
            "auxiliary_state_dict": aux_sd,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "auxiliary_optimizer": self.auxiliary_optimizer.state_dict() if self.auxiliary_optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "config": self.config,
            "bc_mixed_loaded": True if self.bc_states is not None else False,
            "bc_loss_weight": float(self.bc_loss_weight),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        version = checkpoint.get("version", "v1")
        if version not in ("factored",):
            logger.warning(f"Loading non-factored checkpoint (version={version}) - encoder weights only")
        
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
            try:
                res = self.agent.auxiliary_module.load_state_dict(checkpoint["auxiliary_state_dict"], strict=False)
                # `load_state_dict` returns a NamedTuple with missing_keys/unexpected_keys when strict=False
                try:
                    missing = getattr(res, 'missing_keys', None)
                    unexpected = getattr(res, 'unexpected_keys', None)
                    logger.info("Loaded auxiliary module (strict=False). missing=%s unexpected=%s", missing, unexpected)
                except Exception:
                    logger.info("Loaded auxiliary module (strict=False)")
            except TypeError:
                # Older torch versions may not accept strict kwarg in same way; fall back to permissive load
                try:
                    self.agent.auxiliary_module.load_state_dict(checkpoint["auxiliary_state_dict"])
                    logger.info("Loaded auxiliary module")
                except RuntimeError as e:
                    logger.warning("Auxiliary module load_state_dict failed; attempting non-strict load: %s", e)
                    # Try a non-strict manual load by iterating keys
                    state = checkpoint["auxiliary_state_dict"]
                    own_state = self.agent.auxiliary_module.state_dict()
                    for name, param in state.items():
                        if name in own_state:
                            try:
                                own_state[name].copy_(param)
                            except Exception:
                                logger.debug("Failed to copy param %s into auxiliary module", name)
                    logger.info("Partially loaded auxiliary module (manual copy)")
        if self.optimizer and checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer")
        if self.auxiliary_optimizer and checkpoint.get("auxiliary_optimizer"):
            self.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
            logger.info("Loaded auxiliary optimizer")
        if self.scheduler and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.ppo.learning_rate

    def load_bc_dataset(self, npz_path: str, weight: float = 0.0, batch_size: int = 64):
        """Load BC NPZ (states + type/size/amount labels) into trainer memory."""
        try:
            import numpy as _np
            data = _np.load(npz_path, allow_pickle=True)
            self.bc_states = data['states']
            self.bc_type_labels = data['type_labels']
            self.bc_size_labels = data['size_labels']
            self.bc_amount_labels = data['amount_labels']
            # optional binary good/bad labels (per-beat)
            if 'good_bad' in data:
                try:
                    self.bc_good_bad_labels = data['good_bad']
                    logger.info(f"Loaded BC good_bad labels (n={self.bc_good_bad_labels.shape[0]})")
                except Exception:
                    self.bc_good_bad_labels = None
            self.bc_loss_weight = float(weight)
            self.bc_batch_size = int(batch_size)
            logger.info(f"Loaded BC dataset: {npz_path} (n={self.bc_states.shape[0]})")
        except Exception:
            logger.exception(f"Failed to load BC dataset: {npz_path}")

    def bc_pretrain(self, epochs: int = 3, lr: float = 1e-4):
        """Simple supervised pretraining of policy_net on BC dataset.

        Trains only policy parameters (type/size/amount heads + encoder).
        """
        if self.bc_states is None:
            logger.warning("No BC dataset loaded for pretraining")
            return

        self._init_agent(input_dim=self.bc_states.shape[1])
        self.agent.train()
        optimizer = optim.Adam(self.agent.policy_net.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        n = self.bc_states.shape[0]
        bs = min(128, n)

        # Validation split for BC pretraining
        val_frac = getattr(self.config.training, 'bc_val_frac', 0.1)
        val_n = max(1, int(n * val_frac)) if n > 1 else 0
        perm0 = np.random.permutation(n)
        val_idx = perm0[:val_n] if val_n > 0 else np.array([], dtype=np.int64)
        train_idx_all = perm0[val_n:]

        for ep in range(epochs):
            perm = np.random.permutation(train_idx_all)
            losses = []
            for start in range(0, len(perm), bs):
                end = min(start + bs, len(perm))
                idx = perm[start:end]
                batch_states = torch.from_numpy(self.bc_states[idx]).float().to(self.device)
                batch_type = torch.from_numpy(self.bc_type_labels[idx]).long().to(self.device)
                batch_size_lbl = torch.from_numpy(self.bc_size_labels[idx]).long().to(self.device)
                batch_amount = torch.from_numpy(self.bc_amount_labels[idx]).long().to(self.device)

                optimizer.zero_grad()
                encoded = self.agent.policy_net.encoder(batch_states)
                type_logits = self.agent.policy_net.type_head(encoded)
                type_embed = self.agent.policy_net.type_embedding(batch_type)
                size_input = torch.cat([encoded, type_embed], dim=-1)
                size_logits = self.agent.policy_net.size_head(size_input)
                amount_input = torch.cat([encoded, type_embed], dim=-1)
                amount_logits = self.agent.policy_net.amount_head(amount_input)

                loss_type = ce(type_logits, batch_type)
                loss_size = ce(size_logits, batch_size_lbl)
                loss_amount = ce(amount_logits, batch_amount)
                loss = loss_type + loss_size + loss_amount
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            train_loss = float(np.mean(losses)) if losses else 0.0

            # Validation evaluation
            val_loss = 0.0
            if val_n > 0:
                self.agent.eval()
                with torch.no_grad():
                    val_losses = []
                    for start in range(0, len(val_idx), bs):
                        end = min(start + bs, len(val_idx))
                        idx = val_idx[start:end]
                        batch_states = torch.from_numpy(self.bc_states[idx]).float().to(self.device)
                        batch_type = torch.from_numpy(self.bc_type_labels[idx]).long().to(self.device)
                        batch_size_lbl = torch.from_numpy(self.bc_size_labels[idx]).long().to(self.device)
                        batch_amount = torch.from_numpy(self.bc_amount_labels[idx]).long().to(self.device)

                        encoded = self.agent.policy_net.encoder(batch_states)
                        type_logits = self.agent.policy_net.type_head(encoded)
                        type_embed = self.agent.policy_net.type_embedding(batch_type)
                        size_input = torch.cat([encoded, type_embed], dim=-1)
                        size_logits = self.agent.policy_net.size_head(size_input)
                        amount_input = torch.cat([encoded, type_embed], dim=-1)
                        amount_logits = self.agent.policy_net.amount_head(amount_input)

                        v_loss = float(ce(type_logits, batch_type).item() + ce(size_logits, batch_size_lbl).item() + ce(amount_logits, batch_amount).item())
                        val_losses.append(v_loss)
                    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                self.agent.train()

            logger.info(f"BC pretrain epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if self.training_logger is not None:
                self.training_logger.log_scalar("bc/train_loss", train_loss, ep+1)
                self.training_logger.log_scalar("bc/val_loss", val_loss, ep+1)
        # Optional: pretrain binary good/bad classifier on BC labels
        if self.bc_good_bad_labels is not None:
            if self.agent is None:
                self._init_agent(input_dim=self.bc_states.shape[1])
            if self.agent.auxiliary_module is not None and getattr(self.agent.auxiliary_module, 'good_bad_classifier', None) is not None:
                logger.info("Starting BC pretrain for good/bad classifier")
                gb_optimizer = optim.Adam(self.agent.auxiliary_module.good_bad_classifier.parameters(), lr=lr)
                bce = nn.BCEWithLogitsLoss()
                n = self.bc_states.shape[0]
                bs = min(128, n)
                for ep in range(epochs):
                    perm = np.random.permutation(n)
                    losses_gb = []
                    for start in range(0, n, bs):
                        end = min(start + bs, n)
                        idx = perm[start:end]
                        batch_states = torch.from_numpy(self.bc_states[idx]).float().to(self.device)
                        batch_gb = torch.from_numpy(self.bc_good_bad_labels[idx]).float().to(self.device)
                        gb_optimizer.zero_grad()
                        with torch.no_grad():
                            encoded = self.agent.policy_net.encoder(batch_states)
                        preds = self.agent.auxiliary_module.good_bad_classifier(encoded)
                        preds = preds.squeeze(-1)
                        loss_gb = bce(preds, batch_gb)
                        loss_gb.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent.auxiliary_module.good_bad_classifier.parameters(), 1.0)
                        gb_optimizer.step()
                        losses_gb.append(float(loss_gb.item()))
                    logger.info(f"BC classifier pretrain epoch {ep+1}/{epochs} loss={np.mean(losses_gb):.4f}")
            else:
                logger.info("No auxiliary good/bad classifier available for BC pretrain")


def train(
    config: Config,
    data_dir: str,
    n_epochs: int = 1000,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    checkpoint_path: Optional[str] = None,
    bc_pretrain_npz: Optional[str] = None,
    bc_pretrain_epochs: int = 0,
    bc_pretrain_lr: float = 1e-4,
    bc_mixed_npz: Optional[str] = None,
    bc_mixed_weight: float = 0.0,
    bc_mixed_batch: int = 64,
    use_subprocess: bool = False,
    learned_reward_model: Optional[Any] = None,
    max_beats: Optional[int] = None,
    val_audio: Optional[str] = None,
):
    """Main training function.
    
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
    
    mb_display = max_beats if max_beats is not None else "unlimited"
    logger.info(f"Max beats per sample: {mb_display}")
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
        use_augmentation=True,
    )
    if len(dataset) == 0:
        logger.error(f"No data found in {data_dir}")
        return
    logger.info(f"Loaded {len(dataset)} samples (cache: {cache_dir})")
    
    # Load learned reward model if enabled
    if config.reward.use_learned_rewards and learned_reward_model is None:
        reward_integration = LearnedRewardIntegration(config)
        if reward_integration.load_model():
            learned_reward_model = reward_integration.model
            logger.info("Learned reward model loaded for RLHF training")
        else:
            logger.warning("Could not load learned reward model - using dense rewards only")
            config.reward.use_learned_rewards = False
    
    # Trainer
    trainer = PPOTrainer(
        config=config,
        n_envs=n_envs,
        total_epochs=n_epochs,
        use_subprocess=use_subprocess,
        learned_reward_model=learned_reward_model,
    )

    # Attach a TrainingLogger so TensorBoard/W&B outputs are created under config.training.log_dir
    try:
        trainer.training_logger = create_logger(config)
        logger.info(f"Training logger initialized at: {trainer.training_logger.run_dir}")
    except Exception:
        logger.exception("Failed to initialize TrainingLogger; continuing without it")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")

    # Load BC datasets if provided
    if bc_mixed_npz:
        trainer.load_bc_dataset(bc_mixed_npz, weight=bc_mixed_weight, batch_size=bc_mixed_batch)
        logger.info(f"Loaded mixed BC dataset: {bc_mixed_npz} (weight={bc_mixed_weight})")

    # Run BC pretraining if requested (warm-start policy)
    if bc_pretrain_npz and bc_pretrain_epochs > 0:
        if Path(bc_pretrain_npz).exists():
            trainer.load_bc_dataset(bc_pretrain_npz)
            trainer.bc_pretrain(epochs=bc_pretrain_epochs, lr=bc_pretrain_lr)
            logger.info(f"Completed BC pretraining from {bc_pretrain_npz} for {bc_pretrain_epochs} epochs")
        else:
            logger.warning(f"BC pretrain file not found: {bc_pretrain_npz}")
    
    # Training loop
    save_dir = Path(config.training.save_dir)
    # Default maximum beats used when CLI/config does not set `max_beats`.
    # Set to None to allow unlimited when user omits the CLI flag.
    DEFAULT_MAXBEATS = None
    # `max_beats` is an optional train() parameter; if provided it overrides the default.
    effective_global_max = max_beats if max_beats is not None else DEFAULT_MAXBEATS
    effective_display = effective_global_max if effective_global_max is not None else "unlimited"
    logger.info(f"Max beats per sample: {effective_display}")
    
    for epoch in range(trainer.current_epoch, n_epochs):
        epoch_start = time.time()
        
        # === Curriculum parameters ===
        # Start with short segments, gradually increase
        initial_short_beats = 500
        final_short_beats = 2000
        initial_short_prob = 1.0  # 100% short at start
        final_short_prob = 0.2   # 20% short at end
        curriculum_steps = 20000  # Number of epochs to anneal over
        
        progress = min(epoch / curriculum_steps, 1.0)
        short_prob = initial_short_prob * (1 - progress) + final_short_prob * progress
        short_max_beats = int(initial_short_beats + (final_short_beats - initial_short_beats) * progress)
        
        # Log curriculum parameters
        if trainer.training_logger is not None:
            trainer.training_logger.log_scalar("curriculum/short_prob", short_prob, trainer.global_step)
            trainer.training_logger.log_scalar("curriculum/short_max_beats", short_max_beats, trainer.global_step)
        
        # Sample audio states
        data_start = time.time()
        audio_states = []
        idxs = np.random.randint(0, len(dataset), size=n_envs)
        
        for idx in idxs:
            # Determine per-sample max beats based on curriculum and any global override
            if random.random() < short_prob:
                per_sample_max = min(effective_global_max, short_max_beats) if effective_global_max is not None else short_max_beats
            else:
                per_sample_max = effective_global_max
            
            sample = dataset[idx]
            raw_data = sample.get('raw', sample)
            
            beat_times = raw_data["beat_times"]
            beat_features = raw_data["beat_features"]
            
            # Convert to numpy if tensor
            if hasattr(beat_times, 'numpy'):
                beat_times = beat_times.numpy()
            if hasattr(beat_features, 'numpy'):
                beat_features = beat_features.numpy()
            
            # Limit to per-sample max beats (if set)
            if per_sample_max is not None and len(beat_times) > per_sample_max:
                start_idx = np.random.randint(0, len(beat_times) - per_sample_max)
                end_idx = start_idx + per_sample_max
                beat_times = beat_times[start_idx:end_idx]
                beat_features = beat_features[start_idx:end_idx]
                beat_times = beat_times - beat_times[0]  # Reset to 0
            
            tempo = raw_data.get("tempo", 120)
            if hasattr(tempo, 'item'):
                tempo = tempo.item()
            
            audio_state = AudioState(
                beat_index=0,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=tempo,
                raw_audio=raw_data.get("audio"),
                sample_rate=raw_data.get("sample_rate", 22050),
            )
            # Attach pair id and target mel from paired dataset when available
            if isinstance(sample, dict):
                pair_id = sample.get("pair_id")
                if pair_id:
                    audio_state.pair_id = pair_id
                edited = sample.get("edited")
                if edited and "mel" in edited:
                    mel = edited["mel"]
                    # Convert to numpy if tensor
                    if hasattr(mel, 'numpy'):
                        mel = mel.numpy()
                    audio_state.target_mel = mel
            audio_states.append(audio_state)
        # Diagnostics: log sampled beat counts for this epoch (helps detect curriculum / max_beats effects)
        try:
            beats_counts = [len(s.beat_times) for s in audio_states]
            if trainer.training_logger is not None:
                trainer.training_logger.log_scalar("train/avg_sample_beats", float(np.mean(beats_counts)), trainer.global_step)
                trainer.training_logger.log_scalar("train/min_sample_beats", float(np.min(beats_counts)), trainer.global_step)
                trainer.training_logger.log_scalar("train/max_sample_beats", float(np.max(beats_counts)), trainer.global_step)
        except Exception:
            logger.exception("Failed to log sampled beat counts")

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

        # Diagnostic: always emit a minimal per-epoch line to ensure visibility
        try:
            logger.info(f"Epoch {epoch:5d} | Reward: {metrics.get('episode_reward', 0.0):7.2f} | Time: {epoch_time:.1f}s (D:{data_time:.2f}s R:{rollout_time:.1f}s U:{update_time:.1f}s)")
        except Exception:
            logger.exception("Failed to emit epoch diagnostic log")
        
        # Log
        if epoch % 10 == 0:
            aux_loss = metrics.get('auxiliary_loss', 0.0)
            section_dec = metrics.get('section_decisions_per_ep', 0)
            logger.info(
                f"Epoch {epoch:5d} | "
                f"Reward: {metrics['episode_reward']:7.2f} | "
                f"Loss: {metrics['total_loss']:.4f} (P:{metrics['policy_loss']:.4f} V:{metrics['value_loss']:.4f} A:{aux_loss:.4f}) | "
                f"KL: {metrics['approx_kl']:.4f} | "
                f"Sec: {section_dec:.1f} | "
                f"Time: {epoch_time:.1f}s (D:{data_time:.2f}s R:{rollout_time:.1f}s U:{update_time:.1f}s)"
            )
            
            # Log reward breakdown
            breakdown_keys = [k for k in metrics.keys() if k.startswith("breakdown_")]
            if breakdown_keys:
                exclude_suffixes = ("n_actions", "n_creative", "n_keep_ratio", "learned")
                breakdown = [
                    f"{k[10:]}: {metrics[k]:.2f}"
                    for k in breakdown_keys
                    if not any(k.endswith(s) for s in exclude_suffixes)
                ]
                if breakdown:
                    logger.info("  Reward breakdown: " + " | ".join(breakdown))
        
        # Save checkpoints
        checkpoint_interval = getattr(config.training, 'checkpoint_interval', 50)
        if epoch % checkpoint_interval == 0 and epoch > 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(str(ckpt_path))
            # Deterministic evaluation on provided validation audio (cheap diagnostic)
            if val_audio:
                try:
                    audio, sr, audio_state = load_and_process_audio(
                        val_audio,
                        config=config,
                        max_beats=getattr(config.training, 'eval_max_beats', 0),
                        cache_dir=config.data.cache_dir,
                    )
                    actions, action_names, total_reward, aux_preds, final_keep_ratio = run_inference(
                        trainer.agent, config, audio_state, deterministic=True, verbose=False
                    )
                    edited = create_edited_audio(audio, sr, audio_state.beat_times, actions)
                    orig_dur = len(audio) / sr
                    edited_dur = len(edited) / sr
                    edited_pct = 100.0 * edited_dur / orig_dur if orig_dur > 0 else 0.0
                    logger.info(f"Eval deterministic: reward={total_reward:.4f} per-beat-keep_ratio={final_keep_ratio:.3f} edited_pct={edited_pct:.1f}%")
                    if trainer.training_logger is not None:
                        trainer.training_logger.log_scalar("eval/edited_pct", float(edited_pct), trainer.global_step)
                        trainer.training_logger.log_scalar("eval/per-beat-keep_ratio", float(final_keep_ratio), trainer.global_step)
                except Exception:
                    logger.exception("Deterministic eval after checkpoint failed")
        
        # Best model
        if metrics["episode_reward"] > trainer.best_reward:
            trainer.best_reward = metrics["episode_reward"]
            best_path = save_dir / "best.pt"
            trainer.save_checkpoint(str(best_path))
            logger.info(f"New best reward: {trainer.best_reward:.2f}")
            # Run deterministic eval on best model if validation audio provided
            if val_audio:
                try:
                    audio, sr, audio_state = load_and_process_audio(
                        val_audio,
                        config=config,
                        max_beats=getattr(config.training, 'eval_max_beats', 0),
                        cache_dir=config.data.cache_dir,
                    )
                    actions, action_names, total_reward, aux_preds, final_keep_ratio = run_inference(
                        trainer.agent, config, audio_state, deterministic=True, verbose=False
                    )
                    edited = create_edited_audio(audio, sr, audio_state.beat_times, actions)
                    orig_dur = len(audio) / sr
                    edited_dur = len(edited) / sr
                    edited_pct = 100.0 * edited_dur / orig_dur if orig_dur > 0 else 0.0
                    logger.info(f"Eval deterministic (best): reward={total_reward:.4f} per-beat-keep_ratio={final_keep_ratio:.3f} edited_pct={edited_pct:.1f}%")
                    if trainer.training_logger is not None:
                        trainer.training_logger.log_scalar("eval/edited_pct", float(edited_pct), trainer.global_step)
                        trainer.training_logger.log_scalar("eval/per-beat-keep_ratio", float(final_keep_ratio), trainer.global_step)
                except Exception:
                    logger.exception("Deterministic eval after best checkpoint failed")
    
    # Final save
    final_path = save_dir / "final.pt"
    trainer.save_checkpoint(str(final_path))
    logger.info(f"Training complete. Final checkpoint: {final_path}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL audio editor")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Training data directory")
    parser.add_argument("--save_dir", type=str, default="models", help="Save directory")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--n_envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=512, help="Steps per epoch")
    parser.add_argument("--max_beats", type=int, default=None, help="Optional max beats per sample (omit for unlimited)")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--subprocess", action="store_true", help="Use subprocess parallelism")
    parser.add_argument("--use_learned_rewards", action="store_true", help="Enable learned reward model (RLHF)")
    parser.add_argument("--no_lr_decay", action="store_true", help="Disable learning rate decay")
    parser.add_argument("--bc_pretrain_npz", type=str, default=None, help="NPZ file for BC pretraining (states+labels)")
    parser.add_argument("--bc_pretrain_epochs", type=int, default=0, help="Epochs for BC pretraining")
    parser.add_argument("--bc_pretrain_lr", type=float, default=1e-4, help="LR for BC pretraining")
    parser.add_argument("--bc_mixed_npz", type=str, default=None, help="NPZ file for mixed BC supervised loss during PPO")
    parser.add_argument("--bc_mixed_weight", type=float, default=0.0, help="Weight for mixed BC loss added to PPO updates")
    parser.add_argument("--bc_mixed_batch", type=int, default=64, help="Batch size for mixed BC sampling during PPO updates")
    
    args = parser.parse_args()
    
    # Config
    config = get_default_config()
    config.training.save_dir = args.save_dir
    config.ppo.learning_rate = args.lr
    config.ppo.batch_size = args.batch_size
    
    if args.no_lr_decay:
        config.ppo.lr_decay = False
        logger.info("Learning rate decay DISABLED")
    
    if args.use_learned_rewards:
        config.reward.use_learned_rewards = True
        logger.info("Learned rewards ENABLED (RLHF mode)")
    
    train(
        config=config,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        checkpoint_path=args.checkpoint,
        bc_pretrain_npz=args.bc_pretrain_npz,
        bc_pretrain_epochs=args.bc_pretrain_epochs,
        bc_pretrain_lr=args.bc_pretrain_lr,
        bc_mixed_npz=args.bc_mixed_npz,
        bc_mixed_weight=args.bc_mixed_weight,
        bc_mixed_batch=args.bc_mixed_batch,
        use_subprocess=args.subprocess,
        max_beats=args.max_beats,
    )


if __name__ == "__main__":
    main()

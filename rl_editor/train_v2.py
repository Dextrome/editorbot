"""V2 Training Script - Section-level actions with episode rewards.

Based on train_parallel.py with these key changes:
1. Uses AudioEditingEnvV2 with section-level actions (16 actions vs 9)
2. Episode-level rewards (minimize step rewards, maximize end-of-episode quality)
3. Can load V1 weights for transfer learning (encoder transfers, policy head reinitialized)
4. Full action space enabled by default (not restricted to KEEP/CUT)

Key V2 Actions:
- Beat-level: KEEP_BEAT, CUT_BEAT
- Bar-level (4 beats): KEEP_BAR, CUT_BAR
- Phrase-level (8 beats): KEEP_PHRASE, CUT_PHRASE
- Looping: LOOP_BEAT, LOOP_BAR, LOOP_PHRASE
- Reordering: REORDER_BEAT, REORDER_BAR, REORDER_PHRASE
- Navigation: JUMP_BACK_4, JUMP_BACK_8
- Transitions: MARK_SOFT_TRANSITION, MARK_HARD_CUT
"""

# Set multiprocessing start method for Windows compatibility (must be at very top)
import sys
import multiprocessing as mp
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


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
from .environment_v2 import AudioEditingEnvV2
from .actions_v2 import ActionSpaceV2, ActionTypeV2
from .agent import Agent
from .state import AudioState, EditHistory
from .reward import compute_trajectory_return
from .logging_utils import TrainingLogger, create_logger
from .learned_reward_integration import LearnedRewardIntegration
from .auxiliary_tasks import AuxiliaryConfig, AuxiliaryTargetComputer, compute_auxiliary_targets

logger = logging.getLogger(__name__)

# Set multiprocessing start method for Windows compatibility
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


def get_lr_scheduler(optimizer: optim.Optimizer, config: Config, total_epochs: int) -> Optional[LambdaLR]:
    """Create learning rate scheduler based on config."""
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
class RolloutBufferV2:
    """Buffer for storing rollout data with V2 action masks and auxiliary targets."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    masks: List[np.ndarray]
    # V2 specific: track episode-level info
    episode_rewards: List[float]
    episode_lengths: List[int]
    section_decisions: List[int]  # How many section-level actions taken
    # Auxiliary task targets
    beat_indices: List[int]  # Current beat index for each step
    auxiliary_targets: Dict[str, List[np.ndarray]]  # task_name -> list of targets
    
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
        }
    
    def add(self, state, action, reward, value, log_prob, done, mask, beat_index=0, aux_targets=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
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


class VectorizedEnvV2Wrapper:
    """Wrapper for running multiple V2 environments in parallel."""
    
    def __init__(self, config: Config, n_envs: int = 4, learned_reward_model: Optional[Any] = None):
        self.config = config
        self.n_envs = n_envs
        self._executor = ThreadPoolExecutor(max_workers=n_envs)  # Persistent thread pool
        self.envs: List[Optional[AudioEditingEnvV2]] = [None] * n_envs
        self.audio_states: List[Optional[AudioState]] = [None] * n_envs
        self.learned_reward_model = learned_reward_model
        
    def set_learned_reward_model(self, model: Any) -> None:
        """Set the learned reward model for all environments."""
        self.learned_reward_model = model
        for env in self.envs:
            if env is not None and hasattr(env, 'set_learned_reward_model'):
                env.set_learned_reward_model(model)
        
    def set_audio_states(self, audio_states: List[AudioState]):
        """Set audio states for all environments."""
        for i, state in enumerate(audio_states[:self.n_envs]):
            self.audio_states[i] = state
            self.envs[i] = AudioEditingEnvV2(self.config, audio_state=state)
    
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
        """Step all environments with given actions using thread pool for parallelism.
        
        V2 uses episode-level rewards primarily - step rewards are minimal.
        """
        def step_env(args):
            env, action = args
            if env is not None:
                return env.step(action)
            return None, 0.0, False, False, {}
        
        # Parallel step using threads (GIL-friendly for I/O-bound env stepping)
        results = list(self._executor.map(step_env, zip(self.envs, actions)))
        
        next_obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        terminated_list = [r[2] for r in results]
        truncated_list = [r[3] for r in results]
        info_list = [r[4] for r in results]
        
        return next_obs_list, reward_list, terminated_list, truncated_list, info_list
    
    def get_action_masks(self) -> List[np.ndarray]:
        """Get action masks for all V2 environments using thread pool."""
        def get_mask(env):
            if env is not None:
                return env.get_action_mask()
            return np.ones(ActionSpaceV2.N_ACTIONS, dtype=bool)
        
        masks = list(self._executor.map(get_mask, self.envs))
        return masks
    
    def reset_env(self, i: int) -> Tuple[np.ndarray, dict]:
        """Reset a single environment."""
        if self.envs[i] is not None:
            return self.envs[i].reset()
        return None, {}


class ParallelPPOTrainerV2:
    """PPO trainer for V2 environment with section-level actions and episode rewards."""
    
    def __init__(
        self,
        config: Config,
        n_envs: int = 4,
        prefetch_factor: int = 2,
        total_epochs: int = 1000,
        learned_reward_model: Optional[Any] = None,
        use_subprocess: bool = False,
    ):
        self.config = config
        self.n_envs = n_envs
        self.prefetch_factor = prefetch_factor
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.learned_reward_model = learned_reward_model
        self.use_subprocess = use_subprocess
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        # Entropy coefficient (decay over training to reduce exploration as policy improves)
        self.initial_entropy_coeff = config.ppo.entropy_coeff
        self.entropy_coeff_min = getattr(config.ppo, 'entropy_coeff_min', 0.01)
        self.entropy_coeff_decay = getattr(config.ppo, 'entropy_coeff_decay', True)
        
        # Vectorized V2 environments (thread-based or subprocess-based)
        if use_subprocess:
            from .subprocess_vec_env import make_subprocess_vec_env
            self.vec_env = make_subprocess_vec_env(config, n_envs, learned_reward_model=learned_reward_model)
            logger.info(f"Using subprocess-based parallel environments ({n_envs} processes)")
        else:
            self.vec_env = VectorizedEnvV2Wrapper(config, n_envs, learned_reward_model=learned_reward_model)
            logger.info(f"Using thread-based parallel environments ({n_envs} threads)")
        
        # Agent (initialized later when we know input dimensions)
        self.agent: Optional[Agent] = None
        self.policy_optimizer: Optional[optim.Adam] = None
        self.value_optimizer: Optional[optim.Adam] = None
        self.policy_scheduler: Optional[LambdaLR] = None
        self.value_scheduler: Optional[LambdaLR] = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.ppo.use_mixed_precision and torch.cuda.is_available())
        
        # Rollout buffer
        self.buffer = RolloutBufferV2()
        
        # Tracking
        self.global_step = 0
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Target KL for early stopping
        self.target_kl = getattr(config.ppo, 'target_kl', 0.01)
        
        # Auxiliary task target computer
        self.aux_config = AuxiliaryConfig()
        self.aux_target_computer = AuxiliaryTargetComputer(self.aux_config)
        
        # Logging
        self.training_logger: Optional[TrainingLogger] = None
        if config.training.use_tensorboard or config.training.use_wandb:
            self.training_logger = create_logger(config)
        
        # Create directories
        Path(config.training.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ParallelPPOTrainerV2 with {n_envs} environments on {self.device}")
        logger.info(f"V2 Action space: {ActionSpaceV2.N_ACTIONS} actions (section-level)")
        logger.info(f"Target KL for early stopping: {self.target_kl}")
        if config.ppo.lr_decay:
            logger.info(f"LR decay enabled: {config.ppo.lr_decay_type}, warmup={config.ppo.lr_warmup_epochs}")
    
    def _init_agent(self, input_dim: int, n_actions: int, beat_feature_dim: int = 121):
        """Initialize agent and optimizers with auxiliary tasks."""
        self.agent = Agent(
            self.config, 
            input_dim, 
            n_actions,
            beat_feature_dim=beat_feature_dim,
            use_auxiliary_tasks=True,
        )
        
        # Policy optimizer (includes encoder which is shared with auxiliary)
        policy_lr = self.config.ppo.learning_rate
        self.policy_optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=policy_lr
        )
        
        # Value optimizer (same LR as policy)
        self.value_optimizer = optim.Adam(
            self.agent.value_net.parameters(),
            lr=policy_lr
        )
        logger.info(f"Policy/Value LR: {policy_lr:.2e}")
        
        # Auxiliary task optimizer (separate to allow different learning rates)
        self.auxiliary_optimizer = None
        if self.agent.auxiliary_module is not None:
            self.auxiliary_optimizer = optim.Adam(
                self.agent.auxiliary_module.parameters(),
                lr=self.config.ppo.learning_rate * 2.0,  # Slightly higher LR for auxiliary tasks
            )
            logger.info("Auxiliary task optimizer initialized")
        
        # Initialize LR schedulers
        self.policy_scheduler = get_lr_scheduler(
            self.policy_optimizer, self.config, self.total_epochs
        )
        self.value_scheduler = get_lr_scheduler(
            self.value_optimizer, self.config, self.total_epochs
        )
        
        if self.policy_scheduler:
            logger.info(f"LR schedulers initialized for {self.total_epochs} epochs")
    
    def load_v1_weights(self, v1_checkpoint_path: str, strict_encoder: bool = True) -> bool:
        """Load V1 weights for transfer learning.
        
        The encoder (feature extraction) can transfer directly.
        The value network can transfer directly.
        The policy head must be reinitialized for the new action space.
        
        Args:
            v1_checkpoint_path: Path to V1 checkpoint
            strict_encoder: If True, fail if encoder weights don't match exactly
            
        Returns:
            True if weights were loaded successfully
        """
        if not Path(v1_checkpoint_path).exists():
            logger.warning(f"V1 checkpoint not found: {v1_checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(v1_checkpoint_path, map_location=self.device, weights_only=False)
            v1_policy_state = checkpoint.get("policy_state_dict", checkpoint.get("policy_net_state_dict"))
            v1_value_state = checkpoint.get("value_state_dict", checkpoint.get("value_net_state_dict"))
            
            if v1_policy_state is None or v1_value_state is None:
                logger.warning("V1 checkpoint missing policy or value state dict")
                return False
            
            # Load encoder weights (should match)
            encoder_keys = [k for k in v1_policy_state.keys() if 'encoder' in k]
            encoder_state = {k: v for k, v in v1_policy_state.items() if 'encoder' in k}
            
            if encoder_state:
                # Load encoder into policy network
                missing, unexpected = self.agent.policy_net.load_state_dict(encoder_state, strict=False)
                if strict_encoder and missing:
                    missing_encoder = [k for k in missing if 'encoder' in k]
                    if missing_encoder:
                        logger.warning(f"Missing encoder keys: {missing_encoder}")
                        return False
                
                logger.info(f"Loaded {len(encoder_state)} encoder weights from V1")
            
            # Load value network (should match exactly)
            missing, unexpected = self.agent.value_net.load_state_dict(v1_value_state, strict=False)
            if not missing:
                logger.info("Value network weights loaded from V1")
            else:
                logger.warning(f"Value network partial load - missing: {len(missing)} keys")
            
            # Policy head is NOT loaded - it's reinitialized for V2 action space
            logger.info("Policy head reinitialized for V2 action space (13 actions)")
            
            # Also copy encoder weights to value network if it has one
            if hasattr(self.agent.value_net, 'encoder'):
                value_encoder_state = {k.replace('encoder.', ''): v 
                                       for k, v in encoder_state.items()}
                try:
                    self.agent.value_net.encoder.load_state_dict(value_encoder_state, strict=False)
                    logger.info("Encoder weights also copied to value network")
                except Exception as e:
                    logger.debug(f"Could not copy encoder to value net: {e}")
            
            logger.info(f"âœ“ V1 weights loaded for transfer learning from {v1_checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load V1 weights: {e}")
            return False
    
    def step_schedulers(self):
        """Step the learning rate schedulers after each epoch."""
        if self.policy_scheduler:
            self.policy_scheduler.step()
        if self.value_scheduler:
            self.value_scheduler.step()
        self.current_epoch += 1
        
        # Log current learning rates and entropy coefficient
        if self.policy_optimizer and self.value_optimizer:
            policy_lr = self.policy_optimizer.param_groups[0]['lr']
            value_lr = self.value_optimizer.param_groups[0]['lr']
            if self.training_logger:
                self.training_logger.log_scalar("learning_rate/policy", policy_lr, self.current_epoch)
                self.training_logger.log_scalar("learning_rate/value", value_lr, self.current_epoch)
                self.training_logger.log_scalar("learning_rate", policy_lr, self.current_epoch)  # backward compat
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
        """Normalize rewards using running statistics (Welford's algorithm)."""
        for r in rewards.flatten():
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_var += delta * delta2
        
        if self.reward_count > 1:
            std = np.sqrt(self.reward_var / (self.reward_count - 1))
        else:
            std = 1.0
        
        std = max(std, 1e-8)
        normalized = (rewards - self.reward_mean) / std
        return np.clip(normalized, -10.0, 10.0)
    
    def collect_rollouts_parallel(
        self,
        audio_states: List[AudioState],
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect rollouts from multiple V2 environments in parallel."""
        # Clear auxiliary target cache for new audio states
        self.aux_target_computer._cache.clear()
        
        # Setup environments
        self.vec_env.set_audio_states(audio_states)
        obs_list, _ = self.vec_env.reset_all()
        
        # Initialize agent if needed (V2 has different action count)
        if self.agent is None:
            input_dim = obs_list[0].shape[0]
            n_actions = ActionSpaceV2.N_ACTIONS  # V2 action space
            self._init_agent(input_dim, n_actions)
        
        self.buffer.clear()
        episode_rewards = [0.0] * len(obs_list)
        episode_lengths = [0] * len(obs_list)
        section_decisions = [0] * len(obs_list)
        
        # V2: Higher exploration to prevent premature convergence
        # Slower decay - stays high longer
        exploration_rate = max(0.15, 0.5 - self.global_step / 2000000)
        
        for step in range(n_steps):
            # Batch observations - keep on CPU until needed
            obs_batch = np.stack(obs_list)
            
            # Get V2 action masks (parallel via ThreadPool)
            masks = self.vec_env.get_action_masks()
            mask_batch = np.stack(masks)
            
            # Single GPU transfer for batch inference
            obs_tensor = torch.from_numpy(obs_batch).float().to(self.device, non_blocking=True)
            mask_tensor = torch.from_numpy(mask_batch).bool().to(self.device, non_blocking=True)
            
            # Batch inference
            with torch.no_grad():
                actions, log_probs = self.agent.select_action_batch(obs_tensor, mask_tensor)
                values = self.agent.compute_value_batch(obs_tensor)
                # Move to CPU immediately to free GPU memory
                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()
            
            # Epsilon-greedy exploration (vectorized where possible)
            explore_mask = np.random.random(len(obs_list)) < exploration_rate
            for i in np.where(explore_mask)[0]:
                valid_actions = np.where(masks[i])[0]
                if len(valid_actions) > 0:
                    actions_np[i] = np.random.choice(valid_actions)
                    log_probs_np[i] = -np.log(len(valid_actions))
            
            # Step all environments (parallel via ThreadPool)
            next_obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step_all(actions_np.tolist())
            
            # Batch auxiliary target computation
            # For threading mode: access env data directly
            # For subprocess mode: aux targets computed in subprocess and returned in info
            beat_indices = []
            aux_targets_list = []  # Store aux_targets per env
            
            if self.use_subprocess:
                # Subprocess mode - aux targets computed in subprocess workers
                for i in range(len(obs_list)):
                    info = infos[i]
                    beat_idx = info.get("beat", 0)
                    beat_indices.append(beat_idx)
                    # Get pre-computed aux targets from subprocess
                    aux_targets = info.get("aux_targets")
                    if aux_targets is not None:
                        # Reconstruction is a list, convert back to array
                        if "reconstruction" in aux_targets and isinstance(aux_targets["reconstruction"], list):
                            aux_targets["reconstruction"] = np.array(aux_targets["reconstruction"])
                    aux_targets_list.append(aux_targets)
            else:
                # Threading mode - access env data directly and compute here
                for i in range(len(obs_list)):
                    env = self.vec_env.envs[i]
                    if env is not None and env.audio_state is not None:
                        beat_indices.append(env.current_beat)
                        # Compute auxiliary targets using cached computer
                        audio_id = f"env_{i}_epoch"  # Cache per env per epoch
                        aux_targets = self.aux_target_computer.get_targets(
                            audio_id=audio_id,
                            beat_times=env.audio_state.beat_times,
                            beat_features=env.audio_state.beat_features,
                            beat_indices=np.array([env.current_beat]),
                        )
                        aux_targets = {k: v[0] if len(v) > 0 else 0 for k, v in aux_targets.items()}
                        aux_targets_list.append(aux_targets)
                    else:
                        beat_indices.append(0)
                        aux_targets_list.append(None)
            
            # Store in buffer
            for i in range(len(obs_list)):
                self.buffer.add(
                    state=obs_list[i],
                    action=actions_np[i],
                    reward=rewards[i],
                    value=values_np[i],
                    log_prob=log_probs_np[i],
                    done=terminateds[i] or truncateds[i],
                    mask=masks[i],
                    beat_index=beat_indices[i],
                    aux_targets=aux_targets_list[i],
                )
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                
                # Track section-level decisions
                action = actions_np[i]
                if action in [ActionTypeV2.KEEP_BAR.value, ActionTypeV2.CUT_BAR.value,
                             ActionTypeV2.KEEP_PHRASE.value, ActionTypeV2.CUT_PHRASE.value,
                             ActionTypeV2.LOOP_BAR.value, ActionTypeV2.LOOP_PHRASE.value,
                             ActionTypeV2.REORDER_BAR.value, ActionTypeV2.REORDER_PHRASE.value]:
                    section_decisions[i] += 1
            
            # Handle episode ends
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if terminated or truncated:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    self.buffer.section_decisions.append(section_decisions[i])
                    
                    # Log V2 episode info
                    logger.debug(
                        f"V2 Episode complete: len={episode_lengths[i]}, "
                        f"reward={episode_rewards[i]:.2f}, "
                        f"section_decisions={section_decisions[i]}"
                    )
                    
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    section_decisions[i] = 0
                    
                    # Reset this environment (works with both thread and subprocess envs)
                    obs, _ = self.vec_env.reset_env(i)
                    next_obs_list[i] = obs
            
            obs_list = next_obs_list
            self.global_step += len(obs_list)
        
        # === MONTE CARLO RETURNS ===
        # Pure episode returns - NO value function bootstrapping
        # This forces true multi-step credit assignment
        returns, _ = compute_trajectory_return(
            self.buffer.rewards, 
            gamma=self.config.ppo.gamma,
            normalize=False  # Don't normalize yet - do it after advantage computation
        )
        
        # Convert to numpy arrays with NaN handling
        states_arr = np.array(self.buffer.states)
        values_arr = np.array(self.buffer.values)
        returns_arr = np.array(returns)
        log_probs_arr = np.array(self.buffer.log_probs)
        
        # Handle any NaN/Inf values
        states_arr = np.nan_to_num(states_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        values_arr = np.nan_to_num(values_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        returns_arr = np.nan_to_num(returns_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        log_probs_arr = np.nan_to_num(log_probs_arr, nan=-5.0, posinf=0.0, neginf=-10.0)
        
        # === MONTE CARLO ADVANTAGE ===
        # Use mean return as baseline instead of learned value function
        # This reduces variance while keeping Monte Carlo properties
        mean_return = returns_arr.mean()
        advantages = returns_arr - mean_return  # Simple baseline subtraction
        
        # Normalize advantages for stable gradients
        adv_std = advantages.std()
        if adv_std < 1e-8 or np.isnan(adv_std):
            adv_std = 1.0
        advantages = advantages / (adv_std + 1e-8)
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
        """Update policy and value networks using PPO with auxiliary tasks."""
        states = torch.from_numpy(rollout_data["states"]).float().to(self.device)
        actions = torch.from_numpy(rollout_data["actions"]).long().to(self.device)
        returns = torch.from_numpy(rollout_data["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout_data["advantages"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout_data["log_probs"]).float().to(self.device)
        
        # Value targets: use raw returns directly
        # The value network learns to predict the actual expected return scale
        # No normalization needed - just let it learn the natural scale (~50-70)
        
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
        early_stop_epoch = False
        
        # Track auxiliary loss breakdown
        aux_loss_breakdown = {}
        
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
                
                # Get batch auxiliary targets
                batch_aux_targets = {}
                for key, tensor in aux_targets.items():
                    if len(tensor) > len(batch_indices):
                        batch_aux_targets[key] = tensor[batch_indices]
                    elif len(tensor) == len(states):
                        batch_aux_targets[key] = tensor[batch_indices]
                
                # Handle NaN in inputs
                batch_states = torch.nan_to_num(batch_states, nan=0.0)
                batch_returns = torch.nan_to_num(batch_returns, nan=0.0)
                batch_advantages = torch.nan_to_num(batch_advantages, nan=0.0)
                batch_old_log_probs = torch.nan_to_num(batch_old_log_probs, nan=-5.0)
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    # Forward pass
                    new_log_probs, entropy = self.agent.evaluate_actions(batch_states, batch_actions)
                    
                    # Handle NaN in outputs
                    if torch.isnan(new_log_probs).any():
                        new_log_probs = torch.nan_to_num(new_log_probs, nan=-3.0)
                    if torch.isnan(entropy).any():
                        entropy = torch.nan_to_num(entropy, nan=0.0)
                    
                    # Compute ratio with clamping
                    log_ratio = new_log_probs - batch_old_log_probs
                    log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    
                    # Approximate KL divergence for early stopping
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        epoch_kl_divs.append(approx_kl)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - ppo_config.clip_ratio, 1 + ppo_config.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    values = self.agent.compute_value_batch(batch_states)
                    if torch.isnan(values).any():
                        values = torch.nan_to_num(values, nan=0.0)
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    current_entropy_coeff = self.get_current_entropy_coeff()
                    
                    # === AUXILIARY TASK LOSSES ===
                    aux_loss = torch.tensor(0.0, device=self.device)
                    if self.agent.auxiliary_module is not None and batch_aux_targets:
                        aux_loss, aux_breakdown = self.agent.compute_auxiliary_loss(
                            batch_states, 
                            batch_aux_targets,
                            epoch=self.current_epoch,
                        )
                        # Track breakdown for logging
                        for k, v in aux_breakdown.items():
                            if k not in aux_loss_breakdown:
                                aux_loss_breakdown[k] = []
                            aux_loss_breakdown[k].append(v)
                    
                    # Combined loss: policy + value + entropy + auxiliary
                    loss = (
                        policy_loss 
                        + ppo_config.value_loss_coeff * value_loss 
                        + current_entropy_coeff * entropy_loss
                        + aux_loss  # Auxiliary tasks add to total loss
                    )
                
                # Final NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_skipped += 1
                    continue
                
                # Backward pass with mixed precision
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                if self.auxiliary_optimizer is not None:
                    self.auxiliary_optimizer.zero_grad()
                
                # Scaled backward for mixed precision
                self.scaler.scale(loss).backward()
                
                # Unscale before clipping
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.value_optimizer)
                if self.auxiliary_optimizer is not None:
                    self.scaler.unscale_(self.auxiliary_optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), ppo_config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.agent.value_net.parameters(), ppo_config.max_grad_norm)
                if self.agent.auxiliary_module is not None:
                    torch.nn.utils.clip_grad_norm_(self.agent.auxiliary_module.parameters(), ppo_config.max_grad_norm)
                
                # Step optimizers with scaler
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.value_optimizer)
                if self.auxiliary_optimizer is not None:
                    self.scaler.step(self.auxiliary_optimizer)
                self.scaler.update()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                auxiliary_losses.append(aux_loss.item())
            
            # Check KL divergence for early stopping
            if epoch_kl_divs:
                mean_kl = np.mean(epoch_kl_divs)
                approx_kl_divs.append(mean_kl)
                if mean_kl > 1.5 * self.target_kl:
                    logger.debug(f"Early stopping at PPO epoch {epoch+1} due to KL: {mean_kl:.4f}")
                    early_stop_epoch = True
                    early_stop_epoch = True
        
        if nan_batches_skipped > 0:
            logger.warning(f"Skipped {nan_batches_skipped} batches due to NaN/Inf loss")
        
        if not policy_losses:
            logger.error("All batches skipped! Training is not functioning properly.")
            return {
                "policy_loss": 999.0,
                "value_loss": 999.0, 
                "entropy_loss": 0.0,
                "auxiliary_loss": 0.0,
                "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
                "total_loss": 999.0,
                "approx_kl": 0.0,
            }
        
        # Compute mean auxiliary losses
        mean_aux_loss = np.mean(auxiliary_losses) if auxiliary_losses else 0.0
        
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "auxiliary_loss": mean_aux_loss,
            "episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "total_loss": np.mean(policy_losses) + ppo_config.value_loss_coeff * np.mean(value_losses) + mean_aux_loss,
            "approx_kl": np.mean(approx_kl_divs) if approx_kl_divs else 0.0,
            "n_episodes": len(self.episode_rewards),
            "section_decisions_per_ep": np.mean(self.buffer.section_decisions) if self.buffer.section_decisions else 0.0,
        }
        
        # Add auxiliary loss breakdown
        for k, v_list in aux_loss_breakdown.items():
            if v_list:
                metrics[f"aux_{k}"] = np.mean(v_list)
        
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
            self.training_logger.log_scalar("approx_kl", metrics["approx_kl"], self.global_step)
            self.training_logger.log_scalar("section_decisions_per_ep", metrics["section_decisions_per_ep"], self.global_step)
            self.training_logger.log_scalar("auxiliary_loss", mean_aux_loss, self.global_step)
            
            # Log individual auxiliary losses
            for k, v_list in aux_loss_breakdown.items():
                if v_list and k != "warmup_factor":
                    self.training_logger.log_scalar(f"aux/{k}", np.mean(v_list), self.global_step)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "version": "v2",  # Mark as V2 checkpoint
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
            "n_actions": ActionSpaceV2.N_ACTIONS,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved V2 checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Check if this is a V2 checkpoint
        version = checkpoint.get("version", "v1")
        if version != "v2":
            logger.warning(f"Loading V1 checkpoint into V2 trainer - use load_v1_weights for transfer learning")
        
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
        
        # Load scaler state if available and non-empty
        if checkpoint.get("scaler") and len(checkpoint["scaler"]) > 0:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except RuntimeError as e:
                logger.warning(f"Could not load scaler state: {e} - using fresh scaler")
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch}, best_reward={self.best_reward:.2f})")


def train_v2(
    config: Config,
    data_dir: str,
    n_epochs: int = 1000,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    checkpoint_path: Optional[str] = None,
    v1_checkpoint_path: Optional[str] = None,
    target_loss: float = 0.01,
    use_subprocess: bool = False,
):
    """Main V2 training function with parallel environments.
    
    Args:
        config: Configuration object
        data_dir: Path to training data
        n_epochs: Maximum number of epochs
        n_envs: Number of parallel environments
        steps_per_epoch: Steps to collect per epoch
        checkpoint_path: Optional V2 checkpoint to resume from
        v1_checkpoint_path: Optional V1 checkpoint for transfer learning
        target_loss: Stop training when loss reaches this value
        use_subprocess: Use subprocess-based parallelism (true multiprocessing)
    """
    logger.info("=" * 60)
    logger.info("Starting V2 Training (Section Actions + Episode Rewards)")
    logger.info("=" * 60)
    logger.info(f"Parallel environments: {n_envs}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Target loss: {target_loss}")
    if use_subprocess:
        logger.info(f"Parallelism: subprocess")
    else:
        logger.info(f"Parallelism: threading (GIL-bound, auxiliary tasks computed in main process)")
    
    # V2 reward scaling: minimize step rewards, maximize episode rewards
    config.reward.step_reward_scale = 0.001  # Very small step rewards
    config.reward.trajectory_reward_scale = 50.0  # Large episode rewards
    
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
            logger.info("âœ“ Learned reward model loaded for RLHF training")
        else:
            logger.warning("Could not load learned reward model - using dense rewards only")
            config.reward.use_learned_rewards = False
    
    # Create V2 trainer
    trainer = ParallelPPOTrainerV2(
        config, 
        n_envs=n_envs, 
        total_epochs=n_epochs, 
        learned_reward_model=learned_reward_model,
        use_subprocess=use_subprocess,
    )
    
    # Handle checkpoint loading
    # Priority: V2 checkpoint > V1 transfer learning
    if checkpoint_path and Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    elif v1_checkpoint_path and Path(v1_checkpoint_path).exists():
        # Need to initialize agent first to load V1 weights
        # Do a dummy rollout to initialize
        logger.info("Initializing agent for V1 weight transfer...")
        
    # Determine starting epoch
    start_epoch = trainer.current_epoch
    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    start_time = time.time()
    MAX_BEATS = 2500  # Limit beats for tractable action space
    logger.info(f"Max Beats per sample set to {MAX_BEATS} for V2 training")
    
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Sample audio states for parallel environments
        audio_states = []
        for _ in range(n_envs):
            idx = np.random.randint(len(dataset))
            item = dataset[idx]

            raw_data = item["raw"]
            beat_times = raw_data["beat_times"].numpy()
            beat_features = raw_data["beat_features"].numpy()

            # Get ground truth edit labels
            edit_labels = item.get("edit_labels")
            if edit_labels is not None:
                edit_labels = edit_labels.numpy() if hasattr(edit_labels, 'numpy') else np.array(edit_labels)

            # Limit to MAX_BEATS
            if len(beat_times) > MAX_BEATS:
                start_idx = np.random.randint(0, len(beat_times) - MAX_BEATS)
                beat_times = beat_times[start_idx:start_idx + MAX_BEATS]
                beat_features = beat_features[start_idx:start_idx + MAX_BEATS]
                if edit_labels is not None and len(edit_labels) > MAX_BEATS:
                    edit_labels = edit_labels[start_idx:start_idx + MAX_BEATS]
                beat_times = beat_times - beat_times[0]

            audio_state = AudioState(
                beat_index=0,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=raw_data["tempo"].item() if hasattr(raw_data["tempo"], 'item') else raw_data["tempo"],
                target_labels=edit_labels,
            )
            audio_states.append(audio_state)

        # Collect rollouts in parallel
        rollout_start = time.time()
        rollout_data = trainer.collect_rollouts_parallel(audio_states, steps_per_epoch)
        rollout_time = time.time() - rollout_start

        # Load V1 weights after first rollout (agent now initialized)
        if epoch == start_epoch and v1_checkpoint_path and trainer.agent is not None:
            if trainer.load_v1_weights(v1_checkpoint_path):
                logger.info("✓ V1 weights loaded for transfer learning - encoder pretrained, policy head fresh")

        # Update networks
        update_start = time.time()
        metrics = trainer.update(rollout_data)
        update_time = time.time() - update_start

        # Step learning rate schedulers
        trainer.step_schedulers()

        epoch_time = time.time() - epoch_start
        current_lr = trainer.get_current_lr()

        # Log progress
        n_eps = metrics.get('n_episodes', 0)
        section_dec = metrics.get('section_decisions_per_ep', 0)
        aux_loss = metrics.get('auxiliary_loss', 0.0)
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {metrics['total_loss']:.4f} (P: {metrics['policy_loss']:.4f}, V: {metrics['value_loss']:.4f}, Aux: {aux_loss:.4f}) | "
            f"Reward: {metrics['episode_reward']:.2f} (eps: {n_eps}) | "
            f"SectionDec: {section_dec:.1f} | "
            f"LR: {current_lr:.2e} | "
            f"Steps: {trainer.global_step:,} | "
            f"Time: {epoch_time:.1f}s (R:{rollout_time:.1f}s U:{update_time:.1f}s)"
        )

        # NOTE: No early stopping on loss - we want full training for RL
        # The policy loss converging doesn't mean the agent is good at editing!
        # What matters is the episode reward (audio quality metrics)

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_file = Path(config.training.save_dir) / f"checkpoint_v2_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_file))

        # Save best model
        if metrics['episode_reward'] > trainer.best_reward:
            trainer.best_reward = metrics['episode_reward']
            best_file = Path(config.training.save_dir) / "checkpoint_v2_best.pt"
            trainer.save_checkpoint(str(best_file))

    # Save final checkpoint
    final_file = Path(config.training.save_dir) / "checkpoint_v2_final.pt"
    trainer.save_checkpoint(str(final_file))

    total_time = time.time() - start_time
    logger.info(f"V2 Training complete in {total_time/60:.1f} minutes")
    if 'metrics' in dir():
        logger.info(f"Final loss: {metrics['total_loss']:.4f}")
    logger.info(f"Best reward: {trainer.best_reward:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V2 RL Audio Editor Training (Section Actions + Episode Rewards)")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=2048, help="Steps per epoch")
    parser.add_argument("--checkpoint", type=str, default=None, help="V2 checkpoint to resume from")
    parser.add_argument("--v1_checkpoint", type=str, default=None, help="V1 checkpoint for transfer learning")
    parser.add_argument("--target_loss", type=float, default=0.01, help="Target loss to stop training")
    parser.add_argument("--use_learned_rewards", action="store_true", help="Enable learned reward model (RLHF)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no_lr_decay", action="store_true", help="Disable learning rate decay")
    parser.add_argument("--save_dir", type=str, default="models_v2", help="Directory to save checkpoints")
    parser.add_argument("--subprocess", action="store_true", help="Use subprocess-based parallelism (true multiprocessing)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_default_config()
    
    # Set save directory
    config.training.save_dir = args.save_dir
    
    # Override learning rate if specified
    if args.lr is not None:
        config.ppo.learning_rate = args.lr
        logger.info(f"Learning rate set to: {args.lr}")
    
    # Disable LR decay if requested
    if args.no_lr_decay:
        config.ppo.lr_decay = False
        logger.info("Learning rate decay DISABLED (fixed LR)")
    
    # Enable learned rewards if requested
    if args.use_learned_rewards:
        config.reward.use_learned_rewards = True
        logger.info("Learned rewards ENABLED (RLHF mode)")
    
    # Log subprocess mode
    if args.subprocess:
        logger.info("Subprocess parallelism ENABLED (true multiprocessing)")
    
    train_v2(
        config=config,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        checkpoint_path=args.checkpoint,
        v1_checkpoint_path=args.v1_checkpoint,
        target_loss=args.target_loss,
        use_subprocess=args.subprocess,
    )




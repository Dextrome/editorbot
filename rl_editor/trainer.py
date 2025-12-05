"""PPO training loop for RL agent.

Implements proximal policy optimization with support for:
- Offline RL (imitation learning warmup)
- Online RL (human feedback loop)
- Checkpoint/resume
- Gradient accumulation
- Mixed precision training
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from collections import deque

from .config import Config
from .environment import AudioEditingEnv
from .agent import Agent
from .reward import compute_trajectory_return
from .state import AudioState
from .logging_utils import TrainingLogger, create_logger

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer for RL agent."""

    def __init__(self, config: Config) -> None:
        """Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.training.device)

        # Will be initialized when we have env
        self.env: Optional[AudioEditingEnv] = None
        self.agent: Optional[Agent] = None
        self.policy_optimizer: Optional[optim.Adam] = None
        self.value_optimizer: Optional[optim.Adam] = None

        # Tracking
        self.global_step = 0
        self.episode_num = 0
        self.best_reward = -np.inf
        
        # Mixed precision scaler (use device-agnostic API)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.ppo.use_mixed_precision)
        
        # Training logger
        self.training_logger: Optional[TrainingLogger] = None
        if config.training.use_tensorboard or config.training.use_wandb:
            self.training_logger = create_logger(config)

        # Setup paths
        Path(config.training.save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PPO trainer with device {self.device}")

    def initialize_env_and_agent(self, audio_state: AudioState) -> None:
        """Initialize environment and agent.

        Args:
            audio_state: Initial audio state
        """
        # Create environment
        self.env = AudioEditingEnv(self.config, audio_state=audio_state)
        obs, _ = self.env.reset()

        # Create agent
        input_dim = obs.shape[0]
        n_actions = self.env.action_space.n_discrete_actions
        self.agent = Agent(self.config, input_dim, n_actions)

        # Setup optimizers
        ppo_config = self.config.ppo
        self.policy_optimizer = optim.Adam(
            self.agent.get_policy_parameters(), lr=ppo_config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.agent.get_value_parameters(), lr=ppo_config.learning_rate
        )

        logger.info(f"Initialized environment and agent with {input_dim} state dims, {n_actions} actions")

    def train(
        self,
        audio_state: AudioState,
        total_timesteps: Optional[int] = None,
        resume_checkpoint: Optional[str] = None,
    ) -> None:
        """Train RL agent using PPO.

        Args:
            audio_state: Audio state for training
            total_timesteps: Total training timesteps
            resume_checkpoint: Path to checkpoint to resume from
        """
        if total_timesteps is None:
            total_timesteps = self.config.training.total_timesteps

        # Initialize env and agent
        self.initialize_env_and_agent(audio_state)

        # Load checkpoint if specified
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        logger.info(f"Starting training for {total_timesteps} timesteps")

        while self.global_step < total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts(self.config.ppo.n_steps)

            # Update policy and value networks
            metrics = self.update(rollout_data)

            self.episode_num += 1

            # Logging
            if self.episode_num % 10 == 0:
                logger.info(
                    f"Step {self.global_step}/{total_timesteps} | "
                    f"Episode {self.episode_num} | "
                    f"Reward: {metrics.get('episode_reward', 0):.3f} | "
                    f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                    f"Value Loss: {metrics.get('value_loss', 0):.4f}"
                )

            # Checkpointing
            if self.episode_num % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(
                    f"{self.config.training.save_dir}/checkpoint_{self.global_step}.pt"
                )

    def collect_rollouts(self, n_steps: int) -> Dict[str, Any]:
        """Collect rollouts from environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary with trajectory data
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("Environment not initialized")

        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        masks = []
        dones = []

        obs, info = self.env.reset()

        for step in range(n_steps):
            # Convert observation to tensor
            state_tensor = torch.from_numpy(obs).float().to(self.device)
            # states.append(obs)  # Removed duplicate append

            # Get action mask from environment
            action_mask = self.env.action_space.get_action_mask(
                current_beat_index=self.env.audio_state.beat_index,
                remaining_duration=self.env._get_remaining_duration(),
                edited_beats=self.env.edit_history.get_edited_beats(),
                total_beats=len(self.env.audio_state.beat_times),
            )
            action_mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)

            # Select action
            with torch.no_grad():
                action, log_prob = self.agent.select_action(state_tensor, action_mask_tensor)
                value = self.agent.compute_value(state_tensor)

            # Take step
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)

            # Store trajectory
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            masks.append(action_mask)
            dones.append(terminated or truncated)

            obs = next_obs
            self.global_step += 1

            if terminated or truncated:
                obs, info = self.env.reset()

        # Compute returns and advantages
        returns, _ = compute_trajectory_return(rewards, gamma=self.config.ppo.gamma)

        # Compute advantages
        advantages = np.array(returns) - np.array(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "returns": np.array(returns),
            "values": np.array(values),
            "advantages": advantages,
            "log_probs": np.array(log_probs),
            "masks": masks,
            "dones": np.array(dones),
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Update policy and value networks using PPO.

        Args:
            rollout_data: Trajectory data from collect_rollouts

        Returns:
            Dictionary with training metrics
        """
        if self.agent is None or self.policy_optimizer is None or self.value_optimizer is None:
            raise RuntimeError("Agent not initialized")

        states = torch.from_numpy(rollout_data["states"]).float().to(self.device)
        actions = torch.from_numpy(rollout_data["actions"]).long().to(self.device)
        returns = torch.from_numpy(rollout_data["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout_data["advantages"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout_data["log_probs"]).float().to(self.device)

        ppo_config = self.config.ppo
        batch_size = ppo_config.batch_size
        accumulation_steps = ppo_config.gradient_accumulation_steps if ppo_config.use_gradient_accumulation else 1

        # PPO epochs
        policy_losses = []
        value_losses = []

        for epoch in range(ppo_config.n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            for step_idx, i in enumerate(range(0, len(states), batch_size)):
                batch_indices = indices[i : i + batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                with torch.amp.autocast('cuda', enabled=ppo_config.use_mixed_precision):
                    # Policy loss (PPO clipped objective)
                    logits, probs = self.agent.policy_net(batch_states)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - ppo_config.clip_ratio, 1 + ppo_config.clip_ratio)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean() - ppo_config.entropy_coeff * entropy

                    # Value loss
                    batch_values = self.agent.value_net(batch_states)
                    value_loss = nn.functional.mse_loss(batch_values, batch_returns)
                    
                    # Scale loss for accumulation
                    policy_loss = policy_loss / accumulation_steps
                    value_loss = value_loss / accumulation_steps

                # Backward pass with scaler
                self.scaler.scale(policy_loss).backward()
                self.scaler.scale(value_loss).backward()

                # Step if accumulation is done
                if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(range(0, len(states), batch_size)):
                    # Unscale before clipping
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)
                    
                    nn.utils.clip_grad_norm_(
                        self.agent.get_policy_parameters(), ppo_config.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.agent.get_value_parameters(), ppo_config.max_grad_norm
                    )
                    
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()
                    
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()

                policy_losses.append(policy_loss.item() * accumulation_steps)
                value_losses.append(value_loss.item() * accumulation_steps)

        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "episode_reward": float(np.sum(rollout_data["rewards"])),
            "episode_length": len(rollout_data["states"]),
        }
        
        # Log to TensorBoard/W&B
        if self.training_logger:
            self.training_logger.log_training_step(
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                episode_reward=metrics["episode_reward"],
                episode_length=metrics["episode_length"],
                step=self.global_step,
            )
            self.global_step += 1

        return metrics

    def offline_rl_warmup(self, trajectories: List[Dict[str, Any]], n_epochs: int = 10) -> None:
        """Offline RL warmup using imitation learning.

        Args:
            trajectories: List of expert trajectory dicts
            n_epochs: Number of epochs

        TODO: Implement in step 8
        """
        logger.info(f"Starting offline RL warmup with {len(trajectories)} trajectories")
        # Would implement imitation learning from human-edited audio

    def save_checkpoint(self, save_path: str) -> None:
        """Save training checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        if self.agent is None:
            return

        checkpoint = {
            "global_step": self.global_step,
            "episode_num": self.episode_num,
            "agent_state_dict": {
                "policy_net": self.agent.policy_net.state_dict(),
                "value_net": self.agent.value_net.state_dict(),
            },
            "optimizer_states": {
                "policy_optimizer": self.policy_optimizer.state_dict() if self.policy_optimizer else None,
                "value_optimizer": self.value_optimizer.state_dict() if self.value_optimizer else None,
            },
            "config": self.config.to_dict(),
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, load_path: str) -> None:
        """Load training checkpoint.

        Args:
            load_path: Path to load checkpoint
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.global_step = checkpoint["global_step"]
        self.episode_num = checkpoint["episode_num"]

        self.agent.policy_net.load_state_dict(checkpoint["agent_state_dict"]["policy_net"])
        self.agent.value_net.load_state_dict(checkpoint["agent_state_dict"]["value_net"])

        if self.policy_optimizer and checkpoint["optimizer_states"]["policy_optimizer"]:
            self.policy_optimizer.load_state_dict(checkpoint["optimizer_states"]["policy_optimizer"])
        if self.value_optimizer and checkpoint["optimizer_states"]["value_optimizer"]:
            self.value_optimizer.load_state_dict(checkpoint["optimizer_states"]["value_optimizer"])

        logger.info(f"Loaded checkpoint from {load_path} at step {self.global_step}")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy on test set.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with evaluation metrics

        TODO: Implement in step 10
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("Environment and agent not initialized")

        self.agent.eval()

        episode_rewards = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                state_tensor = torch.from_numpy(obs).float().to(self.device)
                action, _ = self.agent.select_action(state_tensor, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)

        self.agent.train()

        eval_metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
        }
        
        # Log evaluation metrics
        if self.training_logger:
            self.training_logger.log_evaluation(
                mean_reward=eval_metrics["mean_reward"],
                std_reward=eval_metrics["std_reward"],
                min_reward=eval_metrics["min_reward"],
                max_reward=eval_metrics["max_reward"],
                step=self.global_step,
            )
        
        return eval_metrics

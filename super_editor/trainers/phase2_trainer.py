"""Phase 2 Trainer: RL-based edit label prediction.

Uses PPO to train an edit predictor that maximizes reconstruction quality
through a frozen Phase 1 model.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Phase2Config, EditLabel
from ..models import ReconstructionModel, ActorCritic
from ..data import PairedMelDataset, collate_fn


class RewardComputer:
    """Computes rewards for predicted edit labels."""

    def __init__(
        self,
        recon_model: ReconstructionModel,
        config: Phase2Config,
    ):
        self.recon_model = recon_model
        self.config = config

        # Freeze reconstruction model
        self.recon_model.eval()
        for param in self.recon_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_reward(
        self,
        raw_mel: torch.Tensor,         # (B, T, n_mels)
        pred_labels: torch.Tensor,     # (B, T)
        target_mel: torch.Tensor,      # (B, T, n_mels)
        target_labels: torch.Tensor,   # (B, T)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute reward for predicted edit labels.

        Args:
            raw_mel: Raw mel spectrogram
            pred_labels: Predicted edit labels
            target_mel: Target edited mel
            target_labels: Ground truth edit labels
            mask: Valid frame mask

        Returns:
            total_reward: Combined reward (B,)
            reward_components: Dictionary of reward components
        """
        B = raw_mel.size(0)

        # 1. Reconstruction quality reward
        pred_mel = self.recon_model(raw_mel, pred_labels, mask)

        # L1 difference (lower is better)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            recon_diff = (torch.abs(pred_mel - target_mel) * mask_expanded).sum(dim=(1, 2))
            recon_diff = recon_diff / (mask_expanded.sum(dim=(1, 2)) + 1e-8)
        else:
            recon_diff = torch.abs(pred_mel - target_mel).mean(dim=(1, 2))

        recon_reward = -recon_diff * self.config.reconstruction_reward_weight

        # 2. Label accuracy reward
        if mask is not None:
            correct = ((pred_labels == target_labels) & mask).float().sum(dim=1)
            total = mask.float().sum(dim=1)
            label_acc = correct / (total + 1e-8)
        else:
            label_acc = (pred_labels == target_labels).float().mean(dim=1)

        label_reward = label_acc * self.config.label_accuracy_reward_weight

        # 3. Duration match reward (similar KEEP ratio)
        if mask is not None:
            pred_keep = ((pred_labels == EditLabel.KEEP) & mask).float().sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)
            target_keep = ((target_labels == EditLabel.KEEP) & mask).float().sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)
        else:
            pred_keep = (pred_labels == EditLabel.KEEP).float().mean(dim=1)
            target_keep = (target_labels == EditLabel.KEEP).float().mean(dim=1)

        duration_diff = torch.abs(pred_keep - target_keep)
        duration_reward = (1 - duration_diff) * self.config.duration_match_reward_weight

        # 4. Smoothness penalty (excessive label changes)
        label_changes = (pred_labels[:, 1:] != pred_labels[:, :-1]).float()
        if mask is not None:
            valid_transitions = mask[:, 1:] & mask[:, :-1]
            changes_rate = (label_changes * valid_transitions.float()).sum(dim=1) / (valid_transitions.float().sum(dim=1) + 1e-8)
        else:
            changes_rate = label_changes.mean(dim=1)

        smoothness_penalty = -changes_rate * self.config.smoothness_penalty_weight

        # Combine rewards
        total_reward = recon_reward + label_reward + duration_reward + smoothness_penalty

        return total_reward, {
            'recon_reward': recon_reward,
            'label_reward': label_reward,
            'duration_reward': duration_reward,
            'smoothness_penalty': smoothness_penalty,
            'recon_diff': recon_diff,
            'label_acc': label_acc,
        }


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self):
        self.raw_mels = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        raw_mel: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        self.raw_mels.append(raw_mel.cpu())
        self.masks.append(mask.cpu())
        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.values.append(value.cpu())
        self.rewards.append(reward.cpu())
        self.dones.append(done.cpu())

    def get(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get all data as tensors on device."""
        return {
            'raw_mels': torch.cat(self.raw_mels, dim=0).to(device),
            'masks': torch.cat(self.masks, dim=0).to(device),
            'actions': torch.cat(self.actions, dim=0).to(device),
            'log_probs': torch.cat(self.log_probs, dim=0).to(device),
            'values': torch.cat(self.values, dim=0).to(device),
            'rewards': torch.cat(self.rewards, dim=0).to(device),
            'dones': torch.cat(self.dones, dim=0).to(device),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return sum(r.size(0) for r in self.rewards)


class Phase2Trainer:
    """PPO trainer for Phase 2 edit label prediction."""

    def __init__(
        self,
        config: Phase2Config,
        recon_model_path: str,
        data_dir: str,
        save_dir: str,
        resume_from: Optional[str] = None,
    ):
        """
        Args:
            config: Phase2Config
            recon_model_path: Path to trained Phase 1 model
            data_dir: Directory with preprocessed data
            save_dir: Directory for checkpoints
            resume_from: Optional checkpoint to resume from
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Load frozen reconstruction model
        print(f"Loading reconstruction model from {recon_model_path}")
        self.recon_model = ReconstructionModel.from_checkpoint(recon_model_path)
        self.recon_model = self.recon_model.to(self.device)
        self.recon_model.eval()
        for param in self.recon_model.parameters():
            param.requires_grad = False

        # Actor-critic network
        self.ac = ActorCritic(config).to(self.device)
        print(f"Actor-Critic parameters: {sum(p.numel() for p in self.ac.parameters()):,}")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=config.learning_rate,
        )

        # Reward computer
        self.reward_computer = RewardComputer(self.recon_model, config)

        # Logging
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.entropy_coeff = config.entropy_coeff

        # Resume
        if resume_from:
            self.load_checkpoint(resume_from)

    def train(self, num_epochs: Optional[int] = None):
        """Run training loop."""
        if num_epochs is None:
            num_epochs = self.config.total_epochs

        # Data loader
        dataset = PairedMelDataset(str(self.data_dir), self._get_phase1_config(), split='train')
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

        print(f"Training with {len(dataset)} samples")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Collect rollouts and update
            metrics = self._train_epoch(loader)

            # Logging
            self._log_epoch(metrics)

            # Entropy decay
            if self.config.entropy_coeff_decay:
                self.entropy_coeff = max(
                    self.config.entropy_coeff_min,
                    self.config.entropy_coeff * (1 - epoch / num_epochs)
                )

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt')

        self._save_checkpoint('final.pt')
        self.writer.close()

    def _get_phase1_config(self):
        """Create a Phase1Config compatible with data loading."""
        from ..config import Phase1Config
        return Phase1Config(
            audio=self.config.audio,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.ac.train()

        buffer = RolloutBuffer()
        total_metrics = {
            'reward': 0, 'recon_diff': 0, 'label_acc': 0,
            'policy_loss': 0, 'value_loss': 0, 'entropy': 0,
        }
        n_updates = 0

        pbar = tqdm(loader, desc=f'Epoch {self.epoch + 1}')
        for batch in pbar:
            # Move to device
            raw_mel = batch['raw_mel'].to(self.device)
            edit_mel = batch['edit_mel'].to(self.device)
            target_labels = batch['edit_labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Get actions from policy
            with torch.no_grad():
                actions, log_probs, _, values = self.ac.get_action_and_value(raw_mel, mask)

            # Compute rewards
            rewards, reward_info = self.reward_computer.compute_reward(
                raw_mel, actions, edit_mel, target_labels, mask
            )

            # Add to buffer
            dones = torch.ones(raw_mel.size(0), device=self.device)  # Episode ends after each sample
            buffer.add(raw_mel, mask, actions, log_probs, values, rewards, dones)

            # Update when buffer is large enough
            if len(buffer) >= self.config.rollout_steps * self.config.batch_size:
                update_metrics = self._ppo_update(buffer)
                buffer.clear()

                for k, v in update_metrics.items():
                    total_metrics[k] += v
                n_updates += 1

                pbar.set_postfix({
                    'reward': f"{update_metrics['reward']:.2f}",
                    'acc': f"{update_metrics['label_acc']:.2%}",
                })

            self.global_step += 1

        # Final update if buffer not empty
        if len(buffer) > 0:
            update_metrics = self._ppo_update(buffer)
            for k, v in update_metrics.items():
                total_metrics[k] += v
            n_updates += 1

        # Average metrics
        if n_updates > 0:
            return {k: v / n_updates for k, v in total_metrics.items()}
        return total_metrics

    def _ppo_update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform PPO update."""
        data = buffer.get(self.device)

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.config.n_epochs_per_update):
            # Get new log probs and values
            _, new_log_probs, entropy, new_values = self.ac.get_action_and_value(
                data['raw_mels'],
                data['masks'],
                action=data['actions'],
            )

            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - data['log_probs'])
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Total loss
            loss = (
                policy_loss +
                self.config.value_coeff * value_loss -
                self.entropy_coeff * entropy
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n_epochs = self.config.n_epochs_per_update

        # Compute average metrics
        with torch.no_grad():
            avg_reward = data['rewards'].mean().item()

            # Compute label accuracy from actions
            # Need target labels - re-fetch from dataset
            acc = 0.0  # Placeholder, tracked in reward

        return {
            'reward': avg_reward,
            'recon_diff': 0.0,  # Tracked in reward_computer
            'label_acc': 0.0,  # Tracked in reward_computer
            'policy_loss': total_policy_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs,
            'entropy': total_entropy / n_epochs,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def _log_epoch(self, metrics: Dict[str, float]):
        """Log epoch metrics."""
        print(f"\nEpoch {self.epoch + 1}: "
              f"Reward={metrics['reward']:.2f}, "
              f"Policy Loss={metrics['policy_loss']:.4f}, "
              f"Value Loss={metrics['value_loss']:.4f}, "
              f"Entropy={metrics['entropy']:.4f}")

        for k, v in metrics.items():
            self.writer.add_scalar(f'train/{k}', v, self.epoch)
        self.writer.add_scalar('train/entropy_coeff', self.entropy_coeff, self.epoch)

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'actor_critic_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'entropy_coeff': self.entropy_coeff,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.entropy_coeff = checkpoint.get('entropy_coeff', self.config.entropy_coeff)
        print(f"Resumed from epoch {self.epoch}")


def train_phase2(
    recon_model_path: str,
    data_dir: str,
    save_dir: str,
    config: Optional[Phase2Config] = None,
    resume_from: Optional[str] = None,
):
    """Convenience function to train Phase 2.

    Args:
        recon_model_path: Path to trained Phase 1 model
        data_dir: Directory with preprocessed data
        save_dir: Directory for checkpoints
        config: Optional config
        resume_from: Optional checkpoint to resume
    """
    if config is None:
        config = Phase2Config()

    trainer = Phase2Trainer(
        config=config,
        recon_model_path=recon_model_path,
        data_dir=data_dir,
        save_dir=save_dir,
        resume_from=resume_from,
    )

    trainer.train()

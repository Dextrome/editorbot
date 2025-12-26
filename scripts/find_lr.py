"""Learning Rate Finder using the LR Range Test (Leslie Smith method).

This script finds the optimal learning rate by:
1. Starting with a very small LR (1e-7)
2. Exponentially increasing LR each batch
3. Tracking loss at each step
4. Finding where loss decreases fastest (optimal LR)
5. Finding where loss starts exploding (max LR)

Usage:
    python -m scripts.find_lr --bc_npz bc_augmented.npz --output lr_finder.png
    python -m scripts.find_lr --bc_npz bc_augmented.npz --min_lr 1e-7 --max_lr 1e-1 --num_steps 200
"""

import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("lr_finder")


class LRFinder:
    """Learning Rate Range Test implementation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        value_net: nn.Module = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.value_net = value_net

        # Store initial state to restore later
        self.initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.initial_optimizer_state = optimizer.state_dict()
        if value_net is not None:
            self.initial_value_state = {k: v.clone() for k, v in value_net.state_dict().items()}
        else:
            self.initial_value_state = None

        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []

    def reset(self):
        """Reset model and optimizer to initial state."""
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        if self.value_net is not None and self.initial_value_state is not None:
            self.value_net.load_state_dict(self.initial_value_state)

    def find(
        self,
        train_loader,
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_steps: int = 200,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
        steps_per_lr: int = 1,
        reset_each_lr: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Run LR range test.

        Args:
            train_loader: DataLoader yielding (states, labels) batches
            min_lr: Starting learning rate
            max_lr: Maximum learning rate to test
            num_steps: Number of LR values to test
            smooth_factor: Exponential smoothing factor for loss
            diverge_threshold: Stop if loss exceeds best_loss * threshold
            steps_per_lr: Number of training steps at each LR (higher = more accurate but slower)
            reset_each_lr: If True, reset model to initial state at each LR (slower but independent)

        Returns:
            Tuple of (learning_rates, losses)
        """
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []

        # Exponential LR schedule: lr = min_lr * (max_lr/min_lr)^(step/num_steps)
        lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

        # Set initial LR
        lr = min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.model.train()
        best_loss = float("inf")
        smoothed_loss = 0.0
        data_iter = iter(train_loader)
        diverged = False

        total_steps = num_steps * steps_per_lr
        mode = "independent (reset each LR)" if reset_each_lr else "progressive"
        logger.info(f"Running LR range test: {min_lr:.2e} -> {max_lr:.2e} over {num_steps} LR values ({steps_per_lr} steps each, {total_steps} total, {mode})")

        for lr_idx in range(num_steps):
            if diverged:
                break

            # Reset model to initial state if using independent mode
            if reset_each_lr:
                self.reset()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # Run steps_per_lr training steps at this LR, accumulate losses
            lr_losses = []
            for _ in range(steps_per_lr):
                # Get batch (cycle through data if needed)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                # Detect if we have rollout data (7 items) or BC data (4 items)
                has_rollout_data = len(batch) == 7

                if has_rollout_data:
                    states, action_types, action_sizes, action_amounts, old_log_probs, returns, advantages = batch
                    states = states.to(self.device)
                    action_types = action_types.to(self.device)
                    action_sizes = action_sizes.to(self.device)
                    action_amounts = action_amounts.to(self.device)
                    old_log_probs = old_log_probs.to(self.device)
                    returns = returns.to(self.device)
                    advantages = advantages.to(self.device)
                else:
                    states, type_labels, size_labels, amount_labels = batch
                    states = states.to(self.device)
                    type_labels = type_labels.to(self.device)
                    size_labels = size_labels.to(self.device)
                    amount_labels = amount_labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                encoded = self.model.encoder(states)
                type_logits = self.model.type_head(encoded)

                if has_rollout_data:
                    # === REAL ROLLOUT DATA - Use actual actions and advantages ===
                    type_embed = self.model.type_embedding(action_types)
                    size_input = torch.cat([encoded, type_embed], dim=-1)
                    size_logits = self.model.size_head(size_input)
                    amount_input = torch.cat([encoded, type_embed], dim=-1)
                    amount_logits = self.model.amount_head(amount_input)

                    # Policy distributions
                    type_dist = torch.distributions.Categorical(logits=type_logits)
                    size_dist = torch.distributions.Categorical(logits=size_logits)
                    amount_dist = torch.distributions.Categorical(logits=amount_logits)

                    # Compute new log probs for the actions that were actually taken
                    new_log_probs = (
                        type_dist.log_prob(action_types) +
                        size_dist.log_prob(action_sizes) +
                        amount_dist.log_prob(action_amounts)
                    )

                    # PPO clipped objective with REAL advantages
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clip_ratio = 0.2
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Entropy bonus
                    entropy = (type_dist.entropy().mean() + size_dist.entropy().mean() + amount_dist.entropy().mean()) / 3.0
                    entropy_coeff = 0.02
                    entropy_loss = -entropy_coeff * entropy

                    # Value loss with REAL returns
                    if self.value_net is not None:
                        values = self.value_net(states).squeeze(-1)
                        value_loss = nn.functional.mse_loss(values, returns)
                        value_loss_coeff = 0.1
                    else:
                        value_loss = torch.tensor(0.0, device=self.device)
                        value_loss_coeff = 0.0

                    # Total PPO loss (no BC loss when using rollout data)
                    loss = policy_loss + value_loss_coeff * value_loss + entropy_loss

                    if lr_idx == 0 and _ == 0:
                        logger.info(f"[ROLLOUT] Loss: Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}*{value_loss_coeff:.2f}={value_loss.item()*value_loss_coeff:.4f}, Entropy={entropy_loss.item():.4f}, Total={loss.item():.4f}")

                else:
                    # === BC DATA - Simulate PPO loss ===
                    type_embed = self.model.type_embedding(type_labels)
                    size_input = torch.cat([encoded, type_embed], dim=-1)
                    size_logits = self.model.size_head(size_input)
                    amount_input = torch.cat([encoded, type_embed], dim=-1)
                    amount_logits = self.model.amount_head(amount_input)

                    # Policy distributions
                    type_dist = torch.distributions.Categorical(logits=type_logits)
                    size_dist = torch.distributions.Categorical(logits=size_logits)
                    amount_dist = torch.distributions.Categorical(logits=amount_logits)

                    # Simulated PPO loss
                    sampled_types = type_dist.sample()
                    sampled_sizes = size_dist.sample()
                    sampled_amounts = amount_dist.sample()
                    new_log_probs = (
                        type_dist.log_prob(sampled_types) +
                        size_dist.log_prob(sampled_sizes) +
                        amount_dist.log_prob(sampled_amounts)
                    )
                    sim_old_log_probs = new_log_probs.detach() + torch.randn_like(new_log_probs) * 0.1
                    sim_advantages = torch.randn(states.shape[0], device=self.device)
                    sim_advantages = (sim_advantages - sim_advantages.mean()) / (sim_advantages.std() + 1e-8)

                    ratio = torch.exp(new_log_probs - sim_old_log_probs)
                    clip_ratio = 0.2
                    surr1 = ratio * sim_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * sim_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # BC Loss
                    ce = nn.CrossEntropyLoss()
                    bc_loss = (ce(type_logits, type_labels) + ce(size_logits, size_labels) + ce(amount_logits, amount_labels)) / 3.0

                    # Entropy bonus
                    entropy = (type_dist.entropy().mean() + size_dist.entropy().mean() + amount_dist.entropy().mean()) / 3.0
                    entropy_coeff = 0.02
                    entropy_loss = -entropy_coeff * entropy

                    # Value loss
                    if self.value_net is not None:
                        values = self.value_net(states).squeeze(-1)
                        sim_returns = values.detach() + torch.randn_like(values) * 5.0
                        value_loss = nn.functional.mse_loss(values, sim_returns)
                        value_loss_coeff = 0.1
                    else:
                        value_loss = torch.tensor(0.0, device=self.device)
                        value_loss_coeff = 0.0

                    # Total loss (PPO + BC)
                    bc_weight = 1.0
                    loss = policy_loss + value_loss_coeff * value_loss + entropy_loss + bc_weight * bc_loss

                    if lr_idx == 0 and _ == 0:
                        logger.info(f"[BC+SIM] Loss: Policy={policy_loss.item():.4f}, Value={value_loss.item()*value_loss_coeff:.4f}, Entropy={entropy_loss.item():.4f}, BC={bc_loss.item():.4f}, Total={loss.item():.4f}")

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"LR {lr_idx}: NaN/Inf loss at LR {lr:.2e}, stopping")
                    diverged = True
                    break

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                lr_losses.append(loss.item())

            if diverged or len(lr_losses) == 0:
                break

            # Average loss at this LR
            avg_loss = sum(lr_losses) / len(lr_losses)

            # Track smoothed loss
            if lr_idx == 0:
                smoothed_loss = avg_loss
            else:
                smoothed_loss = smooth_factor * avg_loss + (1 - smooth_factor) * smoothed_loss

            self.lrs.append(lr)
            self.losses.append(avg_loss)
            self.smoothed_losses.append(smoothed_loss)

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > best_loss * diverge_threshold:
                logger.info(f"LR {lr_idx}: Loss diverged at LR {lr:.2e} (loss={smoothed_loss:.4f}, best={best_loss:.4f})")
                break

            if lr_idx % 20 == 0:
                logger.info(f"LR {lr_idx}/{num_steps}: LR={lr:.2e}, Loss={smoothed_loss:.4f}")

            # Update LR for next iteration
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        logger.info(f"LR range test complete: {len(self.lrs)} steps")
        return self.lrs, self.smoothed_losses

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> Tuple[float, float, float]:
        """Suggest optimal and maximum learning rates.

        Args:
            skip_start: Skip first N points (often noisy)
            skip_end: Skip last N points (often diverging)

        Returns:
            Tuple of (suggested_lr, steepest_lr, max_lr)
            - suggested_lr: Practical LR for training (geometric mean of steepest and max)
            - steepest_lr: LR at steepest loss descent
            - max_lr: LR at loss minimum (before divergence)
        """
        if len(self.lrs) < skip_start + skip_end + 10:
            logger.warning("Not enough data points for reliable suggestion")
            return 1e-4, 1e-4, 1e-3

        lrs = np.array(self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:])
        losses = np.array(self.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.smoothed_losses[skip_start:])

        # Find steepest descent (most negative gradient)
        # Use log scale for LR
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Find minimum gradient (steepest descent)
        min_grad_idx = np.argmin(gradients)
        steepest_lr = lrs[min_grad_idx]

        # Find max LR (where loss is minimum, before divergence)
        min_loss_idx = np.argmin(losses)
        max_lr = lrs[min_loss_idx]

        # Suggested LR: geometric mean between steepest descent and max LR
        # This gives a practical LR in the "sweet spot" of the curve
        # Old method divided steepest by 10, which was too conservative
        suggested_lr = np.sqrt(steepest_lr * max_lr)

        # Clamp suggested to be at least steepest_lr (don't go lower than where learning starts)
        suggested_lr = max(suggested_lr, steepest_lr)

        return suggested_lr, steepest_lr, max_lr

    def plot(self, output_path: Optional[str] = None, show: bool = True):
        """Plot LR vs Loss curve.

        Args:
            output_path: Path to save plot (optional)
            show: Whether to display plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed, cannot plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Loss vs LR (log scale)
        ax1.plot(self.lrs, self.losses, alpha=0.3, label="Raw Loss")
        ax1.plot(self.lrs, self.smoothed_losses, label="Smoothed Loss", linewidth=2)
        ax1.set_xscale("log")
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Loss")
        ax1.set_title("LR Range Test")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add suggested LR annotation
        suggested_lr, steepest_lr, max_lr = self.suggest_lr()
        ax1.axvline(x=steepest_lr, color="b", linestyle=":", alpha=0.7, label=f"Steepest: {steepest_lr:.2e}")
        ax1.axvline(x=suggested_lr, color="g", linestyle="--", linewidth=2, label=f"Suggested: {suggested_lr:.2e}")
        ax1.axvline(x=max_lr, color="r", linestyle="--", label=f"Max: {max_lr:.2e}")
        ax1.legend()

        # Plot 2: Loss gradient vs LR
        if len(self.lrs) > 20:
            log_lrs = np.log10(self.lrs[10:-5])
            losses = np.array(self.smoothed_losses[10:-5])
            gradients = np.gradient(losses, log_lrs)

            ax2.plot(self.lrs[10:-5], gradients)
            ax2.set_xscale("log")
            ax2.set_xlabel("Learning Rate")
            ax2.set_ylabel("Loss Gradient (d_loss / d_log_lr)")
            ax2.set_title("Loss Gradient (negative = loss decreasing)")
            ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Mark minimum gradient
            min_grad_idx = np.argmin(gradients)
            ax2.axvline(x=self.lrs[10 + min_grad_idx], color="g", linestyle="--")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()


class BCDataLoader:
    """Simple DataLoader for BC dataset."""

    def __init__(self, npz_path: str, batch_size: int = 256, normalize: bool = True):
        data = np.load(npz_path, allow_pickle=True)
        self.states = data["states"].astype(np.float32)
        self.type_labels = data["type_labels"].astype(np.int64)
        self.size_labels = data["size_labels"].astype(np.int64)
        self.amount_labels = data["amount_labels"].astype(np.int64)

        # Normalize states
        if normalize:
            mean = self.states.mean(axis=0, keepdims=True)
            std = self.states.std(axis=0, keepdims=True) + 1e-8
            self.states = (self.states - mean) / std
            self.states = np.clip(self.states, -5, 5)

        self.batch_size = batch_size
        self.n_samples = len(self.states)
        self.has_rollout_data = False
        logger.info(f"Loaded BC dataset: {self.n_samples} samples, {self.states.shape[1]} features")


class RolloutDataLoader:
    """DataLoader with real rollout data for accurate LR finding."""

    def __init__(self, data_dir: str, config, policy_net, value_net, device,
                 batch_size: int = 256, n_rollout_steps: int = 2048, bc_npz: str = None):
        from rl_editor.data import PairedAudioDataset
        from rl_editor.environment import AudioEditingEnvFactored
        from rl_editor.state import AudioState

        self.batch_size = batch_size
        self.device = device

        # Load BC data for normalization statistics
        if bc_npz:
            logger.info(f"Loading BC data for normalization from {bc_npz}")
            bc_data = np.load(bc_npz, allow_pickle=True)
            bc_states = bc_data["states"].astype(np.float32)
            self.norm_mean = bc_states.mean(axis=0, keepdims=True)
            self.norm_std = bc_states.std(axis=0, keepdims=True) + 1e-8
            del bc_data, bc_states
            logger.info(f"BC norm stats: mean={self.norm_mean.mean():.4f}, std={self.norm_std.mean():.4f}")
            self.need_initial_norm = False
        else:
            # Will compute running normalization from first states
            self.norm_mean = None
            self.norm_std = None
            self.need_initial_norm = True

        # Load audio files
        logger.info(f"Loading audio files from {data_dir}")
        dataset = PairedAudioDataset(data_dir, config)
        if len(dataset.pairs) == 0:
            raise ValueError(f"No audio pairs found in {data_dir}")

        # Get a sample and create AudioState
        sample = dataset[0]
        raw_data = sample["raw"]
        beat_times = raw_data["beat_times"]
        beat_features = raw_data["beat_features"]
        if hasattr(beat_times, 'numpy'):
            beat_times = beat_times.numpy()
        if hasattr(beat_features, 'numpy'):
            beat_features = beat_features.numpy()
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

        # Create environment with audio_state
        env = AudioEditingEnvFactored(config, audio_state=audio_state)

        # Collect rollout data
        logger.info(f"Collecting {n_rollout_steps} rollout steps...")
        states_list = []
        actions_list = []  # (type, size, amount)
        old_log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []

        state, _ = env.reset()
        policy_net.eval()
        value_net.eval()

        # Collect initial states to compute normalization (if needed)
        if self.need_initial_norm:
            logger.info("Collecting initial states for normalization...")
            initial_states = [state.copy()]
            for _ in range(min(500, n_rollout_steps // 4)):
                # Random action to sample diverse states
                action_arr = np.array([
                    np.random.randint(20),
                    np.random.randint(5),
                    np.random.randint(5)
                ], dtype=np.int64)
                next_state, _, terminated, truncated, _ = env.step(action_arr)
                if terminated or truncated:
                    # Reset with a random sample
                    sample_idx = np.random.randint(len(dataset))
                    sample = dataset[sample_idx]
                    raw_data = sample["raw"]
                    bt = raw_data["beat_times"]
                    bf = raw_data["beat_features"]
                    if hasattr(bt, 'numpy'):
                        bt = bt.numpy()
                    if hasattr(bf, 'numpy'):
                        bf = bf.numpy()
                    t = raw_data.get("tempo", 120)
                    if hasattr(t, 'item'):
                        t = t.item()
                    new_audio_state = AudioState(
                        beat_index=0,
                        beat_times=bt,
                        beat_features=bf,
                        tempo=t,
                        raw_audio=raw_data.get("audio"),
                        sample_rate=raw_data.get("sample_rate", 22050),
                    )
                    state, _ = env.reset(options={"audio_state": new_audio_state})
                else:
                    state = next_state
                initial_states.append(state.copy())

            initial_states = np.array(initial_states, dtype=np.float32)
            self.norm_mean = initial_states.mean(axis=0, keepdims=True)
            self.norm_std = initial_states.std(axis=0, keepdims=True) + 1e-8
            logger.info(f"Initial norm stats: mean={self.norm_mean.mean():.4f}, std={self.norm_std.mean():.4f}")

            # Reset environment for actual rollout
            sample = dataset[0]
            raw_data = sample["raw"]
            bt = raw_data["beat_times"]
            bf = raw_data["beat_features"]
            if hasattr(bt, 'numpy'):
                bt = bt.numpy()
            if hasattr(bf, 'numpy'):
                bf = bf.numpy()
            t = raw_data.get("tempo", 120)
            if hasattr(t, 'item'):
                t = t.item()
            audio_state = AudioState(
                beat_index=0,
                beat_times=bt,
                beat_features=bf,
                tempo=t,
                raw_audio=raw_data.get("audio"),
                sample_rate=raw_data.get("sample_rate", 22050),
            )
            state, _ = env.reset(options={"audio_state": audio_state})

        with torch.no_grad():
            for step in range(n_rollout_steps):
                # Normalize state using computed stats
                if self.norm_mean is not None:
                    state_normalized = (state - self.norm_mean.squeeze()) / self.norm_std.squeeze()
                    state_normalized = np.clip(state_normalized, -5, 5)
                else:
                    state_normalized = state

                state_t = torch.from_numpy(state_normalized).float().unsqueeze(0).to(device)

                # Check for NaN in state
                if torch.isnan(state_t).any():
                    logger.warning(f"Step {step}: NaN in state, replacing with zeros")
                    state_t = torch.nan_to_num(state_t, nan=0.0)

                # Get policy outputs
                encoded = policy_net.encoder(state_t)
                type_logits = policy_net.type_head(encoded)

                # Check for NaN in logits
                if torch.isnan(type_logits).any() or torch.isinf(type_logits).any():
                    logger.warning(f"Step {step}: Invalid type_logits, using uniform")
                    type_logits = torch.zeros_like(type_logits)

                type_dist = torch.distributions.Categorical(logits=type_logits)
                action_type = type_dist.sample()

                type_embed = policy_net.type_embedding(action_type)
                size_input = torch.cat([encoded, type_embed], dim=-1)
                size_logits = policy_net.size_head(size_input)
                if torch.isnan(size_logits).any() or torch.isinf(size_logits).any():
                    size_logits = torch.zeros_like(size_logits)
                size_dist = torch.distributions.Categorical(logits=size_logits)
                action_size = size_dist.sample()

                amount_input = torch.cat([encoded, type_embed], dim=-1)
                amount_logits = policy_net.amount_head(amount_input)
                if torch.isnan(amount_logits).any() or torch.isinf(amount_logits).any():
                    amount_logits = torch.zeros_like(amount_logits)
                amount_dist = torch.distributions.Categorical(logits=amount_logits)
                action_amount = amount_dist.sample()

                log_prob = (
                    type_dist.log_prob(action_type) +
                    size_dist.log_prob(action_size) +
                    amount_dist.log_prob(action_amount)
                )

                value = value_net(state_t).squeeze(-1)

                # Store
                states_list.append(state)
                actions_list.append((action_type.item(), action_size.item(), action_amount.item()))
                old_log_probs_list.append(log_prob.item())
                values_list.append(value.item())

                # Step environment - pass action as numpy array [type, size, amount]
                action_arr = np.array([action_type.item(), action_size.item(), action_amount.item()], dtype=np.int64)
                next_state, reward, terminated, truncated, info = env.step(action_arr)
                done = terminated or truncated
                rewards_list.append(reward)
                dones_list.append(done)

                if done:
                    # Reset with a new random sample from dataset
                    sample_idx = np.random.randint(len(dataset))
                    sample = dataset[sample_idx]
                    raw_data = sample["raw"]
                    bt = raw_data["beat_times"]
                    bf = raw_data["beat_features"]
                    if hasattr(bt, 'numpy'):
                        bt = bt.numpy()
                    if hasattr(bf, 'numpy'):
                        bf = bf.numpy()
                    t = raw_data.get("tempo", 120)
                    if hasattr(t, 'item'):
                        t = t.item()
                    new_audio_state = AudioState(
                        beat_index=0,
                        beat_times=bt,
                        beat_features=bf,
                        tempo=t,
                        raw_audio=raw_data.get("audio"),
                        sample_rate=raw_data.get("sample_rate", 22050),
                    )
                    state, _ = env.reset(options={"audio_state": new_audio_state})
                else:
                    state = next_state

                if (step + 1) % 500 == 0:
                    logger.info(f"  Collected {step + 1}/{n_rollout_steps} steps")

        # Convert to arrays
        self.states = np.array(states_list, dtype=np.float32)
        self.action_types = np.array([a[0] for a in actions_list], dtype=np.int64)
        self.action_sizes = np.array([a[1] for a in actions_list], dtype=np.int64)
        self.action_amounts = np.array([a[2] for a in actions_list], dtype=np.int64)
        self.old_log_probs = np.array(old_log_probs_list, dtype=np.float32)
        self.values = np.array(values_list, dtype=np.float32)
        self.rewards = np.array(rewards_list, dtype=np.float32)
        self.dones = np.array(dones_list, dtype=bool)

        # Compute returns and advantages (GAE)
        gamma = 0.99
        gae_lambda = 0.95
        self.returns = np.zeros_like(self.rewards)
        self.advantages = np.zeros_like(self.rewards)

        last_gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1] if not self.dones[t] else 0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Normalize states using BC stats (if available) for consistency with training
        if self.norm_mean is not None:
            self.states = (self.states - self.norm_mean) / self.norm_std
        else:
            # Fallback to rollout data's own stats
            mean = self.states.mean(axis=0, keepdims=True)
            std = self.states.std(axis=0, keepdims=True) + 1e-8
            self.states = (self.states - mean) / std
        self.states = np.clip(self.states, -5, 5)

        self.n_samples = len(self.states)
        self.has_rollout_data = True
        logger.info(f"Collected rollout data: {self.n_samples} samples, reward mean={self.rewards.mean():.2f}")

    def __iter__(self):
        perm = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            idx = perm[i : i + self.batch_size]
            yield (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.action_types[idx]),
                torch.from_numpy(self.action_sizes[idx]),
                torch.from_numpy(self.action_amounts[idx]),
                torch.from_numpy(self.old_log_probs[idx]),
                torch.from_numpy(self.returns[idx]),
                torch.from_numpy(self.advantages[idx]),
            )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class BCDataLoaderIter:
    """Iterator for BC dataset that also works with rollout mode."""

    def __init__(self, loader):
        self.loader = loader
        self.perm = np.random.permutation(loader.n_samples)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.loader.n_samples:
            raise StopIteration
        batch_idx = self.perm[self.idx : self.idx + self.loader.batch_size]
        self.idx += self.loader.batch_size
        return (
            torch.from_numpy(self.loader.states[batch_idx]),
            torch.from_numpy(self.loader.type_labels[batch_idx]),
            torch.from_numpy(self.loader.size_labels[batch_idx]),
            torch.from_numpy(self.loader.amount_labels[batch_idx]),
        )


# Add __iter__ to BCDataLoader
BCDataLoader.__iter__ = lambda self: BCDataLoaderIter(self)
BCDataLoader.__len__ = lambda self: (self.n_samples + self.batch_size - 1) // self.batch_size


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Finder")
    parser.add_argument("--bc_npz", type=str, default=None, help="Path to BC dataset NPZ (required if no --data_dir)")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory for real rollouts (more accurate)")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR to test")
    parser.add_argument("--max_lr", type=float, default=1e-1, help="Maximum LR to test")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of LR values to test")
    parser.add_argument("--steps_per_lr", type=int, default=1, help="Training steps per LR (higher = more accurate)")
    parser.add_argument("--reset", action="store_true", help="Reset model at each LR (independent mode, slower but more accurate)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load model weights from checkpoint")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--output", type=str, default="lr_finder.png", help="Output plot path")
    parser.add_argument("--no_show", action="store_true", help="Don't display plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Auto-adjust steps_per_lr for reset mode if not explicitly set
    steps_per_lr = args.steps_per_lr
    if args.reset and args.steps_per_lr == 1:
        steps_per_lr = 50  # Need more steps when resetting to see convergence
        logger.info(f"Reset mode: auto-setting steps_per_lr={steps_per_lr}")

    # Validate arguments
    if not args.bc_npz and not args.data_dir:
        raise ValueError("Either --bc_npz or --data_dir must be provided")

    # Create model first (needed for rollout data collection)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rl_editor.config import Config
    from rl_editor.agent import PolicyNetwork, ValueNetwork

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get input_dim from BC data or default
    if args.bc_npz:
        # Peek at BC data for input_dim
        import numpy as _np
        _data = _np.load(args.bc_npz, allow_pickle=True)
        input_dim = _data["states"].shape[1]
        del _data
    else:
        input_dim = 861  # Default for this project

    # Create policy network
    model = PolicyNetwork(config, input_dim=input_dim)
    model.to(device)

    # Create value network (shares encoder architecture)
    value_net = ValueNetwork(config, input_dim=input_dim)
    value_net.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("policy_state_dict"):
            model.load_state_dict(checkpoint["policy_state_dict"])
            logger.info("Loaded policy weights from checkpoint")
        else:
            logger.warning("No policy_state_dict in checkpoint")
        if checkpoint.get("value_state_dict"):
            value_net.load_state_dict(checkpoint["value_state_dict"])
            logger.info("Loaded value weights from checkpoint")

    # Load data (rollout or BC)
    if args.data_dir:
        logger.info("Using REAL ROLLOUT data for LR finding (most accurate)")
        loader = RolloutDataLoader(
            args.data_dir, config, model, value_net, device,
            batch_size=args.batch_size, n_rollout_steps=4096,
            bc_npz=args.bc_npz,  # Pass BC data for normalization stats
        )
    else:
        logger.info("Using BC data for LR finding (simulated PPO loss)")
        loader = BCDataLoader(args.bc_npz, batch_size=args.batch_size)

    # Create optimizer with both policy and value network parameters
    all_params = list(model.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(all_params, lr=args.min_lr)

    # Create LR finder
    finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        value_net=value_net,
    )

    # Run LR range test
    lrs, losses = finder.find(
        train_loader=loader,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps,
        steps_per_lr=steps_per_lr,
        reset_each_lr=args.reset,
    )

    # Get suggestions
    suggested_lr, steepest_lr, max_lr = finder.suggest_lr()
    print("\n" + "=" * 50)
    print("LR FINDER RESULTS")
    print("=" * 50)
    print(f"Suggested LR:  {suggested_lr:.2e}  (USE THIS - geometric mean of steepest & max)")
    print(f"Steepest LR:   {steepest_lr:.2e}  (where loss drops fastest)")
    print(f"Maximum LR:    {max_lr:.2e}  (loss minimum, before divergence)")
    print(f"Conservative:  {steepest_lr:.2e}  (if suggested is unstable)")
    print("=" * 50)

    # Plot
    finder.plot(output_path=args.output, show=not args.no_show)

    # Reset model to initial state
    finder.reset()


if __name__ == "__main__":
    main()

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store initial state to restore later
        self.initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.initial_optimizer_state = optimizer.state_dict()

        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []

    def reset(self):
        """Reset model and optimizer to initial state."""
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)

    def find(
        self,
        train_loader,
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_steps: int = 200,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> Tuple[List[float], List[float]]:
        """Run LR range test.

        Args:
            train_loader: DataLoader yielding (states, labels) batches
            min_lr: Starting learning rate
            max_lr: Maximum learning rate to test
            num_steps: Number of steps to run
            smooth_factor: Exponential smoothing factor for loss
            diverge_threshold: Stop if loss exceeds best_loss * threshold

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
        step = 0
        data_iter = iter(train_loader)

        logger.info(f"Running LR range test: {min_lr:.2e} -> {max_lr:.2e} over {num_steps} steps")

        while step < num_steps:
            # Get batch (cycle through data if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            states, type_labels, size_labels, amount_labels = batch
            states = states.to(self.device)
            type_labels = type_labels.to(self.device)
            size_labels = size_labels.to(self.device)
            amount_labels = amount_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            encoded = self.model.encoder(states)
            type_logits = self.model.type_head(encoded)

            # Use ground-truth type for size/amount (teacher forcing)
            type_embed = self.model.type_embedding(type_labels)
            size_input = torch.cat([encoded, type_embed], dim=-1)
            size_logits = self.model.size_head(size_input)
            amount_input = torch.cat([encoded, type_embed], dim=-1)
            amount_logits = self.model.amount_head(amount_input)

            # Compute loss (normalized by 3 heads)
            ce = nn.CrossEntropyLoss()
            loss_type = ce(type_logits, type_labels)
            loss_size = ce(size_logits, size_labels)
            loss_amount = ce(amount_logits, amount_labels)
            loss = (loss_type + loss_size + loss_amount) / 3.0

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Step {step}: NaN/Inf loss at LR {lr:.2e}, stopping")
                break

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track loss
            loss_val = loss.item()
            if step == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_factor * loss_val + (1 - smooth_factor) * smoothed_loss

            self.lrs.append(lr)
            self.losses.append(loss_val)
            self.smoothed_losses.append(smoothed_loss)

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > best_loss * diverge_threshold:
                logger.info(f"Step {step}: Loss diverged at LR {lr:.2e} (loss={smoothed_loss:.4f}, best={best_loss:.4f})")
                break

            # Update LR
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            step += 1

            if step % 20 == 0:
                logger.info(f"Step {step}/{num_steps}: LR={lr:.2e}, Loss={smoothed_loss:.4f}")

        logger.info(f"LR range test complete: {len(self.lrs)} steps")
        return self.lrs, self.smoothed_losses

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> Tuple[float, float]:
        """Suggest optimal and maximum learning rates.

        Args:
            skip_start: Skip first N points (often noisy)
            skip_end: Skip last N points (often diverging)

        Returns:
            Tuple of (suggested_lr, max_lr)
            - suggested_lr: LR at steepest loss descent (divide by 10 for safety)
            - max_lr: LR just before loss starts increasing
        """
        if len(self.lrs) < skip_start + skip_end + 10:
            logger.warning("Not enough data points for reliable suggestion")
            return 1e-4, 1e-3

        lrs = np.array(self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:])
        losses = np.array(self.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.smoothed_losses[skip_start:])

        # Find steepest descent (most negative gradient)
        # Use log scale for LR
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Find minimum gradient (steepest descent)
        min_grad_idx = np.argmin(gradients)
        steepest_lr = lrs[min_grad_idx]

        # Suggested LR is 1/10th of steepest point (conservative)
        suggested_lr = steepest_lr / 10.0

        # Find max LR (where loss starts increasing significantly)
        min_loss_idx = np.argmin(losses)
        max_lr = lrs[min_loss_idx]

        return suggested_lr, max_lr

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
        suggested_lr, max_lr = self.suggest_lr()
        ax1.axvline(x=suggested_lr, color="g", linestyle="--", label=f"Suggested: {suggested_lr:.2e}")
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
        logger.info(f"Loaded BC dataset: {self.n_samples} samples, {self.states.shape[1]} features")

    def __iter__(self):
        perm = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            idx = perm[i : i + self.batch_size]
            yield (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.type_labels[idx]),
                torch.from_numpy(self.size_labels[idx]),
                torch.from_numpy(self.amount_labels[idx]),
            )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Finder")
    parser.add_argument("--bc_npz", type=str, required=True, help="Path to BC dataset NPZ")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR to test")
    parser.add_argument("--max_lr", type=float, default=1e-1, help="Maximum LR to test")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--output", type=str, default="lr_finder.png", help="Output plot path")
    parser.add_argument("--no_show", action="store_true", help="Don't display plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Load BC data
    loader = BCDataLoader(args.bc_npz, batch_size=args.batch_size)
    input_dim = loader.states.shape[1]

    # Create model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rl_editor.config import Config
    from rl_editor.agent import PolicyNetwork

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = PolicyNetwork(config, input_dim=input_dim)
    model.to(device)

    # Create optimizer with dummy LR (will be overwritten)
    optimizer = optim.Adam(model.parameters(), lr=args.min_lr)

    # Create LR finder
    finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    # Run LR range test
    lrs, losses = finder.find(
        train_loader=loader,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps,
    )

    # Get suggestions
    suggested_lr, max_lr = finder.suggest_lr()
    print("\n" + "=" * 50)
    print("LR FINDER RESULTS")
    print("=" * 50)
    print(f"Suggested LR:  {suggested_lr:.2e}  (use this for training)")
    print(f"Maximum LR:    {max_lr:.2e}  (loss minimum, may be unstable)")
    print(f"Conservative:  {suggested_lr/3:.2e}  (safer, slower convergence)")
    print("=" * 50)

    # Plot
    finder.plot(output_path=args.output, show=not args.no_show)

    # Reset model to initial state
    finder.reset()


if __name__ == "__main__":
    main()

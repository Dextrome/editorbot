"""Learning Rate Finder for Pointer Network.

Implements the LR range test from Leslie Smith's paper:
"Cyclical Learning Rates for Training Neural Networks"

Usage:
    python -m pointer_network.lr_finder --pointer-dir training_data/pointer_sequences

The script will:
1. Start with a very small LR (1e-8)
2. Exponentially increase LR each step
3. Record loss at each step
4. Stop when loss explodes (>4x min loss)
5. Plot loss vs LR and save to lr_finder/

Pick the LR where loss is decreasing fastest (before it starts increasing).
A good rule: use LR about 10x lower than the minimum loss point.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from typing import Optional, List, Tuple
import json

from .models import PointerNetwork, EditOp, STOP_TOKEN
from .data.dataset import PointerDataset, collate_fn
from .config import PointerNetworkConfig


class LRFinder:
    """Learning Rate Finder using exponential LR increase."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16

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
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []

    def find(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-8,
        end_lr: float = 1e-1,
        num_steps: Optional[int] = None,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
    ) -> Tuple[List[float], List[float]]:
        """Run LR finder.

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_steps: Number of steps (default: len(train_loader))
            smooth_factor: Smoothing factor for loss (EMA)
            diverge_threshold: Stop if loss > diverge_threshold * min_loss

        Returns:
            Tuple of (lrs, smoothed_losses)
        """
        if num_steps is None:
            num_steps = len(train_loader)

        # Calculate LR multiplier per step
        lr_mult = (end_lr / start_lr) ** (1 / num_steps)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        self.model.train()
        best_loss = float('inf')
        avg_loss = 0.0

        data_iter = iter(train_loader)
        pbar = tqdm(range(num_steps), desc="LR Finder")

        for step in pbar:
            # Get batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Current LR
            lr = self.optimizer.param_groups[0]['lr']

            # Forward pass
            raw_mel = batch['raw_mel'].to(self.device)
            target_pointers = batch['target_pointers'].to(self.device)

            # Get stems if available
            stems = batch.get('stems')
            if stems is not None:
                stems = {k: v.to(self.device) for k, v in stems.items()}

            # Create target ops
            target_ops = torch.full_like(target_pointers, EditOp.COPY)
            stop_mask = target_pointers == STOP_TOKEN
            target_ops[stop_mask] = EditOp.STOP

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    raw_mel=raw_mel,
                    target_ops=target_ops,
                    target_pointers=target_pointers,
                    stems=stems,
                )
                loss = outputs['loss']

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nLoss is NaN/Inf at LR={lr:.2e}, stopping")
                break

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Record
            loss_val = loss.item()
            self.lrs.append(lr)
            self.losses.append(loss_val)

            # Smoothed loss (exponential moving average)
            if step == 0:
                avg_loss = loss_val
            else:
                avg_loss = smooth_factor * loss_val + (1 - smooth_factor) * avg_loss
            self.smoothed_losses.append(avg_loss)

            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Check divergence
            if avg_loss > diverge_threshold * best_loss:
                print(f"\nLoss diverged at LR={lr:.2e} (loss={avg_loss:.4f} > {diverge_threshold}x best={best_loss:.4f})")
                break

            # Update LR for next step
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult

            pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{loss_val:.4f}', 'smooth': f'{avg_loss:.4f}'})

        return self.lrs, self.smoothed_losses

    def plot(self, save_path: Optional[Path] = None, skip_start: int = 10, skip_end: int = 5):
        """Plot loss vs learning rate.

        Args:
            save_path: Path to save plot
            skip_start: Skip first N points (often noisy)
            skip_end: Skip last N points (often diverged)
        """
        if len(self.lrs) < skip_start + skip_end + 10:
            print("Not enough data points to plot")
            return

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.smoothed_losses[skip_start:]

        # Find suggested LR (steepest negative gradient)
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]

        # Also find min loss LR
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Loss vs LR (log scale)
        ax1.plot(lrs, losses, 'b-', linewidth=1)
        ax1.axvline(suggested_lr, color='r', linestyle='--', label=f'Suggested: {suggested_lr:.2e}')
        ax1.axvline(min_loss_lr, color='g', linestyle=':', label=f'Min loss: {min_loss_lr:.2e}')
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate (log scale)')
        ax1.set_ylabel('Loss (smoothed)')
        ax1.set_title('LR Finder: Loss vs Learning Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss gradient vs LR
        ax2.plot(lrs, gradients, 'b-', linewidth=1)
        ax2.axvline(suggested_lr, color='r', linestyle='--', label=f'Steepest: {suggested_lr:.2e}')
        ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate (log scale)')
        ax2.set_ylabel('Loss Gradient')
        ax2.set_title('LR Finder: Loss Gradient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

        # Print recommendations
        print("\n" + "="*50)
        print("LR FINDER RESULTS")
        print("="*50)
        print(f"Suggested LR (steepest descent): {suggested_lr:.2e}")
        print(f"Min loss LR: {min_loss_lr:.2e}")
        print(f"Conservative LR (10x lower): {suggested_lr / 10:.2e}")
        print(f"Aggressive LR (at min): {min_loss_lr:.2e}")
        print("="*50)
        print("\nRecommendation: Start with the conservative LR and increase if stable.")

        return {
            'suggested_lr': suggested_lr,
            'min_loss_lr': min_loss_lr,
            'conservative_lr': suggested_lr / 10,
        }


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Finder for Pointer Network")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--pointer-dir', default='training_data/pointer_sequences',
                        help='Directory with pointer sequences')
    parser.add_argument('--cache-dir', default='cache',
                        help='Cache directory for mel spectrograms')
    parser.add_argument('--start-lr', type=float, default=1e-8,
                        help='Starting learning rate')
    parser.add_argument('--end-lr', type=float, default=1e-1,
                        help='Ending learning rate')
    parser.add_argument('--num-steps', type=int, default=500,
                        help='Number of steps for LR finder')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--output-dir', default='lr_finder',
                        help='Output directory for plots')
    parser.add_argument('--use-pre-norm', action='store_true', default=True,
                        help='Use Pre-LayerNorm (more stable, default: True)')
    parser.add_argument('--use-edit-ops', action='store_true', default=True,
                        help='Use edit operation tokens (default: True)')
    args = parser.parse_args()

    # Load config from file if provided
    cfg = {}
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        model_cfg = cfg.get('model', {})
    else:
        model_cfg = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model config
    config = PointerNetworkConfig()

    # Override with config file values
    use_pre_norm = model_cfg.get('use_pre_norm', getattr(args, 'use_pre_norm', True))
    use_edit_ops = model_cfg.get('use_edit_ops', getattr(args, 'use_edit_ops', True))
    use_stems = model_cfg.get('use_stems', False)
    n_stems = model_cfg.get('n_stems', 4)
    d_model = model_cfg.get('d_model', config.d_model)
    n_heads = model_cfg.get('n_heads', config.n_heads)
    n_encoder_layers = model_cfg.get('n_encoder_layers', config.n_encoder_layers)
    n_decoder_layers = model_cfg.get('n_decoder_layers', config.n_decoder_layers)
    dropout = model_cfg.get('dropout', config.dropout)
    chunk_size = model_cfg.get('chunk_size', config.chunk_size)

    # Get dirs from config or args
    cache_dir = cfg.get('cache_dir', args.cache_dir)
    pointer_dir = cfg.get('pointer_dir', args.pointer_dir)

    print(f"Model: d_model={d_model}, n_heads={n_heads}, enc={n_encoder_layers}, dec={n_decoder_layers}")
    print(f"Features: pre_norm={use_pre_norm}, edit_ops={use_edit_ops}, stems={use_stems}")

    model = PointerNetwork(
        n_mels=config.n_mels,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        dropout=dropout,
        use_pre_norm=use_pre_norm,
        use_edit_ops=use_edit_ops,
        use_stems=use_stems,
        n_stems=n_stems,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.start_lr, weight_decay=0.01)

    # Create dataset with stems if enabled
    stems_dir = f"{cache_dir}/stems_mel" if use_stems else None
    dataset = PointerDataset(
        cache_dir=cache_dir,
        pointer_dir=pointer_dir,
        chunk_size=chunk_size,
        use_mmap=True,
        preload_pointers=True,
        use_stems=use_stems,
        stems_dir=stems_dir,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Dataset: {len(train_loader)} batches")

    # Run LR finder
    lr_finder = LRFinder(model, optimizer, device=device, use_amp=True)

    print(f"\nRunning LR finder from {args.start_lr:.2e} to {args.end_lr:.2e}...")
    lrs, losses = lr_finder.find(
        train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_steps=args.num_steps,
    )

    # Plot and get recommendations
    results = lr_finder.plot(save_path=output_dir / "lr_finder_plot.png")

    # Save results to JSON
    if results:
        with open(output_dir / "lr_finder_results.json", 'w') as f:
            json.dump({
                'suggested_lr': results['suggested_lr'],
                'min_loss_lr': results['min_loss_lr'],
                'conservative_lr': results['conservative_lr'],
                'num_steps': len(lrs),
                'final_loss': losses[-1] if losses else None,
            }, f, indent=2)
        print(f"\nResults saved to {output_dir / 'lr_finder_results.json'}")


if __name__ == "__main__":
    main()

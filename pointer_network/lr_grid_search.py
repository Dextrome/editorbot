"""LR Grid Search - Run short training sessions at different LRs to find optimal.

More reliable than exponential LR finder because it uses:
- Proper warmup
- Real training dynamics
- Validation metrics

Usage:
    python -m pointer_network.lr_grid_search --config pointer_network/configs/full.json
"""

import torch
import json
import argparse
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import sys

from .trainers.pointer_trainer import PointerNetworkTrainer, load_config
from .config import TrainConfig, PointerNetworkConfig


def run_lr_search(config_path: str, lrs: list, epochs_per_lr: int = 5):
    """Run grid search over learning rates."""

    results = []

    print("=" * 60)
    print("LR GRID SEARCH")
    print("=" * 60)
    print(f"Learning rates: {[f'{lr:.1e}' for lr in lrs]}")
    print(f"Epochs per LR: {epochs_per_lr}")
    print("=" * 60)

    for i, lr in enumerate(lrs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(lrs)}] Testing LR = {lr:.1e}")
        print(f"{'='*60}\n")

        # Load fresh config for each run
        config = load_config(config_path)

        # Modify for this LR test
        config = replace(
            config,
            learning_rate=lr,
            epochs=epochs_per_lr,
            save_dir=f"models/lr_search/lr_{lr:.0e}",
            warmup_steps=min(config.warmup_steps, epochs_per_lr * 50),  # Scale warmup
        )

        # Create trainer
        trainer = PointerNetworkTrainer(config)

        try:
            # Train and capture final validation metrics
            trainer.train()

            # Get metrics (best_val_loss is tracked during training)
            best_val_loss = trainer.best_val_loss

            # Check for best_val_accuracy if tracked, otherwise use 0
            best_val_acc = getattr(trainer, 'best_val_accuracy', 0.0)

            results.append({
                'lr': lr,
                'best_val_loss': best_val_loss,
                'final_val_acc': best_val_acc,
                'status': 'ok'
            })

            print(f"\n>>> [LR={lr:.1e}] best_val_loss: {best_val_loss:.4f}")

        except Exception as e:
            print(f"\n>>> [LR={lr:.1e}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'lr': lr,
                'best_val_loss': float('inf'),
                'final_val_acc': 0.0,
                'status': f'failed: {str(e)[:50]}'
            })

        # Clear CUDA cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up trainer
        del trainer

    # Print summary table
    print("\n" + "=" * 70)
    print("LR GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"{'LR':<12} {'Val Loss':<12} {'Val Acc':<12} {'Status':<20}")
    print("-" * 70)

    for r in results:
        lr_str = f"{r['lr']:.1e}"
        loss_str = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] != float('inf') else "N/A"
        acc_str = f"{r['final_val_acc']:.4f}" if r['final_val_acc'] > 0 else "N/A"
        print(f"{lr_str:<12} {loss_str:<12} {acc_str:<12} {r['status']:<20}")

    print("-" * 70)

    # Find best
    valid_results = [r for r in results if r['status'] == 'ok']
    if valid_results:
        best_by_loss = min(valid_results, key=lambda x: x['best_val_loss'])
        best_by_acc = max(valid_results, key=lambda x: x['final_val_acc'])

        print(f"\nBest by val_loss: LR={best_by_loss['lr']:.1e} (loss={best_by_loss['best_val_loss']:.4f})")
        print(f"Best by val_acc:  LR={best_by_acc['lr']:.1e} (acc={best_by_acc['final_val_acc']:.4f})")

        # Recommendation
        print(f"\n>>> RECOMMENDED LR: {best_by_acc['lr']:.1e}")

    print("=" * 70)

    # Save results
    output_dir = Path("lr_finder")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "lr_grid_search_results.json"

    # Convert for JSON serialization
    json_results = []
    for r in results:
        json_results.append({
            'lr': r['lr'],
            'best_val_loss': r['best_val_loss'] if r['best_val_loss'] != float('inf') else None,
            'final_val_acc': r['final_val_acc'],
            'status': r['status']
        })

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LR Grid Search")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per LR (default: 5)')
    parser.add_argument('--lrs', type=str, default='1e-4,2e-4,3e-4,5e-4,1e-3,4e-3',
                        help='Comma-separated LRs to test')
    args = parser.parse_args()

    # Parse LRs
    lrs = [float(x) for x in args.lrs.split(',')]

    run_lr_search(args.config, lrs, args.epochs)


if __name__ == "__main__":
    main()

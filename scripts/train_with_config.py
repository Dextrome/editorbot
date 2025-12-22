#!/usr/bin/env python
"""Wrapper to run training with programmatic config overrides.

This script imports the project's config and `train.train` function, applies
CLI-provided overrides (learning rate, PPO params, etc.), and runs a short
training job. It's intended to be called by hyperopt scripts as a subprocess
so each trial runs in an isolated process.

Example:
  python scripts/train_with_config.py --data_dir training_data --save_dir models/hpo_trial --epochs 1 --n_envs 4 --steps 256 --lr 1e-5 --clip_ratio 0.2 --entropy_coeff 0.2
"""
import argparse
import json
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="training_data")
    p.add_argument("--save_dir", type=str, default="models/hpo_tmp")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--val_audio", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    # PPO-configurable params
    p.add_argument("--clip_ratio", type=float, default=None)
    p.add_argument("--entropy_coeff", type=float, default=None)
    p.add_argument("--value_loss_coeff", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--subprocess", action="store_true")
    p.add_argument("--bc_mixed_weight", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    # Import training function and config factory
    from rl_editor.config import get_default_config
    from rl_editor import train as train_module

    cfg = get_default_config()
    # Override training-level settings
    cfg.training.save_dir = args.save_dir
    if args.lr is not None:
        cfg.ppo.learning_rate = args.lr
    if args.clip_ratio is not None:
        cfg.ppo.clip_ratio = args.clip_ratio
    if args.entropy_coeff is not None:
        cfg.ppo.entropy_coeff = args.entropy_coeff
    if args.value_loss_coeff is not None:
        cfg.ppo.value_loss_coeff = args.value_loss_coeff
    if args.max_grad_norm is not None:
        cfg.ppo.max_grad_norm = args.max_grad_norm
    if args.batch_size is not None:
        cfg.ppo.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.ppo.gradient_accumulation_steps = args.gradient_accumulation_steps

    # Ensure save dir exists
    Path(cfg.training.save_dir).mkdir(parents=True, exist_ok=True)

    # Run short training (calls train.train)
    train_module.train(
        config=cfg,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        val_audio=args.val_audio,
        checkpoint_path=None,
        bc_mixed_npz=None,
        bc_mixed_weight=args.bc_mixed_weight,
        bc_mixed_batch=64,
        use_subprocess=args.subprocess,
    )


if __name__ == "__main__":
    main()

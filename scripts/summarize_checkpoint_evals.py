#!/usr/bin/env python3
"""Summarize checkpoint evals CSV: moving average, linear trend, save plot.

Usage:
  python scripts/summarize_checkpoint_evals.py --csv output/checkpoint_evals.csv
"""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            epoch = int(r['epoch'])
            reward = None
            if r['reward']:
                try:
                    reward = float(r['reward'])
                except Exception:
                    reward = None
            rows.append((epoch, reward))
    return rows


def moving_average(x, w):
    if w <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='output/checkpoint_evals.csv')
    p.add_argument('--out_png', default='output/checkpoint_evals_trend.png')
    p.add_argument('--ma_window', type=int, default=3)
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print('CSV not found:', path)
        return 2

    rows = read_csv(path)
    rows = sorted([r for r in rows if r[1] is not None], key=lambda x: x[0])
    if not rows:
        print('No numeric rows in csv')
        return 2

    epochs = np.array([r[0] for r in rows], dtype=float)
    rewards = np.array([r[1] for r in rows], dtype=float)

    # moving average
    ma = moving_average(rewards, args.ma_window)

    # linear trend (slope per epoch)
    coeffs = np.polyfit(epochs, rewards, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(epochs, rewards, marker='o', label='reward')
    if len(ma) > 0:
        offset = len(epochs) - len(ma)
        plt.plot(epochs[offset:], ma, label=f'ma{args.ma_window}', color='r')
    # linear fit line
    xs = np.linspace(epochs.min(), epochs.max(), 100)
    plt.plot(xs, slope*xs + intercept, '--', label=f'linear fit (slope={slope:.4f})')
    plt.xlabel('epoch')
    plt.ylabel('deterministic reward')
    plt.title('Checkpoint deterministic evals')
    plt.legend()
    plt.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)

    # summary
    print(f'Rows: {len(epochs)} epochs from {int(epochs.min())} to {int(epochs.max())}')
    print(f'Reward: mean={rewards.mean():.4f} std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}')
    print(f'Linear trend slope per epoch = {slope:.6f} (intercept {intercept:.2f})')
    if slope > 0:
        print('Trend: improving (positive slope)')
    elif slope < 0:
        print('Trend: declining (negative slope)')
    else:
        print('Trend: flat')

    print(f'Wrote plot: {out_png}')
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

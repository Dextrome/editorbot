#!/usr/bin/env python3
"""Extract 'Eval deterministic' lines from train stdout logs and plot trends.

Usage:
  python scripts/extract_deterministic_evals.py --out_csv output/deterministic_evals.csv --out_png output/deterministic_evals.png

"""
import re
import sys
from pathlib import Path
import argparse
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .* - INFO - Eval deterministic(?: \(best\))?: reward=(?P<reward>[-0-9.eE]+) per-beat-keep_ratio=(?P<keep>[0-9.]+) edited_pct=(?P<edited>[0-9.]+)%"
)


def find_log_files(base: Path):
    # common locations: models/**/phase*_train_stdout.txt and any *_train_stdout.txt
    files = list(base.glob('models/**/phase*_train_stdout.txt'))
    files += list(base.glob('**/*_train_stdout.txt'))
    # dedupe
    seen = set()
    out = []
    for f in files:
        p = f.resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def parse_file(path: Path):
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as fh:
        for ln, line in enumerate(fh, start=1):
            m = LINE_RE.search(line)
            if not m:
                continue
            ts = m.group('ts')
            try:
                ts_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')
            except Exception:
                ts_dt = None
            reward = float(m.group('reward'))
            keep = float(m.group('keep'))
            edited = float(m.group('edited'))
            rows.append({'file': str(path), 'line': ln, 'ts': ts_dt, 'reward': reward, 'keep_ratio': keep, 'edited_pct': edited})
    return rows


def moving_average(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_csv', default='output/deterministic_evals.csv')
    p.add_argument('--out_png', default='output/deterministic_evals.png')
    p.add_argument('--workdir', default='.', help='workspace root')
    p.add_argument('--ma_window', type=int, default=5, help='moving average window')
    args = p.parse_args()

    base = Path(args.workdir)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    files = find_log_files(base)
    all_rows = []
    for f in files:
        rows = parse_file(f)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print('No deterministic eval lines found.')
        return 2

    # sort by timestamp if available, else by file+line
    all_rows.sort(key=lambda r: (r['ts'] or datetime.min, r['file'], r['line']))

    # write CSV
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['file', 'line', 'ts', 'reward', 'keep_ratio', 'edited_pct'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                'file': r['file'],
                'line': r['line'],
                'ts': r['ts'].isoformat() if r['ts'] else '',
                'reward': r['reward'],
                'keep_ratio': r['keep_ratio'],
                'edited_pct': r['edited_pct'],
            })

    # Plot rewards with moving average
    rewards = np.array([r['reward'] for r in all_rows], dtype=float)
    xs = np.arange(len(rewards))
    ma = moving_average(rewards, args.ma_window)

    plt.figure(figsize=(10, 4))
    plt.plot(xs, rewards, label='reward', alpha=0.4)
    if len(ma) > 0:
        plt.plot(xs[len(xs)-len(ma):], ma, label=f'ma{args.ma_window}', color='r')
    plt.xlabel('sample (sorted by time)')
    plt.ylabel('Eval deterministic reward')
    plt.title('Deterministic eval rewards (extracted from logs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f'Wrote {out_csv} ({len(all_rows)} rows) and plot {out_png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

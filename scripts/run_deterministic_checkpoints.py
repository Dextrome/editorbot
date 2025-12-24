#!/usr/bin/env python3
"""Run deterministic inference across recent checkpoints and save a CSV of scores.

This script finds checkpoint files under `models/` matching `checkpoint_epoch_*.pt`,
selects the most recent N (by epoch number), runs `python -m rl_editor.infer` with
`--deterministic --n-samples 1` for each, captures the reported cumulative reward,
and writes `output/checkpoint_evals.csv`.

Usage:
  python scripts/run_deterministic_checkpoints.py --input training_data/input/20250809blackhunger_raw.mp3 --n 12
"""
import re
import sys
import subprocess
from pathlib import Path
import csv
import argparse


CKPT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def find_checkpoints(models_dir: Path):
    cks = []
    for p in models_dir.iterdir():
        m = CKPT_RE.search(p.name)
        if m and p.is_file():
            epoch = int(m.group(1))
            cks.append((epoch, p))
    return sorted(cks)


def run_infer(python_exe: str, input_audio: str, checkpoint: Path, out_audio: Path, crossfade_ms: float = 50.0):
    cmd = [python_exe, '-m', 'rl_editor.infer', input_audio, '--checkpoint', str(checkpoint), '--deterministic', '--n-samples', '1', '--crossfade-ms', str(crossfade_ms), '--output', str(out_audio)]
    # Run and capture stdout
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def parse_reward_from_stdout(out: str):
    # prefer the 'Sample i: cumulative reward=' line
    m = re.search(r"Sample \d+: cumulative reward=([\-0-9\.eE]+)", out)
    if m:
        return float(m.group(1))
    # fallback: 'Saved chosen best sample .* \(score=...\)'
    m2 = re.search(r"score=([\-0-9\.eE]+)\)", out)
    if m2:
        return float(m2.group(1))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models_dir', default='models', help='Models directory')
    p.add_argument('--input', required=True, help='Validation input audio')
    p.add_argument('--out_csv', default='output/checkpoint_evals.csv')
    p.add_argument('--out_dir', default='output/checkpoint_evals')
    p.add_argument('--n', type=int, default=12, help='Number of most recent checkpoints to evaluate')
    p.add_argument('--crossfade_ms', type=float, default=50.0)
    args = p.parse_args()

    base = Path('.').resolve()
    py = sys.executable
    models = Path(args.models_dir)
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cks = find_checkpoints(models)
    if not cks:
        print('No checkpoint_epoch_*.pt files found under', models)
        return 2

    # pick most recent N by epoch
    selected = cks[-args.n:]

    rows = []
    for epoch, ck in selected:
        out_name = out_dir / f"edited_epoch_{epoch}.wav"
        print(f'Evaluating checkpoint epoch={epoch} -> {ck}')
        rc, out = run_infer(py, args.input, ck, out_name, crossfade_ms=args.crossfade_ms)
        if rc != 0:
            print(f'Infer failed for {ck} (rc={rc})')
            reward = None
        else:
            reward = parse_reward_from_stdout(out)
        rows.append({'epoch': epoch, 'checkpoint': str(ck), 'reward': reward})

    # write CSV
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['epoch', 'checkpoint', 'reward'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

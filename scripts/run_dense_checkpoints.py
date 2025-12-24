#!/usr/bin/env python3
"""Evaluate checkpoints at a denser temporal resolution.

Selects checkpoint files matching `checkpoint_epoch_*.pt` under --models_dir,
filters epochs by `epoch % step == 0`, then runs deterministic inference
`--repeats` times per checkpoint and records mean/std rewards to CSV.

Example:
  python scripts/run_dense_checkpoints.py --models_dir models/hyperoptV1 --input training_data/input/20250809blackhunger_raw.mp3 --step 100 --repeats 3
"""
import re
import sys
import subprocess
from pathlib import Path
import csv
import argparse
import statistics

CKPT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def find_checkpoints(models_dir: Path):
    cks = []
    for p in models_dir.rglob('checkpoint_epoch_*.pt'):
        m = CKPT_RE.search(p.name)
        if m and p.is_file():
            epoch = int(m.group(1))
            cks.append((epoch, p))
    return sorted(cks)


def run_infer(python_exe: str, input_audio: str, checkpoint: Path, out_audio: Path, crossfade_ms: float = 50.0):
    cmd = [python_exe, '-m', 'rl_editor.infer', input_audio, '--checkpoint', str(checkpoint), '--deterministic', '--n-samples', '1', '--crossfade-ms', str(crossfade_ms), '--output', str(out_audio)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def parse_reward_from_stdout(out: str):
    import re
    m = re.search(r"Sample \d+: cumulative reward=([\-0-9\.eE]+)", out)
    if m:
        return float(m.group(1))
    m2 = re.search(r"score=([\-0-9\.eE]+)\)", out)
    if m2:
        return float(m2.group(1))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models_dir', default='models', help='Models directory')
    p.add_argument('--input', required=True, help='Validation input audio')
    p.add_argument('--out_csv', default='output/dense_checkpoint_evals.csv')
    p.add_argument('--out_dir', default='output/dense_edited')
    p.add_argument('--step', type=int, default=100, help='Epoch step to select checkpoints')
    p.add_argument('--repeats', type=int, default=3, help='Repetitions per checkpoint')
    p.add_argument('--crossfade_ms', type=float, default=50.0)
    args = p.parse_args()

    models = Path(args.models_dir)
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cks = find_checkpoints(models)
    if not cks:
        print('No checkpoints found under', models)
        return 2

    # select epochs matching step
    selected = [(e,p) for e,p in cks if e % args.step == 0]
    if not selected:
        print(f'No checkpoints matching step={args.step} found; available epochs: {[e for e,_ in cks]}')
        return 2

    rows = []
    py = sys.executable
    import random
    for epoch, ck in selected:
        print(f'Evaluating epoch {epoch} ({ck})')
        rewards = []
        for r_i in range(args.repeats):
            # Use a different random seed per repeat to get variability
            seed = random.randint(0, 2**31 - 1)
            out_name = out_dir / f'epoch_{epoch}_run{r_i}.wav'
            cmd = [py, '-m', 'rl_editor.infer', args.input, '--checkpoint', str(ck), '--n-samples', '1', '--crossfade-ms', str(args.crossfade_ms), '--output', str(out_name), '--seed', str(seed)]
            # Prefer stochastic sampling for variance; do not pass --deterministic here
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            rc = proc.returncode
            out = proc.stdout
            if rc != 0:
                print(f'  run {r_i}: infer failed (rc={rc})')
                continue
            reward = parse_reward_from_stdout(out)
            print(f'  run {r_i}: reward={reward}  (seed={seed})')
            if reward is not None:
                rewards.append(reward)

        mean_r = statistics.mean(rewards) if rewards else None
        std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0 if rewards else None
        rows.append({'epoch': epoch, 'checkpoint': str(ck), 'n_runs': len(rewards), 'mean_reward': mean_r, 'std_reward': std_r})

    # write CSV
    import csv
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['epoch','checkpoint','n_runs','mean_reward','std_reward'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

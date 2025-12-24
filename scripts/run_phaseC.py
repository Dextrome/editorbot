#!/usr/bin/env python3
"""Phase C driver: run long confirmation jobs for top-N Phase B candidates.

Usage: python scripts/run_phaseC.py [top_n] [epochs] [val_audio]

Starts sequential runs that resume from the Phase B `best.pt` and saves into
`models/hpo_phaseC/<trial>`. Logs are written to `phaseC_train_stdout.txt`.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASEB = ROOT / 'models' / 'hpo_phaseB'
PHASEC = ROOT / 'models' / 'hpo_phaseC'
CSV = PHASEB / 'phaseB_summary.csv'


def read_summary(csv_path):
    rows = []
    if not csv_path.exists():
        raise SystemExit(f"Missing summary CSV: {csv_path}")
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            br = row.get('best_eval_reward', '')
            try:
                brf = float(br) if br not in (None, '', 'None') else None
            except Exception:
                brf = None
            rows.append((row['trial'], row.get('ckpt', ''), brf))
    return rows


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    val_audio = sys.argv[3] if len(sys.argv) > 3 else 'training_data/input/20251009eh_raw.wav'

    rows = read_summary(CSV)
    rows = [r for r in rows if r[1]]
    rows.sort(key=lambda x: (-(x[2]) if x[2] is not None else float('inf')))
    selected = rows[:top_n]

    PHASEC.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = env.get('PYTHONPATH', '.') or '.'

    for trial, ckpt, br in selected:
        dest = PHASEC / trial
        dest.mkdir(parents=True, exist_ok=True)
        stdout_path = dest / 'phaseC_train_stdout.txt'
        # Allow per-trial hyperparams via HYPERPARAMS_DIR environment variable
        hparams_dir = os.environ.get('HYPERPARAMS_DIR')
        hparams_file = None
        if hparams_dir:
            candidate = Path(hparams_dir) / f"{trial}.json"
            if candidate.exists():
                hparams_file = str(candidate)

        cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'train_with_config.py'),
            '--data_dir', 'training_data',
            '--save_dir', str(dest),
            '--epochs', str(epochs),
            '--n_envs', '4',
            '--steps', '512',
            '--checkpoint', ckpt,
        ]
        if hparams_file:
            cmd.extend(['--hparams_json', hparams_file])
        cmd.extend([
            '--subprocess',
            '--val_audio', val_audio,
        ])
        print(f"Starting Phase C for {trial}: ckpt={ckpt} -> {dest} (epochs={epochs})")
        with open(stdout_path, 'w', encoding='utf-8') as out:
            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
            ret = p.wait()
        print(f"Finished {trial} (exit {ret}), logs at {stdout_path}")


if __name__ == '__main__':
    main()

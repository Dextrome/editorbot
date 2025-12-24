#!/usr/bin/env python3
"""Phase B driver: re-evaluate top trials from Phase A.

Reads models/hpo_phaseA/phaseA_summary.csv, selects top N trials by training best reward,
and resumes training for `epochs` epochs per trial (saving into models/hpo_phaseB/<trial>). 
Runs deterministic eval during training via --val_audio.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASEA_DIR = ROOT / 'models' / 'hpo_phaseA'
PHASEB_DIR = ROOT / 'models' / 'hpo_phaseB'
CSV = PHASEA_DIR / 'phaseA_summary.csv'

def read_summary(csv_path):
    rows = []
    if not csv_path.exists():
        raise SystemExit(f"Missing summary CSV: {csv_path}")
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            name = row['trial']
            ck = row['ckpt']
            br = row['best_reward']
            try:
                brf = float(br) if br not in (None, '', 'None') else None
            except Exception:
                brf = None
            rows.append((name, ck, brf))
    return rows

def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    val_audio = sys.argv[3] if len(sys.argv) > 3 else 'training_data/input/20251009eh_raw.wav'

    rows = read_summary(CSV)
    # keep only entries with ckpt path
    rows = [r for r in rows if r[1]]
    # sort by best_reward descending (None -> very low)
    rows.sort(key=lambda x: (-(x[2]) if x[2] is not None else float('inf')))

    selected = rows[:top_n]
    PHASEB_DIR.mkdir(parents=True, exist_ok=True)

    for name, ckpt, br in selected:
        src_ckpt = Path(ckpt)
        if not src_ckpt.exists():
            print(f"Skipping {name}: checkpoint not found: {src_ckpt}")
            continue
        dest = PHASEB_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        stdout_path = dest / 'phaseB_train_stdout.txt'
        # Check for per-trial hparams file via env var HYPERPARAMS_DIR
        hparams_dir = os.environ.get('HYPERPARAMS_DIR')
        hparams_file = None
        if hparams_dir:
            candidate = Path(hparams_dir) / f"{name}.json"
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
            '--checkpoint', str(src_ckpt),
        ]
        if hparams_file:
            cmd.extend(['--hparams_json', hparams_file])
        cmd.extend([
            '--subprocess',
            '--val_audio', val_audio,
        ])
        env = os.environ.copy()
        env['PYTHONPATH'] = env.get('PYTHONPATH', '.') or '.'
        print(f"Starting Phase B for {name}: ckpt={src_ckpt} -> {dest} (epochs={epochs})")
        with open(stdout_path, 'w', encoding='utf-8') as out:
            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
            ret = p.wait()
        print(f"Finished {name} (exit {ret}), logs at {stdout_path}")

if __name__ == '__main__':
    main()

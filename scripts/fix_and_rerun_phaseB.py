#!/usr/bin/env python3
"""Detect Phase B failures from earlier launcher bug and re-run them correctly.

Scans `models/hpo_phaseB/*/phaseB_train_stdout.txt` for the string
"unrecognized arguments: --val_audio". For each match, moves the old
stdout to `phaseB_train_stdout.bak.TIMESTAMP` and re-invokes
`scripts/train_with_config.py` with the same params (reads checkpoint
path from folder name mapping in `models/hpo_phaseA`).

This script should be run with the workspace venv Python and PYTHONPATH='.'.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PHASEA = ROOT / 'models' / 'hpo_phaseA'
PHASEB = ROOT / 'models' / 'hpo_phaseB'

BUG_SIG = 'unrecognized arguments: --val_audio'


def find_failed_trials():
    failed = []
    if not PHASEB.exists():
        return failed
    for d in PHASEB.iterdir():
        if not d.is_dir():
            continue
        out = d / 'phaseB_train_stdout.txt'
        if not out.exists():
            continue
        txt = out.read_text(errors='ignore')
        if BUG_SIG in txt:
            failed.append((d.name, d, out))
    return failed


def map_phasea_ckpt(trial_name: str):
    # Phase A trial folders use same names under models/hpo_phaseA
    candidate = PHASEA / trial_name / 'best.pt'
    if candidate.exists():
        return candidate
    # fallback: search for any .pt in the phaseA trial dir
    if (PHASEA / trial_name).exists():
        for p in (PHASEA / trial_name).glob('*.pt'):
            return p
    return None


def rerun_trial(trial_name: str, trial_dir: Path, stdout_path: Path, val_audio: str = None):
    ckpt = map_phasea_ckpt(trial_name)
    if ckpt is None:
        print(f"Skipping {trial_name}: no checkpoint found in Phase A")
        return 1

    # backup old stdout
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    bak = stdout_path.with_name(stdout_path.name + f'.bak.{ts}')
    shutil.move(str(stdout_path), str(bak))
    print(f"Backed up {stdout_path} -> {bak}")

    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'train_with_config.py'),
        '--data_dir', 'training_data',
        '--save_dir', str(trial_dir),
        '--epochs', '20',
        '--n_envs', '4',
        '--steps', '512',
        '--checkpoint', str(ckpt),
        '--subprocess',
    ]
    if val_audio:
        cmd += ['--val_audio', val_audio]

    env = os.environ.copy()
    env['PYTHONPATH'] = env.get('PYTHONPATH') or '.'

    with open(stdout_path, 'w', encoding='utf-8') as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        ret = p.wait()
    print(f"Re-run {trial_name} -> exit {ret} (logs: {stdout_path})")
    return ret


def main():
    val_audio = sys.argv[1] if len(sys.argv) > 1 else 'training_data/input/20251009eh_raw.wav'
    failed = find_failed_trials()
    if not failed:
        print('No Phase B failures matching the old bug were found.')
        return
    print(f'Found {len(failed)} failed trials; will re-run them using train_with_config.py')
    for name, d, out in failed:
        print('Re-running', name)
        rerun_trial(name, d, out, val_audio=val_audio)


if __name__ == '__main__':
    main()

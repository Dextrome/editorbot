#!/usr/bin/env python3
"""Aggregate Phase B deterministic eval metrics into a CSV.

Scans `models/hpo_phaseB/*/phaseB_train_stdout.txt` for lines like:
  Eval deterministic (best): reward=-566.6687 per-beat-keep_ratio=0.885 edited_pct=88.1%
and records the best (max) reward per trial along with the corresponding
keep_ratio and edited_pct.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASEB = ROOT / 'models' / 'hpo_phaseB'
OUT = PHASEB / 'phaseB_summary.csv'

EVAL_RE = re.compile(r"Eval deterministic .*: reward=([\-0-9\.eE]+) per-beat-keep_ratio=([0-9\.]+) edited_pct=([0-9\.]+)%")
SAVED_RE = re.compile(r"Saved checkpoint to (.+?\\(?:best|final)\.pt)")


def process_trial(trial_dir: Path):
    out_file = trial_dir / 'phaseB_train_stdout.txt'
    if not out_file.exists():
        return None
    text = out_file.read_text(errors='ignore')
    evals = []
    for m in EVAL_RE.finditer(text):
        r = float(m.group(1))
        keep = float(m.group(2))
        edited = float(m.group(3))
        evals.append((r, keep, edited, m.group(0)))

    saved_ckpt = ''
    m2 = SAVED_RE.search(text)
    if m2:
        saved_ckpt = m2.group(1)

    if not evals:
        return {'trial': trial_dir.name, 'ckpt': saved_ckpt, 'best_eval_reward': '', 'per_beat_keep_ratio': '', 'edited_pct': '', 'raw_eval_line': ''}

    # choose the eval with maximum reward
    best = max(evals, key=lambda x: x[0])
    return {'trial': trial_dir.name, 'ckpt': saved_ckpt, 'best_eval_reward': best[0], 'per_beat_keep_ratio': best[1], 'edited_pct': best[2], 'raw_eval_line': best[3]}


def main():
    rows = []
    if not PHASEB.exists():
        print('No Phase B directory found')
        return
    for d in sorted(PHASEB.iterdir()):
        if not d.is_dir():
            continue
        res = process_trial(d)
        if res:
            rows.append(res)

    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'ckpt', 'best_eval_reward', 'per_beat_keep_ratio', 'edited_pct', 'raw_eval_line'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote summary to {OUT}')


if __name__ == '__main__':
    main()

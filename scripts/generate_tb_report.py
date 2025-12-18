from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default='logs/rl_audio_editor_20251216_233440/tensorboard')
parser.add_argument("--window", type=int, default=5000)
parser.add_argument("--outdir", default='output/reports')
args = parser.parse_args()

logdir = Path(args.logdir)
if not logdir.exists():
    raise SystemExit(f'logdir not found: {logdir}')

ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={'scalars': 0})
print('Loading TB...')
ea.Reload()

tags = ea.Tags().get('scalars', [])
print('Scalar tags found:', len(tags))

target_tags = ['approx_kl', 'train/entropy', 'train/policy_loss', 'train/value_loss', 'learning_rate', 'entropy_coeff', 'grad_norm']
# include reward tags too
reward_tags = [t for t in tags if t.startswith('reward/') or t.startswith('breakdown_')]
all_tags = [t for t in target_tags if t in tags] + reward_tags

if 'approx_kl' not in tags:
    raise SystemExit('approx_kl not found in TB scalars')

# find largest rise in approx_kl
items = ea.Scalars('approx_kl')
steps = np.array([i.step for i in items])
vals = np.array([i.value for i in items])
if len(vals) < 2:
    raise SystemExit('not enough approx_kl points')

diff = np.diff(vals)
idx = np.argmax(diff)
change_step = int(steps[idx+1])
print('Detected approx_kl largest rise at step', change_step, 'diff', float(diff[idx]))

low = max(0, change_step - args.window)
high = change_step + args.window
print('Using window:', low, '-', high)

# collect data
rows = []
for tag in all_tags:
    arr = ea.Scalars(tag)
    for it in arr:
        if low <= it.step <= high:
            rows.append({'step': it.step, 'tag': tag, 'value': it.value})

if not rows:
    raise SystemExit('no scalar points in window')

df = pd.DataFrame(rows)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / f'tb_window_{change_step}.csv'
df.to_csv(csv_path, index=False)
print('Wrote CSV:', csv_path)

# pivot for plotting
pivot = df.pivot_table(index='step', columns='tag', values='value')
plt.figure(figsize=(14,8))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], label=col)
plt.axvline(change_step, color='k', linestyle='--', label='change_step')
plt.legend(loc='upper left', bbox_to_anchor=(1.0,1.0))
plt.xlabel('step')
plt.ylabel('value')
plt.title(f'TensorBoard scalars window around step {change_step}')
plt.tight_layout()
png_path = outdir / f'tb_window_{change_step}.png'
plt.savefig(png_path)
print('Wrote PNG:', png_path)
print('Done')

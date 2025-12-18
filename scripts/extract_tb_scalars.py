from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import numpy as np

logdir = Path('logs/rl_audio_editor_20251216_233440/tensorboard')
if not logdir.exists():
    print('logdir not found:', logdir)
    raise SystemExit(1)

ea = event_accumulator.EventAccumulator(str(logdir))
print('Loading TB events (this may take a while)...')
# load all tags
ea.Reload()

# Scalars of interest
keys = [
    'approx_kl', 'train/entropy', 'train/policy_loss', 'train/value_loss',
    'learning_rate', 'entropy_coeff',
]
# add reward/* tags present
reward_tags = [k for k in ea.Tags().get('scalars', []) if k.startswith('reward/') or k.startswith('breakdown_')]
keys += reward_tags

print('Available scalar tags:', len(ea.Tags().get('scalars', [])))

# helper to get (step, wall_time, value) arrays

def get_scalar(tag):
    if tag not in ea.Tags().get('scalars', []):
        return []
    items = ea.Scalars(tag)
    return [(i.step, i.wall_time, i.value) for i in items]

# collect
scalars = {k: get_scalar(k) for k in keys}

# find approx_kl steps and pick a transition window
if 'approx_kl' in scalars and scalars['approx_kl']:
    steps = [s[0] for s in scalars['approx_kl']]
    # pick median step range
    median_step = int(np.median(steps))
    window = 1000
    low = max(0, median_step - window)
    high = median_step + window
    print(f'approx_kl median step {median_step}, window {low}-{high}')
else:
    print('no approx_kl tag found')
    low, high = 0, 100000000

# print values in window
for k, v in scalars.items():
    if not v:
        continue
    seq = [(step, val) for (step, _, val) in v if low <= step <= high]
    if not seq:
        continue
    print('\nTag:', k)
    for step, val in seq[:20]:
        print(f'  step={step}, val={val:.6f}')
    print('  ... total points in window:', len(seq))

print('\nDone')

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import numpy as np

logdir = Path('logs/rl_audio_editor_20251216_233440/tensorboard')
if not logdir.exists():
    print('logdir not found:', logdir)
    raise SystemExit(1)

ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={'scalars': 0})
print('Loading TB...')
ea.Reload()

tags = ea.Tags().get('scalars', [])
print('Total scalar tags:', len(tags))

target_tags = ['approx_kl', 'train/entropy', 'train/policy_loss', 'train/value_loss', 'learning_rate', 'entropy_coeff']
reward_tags = [t for t in tags if t.startswith('reward/') or t.startswith('breakdown_')]
print('Found reward tags:', reward_tags)

if 'approx_kl' not in tags:
    print('No approx_kl in tags')
    raise SystemExit(1)

items = ea.Scalars('approx_kl')
steps = np.array([i.step for i in items])
vals = np.array([i.value for i in items])

# find largest rise in approx_kl (diff)
diff = np.diff(vals)
idx = np.argmax(diff)
change_step = steps[idx+1]
print('Detected approx_kl largest rise at step', change_step, 'diff', diff[idx])

window = 5000
low = max(0, change_step - window)
high = change_step + window
print('Window:', low, '-', high)

def print_tag(tag):
    if tag not in tags:
        print(' tag not present:', tag); return
    arr = ea.Scalars(tag)
    seq = [(i.step, i.value) for i in arr if low <= i.step <= high]
    if not seq:
        print('  no points for', tag)
        return
    print('\nTag:', tag)
    for s,v in seq[:50]:
        print(f'  step={s}, val={v:.6f}')
    print('  ... count in window:', len(seq))

for t in target_tags + reward_tags:
    print_tag(t)

print('\nDone')

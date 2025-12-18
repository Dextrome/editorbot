from tensorboard.backend.event_processing import event_accumulator
import sys
import os

if len(sys.argv) < 2:
    print('Usage: inspect_tb.py <path_to_event_file>')
    sys.exit(1)

path = sys.argv[1]
if not os.path.exists(path):
    print('File not found:', path)
    sys.exit(2)

ea = event_accumulator.EventAccumulator(path)
# Load all tags (may be slow)
print('Loading scalars...')
ea.Reload()

scalars = ea.Tags().get('scalars', [])
reward_tags = [t for t in scalars if t.startswith('reward/')]
print(f'Found {len(reward_tags)} reward tags')

results = {}
for tag in reward_tags:
    events = ea.Scalars(tag)
    if not events:
        continue
    values = [e.value for e in events]
    steps = [e.step for e in events]
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values)/len(values)
    min_idx = values.index(min_val)
    results[tag] = {'min': min_val, 'max': max_val, 'mean': mean_val, 'min_step': steps[min_idx]}

# Print sorted by min value ascending
items = sorted(results.items(), key=lambda kv: kv[1]['min'])
for tag, info in items[:20]:
    print(f"{tag}: min={info['min']:.4f} at step={info['min_step']}, mean={info['mean']:.4f}, max={info['max']:.4f}")

# Also print train/episode_reward extremes
if 'train/episode_reward' in scalars:
    events = ea.Scalars('train/episode_reward')
    vals = [e.value for e in events]
    steps = [e.step for e in events]
    print('\ntrain/episode_reward: min={:.2f} at step={}, max={:.2f}'.format(min(vals), steps[vals.index(min(vals))], max(vals)))

print('\nDone')

import os
from tensorboard.backend.event_processing import event_accumulator

root = 'f:/editorbot/logs'

runs = sorted([d for d in os.listdir(root) if d.startswith('rl_audio_editor_')])
print(f'Found {len(runs)} runs')

for run in runs[-50:]:
    tb_dir = os.path.join(root, run, 'tensorboard')
    if not os.path.isdir(tb_dir):
        continue
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.startswith('events.out.tfevents')]
    if not files:
        continue
    # pick latest file
    files = sorted(files)
    path = files[-1]
    try:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        scalars = ea.Tags().get('scalars', [])
        reward_tags = [t for t in scalars if t.startswith('reward/')]
        if reward_tags:
            print(f"Run: {run} -> {len(reward_tags)} reward tags (file: {os.path.basename(path)})")
            # print top negative mins
            results = {}
            for tag in reward_tags:
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                steps = [e.step for e in events]
                results[tag] = (min(values), steps[values.index(min(values))])
            items = sorted(results.items(), key=lambda kv: kv[1][0])
            for tag, (mn, st) in items[:10]:
                print(f"  {tag}: min={mn:.4f} at step={st}")
    except Exception as e:
        print(f"Run {run} failed to read: {e}")

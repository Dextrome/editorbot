import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

in_csv = 'output/collapse_range_evals.csv'
out_png = 'output/collapse_range_evals.png'

epochs = []
rewards = []
notes = []

with open(in_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            epoch = int(row['epoch'])
        except Exception:
            continue
        r = row.get('reward','').strip()
        val = None
        if r == '' or r.upper().startswith('ERROR'):
            val = float('nan')
        else:
            try:
                val = float(r)
            except Exception:
                val = float('nan')
        epochs.append(epoch)
        rewards.append(val)
        notes.append(row.get('notes',''))

if not epochs:
    print('No data found in', in_csv)
    raise SystemExit(1)

# sort by epoch
sorted_idx = np.argsort(epochs)
epochs = np.array(epochs)[sorted_idx]
rewards = np.array(rewards)[sorted_idx]

plt.figure(figsize=(10,4))
plt.plot(epochs, rewards, '-o', color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Deterministic Reward')
plt.title('Deterministic Reward vs Epoch (4400-4636)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_png)
print('Saved plot to', out_png)

# print simple stats
valid = rewards[~np.isnan(rewards)]
if valid.size:
    print('Epochs:', epochs.min(), '->', epochs.max())
    print('Reward min/max/mean: {:.2f}/{:.2f}/{:.2f}'.format(np.min(valid), np.max(valid), np.mean(valid)))
else:
    print('No valid reward values found')

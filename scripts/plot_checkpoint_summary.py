import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path('output/reports/batch_compare')
p.mkdir(parents=True, exist_ok=True)
csv='output/reports/batch_compare/all_checkpoints_summary.csv'
if not Path(csv).exists():
    print('CSV not found:', csv)
    raise SystemExit(1)
df=pd.read_csv(csv)
# Convert mean_n_actions to numeric
if 'mean_n_actions' not in df.columns:
    print('mean_n_actions column missing')
    raise SystemExit(1)
df['mean_n_actions']=pd.to_numeric(df['mean_n_actions'], errors='coerce')
# Plot mean_score vs mean_n_actions
plt.figure(figsize=(10,5))
plt.scatter(df['mean_n_actions'], df['mean_score'])
for i,r in df.iterrows():
    label=Path(r['checkpoint']).stem
    plt.text(r['mean_n_actions'], r['mean_score'], label, fontsize=8)
plt.xlabel('mean_n_actions')
plt.ylabel('mean_score')
plt.title('Checkpoint comparison: mean_score vs mean_n_actions')
plt.grid(True)
out='output/reports/batch_compare/summary_plot.png'
plt.tight_layout()
plt.savefig(out)
print('Wrote', out)

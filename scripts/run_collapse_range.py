"""
Run deterministic inference across a range of checkpoint epochs and write CSV results.
Usage: python scripts/run_collapse_range.py
"""
import subprocess, csv, re, os, sys
models_dir = 'models/hyperoptV1'
start_epoch = 4400
end_epoch = 4636
input_audio = 'training_data/input/20250809blackhunger_raw.mp3'
out_csv = 'output/collapse_range_evals.csv'
cmd_base = [sys.executable, '-m', 'rl_editor.infer', input_audio, '--deterministic']
if not os.path.exists('output'):
    os.makedirs('output')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','checkpoint_path','reward','notes'])
    for epoch in range(start_epoch, end_epoch+1):
        ck = os.path.join(models_dir, f'checkpoint_epoch_{epoch}.pt')
        if not os.path.exists(ck):
            continue
        cmd = cmd_base + ['--checkpoint', ck]
        print(f'Running epoch {epoch}...')
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception as e:
            writer.writerow([epoch, ck, 'ERROR', str(e)])
            continue
        out = p.stdout + '\n' + p.stderr
        # Try several common patterns for reward
        m = re.search(r'Eval deterministic[: ]+([-+]?\d*\.?\d+)', out)
        if not m:
            m = re.search(r'Total deterministic reward[: ]+([-+]?\d*\.?\d+)', out)
        if not m:
            # fallback: search for 'Reward:' lines with a number
            m = re.search(r'\bReward[: ]+([-+]?\d*\.?\d+)', out)
        reward = m.group(1) if m else ''
        # capture last non-empty stdout line as note
        last_line = ''
        for line in (p.stdout + '\n' + p.stderr).splitlines()[::-1]:
            if line.strip():
                last_line = line.strip()
                break
        writer.writerow([epoch, ck, reward, last_line])
print('Done — results written to', out_csv)

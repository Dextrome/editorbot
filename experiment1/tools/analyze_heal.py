import soundfile as sf
import numpy as np

fn = 'output/new_heal_250.wav'
arr, sr = sf.read(fn)
mono = np.mean(np.abs(arr), axis=1)
thresholds = [1e-6, 1e-4, 1e-3, 1e-2]
for threshold in thresholds:
    small = mono < threshold
    runs = []
    start = None
    for i, v in enumerate(small):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(small) - start))
    large_runs = [r for r in runs if r[1] >= int(sr * 0.1)]
    print('\nThreshold', threshold, 'Total near-zero runs:', len(runs))
    print('Runs >=100ms:', len(large_runs))
    if large_runs:
        for s, l in large_runs[:20]:
            print('Run at', s, 'len', l, 'ms {:.1f}'.format(l/sr*1000))
    else:
        print('No runs >= 100ms')
    print('Max run len:', max(runs, key=lambda x: x[1])[1] if runs else 0)
    
print('\nSummary:')
print('Longest sample-level near-zero run (threshold=1e-6):', max(runs, key=lambda x: x[1])[1] if runs else 0)

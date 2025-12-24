import sys
import soundfile as sf
import numpy as np


def analyze(path, diff_thresh=0.02):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    diffs = np.abs(np.diff(y))
    max_jump = float(diffs.max())
    jumps = int((diffs > diff_thresh).sum())
    jumps_per_sec = jumps / (len(y)/sr)
    return {'path': path, 'sr': sr, 'duration_s': len(y)/sr, 'max_jump': max_jump, 'jumps': jumps, 'jumps_per_sec': jumps_per_sec}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/analyze_clicks.py <file1> [file2 ...]')
        sys.exit(1)
    for p in sys.argv[1:]:
        try:
            r = analyze(p)
            print(f"{r['path']}: duration={r['duration_s']:.2f}s sr={r['sr']} max_jump={r['max_jump']:.6f} jumps={r['jumps']} jumps/s={r['jumps_per_sec']:.3f}")
        except Exception as e:
            print(p, 'ERROR', e)

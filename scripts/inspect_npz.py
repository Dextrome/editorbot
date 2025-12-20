import sys
import numpy as np
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if p is None or not p.exists():
    print('Usage: python inspect_npz.py <path.npz>')
    sys.exit(2)

print('Inspecting', p)
with np.load(str(p)) as d:
    for k in d.files:
        arr = d[k]
        print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size > 0 and arr.ndim <= 2:
            flat = arr.ravel()
            to_show = flat[:10]
            print(f"  sample: {to_show} (total {flat.size} elements)")
            
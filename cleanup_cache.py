"""Delete cached features that have only 109 dims (missing stem features)."""
import numpy as np
from pathlib import Path

cache_dir = Path('f:\\editorbot\\rl_editor\\cache\\features')
files = list(cache_dir.glob('*.npz'))
deleted = 0

for f in files:
    try:
        d = np.load(f)
        dim = d['beat_features'].shape[-1]
        if dim == 109:
            print(f'Deleting {f.name} ({dim} dims)')
            f.unlink()
            deleted += 1
    except Exception as e:
        print(f'Error processing {f}: {e}')

print(f'\nDeleted {deleted} files with 109 dims')

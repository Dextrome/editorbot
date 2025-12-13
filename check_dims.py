import numpy as np
from pathlib import Path

files = list(Path('f:\\editorbot\\rl_editor\\cache\\features').glob('*.npz'))
count_109 = 0
count_121 = 0

for f in files:
    d = np.load(f)
    dim = d['beat_features'].shape[-1]
    if dim == 109:
        print(f'{f.name}: {dim}')
        count_109 += 1
    else:
        count_121 += 1

print(f'\nTotal: {count_109} files with 109 dims, {count_121} files with 121 dims')

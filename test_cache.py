import sys
sys.path.insert(0, 'f:\\editorbot')

from rl_editor.cache import FeatureCache
from pathlib import Path

cache = FeatureCache('f:\\editorbot\\rl_editor\\cache')
files = list(Path('./training_data/input').glob('*_raw.*'))[:3]  # Use relative path like training does

print(f'Found {len(files)} files')
for f in files:
    # Resolve to absolute first, like the real code does
    abs_f = f.absolute()
    print(f'File: {abs_f}')
    print(f'Cache path would be: {cache.get_features_path(abs_f)}')
    cached = cache.load_features(abs_f)
    if cached is not None:
        bf = cached['beat_features']
        print(f'{f.name}: shape={bf.shape}, last_dim={bf.shape[-1]}')
    else:
        print(f'{f.name}: NO CACHE')

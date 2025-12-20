"""Test loading a paired item and print key shapes."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset

cfg = get_default_config()
# Use default data dir from config; override if needed
ds = PairedAudioDataset(cfg.data.data_dir, cfg, cache_dir=cfg.data.cache_dir, use_augmentation=False)

if len(ds.pairs) == 0:
    print("No pairs found in dataset")
    sys.exit(0)

item = ds[0]
raw = item['raw']
edited = item['edited']
print('pair_id:', item.get('pair_id'))

def show(name, obj):
    if isinstance(obj, dict):
        for k,v in obj.items():
            try:
                shape = v.shape
            except Exception:
                shape = type(v)
            print(f"  {name}.{k}: {shape}")
    else:
        try:
            print(f"{name}: {obj.shape}")
        except Exception:
            print(f"{name}: {type(obj)}")

print('RAW:')
show('raw', raw)
print('EDITED:')
show('edited', edited)
print('edit_labels shape:', item['edit_labels'].shape)

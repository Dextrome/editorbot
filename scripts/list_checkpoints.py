import torch
from pathlib import Path
import numpy as np


def safe_load(path):
    """Try a fast weights-only load then fall back to full load with safe globals.

    Returns the loaded checkpoint dict or raises the last exception.
    """
    try:
        return torch.load(str(path), map_location='cpu')
    except Exception as e:
        # Fallback: allowlist numpy scalar global used by older pickles
        try:
            with torch.serialization.add_safe_globals([np._core.multiarray.scalar]):
                return torch.load(str(path), map_location='cpu', weights_only=False)
        except Exception:
            raise e


ckpt_dir = Path('models/modelV1')
files = sorted([p for p in ckpt_dir.iterdir() if p.name.startswith('checkpoint_epoch') or p.name in ('best.pt','final.pt')])
print('Found', len(files), 'checkpoint files')
for p in files:
    try:
        ckpt = safe_load(p)
        gs = ckpt.get('global_step', ckpt.get('current_step', None))
        epoch = ckpt.get('current_epoch', None)
        print(p.name, 'epoch=', epoch, 'global_step=', gs)
    except Exception as e:
        print('Failed to read', p.name, e)

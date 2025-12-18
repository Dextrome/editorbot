import torch
import sys
import os
# ensure project root is importable for any custom classes during unpickle
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

p = sys.argv[1]
try:
    ck = torch.load(p, map_location='cpu')
except Exception as e:
    print('Initial load failed:', e)
    # Try safe allowlisting for known numpy scalar global (recommended safe approach)
    try:
        import numpy as _np
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
            print('Attempting safe_globals allowlist for numpy scalar...')
            try:
                with torch.serialization.safe_globals([_np._core.multiarray.scalar]):
                    ck = torch.load(p, map_location='cpu')
            except Exception:
                raise
        else:
            raise
    except Exception:
        # Fallback: retry with full (unsafe) load if user trusts checkpoint
        try:
            print('safe_globals failed or unavailable; retrying with weights_only=False (unsafe)')
            ck = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e2:
            print('Retry failed:', e2)
            raise

print('loaded type:', type(ck))
if isinstance(ck, dict):
    print('keys:', list(ck.keys()))
    for k in list(ck.keys()):
        v = ck[k]
        print(k, '->', type(v))
else:
    print('repr:', repr(ck)[:500])

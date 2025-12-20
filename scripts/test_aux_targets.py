import time
import numpy as np
import os
import sys
# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_editor.auxiliary_tasks import AuxiliaryTargetComputer, AuxiliaryConfig

cfg = AuxiliaryConfig()
comp = AuxiliaryTargetComputer(cfg)

n_beats = 500
n_mels = cfg.mel_dim
n_frames = 16000  # large

# synthetic mel: (n_mels, n_frames)
mel = np.random.rand(n_mels, n_frames).astype(np.float32)
beat_times = np.linspace(0, 120.0, n_beats)
beat_features = np.random.rand(n_beats, 12).astype(np.float32)
beat_indices = np.random.randint(0, n_beats, size=(64,))

start = time.time()
targets = comp.get_targets('test_audio', beat_times, beat_features, beat_indices, edited_mel=mel)
end = time.time()
print('get_targets time:', end - start)
print('mel_reconstruction shape:', targets.get('mel_reconstruction').shape if 'mel_reconstruction' in targets else None)

# Call again to confirm cached speed
start = time.time()
targets = comp.get_targets('test_audio', beat_times, beat_features, beat_indices, edited_mel=mel)
end = time.time()
print('get_targets time (cached):', end - start)

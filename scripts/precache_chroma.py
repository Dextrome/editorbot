"""Precompute per-track per-beat mel and chroma and save to training_data/feature_cache/chroma.

This script:
- Loads feature .pt files under feature_cache
- Attempts to extract beat_times and beat_features
- Finds the matching audio file in training_data/input by stem
- Computes a mel spectrogram with n_mels from AuxiliaryConfig
- Computes per-beat mel vectors by averaging frames per beat
- Computes chroma per beat via mel->chroma linear mapping and saves to .npy
- Also calls AuxiliaryTargetComputer.get_targets() to populate its in-memory cache (useful when running inside same process)

Run from repo root: `python scripts/precache_chroma.py`
"""
import os
import sys
import glob
import json
import numpy as np
import torch

# Ensure repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util
aux_mod_path = os.path.join(ROOT, 'rl_editor', 'auxiliary_tasks.py')
spec = importlib.util.spec_from_file_location('auxiliary_tasks', aux_mod_path)
aux_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux_mod)
AuxiliaryConfig = aux_mod.AuxiliaryConfig
AuxiliaryTargetComputer = aux_mod.AuxiliaryTargetComputer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_FC_DIR = os.path.join(ROOT, 'feature_cache', 'rl_editor', 'paired')
AUDIO_DIR = os.path.join(ROOT, 'training_data', 'input')
# OUT_DIR will be resolved from AuxiliaryConfig.mel_chroma_cache after creating config

config = AuxiliaryConfig()
aux = AuxiliaryTargetComputer(config=config)

# Resolve output cache directory from config (absolute or repo-relative)
cache_cfg = getattr(config, 'mel_chroma_cache', None)
if cache_cfg is None:
    OUT_DIR = os.path.join(ROOT, 'feature_cache', 'chroma')
else:
    OUT_DIR = cache_cfg if os.path.isabs(cache_cfg) else os.path.join(ROOT, cache_cfg)

# Iterate audio files in AUDIO_DIR (process all tracks). If a feature .pt exists for a track, use it.
audio_files = []
for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg'):
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))

os.makedirs(OUT_DIR, exist_ok=True)
if len(audio_files) == 0:
    print('No audio files found in', AUDIO_DIR)
    sys.exit(0)

try:
    import librosa
    import librosa.display
except Exception as e:
    print('librosa not available:', e)
    print('This script requires librosa to compute mel spectrograms. Aborting.')
    sys.exit(1)

for audio_path in audio_files:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    print('Processing', stem)

    # Skip if both cached outputs already exist
    out_mel_path = os.path.join(OUT_DIR, f'{stem}_mel.npy')
    out_chroma_path = os.path.join(OUT_DIR, f'{stem}_chroma.npy')
    if os.path.exists(out_mel_path) and os.path.exists(out_chroma_path):
        print('  cached, skipping')
        continue

    # Try to find a matching feature .pt file for beat_times / beat_features
    feature_candidates = glob.glob(os.path.join(TRAIN_FC_DIR, stem + '*.pt'))
    data = None
    if feature_candidates:
        p = feature_candidates[0]
        try:
            data = torch.load(p, map_location='cpu')
        except Exception as e:
            print('  failed to load feature file', p, e)
            data = None

    # Try common keys for beat_times and beat_features
    beat_times = None
    beat_features = None
    for k in ('beat_times', 'beats', 'beat_times_sec'):
        if isinstance(data, dict) and k in data:
            beat_times = np.asarray(data[k])
            break
    for k in ('beat_features', 'features', 'beat_feats'):
        if isinstance(data, dict) and k in data:
            beat_features = np.asarray(data[k])
            break

    if beat_times is None or beat_features is None:
        print('  missing beat_times or beat_features for', stem, 'keys:', list(data.keys()) if isinstance(data, dict) else 'unknown')
        # still attempt to find beats by running beat tracker on audio

    # audio_path is already known from audio_files iteration
    print('  audio:', audio_path)

    try:
        y, sr = librosa.load(audio_path, sr=config.mel_sample_rate, mono=True)
    except Exception as e:
        print('  failed to load audio', e)
        continue

    # Compute mel spectrogram (power)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.mel_dim, power=2.0)
    # Convert to dB (optional but consistent with many pipelines)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    # S_db shape: (n_mels, n_frames)

    # If beat_times not present, estimate beats via librosa.beat
    if beat_times is None:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print('  estimated beats:', len(beat_times))

    n_beats = len(beat_times)
    n_mels, n_frames = S_db.shape
    # Build per-beat mel by segmenting frames evenly across beats
    frames_per_beat = int(np.ceil(n_frames / max(1, n_beats)))
    target_len = frames_per_beat * n_beats
    if target_len != n_frames:
        pad = target_len - n_frames
        S_db = np.pad(S_db, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)
        n_frames = S_db.shape[1]

    try:
        mel_reshaped = S_db.reshape(n_mels, n_beats, frames_per_beat)
        per_beat = mel_reshaped.mean(axis=2).T.astype(np.float32)  # (n_beats, n_mels)
    except Exception as e:
        print('  failed to reshape mel -> per_beat', e)
        continue

    # Save per-beat mel and compute chroma mapping
    out_mel_path = os.path.join(OUT_DIR, f'{stem}_mel.npy')
    np.save(out_mel_path, per_beat)

    # Compute mel->chroma mapping
    try:
        mel_freqs = librosa.mel_frequencies(n_mels=config.mel_dim, fmin=0.0, fmax=sr / 2.0)
        M = np.zeros((config.mel_dim, 12), dtype=np.float32)
        for i, f in enumerate(mel_freqs):
            if f <= 0:
                continue
            midi = 69 + 12.0 * np.log2(f / 440.0)
            chroma_idx = int(np.round(midi)) % 12
            M[i, chroma_idx] = 1.0
    except Exception:
        M = np.zeros((config.mel_dim, 12), dtype=np.float32)
        for i in range(config.mel_dim):
            M[i, i % 12] = 1.0

    chroma_per_beat = np.dot(per_beat, M).astype(np.float32)  # (n_beats, 12)
    out_chroma_path = os.path.join(OUT_DIR, f'{stem}_chroma.npy')
    np.save(out_chroma_path, chroma_per_beat)

    # Populate in-memory cache via AuxiliaryTargetComputer (use beat_indices all beats)
    try:
        aux.get_targets(audio_id=stem, beat_times=beat_times, beat_features=beat_features if beat_features is not None else np.zeros((n_beats, 1), dtype=np.float32), beat_indices=np.arange(n_beats), edited_mel=S_db)
        # Also store chroma in the module cache for convenience
        try:
            aux._cache[stem]['chroma_reconstruction_full'] = chroma_per_beat
        except Exception:
            pass
    except Exception as e:
        print('  failed to populate AuxiliaryTargetComputer cache:', e)

    print('  saved:', out_mel_path, out_chroma_path)

print('Done.')

"""CPU-only auxiliary target computation for subprocess workers.

This module contains only numpy-based code, avoiding torch imports
to reduce memory footprint in worker processes.
"""

import numpy as np
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AuxiliaryConfigCPU:
    """Lightweight config for CPU-only target computation."""
    use_tempo_prediction: bool = True
    use_energy_prediction: bool = True
    use_phrase_detection: bool = True
    use_beat_reconstruction: bool = True
    use_mel_reconstruction: bool = True
    use_chroma_continuity: bool = False

    tempo_min: float = 60.0
    tempo_max: float = 180.0
    tempo_bins: int = 20
    energy_bins: int = 10
    mel_dim: int = 128


def compute_tempo_targets(
    beat_times: np.ndarray,
    tempo_min: float = 60.0,
    tempo_max: float = 180.0,
    n_bins: int = 20,
) -> np.ndarray:
    """Compute tempo bin targets from beat times (numpy-only)."""
    n_beats = len(beat_times)
    targets = np.zeros(n_beats, dtype=np.int64)
    min_dt = 0.1

    for i in range(n_beats):
        if i > 0 and i < n_beats - 1:
            dt = beat_times[min(i+1, n_beats-1)] - beat_times[max(i-1, 0)]
            local_tempo = 120.0 / (dt / 2.0) if dt > min_dt else 120.0
        elif i == 0 and n_beats > 1:
            dt = beat_times[1] - beat_times[0]
            local_tempo = 60.0 / dt if dt > min_dt else 120.0
        else:
            local_tempo = 120.0

        local_tempo = np.clip(local_tempo, tempo_min, tempo_max)
        bin_idx = int((local_tempo - tempo_min) / (tempo_max - tempo_min) * (n_bins - 1))
        targets[i] = np.clip(bin_idx, 0, n_bins - 1)

    return targets


def compute_energy_targets(
    beat_features: np.ndarray,
    n_bins: int = 10,
    energy_feature_idx: int = 3,
) -> np.ndarray:
    """Compute energy bin targets from beat features (numpy-only)."""
    if beat_features.ndim == 1:
        beat_features = beat_features.reshape(1, -1)

    if beat_features.shape[1] <= energy_feature_idx:
        return np.zeros(len(beat_features), dtype=np.int64)

    energies = beat_features[:, energy_feature_idx]
    e_min, e_max = energies.min(), energies.max()

    if e_max - e_min < 1e-8:
        return np.zeros(len(beat_features), dtype=np.int64)

    normalized = (energies - e_min) / (e_max - e_min)
    targets = (normalized * (n_bins - 1)).astype(np.int64)
    return np.clip(targets, 0, n_bins - 1)


def compute_phrase_targets(beat_features: np.ndarray, window: int = 4) -> np.ndarray:
    """Detect phrase boundaries from feature changes (numpy-only)."""
    n_beats = len(beat_features)
    targets = np.zeros(n_beats, dtype=np.float32)

    if n_beats < window * 2:
        return targets

    for i in range(window, n_beats - window):
        before = beat_features[i-window:i].mean(axis=0)
        after = beat_features[i:i+window].mean(axis=0)
        diff = np.linalg.norm(after - before)
        targets[i] = diff

    if targets.max() > 0:
        targets = targets / targets.max()

    threshold = 0.5
    targets = (targets > threshold).astype(np.float32)

    return targets


def compute_reconstruction_targets(
    beat_features: np.ndarray,
    beat_indices: np.ndarray,
) -> tuple:
    """Compute next-beat reconstruction targets (numpy-only)."""
    n_beats = len(beat_features)
    batch_size = len(beat_indices)
    feature_dim = beat_features.shape[1] if beat_features.ndim > 1 else 1

    targets = np.zeros((batch_size, feature_dim), dtype=np.float32)
    mask = np.zeros(batch_size, dtype=np.float32)

    for i, idx in enumerate(beat_indices):
        if idx + 1 < n_beats:
            targets[i] = beat_features[idx + 1]
            mask[i] = 1.0

    return targets, mask


class AuxiliaryTargetComputerCPU:
    """Lightweight CPU-only auxiliary target computer for workers."""

    def __init__(self, config: Optional[AuxiliaryConfigCPU] = None):
        self.config = config or AuxiliaryConfigCPU()
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._cache_max_size = 200

    def _cache_put(self, key: str, value: Dict[str, np.ndarray]):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

    def clear_cache(self):
        self._cache.clear()

    def get_targets(
        self,
        audio_id: str,
        beat_times: np.ndarray,
        beat_features: np.ndarray,
        beat_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get auxiliary targets using cache (numpy-only)."""
        if audio_id not in self._cache:
            self._cache_put(audio_id, self._compute_full_targets(beat_times, beat_features))
        else:
            self._cache.move_to_end(audio_id)

        cached = self._cache[audio_id]
        n_beats = len(beat_times)
        targets = {}

        for key, full_targets in cached.items():
            if key == "reconstruction_full":
                continue
            safe_indices = np.clip(beat_indices, 0, n_beats - 1)
            targets[key] = full_targets[safe_indices]

        if self.config.use_beat_reconstruction:
            recon_targets, recon_mask = compute_reconstruction_targets(
                beat_features, beat_indices
            )
            targets["reconstruction"] = recon_targets
            targets["reconstruction_mask"] = recon_mask

        return targets

    def _compute_full_targets(
        self,
        beat_times: np.ndarray,
        beat_features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute targets for all beats (for caching)."""
        targets: Dict[str, np.ndarray] = {}

        if self.config.use_tempo_prediction:
            targets["tempo"] = compute_tempo_targets(
                beat_times, self.config.tempo_min, self.config.tempo_max, self.config.tempo_bins
            )

        if self.config.use_energy_prediction:
            targets["energy"] = compute_energy_targets(
                beat_features, self.config.energy_bins
            )

        if self.config.use_phrase_detection:
            targets["phrase"] = compute_phrase_targets(beat_features)

        return targets

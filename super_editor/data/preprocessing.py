"""Preprocessing utilities for Super Editor.

Includes mel spectrogram extraction and edit label inference.

Updated for BigVGAN v2 compatibility:
- Uses torch.stft based extraction (matching BigVGAN's preprocessing exactly)
- Volume normalization to 0.95 (like BigVGAN training)
- Log compression with clipping at 1e-5
- Normalization to [0, 1] using BigVGAN's typical range [-11.5, 2.5]
"""

import os
import numpy as np
import librosa
import torch
from typing import Optional, Tuple, List
from pathlib import Path

from ..config import AudioConfig, EditLabel

# Import BigVGAN-compatible extraction from shared module
from shared.audio_utils import (
    compute_mel_spectrogram_bigvgan,
    compute_mel_spectrogram_bigvgan_from_file,
    normalize_mel_for_model,
    denormalize_mel_for_vocoder as _denormalize_mel,
    BIGVGAN_MEL_MIN,
    BIGVGAN_MEL_MAX,
)

# Re-export constants for backward compatibility
MEL_LOG_MIN = BIGVGAN_MEL_MIN  # -11.5
MEL_LOG_MAX = BIGVGAN_MEL_MAX  # 2.5
MEL_LOG_RANGE = MEL_LOG_MAX - MEL_LOG_MIN


class MelExtractor:
    """Extract mel spectrograms using BigVGAN-compatible preprocessing.

    Uses torch.stft based extraction that matches BigVGAN's preprocessing exactly,
    ensuring the model trains on mel spectrograms compatible with BigVGAN vocoding.
    """

    def __init__(self, config: AudioConfig):
        self.config = config

    def extract(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            mel: Mel spectrogram (T, n_mels) normalized to [0, 1]
        """
        # Use BigVGAN-compatible extraction
        mel_log, _ = compute_mel_spectrogram_bigvgan_from_file(
            audio_path,
            config=self.config,
            normalize_volume=True,  # Volume normalize like BigVGAN
            device='cpu',
        )
        # mel_log is (n_mels, T), transpose to (T, n_mels)
        mel_log = mel_log.T  # (T, n_mels)

        # Normalize to [0, 1]
        mel_norm = normalize_mel_for_model(mel_log)

        return mel_norm.numpy().astype(np.float32)

    def extract_with_beats(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mel spectrogram and detect beats.

        Args:
            audio_path: Path to audio file

        Returns:
            mel: Mel spectrogram (T, n_mels) normalized to [0, 1]
            beat_times: Beat times in seconds
        """
        # Get mel using BigVGAN-compatible extraction
        mel_log, audio = compute_mel_spectrogram_bigvgan_from_file(
            audio_path,
            config=self.config,
            normalize_volume=True,
            device='cpu',
        )
        mel_log = mel_log.T  # (T, n_mels)
        mel_norm = normalize_mel_for_model(mel_log)

        # Detect beats from the audio
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=self.config.sample_rate
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=self.config.sample_rate, hop_length=self.config.hop_length
        )

        return mel_norm.numpy().astype(np.float32), beat_times.astype(np.float32)

    def save(self, mel: np.ndarray, output_path: str, beat_times: Optional[np.ndarray] = None):
        """Save mel spectrogram to file."""
        data = {'mel': mel}
        if beat_times is not None:
            data['beat_times'] = beat_times
        np.savez_compressed(output_path, **data)


def denormalize_for_vocoder(mel_norm: np.ndarray) -> np.ndarray:
    """Convert normalized mel [0, 1] back to BigVGAN log scale.

    Args:
        mel_norm: Mel spectrogram normalized to [0, 1]

    Returns:
        mel_log: Mel spectrogram in log scale for BigVGAN vocoder
    """
    mel_tensor = torch.from_numpy(mel_norm).float()
    return _denormalize_mel(mel_tensor).numpy()


class EditLabelInferencer:
    """Infer edit labels by comparing raw and edited audio."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.hop_length = config.hop_length
        self.sample_rate = config.sample_rate

    def infer_labels(self, raw_mel: np.ndarray, edit_mel: np.ndarray,
                     raw_beat_times: Optional[np.ndarray] = None,
                     similarity_threshold: float = 0.8) -> np.ndarray:
        T_raw, T_edit = len(raw_mel), len(edit_mel)
        labels = np.ones(T_raw, dtype=np.int64)
        try:
            from librosa.sequence import dtw
        except ImportError:
            return self._simple_alignment(raw_mel, edit_mel)
        if T_raw * T_edit > 50_000_000:
            return self._simple_alignment(raw_mel, edit_mel)
        try:
            raw_norm = raw_mel / (np.linalg.norm(raw_mel, axis=1, keepdims=True) + 1e-8)
            edit_norm = edit_mel / (np.linalg.norm(edit_mel, axis=1, keepdims=True) + 1e-8)
            cost = 1 - np.dot(raw_norm, edit_norm.T)
        except MemoryError:
            return self._simple_alignment(raw_mel, edit_mel)
        try:
            D, wp = dtw(C=cost, subseq=True, backtrack=True)
        except Exception:
            return self._simple_alignment(raw_mel, edit_mel)
        aligned_raw = set()
        for i in range(len(wp)):
            raw_idx, edit_idx = int(wp[i, 0]), int(wp[i, 1])
            if raw_idx < T_raw and edit_idx < T_edit and cost[raw_idx, edit_idx] < (1 - similarity_threshold):
                aligned_raw.add(raw_idx)
        for i in range(T_raw):
            if i not in aligned_raw:
                labels[i] = EditLabel.CUT
        edit_to_raw = {}
        for i in range(len(wp)):
            raw_idx, edit_idx = int(wp[i, 0]), int(wp[i, 1])
            if raw_idx < T_raw and edit_idx < T_edit:
                edit_to_raw.setdefault(edit_idx, []).append(raw_idx)
        for raw_indices in edit_to_raw.values():
            if len(raw_indices) > 1:
                for raw_idx in raw_indices:
                    if labels[raw_idx] == EditLabel.KEEP:
                        labels[raw_idx] = EditLabel.LOOP
        return labels

    def _simple_alignment(self, raw_mel: np.ndarray, edit_mel: np.ndarray) -> np.ndarray:
        T_raw, T_edit = len(raw_mel), len(edit_mel)
        labels = np.ones(T_raw, dtype=np.int64)
        if T_edit < T_raw * 0.9:
            n_cut = int(T_raw * (1 - T_edit / T_raw))
            labels[np.random.choice(T_raw, n_cut, replace=False)] = EditLabel.CUT
        elif T_edit > T_raw * 1.1:
            n_loop = int(T_raw * ((T_edit / T_raw) - 1))
            labels[np.random.choice(T_raw, min(n_loop, T_raw), replace=False)] = EditLabel.LOOP
        return labels

    def infer_from_edits(self, raw_length: int, edit_regions: List[Tuple[int, int, int]]) -> np.ndarray:
        labels = np.ones(raw_length, dtype=np.int64)
        for start, end, label in edit_regions:
            labels[max(0, start):min(raw_length, end)] = label
        return labels


def process_audio_pair(raw_audio_path: str, edit_audio_path: str, output_dir: str,
                       pair_id: str, config: AudioConfig, infer_labels: bool = True) -> str:
    output_dir = Path(output_dir)
    (output_dir / 'features').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    extractor = MelExtractor(config)
    raw_mel, raw_beats = extractor.extract_with_beats(raw_audio_path)
    extractor.save(raw_mel, output_dir / 'features' / f'{pair_id}_raw.npz', raw_beats)
    edit_mel, edit_beats = extractor.extract_with_beats(edit_audio_path)
    extractor.save(edit_mel, output_dir / 'features' / f'{pair_id}_edit.npz', edit_beats)
    if infer_labels:
        labels = EditLabelInferencer(config).infer_labels(raw_mel, edit_mel, raw_beats)
        np.save(output_dir / 'labels' / f'{pair_id}_labels.npy', labels)
    return pair_id


def preprocess_dataset(pairs: List[Tuple[str, str, str]], output_dir: str,
                       config: AudioConfig, num_workers: int = 4, infer_labels: bool = True):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    def process_one(args):
        try:
            return process_audio_pair(args[0], args[1], output_dir, args[2], config, infer_labels)
        except Exception as e:
            print(f"Error processing {args[2]}: {e}")
            return None
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, pair) for pair in pairs]
        for future in tqdm(as_completed(futures), total=len(pairs), desc="Processing"):
            future.result()
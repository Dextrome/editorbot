"""Core audio processing functionality."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa


class AudioProcessor:
    """Handles loading, saving, and basic audio processing operations."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio processor.

        Args:
            sample_rate: Target sample rate for audio processing.
        """
        self.sample_rate = sample_rate
        self.audio_data: Optional[np.ndarray] = None
        self.original_sr: Optional[int] = None

    def load(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio_data, sr = librosa.load(str(file_path), sr=self.sample_rate)
        self.audio_data = audio_data
        self.original_sr = sr
        return audio_data, sr

    def save(self, file_path: str | Path, audio_data: Optional[np.ndarray] = None) -> None:
        """
        Save audio data to a file.

        Args:
            file_path: Output file path.
            audio_data: Audio data to save. Uses stored data if not provided.
        """
        data = audio_data if audio_data is not None else self.audio_data
        if data is None:
            raise ValueError("No audio data to save")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(file_path), data, self.sample_rate)

    def normalize(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.

        Args:
            audio_data: Audio data to normalize. Uses stored data if not provided.

        Returns:
            Normalized audio data.
        """
        data = audio_data if audio_data is not None else self.audio_data
        if data is None:
            raise ValueError("No audio data to normalize")

        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    def trim_silence(
        self, audio_data: Optional[np.ndarray] = None, threshold_db: float = -40.0
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio_data: Audio data to trim. Uses stored data if not provided.
            threshold_db: Silence threshold in decibels.

        Returns:
            Trimmed audio data.
        """
        data = audio_data if audio_data is not None else self.audio_data
        if data is None:
            raise ValueError("No audio data to trim")

        trimmed, _ = librosa.effects.trim(data, top_db=abs(threshold_db))
        return trimmed

    def resample(self, target_sr: int, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Resample audio to a different sample rate.

        Args:
            target_sr: Target sample rate.
            audio_data: Audio data to resample. Uses stored data if not provided.

        Returns:
            Resampled audio data.
        """
        data = audio_data if audio_data is not None else self.audio_data
        if data is None:
            raise ValueError("No audio data to resample")

        return librosa.resample(data, orig_sr=self.sample_rate, target_sr=target_sr)

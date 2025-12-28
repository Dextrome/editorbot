"""Unified audio configuration and edit label vocabulary.

Shared between rl_editor and super_editor modules.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List


@dataclass
class AudioConfig:
    """Audio processing configuration.

    Unified config that includes all fields needed by both rl_editor and super_editor.

    Updated for BigVGAN v2 compatibility (44kHz, 128 mels, full frequency range).
    """
    sample_rate: int = 44100  # Changed from 22050 for BigVGAN v2
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048  # Used by super_editor for STFT
    fmin: float = 0.0       # Min frequency for mel filterbank
    fmax: float = None      # None = Nyquist (22050 Hz for 44kHz audio) - BigVGAN compatible

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def frame_duration_ms(self) -> float:
        """Duration of each frame in milliseconds."""
        return 1000 * self.hop_length / self.sample_rate

    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.sample_rate / self.hop_length)

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx * self.hop_length / self.sample_rate


class EditLabel(IntEnum):
    """Edit label vocabulary.

    Defines the 8-class edit labels used for training.
    Both modules use the same vocabulary.
    """
    CUT = 0         # Remove this frame entirely
    KEEP = 1        # Keep this frame unchanged
    LOOP = 2        # Repeat this frame/section
    FADE_IN = 3     # Apply fade in effect
    FADE_OUT = 4    # Apply fade out effect
    EFFECT = 5      # Apply audio effect (reverb, filter, etc.)
    TRANSITION = 6  # Crossfade region between sections
    PAD = 7         # Padding token (for batching)

    @classmethod
    def names(cls) -> List[str]:
        """Get list of label names in order."""
        return [label.name for label in cls]

    @classmethod
    def to_name(cls, label: int) -> str:
        """Convert label index to name string."""
        try:
            return cls(label).name
        except ValueError:
            return 'UNKNOWN'

    @classmethod
    def from_name(cls, name: str) -> 'EditLabel':
        """Convert name string to EditLabel."""
        return cls[name.upper()]

    @classmethod
    def num_labels(cls) -> int:
        """Number of edit labels (excluding PAD for some use cases)."""
        return len(cls)

    @classmethod
    def num_action_labels(cls) -> int:
        """Number of actionable labels (excluding PAD)."""
        return len(cls) - 1  # Exclude PAD


# Convenience constants for backward compatibility
SAMPLE_RATE = 44100  # Updated for BigVGAN v2
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def get_default_audio_config() -> AudioConfig:
    """Get default audio configuration."""
    return AudioConfig()

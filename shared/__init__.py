"""
Shared audio processing utilities.

This module consolidates common code between rl_editor and super_editor:
- AudioConfig: Unified audio processing configuration
- EditLabel: Edit label vocabulary (8 classes)
- Audio utilities: mel extraction, beat detection, etc.
- Logging: Training logger with TensorBoard/W&B support
- Caching: Feature caching for faster training
"""
from .demucs_wrapper import DemucsSeparator
from .audio_config import (
    AudioConfig,
    EditLabel,
    get_default_audio_config,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)
from .audio_utils import (
    # Audio I/O
    load_audio,
    save_audio,
    set_audio_cache_enabled,
    clear_audio_cache,
    get_audio_cache_stats,
    # Mel spectrogram
    compute_mel_spectrogram,
    compute_mel_spectrogram_from_config,
    MelExtractor,
    # Beat detection
    detect_beats,
    detect_beats_from_audio,
    # Tempo and energy
    estimate_tempo,
    get_energy_contour,
    get_spectral_centroid,
    # Utilities
    set_seed,
    griffin_lim,
)
from .logging_utils import (
    TrainingLogger,
    create_logger,
    create_logger_from_config,
)
from .cache import (
    FeatureCache,
    get_cache,
    set_cache_enabled,
)
from .normalization import (
    RunningMeanStd,
    TorchRunningMeanStd,
    ObservationNormalizer,
    RewardNormalizer,
)

__all__ = [
    # Audio config
    'AudioConfig',
    'EditLabel',
    'get_default_audio_config',
    'SAMPLE_RATE',
    'N_MELS',
    'N_FFT',
    'HOP_LENGTH',
    # Demucs
    'DemucsSeparator',
    # Audio I/O
    'load_audio',
    'save_audio',
    'set_audio_cache_enabled',
    'clear_audio_cache',
    'get_audio_cache_stats',
    # Mel spectrogram
    'compute_mel_spectrogram',
    'compute_mel_spectrogram_from_config',
    'MelExtractor',
    # Beat detection
    'detect_beats',
    'detect_beats_from_audio',
    # Tempo and energy
    'estimate_tempo',
    'get_energy_contour',
    'get_spectral_centroid',
    # Utilities
    'set_seed',
    'griffin_lim',
    # Logging
    'TrainingLogger',
    'create_logger',
    'create_logger_from_config',
    # Caching
    'FeatureCache',
    'get_cache',
    'set_cache_enabled',
    # Normalization
    'RunningMeanStd',
    'TorchRunningMeanStd',
    'ObservationNormalizer',
    'RewardNormalizer',
]

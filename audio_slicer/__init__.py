"""Audio Slicer - FaceSwap-style audio editing.

Like FaceSwap:
- Shared encoder learns content features
- Decoder_raw learns to output raw style
- Decoder_edited learns to output edited style

At inference:
- Score raw segments by transformation quality
- Keep segments that transform cleanly to "edited" style
- Discard segments that don't fit the pattern
"""

from .config import SlicerConfig, TrainConfig, ModelConfig
from .models import DualAutoencoder, QualityAutoencoder
from .trainers import Trainer
from .inference import AudioSlicer

__all__ = [
    'SlicerConfig',
    'TrainConfig',
    'ModelConfig',
    'DualAutoencoder',
    'QualityAutoencoder',
    'Trainer',
    'AudioSlicer',
]

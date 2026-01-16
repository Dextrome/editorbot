"""Mel-to-Mel Editor - Direct audio transformation learning.

No labels, no predefined effects - learns end-to-end from paired data.
"""

from .config import TrainConfig, ModelConfig, LossConfig
from .models import MelUNet
from .losses import CombinedLoss
from .data import PairedMelDataset
from .trainers import Trainer
from .inference import MelToMelPipeline

__all__ = [
    'TrainConfig',
    'ModelConfig',
    'LossConfig',
    'MelUNet',
    'CombinedLoss',
    'PairedMelDataset',
    'Trainer',
    'MelToMelPipeline',
]

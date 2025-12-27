"""Super Editor - Two-phase supervised audio editing system.

Phase 1: Train autoencoder to reconstruct edited audio from (raw + edit_labels)
Phase 2: Train RL agent to predict optimal edit_labels
"""

from .config import Phase1Config, Phase2Config, AudioConfig, EditLabel

__all__ = [
    # Config
    'Phase1Config',
    'Phase2Config',
    'AudioConfig',
    'EditLabel',
]

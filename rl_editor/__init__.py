"""RL-based audio editing framework (V2).

This module provides an end-to-end reinforcement learning system for music editing,
with section-level actions (KEEP_PHRASE, CUT_BAR, LOOP, etc.).
Crossfades are applied automatically at edit boundaries.
"""

__version__ = "2.0.0"

from .config import Config, get_default_config
from .state import AudioState, EditHistory, StateRepresentation
from .agent import Agent, PolicyNetwork, ValueNetwork
from .reward import RewardCalculator, RewardComponents
from .data import AudioDataset, PairedAudioDataset, create_dataloader
from .cache import FeatureCache, get_cache
from .logging_utils import TrainingLogger, create_logger

# V2 imports (section-level actions + episode rewards)
from .actions_v2 import ActionTypeV2, ActionSpaceV2, EditHistoryV2
from .environment_v2 import AudioEditingEnvV2

__all__ = [
    # Config
    "Config",
    "get_default_config",
    # State
    "AudioState",
    "EditHistory",
    "StateRepresentation",
    # Agent
    "Agent",
    "PolicyNetwork",
    "ValueNetwork",
    # Reward
    "RewardCalculator",
    "RewardComponents",
    # Data
    "AudioDataset",
    "PairedAudioDataset",
    "create_dataloader",
    # Cache
    "FeatureCache",
    "get_cache",
    # Logging
    "TrainingLogger",
    "create_logger",
    # V2 (section-level actions + episode rewards)
    "ActionTypeV2",
    "ActionSpaceV2",
    "EditHistoryV2",
    "AudioEditingEnvV2",
]

"""RL-based audio editing framework.

This module provides an end-to-end reinforcement learning system for music editing,
with section-level actions (KEEP_PHRASE, CUT_BAR, LOOP, etc.).
Crossfades are applied automatically at edit boundaries.

Supports both:
- V2: Single-head discrete action space (39 actions)
- Factored: 3-head action space (type × size × amount = 450 combinations from 28 outputs)
"""

__version__ = "3.0.0"

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

# Factored action space imports (3-head policy)
from .actions_factored import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
)
from .agent_factored import FactoredAgent, FactoredPolicyNetwork
from .environment_factored import AudioEditingEnvFactored

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
    # Factored (3-head action space)
    "ActionType",
    "ActionSize",
    "ActionAmount",
    "FactoredAction",
    "FactoredActionSpace",
    "EditHistoryFactored",
    "N_ACTION_TYPES",
    "N_ACTION_SIZES",
    "N_ACTION_AMOUNTS",
    "FactoredAgent",
    "FactoredPolicyNetwork",
    "AudioEditingEnvFactored",
]

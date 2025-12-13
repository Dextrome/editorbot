"""RL-based audio editing framework.

This module provides an end-to-end reinforcement learning system for music editing,
with section-level actions (KEEP_PHRASE, CUT_BAR, LOOP, etc.).
Crossfades are applied automatically at edit boundaries.

Uses factored 3-head action space:
- Type head: What action (20 types: KEEP, CUT, LOOP, FADE, SPEED, PITCH, etc.)
- Size head: How many beats (5 sizes: BEAT, BAR, PHRASE, etc.)
- Amount head: Intensity/direction (5 amounts: -3dB to +3dB, etc.)

This gives 450 possible combinations from just 28 network outputs.
"""

__version__ = "3.0.0"

from .config import Config, get_default_config
from .state import AudioState, EditHistory, StateRepresentation
from .agent import Agent, PolicyNetwork, ValueNetwork
from .reward import RewardCalculator, RewardComponents
from .data import AudioDataset, PairedAudioDataset, create_dataloader
from .cache import FeatureCache, get_cache
from .logging_utils import TrainingLogger, create_logger

# Factored action space (3-head policy)
from .actions import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
)
from .environment import AudioEditingEnvFactored

# Aliases for backward compatibility
FactoredAgent = Agent
FactoredPolicyNetwork = PolicyNetwork

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
    # Factored Action Space (primary implementation)
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

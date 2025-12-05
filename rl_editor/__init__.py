"""RL-based audio editing framework.

This module provides an end-to-end reinforcement learning system for music editing,
with actions like KEEP, CUT, LOOP, and REORDER.
Crossfades are applied automatically at edit boundaries.
"""

__version__ = "0.1.0"

from .config import Config, get_default_config
from .actions import ActionType, Action, KeepAction, CutAction, LoopAction, ReorderAction, ActionSpace
from .state import AudioState, EditHistory, StateRepresentation
from .environment import AudioEditingEnv
from .agent import Agent, PolicyNetwork, ValueNetwork
from .trainer import PPOTrainer
from .reward import RewardCalculator, RewardComponents
from .reward_model import RewardModel
from .data import AudioDataset, PairedAudioDataset, create_dataloader
from .cache import FeatureCache, get_cache
from .evaluation import Evaluator
from .logging_utils import TrainingLogger, create_logger

__all__ = [
    # Config
    "Config",
    "get_default_config",
    # Actions
    "ActionType",
    "Action",
    "KeepAction",
    "CutAction",
    "LoopAction",
    "ReorderAction",
    "ActionSpace",
    # State
    "AudioState",
    "EditHistory",
    "StateRepresentation",
    # Environment
    "AudioEditingEnv",
    # Agent
    "Agent",
    "PolicyNetwork",
    "ValueNetwork",
    # Training
    "PPOTrainer",
    # Reward
    "RewardCalculator",
    "RewardComponents",
    "RewardModel",
    # Data
    "AudioDataset",
    "PairedAudioDataset",
    "create_dataloader",
    # Cache
    "FeatureCache",
    "get_cache",
    # Evaluation
    "Evaluator",
    # Logging
    "TrainingLogger",
    "create_logger",
]

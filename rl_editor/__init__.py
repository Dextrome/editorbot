"""RL-based audio editing framework.

This module provides an end-to-end reinforcement learning system for music editing,
with section-level actions (KEEP_PHRASE, CUT_BAR, LOOP, etc.).
Crossfades are applied automatically at edit boundaries.

Uses factored 3-head action space:
- Type head: What action (20 types: KEEP, CUT, LOOP, FADE, SPEED, PITCH, etc.)
- Size head: How many beats (5 sizes: BEAT, BAR, PHRASE, etc.)
- Amount head: Intensity/direction (5 amounts: -3dB to +3dB, etc.)

This gives 450 possible combinations from just 28 network outputs.

NOTE: This module uses lazy imports to reduce memory in subprocess workers.
Only import what you need directly, e.g.:
    from rl_editor.config import Config
    from rl_editor.environment import AudioEditingEnvFactored
"""

__version__ = "3.0.0"

# Lazy imports - modules are only loaded when accessed
def __getattr__(name):
    """Lazy import to avoid loading torch in subprocess workers."""
    _imports = {
        # Config
        "Config": ("config", "Config"),
        "get_default_config": ("config", "get_default_config"),
        # State
        "AudioState": ("state", "AudioState"),
        "EditHistory": ("state", "EditHistory"),
        "StateRepresentation": ("state", "StateRepresentation"),
        # Agent (heavy - imports torch)
        "Agent": ("agent", "Agent"),
        "PolicyNetwork": ("agent", "PolicyNetwork"),
        "ValueNetwork": ("agent", "ValueNetwork"),
        # Reward
        "RewardCalculator": ("reward", "RewardCalculator"),
        "RewardComponents": ("reward", "RewardComponents"),
        # Data (heavy - imports torch)
        "AudioDataset": ("data", "AudioDataset"),
        "PairedAudioDataset": ("data", "PairedAudioDataset"),
        "create_dataloader": ("data", "create_dataloader"),
        # Cache
        "FeatureCache": ("cache", "FeatureCache"),
        "get_cache": ("cache", "get_cache"),
        # Logging
        "TrainingLogger": ("logging_utils", "TrainingLogger"),
        "create_logger": ("logging_utils", "create_logger"),
        # Factored Action Space
        "ActionType": ("actions", "ActionType"),
        "ActionSize": ("actions", "ActionSize"),
        "ActionAmount": ("actions", "ActionAmount"),
        "FactoredAction": ("actions", "FactoredAction"),
        "FactoredActionSpace": ("actions", "FactoredActionSpace"),
        "EditHistoryFactored": ("actions", "EditHistoryFactored"),
        "N_ACTION_TYPES": ("actions", "N_ACTION_TYPES"),
        "N_ACTION_SIZES": ("actions", "N_ACTION_SIZES"),
        "N_ACTION_AMOUNTS": ("actions", "N_ACTION_AMOUNTS"),
        # Environment
        "AudioEditingEnvFactored": ("environment", "AudioEditingEnvFactored"),
    }

    if name in _imports:
        module_name, attr_name = _imports[name]
        import importlib
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, attr_name)

    # Aliases
    if name == "FactoredAgent":
        return __getattr__("Agent")
    if name == "FactoredPolicyNetwork":
        return __getattr__("PolicyNetwork")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

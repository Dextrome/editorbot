"""Model components for Super Editor."""

from .encoder import EditEncoder, ConvEditEncoder, PositionalEncoding
from .decoder import MelDecoder, MultiScaleMelDecoder
from .reconstruction import ReconstructionModel, MultiScaleReconstructionModel
from .dsp_editor import DSPEditor
from .edit_predictor import EditPredictor, ValueNetwork, ActorCritic

__all__ = [
    # Encoder
    'EditEncoder',
    'ConvEditEncoder',
    'PositionalEncoding',
    # Decoder
    'MelDecoder',
    'MultiScaleMelDecoder',
    # Reconstruction (Phase 1) - legacy
    'ReconstructionModel',
    'MultiScaleReconstructionModel',
    # DSP Editor (replaces Phase 1)
    'DSPEditor',
    # Edit Predictor (Phase 2)
    'EditPredictor',
    'ValueNetwork',
    'ActorCritic',
]

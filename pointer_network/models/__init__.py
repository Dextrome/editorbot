"""Pointer network models."""
from .encoder import MelEncoder, TransformerMelEncoder, PositionalEncoding
from .decoder import PointerDecoder, PointerAttention
from .pointer_network import (
    PointerNetwork,
    MultiScaleEncoder,
    MusicAwarePositionalEncoding,
    HierarchicalAttention,
    SparseAttention,
    LengthPredictor,
    DurationConditioning,
    EditStyleVAE,
    StructurePredictionHead,
    CachedMultiHeadAttention,
    STOP_TOKEN,
    PAD_TOKEN,
)

__all__ = [
    # Legacy components (from encoder/decoder modules)
    'MelEncoder',
    'TransformerMelEncoder',
    'PositionalEncoding',
    'PointerDecoder',
    'PointerAttention',
    # Consolidated model components
    'PointerNetwork',
    'MultiScaleEncoder',
    'MusicAwarePositionalEncoding',
    'HierarchicalAttention',
    'SparseAttention',
    'LengthPredictor',
    'DurationConditioning',
    'EditStyleVAE',
    'StructurePredictionHead',
    'CachedMultiHeadAttention',
    # Constants
    'STOP_TOKEN',
    'PAD_TOKEN',
]

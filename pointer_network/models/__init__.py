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
    # V2 features integrated into V1
    EditOp,
    PreNormTransformerEncoderLayer,
    PreNormTransformerDecoderLayer,
    StemEncoder,
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
    # V2 features (Pre-LayerNorm, Edit Ops, Stems)
    'EditOp',
    'PreNormTransformerEncoderLayer',
    'PreNormTransformerDecoderLayer',
    'StemEncoder',
    # Constants
    'STOP_TOKEN',
    'PAD_TOKEN',
]

# Pointer Network for Audio Editing
# Learns to generate sequences of frame pointers from raw audio
# Now includes: Pre-LayerNorm, Edit Operations, Multi-Stem support

from .models import (
    PointerNetwork,
    MultiScaleEncoder,
    MusicAwarePositionalEncoding,
    HierarchicalAttention,
    SparseAttention,
    EditStyleVAE,
    StructurePredictionHead,
    # V2 features (now integrated)
    EditOp,
    PreNormTransformerEncoderLayer,
    PreNormTransformerDecoderLayer,
    StemEncoder,
    # Constants
    STOP_TOKEN,
    PAD_TOKEN,
)
from .data.dataset import PointerDataset, collate_fn, create_dataloader
from .trainers import PointerNetworkTrainer, train_pointer_network
from .config import PointerNetworkConfig, TrainConfig

__all__ = [
    # Model
    'PointerNetwork',
    'MultiScaleEncoder',
    'MusicAwarePositionalEncoding',
    'HierarchicalAttention',
    'SparseAttention',
    'EditStyleVAE',
    'StructurePredictionHead',
    # V2 features (Pre-LayerNorm, Edit Ops, Stems)
    'EditOp',
    'PreNormTransformerEncoderLayer',
    'PreNormTransformerDecoderLayer',
    'StemEncoder',
    # Constants
    'STOP_TOKEN',
    'PAD_TOKEN',
    # Data
    'PointerDataset',
    'collate_fn',
    'create_dataloader',
    # Training
    'PointerNetworkTrainer',
    'train_pointer_network',
    # Config
    'PointerNetworkConfig',
    'TrainConfig',
]

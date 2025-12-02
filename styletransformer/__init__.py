# Style Transfer Module for AI-based song transformation
"""
AI Style Transfer for Audio

Transform raw recordings into polished songs by learning from reference tracks.

Components:
- StyleEncoder: Extracts style embeddings from songs (structure, energy, harmony)
- RemixPolicy: Generates remix decisions based on target style
- StyleDiscriminator: Scores how well output matches target style
- IterativeStyleTransfer: Main interface with refinement loop

Usage:
    from style_transfer import IterativeStyleTransfer
    
    transfer = IterativeStyleTransfer()
    transfer.load_models("models/")
    
    result = transfer.transform(
        source="my_jam.wav",
        target_style="reference_song.wav", 
        output="output.wav"
    )
"""

from .style_encoder import StyleEncoder, StyleEncoderNet, FeatureExtractor
from .remix_policy import RemixPolicy, RemixPolicyNet, RemixAction, RemixPlan
from .discriminator import StyleDiscriminator, MultiScaleDiscriminator
from .iterative_transfer import IterativeStyleTransfer, TransferResult
from .trainer import StyleTransferTrainer, TrainingConfig, train_style_transfer

__all__ = [
    # High-level interfaces
    'IterativeStyleTransfer',
    'TransferResult',
    'StyleEncoder',
    'StyleDiscriminator', 
    'RemixPolicy',
    
    # Training
    'StyleTransferTrainer',
    'TrainingConfig',
    'train_style_transfer',
    
    # Data structures
    'RemixAction',
    'RemixPlan',
    
    # Low-level (for advanced use)
    'StyleEncoderNet',
    'RemixPolicyNet',
    'MultiScaleDiscriminator',
    'FeatureExtractor',
]

#!/usr/bin/env python
"""Test enhanced training with richer features and augmentation."""

import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("ENHANCED TRAINING TEST")
print("=" * 60)

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import modules
from rl_editor.config import Config
from rl_editor.data import PairedAudioDataset, HAS_ENHANCED_FEATURES, HAS_AUGMENTATION, HAS_CACHE
from rl_editor.features import BeatFeatureExtractor, get_enhanced_feature_config

print(f"\nModule availability:")
print(f"  HAS_ENHANCED_FEATURES: {HAS_ENHANCED_FEATURES}")
print(f"  HAS_AUGMENTATION: {HAS_AUGMENTATION}")
print(f"  HAS_CACHE: {HAS_CACHE}")

# Setup config
config = Config()
config.features.feature_mode = "enhanced"
config.data.use_stems = False
config.augmentation.enabled = False  # Test without augmentation first

print(f"\nConfiguration:")
print(f"  Feature mode: {config.features.feature_mode}")
print(f"  Cache dir: {config.data.cache_dir}")
print(f"  Use stems: {config.data.use_stems}")

# Test feature extractor
feat_config = get_enhanced_feature_config()
extractor = BeatFeatureExtractor(
    sr=config.audio.sample_rate,
    hop_length=config.audio.hop_length,
    n_fft=config.audio.n_fft,
    n_mels=config.audio.n_mels,
    config=feat_config,
)
print(f"\nFeature extractor:")
print(f"  Feature dimension: {extractor.get_feature_dim()}")

# Load dataset
print("\n" + "-" * 40)
print("LOADING DATASET")
print("-" * 40)

dataset = PairedAudioDataset(
    data_dir="./training_data",
    config=config,
    include_reference=False,
    use_augmentation=False,
)
print(f"Dataset size: {len(dataset)} pairs")

# Load first sample
print("\nLoading first sample...")
sample = dataset[0]

raw_data = sample["raw"]
edit_labels = sample["edit_labels"]

print(f"\nSample info:")
print(f"  Pair ID: {sample['pair_id']}")
print(f"  Beat features shape: {raw_data['beat_features'].shape}")
print(f"  Num beats: {len(raw_data['beats'])}")
print(f"  Duration: {raw_data['duration']:.1f}s")
print(f"  Tempo: {raw_data['tempo']:.1f} BPM")
print(f"  Edit labels shape: {edit_labels.shape}")
print(f"  Keep ratio: {edit_labels.mean():.1%}")

# Verify feature dimension
expected_dim = extractor.get_feature_dim()
actual_dim = raw_data["beat_features"].shape[-1]
print(f"\nFeature dimension check:")
print(f"  Expected: {expected_dim}")
print(f"  Actual: {actual_dim}")
assert actual_dim == expected_dim, f"Dimension mismatch! {actual_dim} != {expected_dim}"
print("  ✓ PASSED")

# Test augmentation
print("\n" + "-" * 40)
print("TESTING AUGMENTATION")
print("-" * 40)

config.augmentation.enabled = True
aug_dataset = PairedAudioDataset(
    data_dir="./training_data",
    config=config,
    include_reference=False,
    use_augmentation=True,
)
print("Augmentation dataset created")

# Load augmented sample
aug_sample = aug_dataset[0]
aug_raw = aug_sample["raw"]
print(f"Augmented beat features shape: {aug_raw['beat_features'].shape}")
print("  ✓ Augmentation works")

# Test behavioral cloning readiness
print("\n" + "-" * 40)
print("TESTING BC TRAINING READINESS")
print("-" * 40)

from rl_editor.behavioral_cloning import BehavioralCloningDataset

bc_dataset = BehavioralCloningDataset(
    data_dir="./training_data",
    config=config,
    max_beats=500,
)
print(f"BC dataset size: {len(bc_dataset)} samples")

if len(bc_dataset) > 0:
    sample = bc_dataset[0]
    state = sample["observation"]
    label = sample["label"]
    print(f"State shape: {state.shape}")
    print(f"Label value: {label.item()}")
    print("  ✓ BC dataset ready")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print(f"\nEnhanced features: {actual_dim} dimensions")
print(f"Training samples: {len(bc_dataset)}")
print(f"Ready for training with: python -m rl_editor.behavioral_cloning")

# RL Editor - Reinforcement Learning Audio Editor

A complete end-to-end reinforcement learning system for automated music editing. The agent learns to make edit decisions (KEEP, CUT, LOOP, REORDER) by imitating human edits and optimizing for audio quality.

## Overview

Traditional audio editing pipelines use hand-crafted rules:
```
Input Audio → [ML: Score beats] → [Rules: Threshold, Merge, Concat] → Output
```

This RL-based approach learns the full editing policy:
```
Input Audio → [RL Agent: Choose actions] → Output
                     ↑
            Actions: KEEP, CUT, LOOP, REORDER
                     ↑
            Reward: Human feedback / Audio quality metrics
```

## Features

- **End-to-end learned editing** - No hand-crafted decision rules
- **Rich audio features** - 109-121 dimensions including spectral, MFCCs, chroma, rhythmic
- **Stem separation** - Demucs integration for drums/bass/vocals/other analysis
- **Behavioral cloning warmup** - Initialize from human demonstrations
- **PPO fine-tuning** - Refine with automatic quality rewards
- **Comprehensive caching** - Pre-compute features for fast training
- **Data augmentation** - Pitch shift, time stretch, noise, gain, EQ

## Installation

```bash
# Core dependencies
pip install torch torchaudio librosa soundfile numpy gymnasium

# For stem separation (optional, slow without GPU)
pip install demucs

# For training visualization
pip install tensorboard wandb
```

## Quick Start

### 1. Prepare Training Data

Organize your data as:
```
training_data/
├── input/              # Raw audio files (*_raw.wav)
├── desired_output/     # Human-edited versions (*_edit.wav)
└── reference/          # Additional finished tracks (optional)
```

### 2. Pre-cache Features (Recommended)

```bash
# Basic features (fast, 4 dims)
python -m rl_editor.precache --data_dir ./training_data --feature_mode basic

# Enhanced features (recommended, 109 dims)
python -m rl_editor.precache --data_dir ./training_data --feature_mode enhanced

# Full features with stems (slow, 121 dims, requires GPU)
python -m rl_editor.precache --data_dir ./training_data --feature_mode full --stems
```

### 3. Train with Behavioral Cloning

```bash
python -m rl_editor.behavioral_cloning \
    --data_dir ./training_data \
    --epochs 100 \
    --save_dir ./models
```

### 4. Fine-tune with PPO

```bash
python -m rl_editor.train_unified \
    --data_dir ./training_data \
    --checkpoint ./models/bc_best.pt \
    --total_timesteps 1000000
```

### 5. Run Inference

```bash
python -m rl_editor.infer \
    --input ./raw_audio.wav \
    --output ./edited_audio.wav \
    --checkpoint ./models/best.pt
```

## Architecture

### Module Structure

```
rl_editor/
├── config.py           # All hyperparameters (dataclass Config pattern)
├── actions.py          # KEEP, CUT, LOOP, REORDER action definitions
├── state.py            # AudioState, EditHistory, StateRepresentation
├── environment.py      # Gymnasium-compatible RL environment
├── agent.py            # Policy and Value networks (Transformer-based)
├── trainer.py          # PPO training loop
├── behavioral_cloning.py # Supervised pre-training from demonstrations
├── reward.py           # Dense reward signals (tempo, energy, transitions)
├── reward_model.py     # Learned reward from human preferences (RLHF)
├── features.py         # Enhanced feature extraction (109+ dims)
├── augmentation.py     # Data augmentation preserving edit labels
├── data.py             # Dataset classes with caching integration
├── cache.py            # Centralized disk caching system
├── precache.py         # CLI for pre-computing all features
├── evaluation.py       # Evaluation metrics
├── infer.py            # Inference/export
├── utils.py            # Audio I/O, mel spectrograms
└── logging_utils.py    # TensorBoard/W&B logging
```

### Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| **KEEP** | Include beat in output | beat_index |
| **CUT** | Remove beat from output | beat_index |
| **LOOP** | Repeat beat 2x, 3x, or 4x | beat_index, n_times |
| **REORDER** | Move beat to new position | beat_index, target_position |

Crossfades are applied automatically at edit boundaries (default 50ms).

### State Representation

The agent observes:
- **Beat features** - Spectral, rhythmic, timbral features for current and context beats
- **Edit history** - What actions have been taken so far
- **Global features** - Tempo, key, energy contour
- **Progress** - Position in track, remaining duration

### Reward Signals

Three complementary reward sources:

1. **Dense rewards** (automatic metrics):
   - Tempo consistency - Stable BPM throughout
   - Energy flow - Smooth dynamics, no jarring transitions
   - Phrase completeness - Respect musical phrase boundaries
   - Transition quality - Beats align well at edit points
   - Loop ratio penalty - Discourage excessive looping (max 15%)

2. **Sparse rewards** - Human ratings (1-10) of final edits

3. **Learned rewards** (RLHF) - Train preference model from A/B comparisons

## Feature Extraction

### Feature Modes

| Mode | Dimensions | Features Included |
|------|------------|-------------------|
| `basic` | 4 | onset, RMS, spectral centroid, ZCR |
| `enhanced` | 109 | + rolloff, bandwidth, flatness, contrast (7), MFCCs (13+13δ), chroma (12), rhythmic (5), deltas (52) |
| `full` | 121 | + stem features (4 stems × 3 features) |

### Enhanced Features Breakdown

```
Basic spectral:      4   (onset, rms, centroid, zcr)
Extended spectral:   3   (rolloff, bandwidth, flatness)
Spectral contrast:   7   (frequency band contrasts)
MFCCs:              13   (timbral coefficients)
MFCC deltas:        13   (temporal derivatives)
Chroma:             12   (pitch class distribution)
Rhythmic:            5   (beat phase ×3, tempo deviation, IBI ratio)
Delta features:     52   (temporal derivatives of spectral)
───────────────────────
Total enhanced:    109
Stem features:      12   (4 stems × energy, centroid, ratio)
───────────────────────
Total full:        121
```

## Data Augmentation

Augmentations preserve edit labels by applying identical transforms to both raw and edited audio:

| Augmentation | Range | Default Probability |
|--------------|-------|---------------------|
| Pitch shift | ±2 semitones | 50% |
| Time stretch | 0.9x - 1.1x | 50% |
| Noise | 20-40 dB SNR | 30% |
| Gain | ±6 dB | 50% |
| EQ | ±6 dB bands | 30% |

Enable with `use_augmentation=True` in dataset or `augmentation.enabled=True` in config.

## Caching System

All computed features are cached to `rl_editor/cache/`:

```
rl_editor/cache/
├── features/     # Beat-level features (109 dims)
│   └── {name}_{hash}.npz
├── stems/        # Demucs separated stems
│   └── {name}_{hash}_stems.npz
├── mel/          # Mel spectrograms
│   └── {name}_{hash}_mel.npz
├── labels/       # Edit labels for raw/edited pairs
│   └── {name}_{hash}_{hash}_labels.npz
└── metadata/     # Track metadata
```

Cache keys include file modification time for automatic invalidation.

## Configuration

All hyperparameters use the dataclass Config pattern in `config.py`:

```python
from rl_editor import Config

config = Config()

# Audio processing
config.audio.sample_rate = 22050
config.audio.hop_length = 512

# Action space
config.action_space.max_loop_times = 4
config.action_space.default_crossfade_ms = 50

# Reward shaping
config.reward.target_max_duration_s = 600.0  # 10 min max output
config.reward.max_loop_ratio = 0.15  # Max 15% of beats looped

# PPO training
config.ppo.learning_rate = 1e-4
config.ppo.clip_ratio = 0.2
config.ppo.entropy_coeff = 0.05

# Data
config.data.cache_dir = "./rl_editor/cache"
config.data.use_stems = False  # Enable only if pre-cached

# Features
config.features.feature_mode = "enhanced"  # basic, enhanced, or full

# Augmentation
config.augmentation.enabled = True
config.augmentation.pitch_shift_prob = 0.5
```

## Training Pipeline

### Phase 1: Behavioral Cloning (BC)

Supervised learning from human demonstrations:
- Input: Beat features + context
- Target: KEEP/CUT labels from raw/edited pairs
- Loss: Cross-entropy

This creates a reasonable initial policy that respects musical structure.

### Phase 2: PPO Fine-tuning

Reinforcement learning with automatic rewards:
- Collect trajectories using current policy
- Compute advantages with GAE (λ=0.95)
- Update policy with clipped objective
- Update value network

### Phase 3: RLHF (Optional)

Learn reward model from human preferences:
1. Generate candidate edits with different policies
2. Collect A/B preference judgments from humans
3. Train reward model on preference data
4. Use learned reward in PPO training

## GPU Optimization

The system is optimized for GPU training:

- **CUDA acceleration** - All models use GPU by default
- **Mixed precision** - Optional FP16 training (experimental)
- **Gradient accumulation** - Train with large effective batch sizes
- **Parallel data loading** - Multi-worker DataLoader
- **Feature caching** - Pre-compute expensive features offline

## API Reference

### Core Classes

```python
from rl_editor import (
    # Configuration
    Config, get_default_config,
    
    # Actions
    ActionType, ActionSpace, KeepAction, CutAction, LoopAction, ReorderAction,
    
    # State
    AudioState, EditHistory, StateRepresentation,
    
    # Environment
    AudioEditingEnv,
    
    # Agent
    Agent, PolicyNetwork, ValueNetwork,
    
    # Training
    PPOTrainer,
    
    # Reward
    RewardCalculator, RewardComponents, RewardModel,
    
    # Data
    AudioDataset, PairedAudioDataset, create_dataloader,
    
    # Cache
    FeatureCache, get_cache,
    
    # Evaluation
    Evaluator,
    
    # Logging
    TrainingLogger, create_logger,
)
```

### Example: Custom Training Loop

```python
from rl_editor import Config, AudioEditingEnv, Agent, AudioState
import torch

# Setup
config = Config()
config.training.device = "cuda"

# Load audio
audio_state = AudioState.from_file("input.wav", config)

# Create environment
env = AudioEditingEnv(config, audio_state=audio_state)
obs, info = env.reset()

# Create agent
agent = Agent(config, input_dim=obs.shape[0], n_actions=env.action_space.n_discrete_actions)

# Training loop
for episode in range(1000):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, log_prob, value = agent.select_action(torch.from_numpy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition, update agent...
```

## Development

### Running Tests

```bash
pytest rl_editor/tests/
```

### Code Style

- Follow PEP 8
- Use type hints for all functions
- Document all public APIs
- Keep modules focused and modular

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this code, please cite:

```bibtex
@software{rl_editor,
  title = {RL Editor: Reinforcement Learning Audio Editor},
  author = {[Author]},
  year = {2025},
  url = {https://github.com/[repo]}
}
```

## Acknowledgments

- [librosa](https://librosa.org/) for audio analysis
- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [Gymnasium](https://gymnasium.farama.org/) for RL environment interface
- [PyTorch](https://pytorch.org/) for deep learning

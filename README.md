# RL Audio Editor

A deep learning system for automated music editing. The agent learns to make intelligent editing decisions by training on paired raw/edited audio examples.

## Overview

Traditional audio editing uses hand-crafted rules. This system learns the full editing policy end-to-end using two approaches:

### Approach 1: Supervised Mel Reconstruction
```
Raw Audio + Edit Labels → [Encoder] → Latent → [Decoder] → Edited Audio
```
Direct reconstruction using multi-scale perceptual losses. Faster to train, simpler architecture.

### Approach 2: RL with Factored Actions
```
Input Audio → [RL Agent] → Edited Output
                  ↑
         Factored Actions:
         - Type: KEEP, CUT, LOOP, FADE, GAIN, PITCH, SPEED...
         - Size: BEAT, BAR, PHRASE, TWO_BARS...
         - Amount: intensity/direction
```
Sequential decision making with PPO + Behavioral Cloning.

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch torchaudio librosa soundfile numpy gymnasium tensorboard
pip install demucs  # Optional: for stem separation
```

### 2. Prepare Training Data

```
training_data/
├── input/           # Raw audio files (*_raw.wav)
├── desired_output/  # Human-edited versions (*_edit.wav)
└── reference/       # Additional finished tracks (optional)
```

### 3. Train

**Option A: Supervised Reconstruction (Recommended to start)**
```bash
python -m rl_editor.supervised_trainer \
    --data-dir training_data \
    --save-dir models/supervised \
    --epochs 100 \
    --batch-size 8
```

**Option B: RL with BC Mixed Training**
```bash
# Generate BC dataset first
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz

# Train with BC + PPO
python -m rl_editor.train \
    --save-dir models/rl_model \
    --bc-mixed bc_rich.npz \
    --bc-weight 0.3 \
    --epochs 5000 \
    --subprocess
```

### 4. Inference

```bash
python -m rl_editor.infer "input.wav" --checkpoint "models/rl_model/best.pt" --output "edited.wav"
```

## Architecture

### Supervised Model

Inspired by FaceSwap's autoencoder architecture:

| Component | Description |
|-----------|-------------|
| **EditEncoder** | Transformer encoder processing raw mel + edit label embeddings |
| **MelDecoder** | Transformer decoder with residual connection to raw input |
| **Multi-Scale STFT Loss** | Frequency-aware reconstruction at 512/1024/2048 FFT sizes |
| **Edit Consistency Loss** | Preserves unedited regions |

### RL Model - Factored Action Space

Instead of 500 discrete actions, uses 3 factored heads:

| Head | Outputs | Examples |
|------|---------|----------|
| **Type** | 20 | KEEP, CUT, LOOP, FADE_IN, FADE_OUT, GAIN, PITCH, SPEED... |
| **Size** | 5 | BEAT, BAR, TWO_BARS, PHRASE, TWO_PHRASES |
| **Amount** | 5 | NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE |

This yields 500 combinations from only 30 network outputs.

### Neural Network (RL)

- **Encoder**: HybridNATTENEncoder (neighborhood attention + global pooling)
- **Policy**: 3-head output (type, size, amount distributions)
- **Value**: Separate value network for PPO
- **Auxiliary Tasks**: Tempo prediction, energy classification, phrase detection

### Features (121 dimensions)

```
Basic spectral:      4   (onset, rms, centroid, zcr)
Extended spectral:   3   (rolloff, bandwidth, flatness)
Spectral contrast:   7   (frequency band contrasts)
MFCCs:              26   (13 coefficients + 13 deltas)
Chroma:             12   (pitch class distribution)
Rhythmic:            5   (beat phase, tempo deviation)
Delta features:     52   (temporal derivatives)
Stem features:      12   (4 stems × 3 features, if enabled)
```

## Project Structure

```
editorbot/
├── rl_editor/              # Main package
│   ├── supervised_trainer.py  # Supervised mel reconstruction
│   ├── train.py            # PPO training loop
│   ├── agent.py            # Policy/Value networks
│   ├── environment.py      # Gymnasium RL environment
│   ├── actions.py          # Factored action space (20×5×5)
│   ├── config.py           # All hyperparameters
│   ├── features.py         # Audio feature extraction
│   ├── reward.py           # Reward computation
│   ├── auxiliary_tasks.py  # Multi-task learning heads
│   └── infer.py            # Inference/export
├── scripts/                # Utilities
│   ├── infer_rich_bc_labels.py  # Generate BC dataset
│   └── augment_bc_with_synthetic.py  # Augment rare actions
├── training_data/          # Paired audio for training
├── models/                 # Saved checkpoints
├── logs/                   # TensorBoard logs
└── CLAUDE.md              # Development guidelines
```

## Training Configuration

### Supervised (supervised_trainer.py)

```python
encoder_dim: 512
decoder_dim: 512
n_layers: 6
n_heads: 8
learning_rate: 1e-4
batch_size: 8
l1_weight: 1.0
multiscale_stft_weight: 1.0
edit_consistency_weight: 0.5
```

### RL (config.py)

```python
# PPO
learning_rate: 1e-4
entropy_coeff: 0.02
clip_ratio: 0.2
target_kl: 0.02

# Model
policy_hidden_dim: 512   # Must be divisible by natten_n_heads
natten_n_heads: 8
natten_n_layers: 3
natten_kernel_size: 33   # Must be odd

# Training
batch_size: 2048
gradient_accumulation_steps: 4
use_mixed_precision: true
```

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

### Supervised Metrics
- `train/l1_loss` - Should decrease
- `train/msstft_loss` - Should decrease
- `val/total_loss` - Should decrease (watch for overfitting)

### RL Metrics
- `train/episode_reward` - Should trend upward
- `train/entropy` - Should NOT be at maximum (-6.2)
- `approx_kl` - Should stay near target_kl (0.02)

## Tips

- **Start with supervised training** - faster iteration, simpler debugging
- Use `--subprocess` on Windows for true multiprocessing (RL only)
- Pre-cache features for faster training: features are cached in `rl_editor/cache/`
- Watch for overfitting in supervised training (val loss increasing)
- For RL, watch for action type collapse (>90% one type)

## Development

See `CLAUDE.md` for detailed development guidelines, constraints, and architecture notes.

```bash
# Run tests
pytest rl_editor/tests/

# Check a checkpoint
python -c "import torch; c = torch.load('models/best.pt'); print(c.keys())"
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended)
- ~8GB+ VRAM for training
- ~16GB RAM for 16 parallel environments (RL)

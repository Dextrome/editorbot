# RL Audio Editor

A reinforcement learning system for automated music editing. The agent learns to make intelligent editing decisions (KEEP, CUT, LOOP, effects) by training on paired raw/edited audio examples.

## Overview

Traditional audio editing uses hand-crafted rules. This RL-based approach learns the full editing policy end-to-end:

```
Input Audio → [RL Agent] → Edited Output
                  ↑
         Factored Actions:
         - Type: KEEP, CUT, LOOP, FADE, GAIN, PITCH, SPEED...
         - Size: BEAT, BAR, PHRASE, TWO_BARS...
         - Amount: intensity/direction
```

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

```bash
# Fresh training with subprocess parallelism (recommended for Windows)
python -m rl_editor.train --save_dir models/my_model --epochs 30000 --n_envs 16 --steps 512 --subprocess

# Resume from checkpoint
python -m rl_editor.train --save_dir models/my_model --checkpoint models/my_model/best.pt --epochs 30000 --subprocess
```

### 4. Inference

```bash
python -m rl_editor.infer "input.wav" --checkpoint "models/my_model/best.pt" --output "edited.wav"
```

## Architecture

### Factored Action Space

Instead of 500 discrete actions, uses 3 factored heads:

| Head | Outputs | Examples |
|------|---------|----------|
| **Type** | 20 | KEEP, CUT, LOOP, FADE_IN, FADE_OUT, GAIN, PITCH, SPEED... |
| **Size** | 5 | BEAT, BAR, TWO_BARS, PHRASE, TWO_PHRASES |
| **Amount** | 5 | NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE |

This yields 500 combinations from only 30 network outputs.

### Neural Network

- **Encoder**: HybridNATTENEncoder (neighborhood attention + global pooling)
- **Policy**: 3-head output (type, size, amount distributions)
- **Value**: Separate value network for PPO
- **Auxiliary Tasks**: Tempo prediction, energy classification, phrase detection, mel reconstruction

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
│   ├── train.py            # PPO training loop
│   ├── agent.py            # Policy/Value networks
│   ├── environment.py      # Gymnasium RL environment
│   ├── actions.py          # Factored action space (20×5×5)
│   ├── config.py           # All hyperparameters
│   ├── features.py         # Audio feature extraction
│   ├── reward.py           # Reward computation
│   ├── auxiliary_tasks.py  # Multi-task learning heads
│   ├── cache.py            # Feature caching system
│   ├── subprocess_vec_env.py # Parallel environments
│   └── infer.py            # Inference/export
├── training_data/          # Paired audio for training
├── models/                 # Saved checkpoints
├── logs/                   # TensorBoard logs
└── CLAUDE.md              # Development guidelines
```

## Training Configuration

Key hyperparameters in `rl_editor/config.py`:

```python
# PPO
learning_rate: 4e-5
entropy_coeff: 0.15      # Lower = more deterministic policy
entropy_coeff_min: 0.02  # Final entropy after decay
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

## Behavioral Cloning (Optional)

Pre-train from expert demonstrations before RL:

```bash
# 1. Generate rich BC dataset with action detection
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz

# 2. Pretrain policy on BC dataset
python -m rl_editor.train --bc_pretrain_npz bc_rich.npz --bc_pretrain_epochs 10 --save_dir models/bc_init

# 3. Or mix BC loss during PPO training
python -m rl_editor.train --bc_mixed_npz bc_rich.npz --bc_mixed_weight 0.1 --subprocess
```

The rich BC generator (`scripts/infer_rich_bc_labels.py`) detects:
- **KEEP/CUT**: From ground-truth alignment labels
- **GAIN**: RMS energy differences (>1.5 dB)
- **PITCH_UP/DOWN**: Spectral centroid shifts (>0.8 semitones)
- **EQ_HIGH/LOW**: Band energy ratio changes (>1.5 dB)
- **REORDER**: Position inversions in alignment mapping
- **Multi-beat sizes**: Consecutive runs of 4+ (BAR), 8+ (TWO_BARS), 16+ (PHRASE) beats

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Key metrics to watch:
- `train/episode_reward` - Should trend upward
- `train/entropy` - Logged as negative; should NOT be at maximum (-6.2)
- `learning_rate` - Should be at expected value, not decayed to 0
- `approx_kl` - Should stay near target_kl (0.02)

## Tips

- Use `--subprocess` on Windows for true multiprocessing
- Set `--epochs` high (30000) to prevent premature LR decay when resuming
- Monitor entropy - if at maximum, policy is random (lower entropy_coeff)
- Pre-cache features for faster training: features are cached in `rl_editor/cache/`
- Watch for action type collapse (>90% one type) - indicates reward issues

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
- ~16GB RAM for 16 parallel environments


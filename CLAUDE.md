# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based music editor that uses deep reinforcement learning to automatically edit raw music recordings into polished tracks. The agent learns intelligent editing decisions (KEEP, CUT, LOOP, REORDER, effects) through behavioral cloning and PPO optimization.

**Stack**: Python 3.10+, PyTorch, librosa, NATTEN (neighborhood attention), Demucs (stem separation), Gymnasium

## Commands

```bash
# Training (threading mode - default)
python -m rl_editor.train --save_dir models --epochs 30000 --steps 512 --n_envs 16 --lr 5e-5

# Training (subprocess mode - true multiprocessing for Windows)
python -m rl_editor.train --save_dir models --epochs 30000 --steps 512 --n_envs 16 --subprocess

# Resume from checkpoint
python -m rl_editor.train --checkpoint models/best.pt --save_dir models

# Inference
python -m rl_editor.infer "input.wav" --checkpoint "models/best.pt" --output "output.wav"

# Tests
pytest rl_editor/tests/
```

## Architecture

### Factored 3-Head Action Space (Key Innovation)

The system uses a factored action space instead of discrete actions:
- **Type head** (18→20 outputs): Action type (KEEP, CUT, LOOP, GAIN, PITCH, etc.)
- **Size head** (5 outputs): Duration (BEAT, BAR, PHRASE, TWO_BARS, TWO_PHRASES)
- **Amount head** (5 outputs): Intensity (NEG_LARGE to POS_LARGE)

This yields 500 action combinations from 28 network outputs vs 500 discrete outputs.

### Core Components

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized hyperparameters (dataclass pattern) |
| `train.py` | PPO training with parallel envs |
| `environment.py` | Gymnasium RL env with factored actions |
| `agent.py` | 3-head PolicyNetwork + ValueNetwork with HybridNATTENEncoder |
| `actions.py` | Factored action definitions (20 types × 5 sizes × 5 amounts) |
| `features.py` | 121-dim beat features (spectral, MFCCs, chroma, stems) |
| `auxiliary_tasks.py` | Multi-task learning (tempo, energy, phrase, mel reconstruction) |
| `cache.py` | Disk-based feature caching |
| `subprocess_vec_env.py` | Subprocess parallelism for Windows |

### Reward Structure

Episode-end Monte Carlo rewards with components:
- Keep ratio (target ~45% cut), section coherence, phrase alignment
- Edit structure quality, audio flow, action diversity
- Penalties: excessive loops (budget: 2 base + 1 per 3 cuts), excessive jumps (>25%)

## Key Constraints

- **policy_hidden_dim must be divisible by natten_n_heads** (e.g., 512 with 8 heads)
- **NATTEN kernel_size must be odd** (e.g., 31, 33)
- **Mixed precision disabled** - Large value losses cause FP16 overflow
- **High entropy** - Use entropy_coeff 0.5-0.75 until policy stabilizes
- **Feature caching** - Always pre-cache features for training speed

## Development Guidelines

- Always use CUDA when available
- Use `--subprocess` flag for true multiprocessing on Windows
- Action type collapse (>90% one type) indicates reward structure problem
- Monitor action distribution during inference to diagnose training issues
- Log reward breakdowns to diagnose issues; watch for type collapse
- Typical batch size 512-1024 for RTX 4070 Ti, n_envs 16-24

## Data Structure

```
training_data/
├── input/           # Raw audio (*_raw.wav)
├── desired_output/  # Human-edited versions (*_edit.wav)
└── reference/       # Additional finished tracks (optional)
```

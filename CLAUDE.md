# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based music editor that learns to automatically edit raw music recordings into polished tracks. The agent learns intelligent editing decisions (KEEP, CUT, LOOP, REORDER, effects) through **Behavioral Cloning (BC) combined with PPO optimization**.

**Stack**: Python 3.10+, PyTorch, librosa, NATTEN (neighborhood attention), Demucs (stem separation), Gymnasium

## Training Approach

The system uses a **hybrid BC + PPO approach**:

1. **Behavioral Cloning (BC)**: Supervised learning from detected edit patterns in training pairs
2. **PPO**: Reinforcement learning to optimize episode rewards
3. **BC Mixed Training**: BC loss added during PPO to guide the policy

### Why BC is Critical

The factored 3-head action space (20×5×5 = 500 combinations) makes pure RL exploration inefficient. BC provides:
- Supervision for underrepresented action types
- Faster convergence than pure exploration
- Prevents entropy collapse to uniform distributions

## Commands

```bash
# Generate BC dataset with rich action labels
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz

# Augment BC dataset with synthetic examples for rare action types
python -m scripts.augment_bc_with_synthetic --input bc_rich.npz --output bc_augmented.npz --min_samples 1000

# Training with BC mixed loss (recommended)
python -m rl_editor.train --bc_mixed_npz bc_augmented.npz --bc_mixed_weight 0.5 --save_dir models/v1 --subprocess

# Pure BC pretraining before PPO
python -m rl_editor.train --bc_pretrain_npz bc_augmented.npz --bc_pretrain_epochs 50 --save_dir models/v1

# Resume from checkpoint
python -m rl_editor.train --checkpoint models/v1/best.pt --bc_mixed_npz bc_augmented.npz --bc_mixed_weight 0.5

# Inference
python -m rl_editor.infer "input.wav" --checkpoint "models/v1/best.pt" --output "output.wav"

# Tests
pytest rl_editor/tests/
```

## Architecture

### Factored 3-Head Action Space

The system uses a factored action space instead of 500 discrete actions:
- **Type head** (20 outputs): KEEP, CUT, LOOP, REORDER, JUMP_BACK, SKIP, FADE_IN/OUT, GAIN, SPEED_UP/DOWN, REVERSE, PITCH_UP/DOWN, EQ_LOW/HIGH, DISTORTION, REVERB, REPEAT_PREV, SWAP_NEXT
- **Size head** (5 outputs): BEAT, BAR, PHRASE, TWO_BARS, TWO_PHRASES
- **Amount head** (5 outputs): NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE

**Important**: Entropy is normalized by dividing by 3 (number of heads). Without this, max entropy = ln(20)+ln(5)+ln(5) ≈ 6.2, which overwhelms reward signals.

### Core Components

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized hyperparameters (dataclass pattern) |
| `train.py` | PPO training with BC mixed loss, state normalization |
| `environment.py` | Gymnasium RL env with factored actions |
| `agent.py` | 3-head PolicyNetwork + ValueNetwork with HybridNATTENEncoder |
| `actions.py` | Factored action definitions (20 types × 5 sizes × 5 amounts) |
| `features.py` | 121-dim beat features (spectral, MFCCs, chroma, stems) |
| `state.py` | StateRepresentation (861-dim observation from beat context) |
| `auxiliary_tasks.py` | Multi-task learning (tempo, energy, phrase, mel reconstruction) |
| `scripts/infer_rich_bc_labels.py` | Detect action labels from training pairs |
| `scripts/augment_bc_with_synthetic.py` | Generate synthetic BC samples for rare actions |

### State Normalization (Critical)

**Raw states have values up to ~10,000 which cause NaN during training.**

The trainer automatically normalizes:
1. BC states normalized when loaded (mean=0, std=1, clipped to [-5, 5])
2. Rollout states normalized using BC stats during collection and training

This ensures BC and rollout states have the same distribution, preventing the model from learning different behaviors for each.

## Key Metrics to Monitor (TensorBoard)

| Metric | Target | Problem If |
|--------|--------|-----------|
| `train/entropy` | Decreasing from ~2.0 | Stuck at max (~2.05) = uniform policy |
| `train/bc_entropy` | Decreasing toward ~1.0 | Stuck at max = BC not learning |
| `train/bc_loss` | Decreasing below 1.0 | Stuck at ~1.3+ = not learning from BC |
| `approx_kl` | Near 0.02 | >0.05 = updates too aggressive |
| `train/episode_reward` | Increasing | Declining = reward/BC conflict |
| `counters/n_keep_ratio` | 40-60% | >90% = action collapse |

## Key Constraints

- **policy_hidden_dim must be divisible by natten_n_heads** (e.g., 512 with 8 heads)
- **NATTEN kernel_size must be odd** (e.g., 31, 33)
- **State normalization required** - Raw features cause NaN without normalization
- **Low entropy_coeff** - Use 0.02 or lower when BC is active (BC provides supervision)
- **BC entropy penalty** - Counteracts entropy bonus on BC states to prevent distribution mismatch

## Entropy Issues (Historical Context)

The factored action space caused entropy problems:

1. **Summed entropy was too high**: ln(20)+ln(5)+ln(5) ≈ 6.2 overwhelmed rewards
   - **Fix**: Normalize by dividing by 3: `(type_ent + size_ent + amount_ent) / 3.0`

2. **BC learned but rollout didn't**: Model output peaked distributions on BC states but uniform on rollout states
   - **Fix**: Normalize both BC and rollout states with same stats
   - **Fix**: Add BC entropy penalty to counteract entropy bonus

3. **NaN during training**: Unnormalized states (values up to 10,000) caused overflow
   - **Fix**: Normalize states to mean=0, std=1, clip to [-5, 5]

## BC Dataset Structure

The BC dataset (`bc_augmented.npz`) contains:
- `states`: (N, 861) float32 - normalized state observations
- `type_labels`: (N,) int64 - action type indices (0-19)
- `size_labels`: (N,) int64 - action size indices (0-4)
- `amount_labels`: (N,) int64 - action amount indices (0-4)
- `good_bad`: (N,) int64 - binary label (1=keep, 0=cut)
- `pair_ids`: (N,) object - source track identifiers

Action type distribution should cover all 20 types (use augmentation script for rare types).

## Data Structure

```
training_data/
├── input/           # Raw audio (*_raw.wav)
├── desired_output/  # Human-edited versions (*_edit.wav)
└── reference/       # Additional finished tracks (optional)

editorbot/
├── bc_rich.npz      # BC dataset from infer_rich_bc_labels
├── bc_augmented.npz # BC dataset with synthetic augmentation
├── models/          # Saved checkpoints
└── logs/            # TensorBoard logs
```

## Development Guidelines

- Always use CUDA when available
- Use `--subprocess` flag for true multiprocessing on Windows
- Start fresh training after changing normalization (weights incompatible)
- Action type collapse (>90% one type) indicates reward structure problem
- Monitor `train/bc_entropy` - should decrease if BC is working
- Typical batch size 2048, n_envs 16 for RTX 4070 Ti

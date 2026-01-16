# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-based music editor that learns to automatically edit raw music recordings into polished tracks. The system supports **three training paradigms**:

1. **Supervised Mel Reconstruction** - Direct reconstruction of edited audio (faster, simpler)
2. **Behavioral Cloning + PPO** - Sequential action prediction with reinforcement learning
3. **Pointer Network** - Frame-level alignment that preserves audio quality (current focus)

**Stack**: Python 3.10+, PyTorch, librosa, NATTEN (neighborhood attention), Demucs (stem separation), Gymnasium

## Training Approaches

### Approach 1: Supervised Mel Reconstruction (Recommended for starting)

Inspired by FaceSwap's autoencoder architecture. Directly learns to reconstruct edited mel spectrograms from raw audio + edit labels.

```
Raw Mel + Edit Labels → Encoder → Latent → Decoder → Edited Mel
```

**Advantages:**
- Much simpler than RL - direct supervision
- Faster convergence
- Better sample efficiency
- Multi-scale perceptual losses

**Losses:**
- L1/L2 on mel spectrogram
- Multi-scale STFT loss (phase-aware)
- Edit consistency loss (preserve unedited regions)

```bash
# Train supervised reconstruction
python -m rl_editor.supervised_trainer --data-dir training_data --save-dir models/supervised --epochs 100
```

### Approach 2: Hybrid BC + PPO (For sequential decision making)

Uses a **hybrid BC + PPO approach** for action-by-action editing:

1. **Behavioral Cloning (BC)**: Supervised learning from detected edit patterns
2. **PPO**: Reinforcement learning to optimize episode rewards
3. **BC Mixed Training**: BC loss added during PPO for guidance

**Why BC is Critical for PPO:**
The factored 3-head action space (20×5×5 = 500 combinations) makes pure RL exploration inefficient. BC provides:
- Supervision for underrepresented action types
- Faster convergence than pure exploration
- Prevents entropy collapse to uniform distributions

### Approach 3: Pointer Network (Current Focus)

Instead of generating audio, the pointer network **points to frames** in the original raw audio to construct the edit. For each output frame, it predicts which raw frame to copy.

```
Raw Mel (T_raw frames) → Encoder → Pointer Decoder → Pointer Sequence (T_edit indices)
```

**Key Insight**: Editing is mostly rearrangement (cuts, loops, reordering) not generation. By copying frames from the original, we preserve 100% audio quality.

**Advantages:**
- **Perfect quality** - Copies original frames, no generation artifacts
- **Learns edit patterns** - Cuts, loops, reordering from examples
- **Variable output length** - STOP token handles different edit lengths
- **Style diversity** - VAE latent enables multiple valid edits

**Architecture:**
- Multi-scale encoder (frame/beat/bar levels)
- Music-aware positional encoding
- Hierarchical decoder with sparse attention
- Pointer head + STOP prediction

**Data Pipeline:**
1. Align raw ↔ edited audio pairs via cross-correlation
2. Generate pointer sequences mapping each edit frame → raw frame
3. Train pointer network to predict these sequences

## Commands

```bash
# === Supervised Training ===
python -m rl_editor.supervised_trainer --data-dir training_data --save-dir models/supervised

# === BC + PPO Training ===
# Generate BC dataset with rich action labels
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz

# Augment BC dataset with synthetic examples for rare action types
python -m scripts.augment_bc_with_synthetic --input bc_rich.npz --output bc_augmented.npz --min_samples 1000

# Training with BC mixed loss (recommended)
python -m rl_editor.train --bc-mixed bc_augmented.npz --bc-weight 0.5 --save-dir models/v1 --subprocess

# Pure BC pretraining before PPO
python -m rl_editor.train --bc-pretrain bc_augmented.npz --bc-pretrain-epochs 50 --save-dir models/v1

# Resume from checkpoint
python -m rl_editor.train --checkpoint models/v1/best.pt --bc-mixed bc_augmented.npz --bc-weight 0.5

# === Inference ===
python -m rl_editor.infer "input.wav" --checkpoint "models/v1/best.pt" --output "output.wav"

# === Tests ===
pytest rl_editor/tests/

# === Pointer Network ===
# Generate pointer sequences (aligns raw ↔ edited audio frame-by-frame)
python -m pointer_network.generate_pointer_sequences

# Train pointer network
python -m pointer_network.trainers.pointer_trainer \
    --cache-dir cache \
    --pointer-dir training_data/pointer_sequences \
    --save-dir models/pointer_network \
    --epochs 100
```

## Architecture

### Supervised Model (supervised_trainer.py)

```
┌─────────────────────────────────────────────────────────┐
│                    EditEncoder                          │
│  Raw Mel ─→ MelProj ─┐                                  │
│                      ├─→ Combine ─→ Transformer ─→ Latent
│  Edit Labels ─→ Embed ┘                                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    MelDecoder                           │
│  Latent ─→ Transformer ─→ OutProj ─┐                    │
│                                    ├─→ Edited Mel       │
│  Raw Mel ────────────→ Residual ───┘                    │
└─────────────────────────────────────────────────────────┘
```

### PPO Model - Factored 3-Head Action Space

The system uses a factored action space instead of 500 discrete actions:
- **Type head** (20 outputs): KEEP, CUT, LOOP, REORDER, JUMP_BACK, SKIP, FADE_IN/OUT, GAIN, SPEED_UP/DOWN, REVERSE, PITCH_UP/DOWN, EQ_LOW/HIGH, DISTORTION, REVERB, REPEAT_PREV, SWAP_NEXT
- **Size head** (5 outputs): BEAT, BAR, PHRASE, TWO_BARS, TWO_PHRASES
- **Amount head** (5 outputs): NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE

**Important**: Entropy is normalized by dividing by 3 (number of heads). Without this, max entropy = ln(20)+ln(5)+ln(5) ≈ 6.2, which overwhelms reward signals.

### Core Components

| Module | Purpose |
|--------|---------|
| `supervised_trainer.py` | Supervised mel reconstruction (FaceSwap-inspired) |
| `config.py` | Centralized hyperparameters (dataclass pattern) |
| `train.py` | PPO training with BC mixed loss, state normalization |
| `environment.py` | Gymnasium RL env with factored actions |
| `agent.py` | 3-head PolicyNetwork + ValueNetwork with HybridNATTENEncoder |
| `actions.py` | Factored action definitions (20 types × 5 sizes × 5 amounts) |
| `features.py` | 121-dim beat features (spectral, MFCCs, chroma, stems) |
| `state.py` | StateRepresentation (861-dim observation from beat context) |
| `auxiliary_tasks.py` | Multi-task learning (tempo, energy, phrase, mel reconstruction) |
| `pointer_network/` | Frame-level pointer model for quality-preserving edits |

## Key Metrics to Monitor (TensorBoard)

### Supervised Training
| Metric | Target | Problem If |
|--------|--------|------------|
| `train/l1_loss` | Decreasing | Stuck = not learning |
| `train/msstft_loss` | Decreasing | High = frequency artifacts |
| `val/total_loss` | Decreasing | Increasing = overfitting |

### PPO Training
| Metric | Target | Problem If |
|--------|--------|------------|
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

## Loss Functions (Supervised)

Inspired by FaceSwap's multi-scale perceptual losses:

1. **L1/MSE** - Pixel-level reconstruction
2. **Multi-Scale STFT** - Frequency-aware loss at multiple resolutions (512, 1024, 2048 FFT)
3. **Edit Consistency** - Penalize changes where edit_labels say KEEP
4. **Feature Loss** (optional) - Perceptual loss using audio embeddings

## Data Structure

```
editorbot/
├── cache/              # Centralized cache for all editors (gitignored contents)
│   ├── features/       # Cached mel spectrograms (.npz files)
│   └── labels/         # Cached edit labels (.npy files)
├── training_data/      # Training audio pairs (gitignored)
│   ├── input/          # Raw audio (*_raw.wav)
│   ├── desired_output/ # Human-edited versions (*_edit.wav)
│   ├── reference/      # Additional finished tracks (optional)
│   └── pointer_sequences/  # Generated pointer sequences
├── rl_editor/          # RL-based editor (BC + PPO)
├── super_editor/       # Supervised mel reconstruction (Phase 1 & 2)
├── audio_slicer/       # FaceSwap-style audio segmentation
├── mel_to_mel_editor/  # Direct mel transformation
├── pointer_network/    # Pointer network for frame alignment
├── scripts/            # Utility scripts
├── models/             # Saved checkpoints (gitignored)
├── logs/               # TensorBoard logs (gitignored)
├── bc_rich.npz         # BC dataset from infer_rich_bc_labels
└── bc_augmented.npz    # BC dataset with synthetic augmentation
```

## Development Guidelines

- Always use CUDA when available
- Use `--subprocess` flag for true multiprocessing on Windows (PPO)
- Start fresh training after changing normalization (weights incompatible)
- For supervised training, start with small models and scale up
- Monitor validation loss to detect overfitting
- Typical batch size: 8-16 for supervised, 2048 for PPO

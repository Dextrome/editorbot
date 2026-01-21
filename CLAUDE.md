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

### Approach 3: Pointer Network V2 (Current Focus)

Instead of generating audio, the pointer network **points to frames** in the original raw audio to construct the edit. For each output frame, it predicts which raw frame to copy.

```
Raw Mel (T_raw frames) → Linear Attn Encoder → Windowed Cross-Attn → Delta Prediction → Pointers
```

**Key Insight**: Editing is mostly rearrangement (cuts, loops, reordering) not generation. By copying frames from the original, we preserve 100% audio quality.

**V2 Architecture (Full-Sequence Delta Prediction):**
- **Linear Attention Encoder** - O(n) complexity via ELU+1 feature map, handles 50k+ frames
- **Global Summary Tokens** - ~64 tokens for long-range context (cuts/loops)
- **Position-Aware Windowed Cross-Attention** - Only attend to window around expected position
- **Delta Prediction Head** - Predicts offset from expected position (not absolute)
  - Small deltas: [-64, +64] for sequential edits (97-99% of cases)
  - Jump buckets: For large jumps (cuts, loops)
  - use_jump: Binary decision to use delta or jump
  - stop: End-of-sequence prediction
- **Edit Ops (optional auxiliary task)** - COPY, LOOP, SKIP, FADE labels

**Why Delta Prediction?**
- Expected position = output_position / compression_ratio (~0.67)
- Network learns WHEN to deviate, not memorizing absolute positions
- Compression ratio encodes prior that edited track is ~67% of raw length

**Data Pipeline:**
1. Align raw ↔ edited audio pairs via cross-correlation
2. Generate pointer sequences mapping each edit frame → raw frame
3. (Optional) Generate edit operation labels
4. Train pointer network with delta prediction

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

# Step 1: Precache new samples (mel spectrograms + pointer sequences)
# For specific samples:
python -m pointer_network.precache_samples bubbletron 20180927darkjam

# For all uncached samples:
python -m pointer_network.precache_samples

# Step 2: (Optional) Precache Demucs stems (only needed if use_stems=True)
# 2a. Extract stems from audio (creates raw audio waveforms in cache/stems/)
python scripts/precache_stems.py --data_dir training_data --cache_dir cache

# 2b. Convert stem audio to mel spectrograms (creates mel specs in cache/stems_mel/)
python scripts/convert_stems_to_mel.py

# Step 3: Train pointer network
python -m pointer_network.trainers.pointer_trainer \
    --config pointer_network/configs/full.json

# Or with explicit paths:
python -m pointer_network.trainers.pointer_trainer \
    --cache-dir cache \
    --pointer-dir training_data/pointer_sequences \
    --save-dir models/pointer_network \
    --epochs 100

# Resume training from checkpoint:
python -m pointer_network.trainers.pointer_trainer \
    --config pointer_network/configs/full.json \
    --resume models/pointer_network_full_v2/latest.pt

# Inference:
python -m pointer_network.infer input.wav --checkpoint models/pointer_network_full_v2/best.pt
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

### Pointer Network V2 Training
| Metric | Target | Problem If |
|--------|--------|------------|
| `train/delta_loss` | Decreasing | Stuck = not learning delta offsets |
| `train/jump_loss` | Decreasing | High = can't predict large jumps |
| `train/use_jump_loss` | Decreasing | High = can't decide delta vs jump |
| `train/stop_loss` | Decreasing | High = poor sequence ending |
| `val_delta_accuracy` | >90% | Low = model not learning patterns |
| `grad_norm` | <10 after clip | >100 = exploding gradients |

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

All paths are relative to `F:\editorbot\`:

```
F:\editorbot\
├── cache\                              # Centralized cache (gitignored contents)
│   ├── features\                       # Mel spectrograms: {sample}_raw.npz, {sample}_edit.npz
│   │   └── *.npz                       # Contains 'mel' array (128, time), normalized (mel_db+80)/80
│   ├── labels\                         # Cached edit labels (.npy files)
│   ├── stems\                          # Raw audio waveforms from Demucs (intermediate)
│   │   └── {sample}_stems.npz          # Contains drums, bass, vocals, other audio arrays
│   └── stems_mel\                      # Mel spectrograms of stems (4 channels)
│       └── {sample}_stems.npz          # Contains drums, bass, vocals, other mel arrays (128, time)
│
├── training_data\                      # Training audio pairs (gitignored)
│   ├── input\                          # Raw recordings
│   │   └── {sample}_raw.wav            # e.g., 20180927darkjam_raw.wav
│   ├── desired_output\                 # Human-edited versions
│   │   └── {sample}_edit.wav           # e.g., 20180927darkjam_edit.wav
│   ├── reference\                      # Additional finished tracks (optional)
│   └── pointer_sequences\              # Generated pointer/ops data
│       ├── {sample}_pointers.npy       # Frame indices mapping edit→raw
│       ├── {sample}_ops.npy            # Edit operation codes per frame
│       ├── {sample}_info.json          # Alignment metadata
│       └── {sample}_alignment.png      # Visual alignment plot
│
├── pointer_network\                    # Pointer network code
│   ├── models\                         # Model definitions
│   │   └── pointer_network.py          # Main PointerNetwork class
│   ├── data\                           # Dataset classes
│   │   └── dataset.py                  # PointerDataset
│   ├── trainers\                       # Training code
│   │   └── pointer_trainer.py          # PointerNetworkTrainer
│   ├── configs\                        # JSON config files
│   │   └── full.json                   # Full training config with stems
│   └── infer.py                        # Inference script
│
├── models\                             # Saved checkpoints (gitignored)
│   └── pointer_network_full_v2\        # Current best model
│       ├── best.pt                     # Best validation loss checkpoint
│       └── latest.pt                   # Most recent checkpoint
│
├── logs\                               # TensorBoard logs (gitignored)
├── rl_editor\                          # RL-based editor (BC + PPO)
├── super_editor\                       # Supervised mel reconstruction
├── audio_slicer\                       # Audio segmentation
├── mel_to_mel_editor\                  # Direct mel transformation
└── scripts\                            # Utility scripts
```

### Key File Formats

| File Type | Location | Format |
|-----------|----------|--------|
| Raw audio | `training_data/input/{sample}_raw.wav` | WAV, any sample rate |
| Edited audio | `training_data/desired_output/{sample}_edit.wav` | WAV, any sample rate |
| Mel cache | `cache/features/{sample}_{raw|edit}.npz` | `mel`: (128, time) float32 |
| Stem mel cache | `cache/stems_mel/{sample}_stems.npz` | `drums, bass, vocals, other`: (128, time) each |
| Pointers | `training_data/pointer_sequences/{sample}_pointers.npy` | int64 array (edit_frames,) |
| Edit ops | `training_data/pointer_sequences/{sample}_ops.npy` | int8 array (edit_frames,) |

## Development Guidelines

- Always use CUDA when available
- Use `--subprocess` flag for true multiprocessing on Windows (PPO)
- Start fresh training after changing normalization (weights incompatible)
- For supervised training, start with small models and scale up
- Monitor validation loss to detect overfitting
- Typical batch size: 8-16 for supervised, 2048 for PPO

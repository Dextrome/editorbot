# Audio Editor

A deep learning system for automated music editing. The agent learns to make intelligent editing decisions by training on paired raw/edited audio examples.

## Overview

This project explores multiple approaches to learning audio editing from examples:

| Approach | Status | Description |
|----------|--------|-------------|
| **Pointer Network V2** | **Current Focus** | Delta prediction + linear attention for full sequences |
| Supervised Mel Reconstruction | Implemented | Direct mel spectrogram reconstruction |
| RL with Factored Actions | Implemented | Sequential decision making with PPO |

## Pointer Network V2 (Current Focus)

Instead of generating new audio, the pointer network learns to **copy and reorder frames** from the original recording. This preserves audio quality while learning editing patterns.

**Key Features:**
- **O(n) Linear Attention** - Handles 50k+ frame sequences
- **Delta Prediction** - Predicts offset from expected position, not absolute
- **Full-Sequence Training** - No train/inference mismatch
- **Multi-Stem Support** - Optional drums/bass/vocals/other encoding

```
Raw Mel → Linear Attention Encoder → Windowed Cross-Attention → Delta Prediction → Pointers
```

**Why Pointers?** Editing is mostly selection/reordering, not generation. Copying frames preserves 100% audio quality.

> **Full documentation:** See [`pointer_network/README.md`](pointer_network/README.md) for architecture details, configuration, and training guide.

### Quick Start

```bash
# 1. Precache mel spectrograms and generate pointer sequences
python -m pointer_network.precache_samples

# 2. (Optional) Precache stems for multi-track mode
python scripts/precache_stems.py && python scripts/convert_stems_to_mel.py

# 3. Train
python -m pointer_network.trainers.pointer_trainer --config pointer_network/configs/full.json

# 4. Inference
python -m pointer_network.infer input.wav --checkpoint models/pointer_network_full_v2/best.pt
```

## Setup

### Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install torch torchaudio librosa soundfile numpy gymnasium tensorboard tqdm
pip install demucs  # Stem separation (optional)
```

### Training Data

```
training_data/
├── input/           # Raw audio files (*_raw.wav)
├── desired_output/  # Human-edited versions (*_edit.wav)
└── reference/       # Additional finished tracks (optional)
```

Files are matched by name prefix (e.g., `song1_raw.wav` ↔ `song1_edit.wav`).

## Project Structure

```
editorbot/
├── pointer_network/           # Pointer-based editing (CURRENT FOCUS)
│   ├── README.md              # Detailed V2 documentation
│   ├── models/pointer_network.py
│   ├── data/dataset.py
│   ├── trainers/pointer_trainer.py
│   └── configs/full.json
│
├── super_editor/              # Two-phase supervised editor
│   └── README.md              # Phase 1 reconstruction + Phase 2 RL
├── audio_slicer/              # Anomaly-based editing (experimental)
│   └── README.md              # FaceSwap-style approaches & lessons
├── rl_editor/                 # RL-based editing (PPO + BC)
├── mel_to_mel_editor/         # Direct mel transformation
├── scripts/                   # Utility scripts
│
├── cache/                     # Cached mel spectrograms
├── training_data/             # Audio pairs + pointer sequences
├── models/                    # Saved checkpoints
└── logs/                      # TensorBoard logs
```

> **Alternative approach docs:** [`super_editor/README.md`](super_editor/README.md) (two-phase supervised) | [`audio_slicer/README.md`](audio_slicer/README.md) (anomaly detection)

## Alternative Approaches

### Supervised Mel Reconstruction

Direct reconstruction using multi-scale perceptual losses. See [`super_editor/README.md`](super_editor/README.md) for two-phase approach details.

```bash
python -m rl_editor.supervised_trainer --data-dir training_data --save-dir models/supervised
```

### RL with Factored Actions

Sequential decision making with PPO + Behavioral Cloning:

```bash
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz
python -m rl_editor.train --save-dir models/rl --bc-mixed bc_rich.npz --bc-weight 0.3
```

The factored action space uses 3 heads (20×5×5 = 500 combinations):
- **Type** (20): KEEP, CUT, LOOP, FADE, GAIN, PITCH, SPEED...
- **Size** (5): BEAT, BAR, TWO_BARS, PHRASE, TWO_PHRASES
- **Amount** (5): NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE

## Monitoring

```bash
tensorboard --logdir logs
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended (8GB+ VRAM for full model)

## Lessons Learned

| Approach | Why It Failed |
|----------|---------------|
| FaceSwap-style dual autoencoder | Raw/edited audio too similar |
| Binary classifier (edited=good) | Good content exists in both datasets |
| Direct mel generation | Quality degradation |
| Pure RL (no BC) | 500 action combinations too large for exploration |

**Key insight**: Editing is mostly **selection/reordering**, not generation.

> **Detailed analysis:** See [`audio_slicer/README.md`](audio_slicer/README.md) for in-depth documentation of failed approaches and what we learned from them.

## Development

See [`CLAUDE.md`](CLAUDE.md) for development guidelines and [`pointer_network/README.md`](pointer_network/README.md) for V2 architecture details.

# Super Editor - Supervised Audio Editing System

A two-phase approach to learning audio editing:
1. **Phase 1**: Supervised autoencoder learns to reconstruct edited audio from (raw + edit_labels)
2. **Phase 2**: RL agent learns to predict optimal edit_labels

## Why This Approach?

The current PPO approach tries to learn **both** what edits to make AND how to execute them simultaneously. This is hard because:
- 500 action combinations make exploration inefficient
- Sparse rewards (only at episode end) provide weak signal
- No direct supervision on output quality

The **Super Editor** approach separates these concerns:
- **Phase 1**: Learn a strong audio reconstruction model (supervised, dense loss)
- **Phase 2**: Learn edit prediction using the frozen reconstruction model as reward

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: RECONSTRUCTION                      │
│                                                                      │
│   Raw Audio ──┐                                                      │
│               ├──→ [Encoder] ──→ Latent ──→ [Decoder] ──→ Edited Mel │
│   Edit Labels ┘                                                      │
│                                                                      │
│   Loss: L1 + Multi-Scale STFT + Edit Consistency                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: EDIT PREDICTION                     │
│                                                                      │
│   Raw Audio ──→ [Edit Predictor] ──→ Predicted Labels                │
│                        │                    │                        │
│                        │                    ▼                        │
│                        │         [Frozen Reconstruction Model]       │
│                        │                    │                        │
│                        │                    ▼                        │
│                        └────────→ Reward = Quality(output)           │
│                                                                      │
│   Training: PPO on edit prediction, reconstruction model as reward   │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
super_editor/
├── README.md                    # This file
├── PHASE1_PLAN.md              # Detailed Phase 1 implementation plan
├── PHASE2_PLAN.md              # Detailed Phase 2 implementation plan
├── config.py                   # Configuration dataclasses
├── models/
│   ├── encoder.py              # Audio encoder (raw mel → latent)
│   ├── decoder.py              # Audio decoder (latent → edited mel)
│   ├── edit_predictor.py       # Edit label predictor for Phase 2
│   └── vocoder.py              # Optional: latent → waveform
├── losses/
│   ├── reconstruction.py       # L1, MSE, spectral losses
│   ├── perceptual.py          # Feature-based perceptual loss
│   └── consistency.py         # Edit consistency loss
├── data/
│   ├── dataset.py             # Paired audio dataset
│   ├── augmentation.py        # Audio augmentation
│   └── preprocessing.py       # Mel extraction, normalization
├── trainers/
│   ├── phase1_trainer.py      # Supervised reconstruction training
│   └── phase2_trainer.py      # RL edit prediction training
├── inference/
│   ├── reconstruct.py         # Run reconstruction model
│   └── full_pipeline.py       # End-to-end inference
└── checkpoints/               # Saved models
```

## Quick Start

```bash
# Phase 1: Train reconstruction model
python -m super_editor.trainers.phase1_trainer \
    --data-dir training_data \
    --save-dir super_editor/checkpoints/phase1 \
    --epochs 100

# Phase 2: Train edit predictor (after Phase 1 converges)
python -m super_editor.trainers.phase2_trainer \
    --data-dir training_data \
    --reconstruction-model super_editor/checkpoints/phase1/best.pt \
    --save-dir super_editor/checkpoints/phase2 \
    --epochs 1000

# Inference
python -m super_editor.inference.full_pipeline \
    --input raw_audio.wav \
    --edit-model super_editor/checkpoints/phase2/best.pt \
    --recon-model super_editor/checkpoints/phase1/best.pt \
    --output edited_audio.wav
```

## Key Differences from RL Approach

| Aspect | RL (PPO) Approach | Super Editor |
|--------|-------------------|--------------|
| **Supervision** | Sparse episode rewards | Dense reconstruction loss |
| **Action space** | 500 discrete actions | Binary edit labels per beat |
| **Learning** | End-to-end, unstable | Two-phase, stable |
| **Sample efficiency** | Low (needs exploration) | High (direct supervision) |
| **Debugging** | Hard (reward hacking) | Easy (can visualize reconstructions) |

## Hardware Requirements

- **Phase 1**: ~8GB VRAM, trains in ~6-12 hours on RTX 4070 Ti
- **Phase 2**: ~4GB VRAM, trains in ~2-4 hours
- **Inference**: ~2GB VRAM, real-time capable

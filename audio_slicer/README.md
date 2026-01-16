# Audio Slicer - Anomaly-Based Audio Editing

## Overview

A **completely label-free** approach to automatic audio editing. Instead of predicting edit labels or doing frame-to-frame mapping, this system learns **what good audio sounds like** from your edited examples, then detects "anomalies" (bad segments) in raw audio by reconstruction error.

## Core Insight: Anomaly Detection

The key realization: **Edited audio is a subset of raw audio.** Human editors SELECT good content and CUT bad content.

Instead of trying to classify "good vs bad" (which fails because good content exists in both raw and edited), we:
1. Train an autoencoder **only on edited (good) audio**
2. For raw audio, segments with **high reconstruction error** are anomalies (bad)
3. Keep segments with low error, cut segments with high error

## Why This Works

The autoencoder only sees "good" audio during training. When presented with:
- **Good segments**: Model recognizes patterns, low reconstruction error
- **Bad segments**: Model hasn't seen these patterns, high reconstruction error

This is classic anomaly detection applied to audio editing.

## What We Tried (and Why They Failed)

### FaceSwap-Style Dual Autoencoder
- Trained shared encoder + separate decoders for raw/edited
- **Failed**: All segments scored similarly (0.81-0.89)
- **Why**: Raw and edited audio share too much similarity (same melody, key, tempo)

### Binary Classifier (edited=good, raw=bad)
- Trained classifier to distinguish edited from raw
- **Failed**: Classifier learned "music-like features" not editing patterns
- **Why**: Good content exists in both datasets, can't discriminate

### Anomaly Detection (WORKS)
- Train autoencoder only on edited audio
- Score by reconstruction error
- **Success**: Clear separation between good and bad segments

## Architecture (Anomaly Detector)

```
                    ┌─────────────────┐
Edited Audio ─────► │    Encoder      │ ────► Latent (z)
(training only)     │  (conv + pool)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Decoder      │ ────► Reconstruction
                    │ (conv transpose)│
                    └─────────────────┘

Inference:
                    ┌─────────────────┐
Raw Segment ──────► │   Autoencoder   │ ────► Reconstruction
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  MSE(input,     │ ────► Error Score
                    │  reconstruction)│      (high = anomaly)
                    └─────────────────┘
```

## Training

### Data Requirements
- Paired raw + edited audio files (but only edited files are used for training)
- Same as before: `training_data/input/*.wav` and `training_data/desired_output/*.wav`
- **No labels needed** - just the edited audio

### Training Process
1. Extract random segments from edited audio only
2. Train autoencoder to reconstruct these segments
3. Model learns the distribution of "good" audio

```bash
python _train_anomaly.py
```

### Key Hyperparameters
- `segment_frames`: Length of segments (64 = ~0.75 seconds)
- `hidden_dims`: Encoder/decoder capacity ([32, 64, 128])
- `latent_dim`: Bottleneck size (64)
- `epochs`: Training iterations (100)

## Inference

### Segment Scoring
For each segment of raw audio:
1. Feed through autoencoder
2. Compute MSE between input and reconstruction
3. High error = anomaly (bad segment)

### Selection Process
```python
# Using the anomaly detector
from audio_slicer.trainers.anomaly_trainer import SimpleAutoencoder

model = SimpleAutoencoder.from_checkpoint('models/anomaly_detector/best.pt')

# Score segment
error = ((model(segment)['reconstruction'] - segment) ** 2).mean()

# Low error = good segment, high error = bad segment
```

### Threshold Selection
- Use percentile-based threshold (e.g., 40th percentile of errors)
- Segments below threshold are kept
- Adjust based on desired compression ratio

## Key Differences from Previous Approaches

| Approach | Labels | Frame Mapping | Can Cut | Works |
|----------|--------|---------------|---------|-------|
| super_editor (Phase 1) | Yes | 1:1 | Via DSP | Limited |
| super_editor (Phase 2) | Predicted | 1:1 | Via labels | Collapsed |
| mel_to_mel | No | 1:1 | No | Can't cut |
| FaceSwap-style | No | No | Yes | Failed (no discrimination) |
| Binary classifier | No | No | Yes | Failed (all high scores) |
| **Anomaly detector** | **No** | **No** | **Yes** | **Works!** |

## Future Ideas

### 1. Contrastive Loss
Add contrastive term: edited segments should cluster together in latent space.

### 2. Adversarial Training
Add discriminator to distinguish real edited audio from transformed raw audio.

### 3. Multi-scale Processing
Process at multiple time scales (beat, bar, phrase level).

### 4. Beat-aligned Segments
Align segment boundaries to detected beats for smoother cuts.

### 5. Transformer Encoder
Replace conv encoder with transformer for better long-range dependencies.

### 6. Perceptual Loss
Add VGG-style perceptual loss using audio embeddings (like CLAP or wav2vec).

## Troubleshooting

### Low quality scores everywhere
- Model may not have learned good representations
- Try training longer
- Increase model capacity

### Everything gets cut
- Threshold too high
- Raw audio very different from edited style
- Lower threshold or check data quality

### Nothing gets cut
- Threshold too low
- Model learned identity mapping
- Check that raw and edited are actually different

## Files

```
audio_slicer/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── README.md                # This file
├── IDEAS.md                 # Insights and future directions
├── models/
│   ├── scorer.py            # Quality scorer (binary classifier - failed)
│   ├── autoencoder.py       # Single autoencoder
│   └── dual_autoencoder.py  # FaceSwap-style dual AE (failed)
├── data/
│   └── dataset.py           # Segment extraction from cache
├── trainers/
│   ├── trainer.py           # Dual AE training (failed)
│   ├── classifier_trainer.py # Binary classifier training (failed)
│   └── anomaly_trainer.py   # Anomaly detector training (WORKS)
└── inference/
    └── slicer.py            # Segment scoring and selection

# Training scripts
_train_anomaly.py            # Train the anomaly detector
_test_anomaly.py             # Test segment selection

# Output
models/anomaly_detector/     # Trained model checkpoints
```

## Results

Using the anomaly detector on a 641-second raw audio file:
- **Input**: 641 seconds (raw)
- **Output**: 313 seconds (49% of original)
- **Target**: ~42% (based on actual edited audio ratio)

The model successfully identifies:
- **Anomalies (high error)**: Transitions, problematic sections
- **Good segments (low error)**: Clean, consistent audio

---

## Current Limitation: Model Doesn't Actually Learn to Edit

**Important realization**: The anomaly detector approach has a fundamental limitation.

What the model does:
- Learns what "good" audio sounds like
- Scores segments by reconstruction error

What the model does NOT do:
- Learn WHERE humans cut
- Learn WHAT to keep/repeat
- Learn HOW to structure the edit

**All editing logic is manual code** in the inference script:
- Where to cut → manual (energy-based, beat-aligned, onset detection)
- Minimum segment length → manual (8 seconds)
- How to transition → manual (crossfade mixing)

The model is a quality scorer, not an editor.

---

## NEXT: Pointer-Based Sequence Model

To build a system that actually **learns editing behavior**, we're planning a pointer network approach.

### Concept

```
Input:  Raw audio + Edited audio (as reference)
Output: Sequence of frame pointers [1000, 1001, ..., 2000, 1000, ...]
```

The model outputs which frames from raw audio to use, in what order. This allows:
- **Cutting**: Skip frames
- **Keeping**: Include frames
- **Looping**: Point to same frames multiple times
- **Reordering**: Change sequence

### Architecture

```
Raw Audio  ───→ Encoder ───→ ┐
                             ├─→ Cross-attention ─→ Pointer Decoder ─→ [indices]
Edited Audio ─→ Encoder ───→ ┘
```

The cross-attention implicitly learns alignment (which parts of raw appear in edited).

### Key Challenges

1. **Long sequences**: 10 min audio = 50k frames to point into
2. **Variable output length**: Model must decide when to stop
3. **Training signal**: Need aligned raw↔edited pairs
4. **Multiple valid edits**: No single "correct" answer

See `IDEAS.md` for full details on limitations and possible improvements.

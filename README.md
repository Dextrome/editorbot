# Audio Editor

A deep learning system for automated music editing. The agent learns to make intelligent editing decisions by training on paired raw/edited audio examples.

## Overview

This project explores multiple approaches to learning audio editing from examples:

| Approach | Status | Description |
|----------|--------|-------------|
| **Pointer Network** | **Current Focus** | Learns to reorder/select frames from raw audio |
| Supervised Mel Reconstruction | Implemented | Direct mel spectrogram reconstruction |
| RL with Factored Actions | Implemented | Sequential decision making with PPO |
| Mel-to-Mel Editor | Experimental | Transformer-based mel transformation |

## Pointer Network (Current Focus)

Instead of generating new audio, the pointer network learns to **copy and reorder frames** from the original recording. This preserves audio quality while learning editing patterns.

```
Raw Mel (T_raw frames)
        │
        ▼
┌───────────────────────┐
│   Multi-Scale Encoder │  ← Frame/Beat/Bar level encoding
│   + Music-Aware PE    │  ← Beat, bar, phrase structure
│   + Edit Style VAE    │  ← Latent space for edit diversity
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Hierarchical Decoder │  ← Bar → Beat → Frame prediction
│   + Sparse Attention  │  ← O(n) not O(n²)
│   + KV Caching        │  ← Fast inference
└───────────────────────┘
        │
        ▼
Pointer Sequence (T_edit indices into raw)
```

### Why Pointers?

- **Preserves quality**: Copies frames, never generates
- **Learns patterns**: Cuts, loops, reordering from examples
- **Hierarchical**: Coarse-to-fine (bar → beat → frame) for musical coherence
- **Variable length**: STOP token handles different output lengths

### Special Tokens

| Token | Value | Purpose |
|-------|-------|---------|
| `STOP_TOKEN` | -2 | End of output sequence |
| `PAD_TOKEN` | -1 | Padding for batching |

### Quick Start (Pointer Network)

```bash
# 1. Generate pointer sequences from paired audio
python -m pointer_network.generate_pointer_sequences

# 2. Train the pointer network
python -m pointer_network.trainers.pointer_trainer \
    --cache-dir cache \
    --pointer-dir training_data/pointer_sequences \
    --save-dir models/pointer_network \
    --epochs 100
```

### Inference Example

```python
from pointer_network import PointerNetwork
import torch

# Load trained model
model = PointerNetwork(d_model=256, n_heads=8).cuda()
checkpoint = torch.load('models/pointer_network/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference on raw mel spectrogram
with torch.no_grad():
    result = model.generate(raw_mel.unsqueeze(0).cuda(), max_length=10000)
    pointers = result['pointers'][0]  # Frame indices into raw audio

# Reconstruct edited audio by selecting frames
edited_mel = raw_mel[:, pointers[pointers >= 0]]
```

## Setup

### Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install torch torchaudio librosa soundfile numpy gymnasium tensorboard tqdm
pip install natten  # Neighborhood attention (optional, for RL approach)
pip install demucs  # Stem separation (optional)
```

### External Dependencies

The vocoder modules (for mel-to-audio conversion) should be cloned separately:

```bash
cd vocoder
git clone https://github.com/NVIDIA/BigVGAN.git
git clone https://github.com/jik876/hifi-gan.git
```

### Training Data

```
training_data/
├── input/           # Raw audio files (*_raw.wav or *_raw.mp3)
├── desired_output/  # Human-edited versions (*_edit.wav)
└── reference/       # Additional finished tracks (optional)
```

Files are matched by name prefix (e.g., `song1_raw.wav` ↔ `song1_edit.wav`).

## Project Structure

```
editorbot/
├── pointer_network/           # Pointer-based editing (CURRENT FOCUS)
│   ├── models/
│   │   └── pointer_network.py # Main model with hierarchical pointers
│   ├── data/
│   │   └── dataset.py         # PointerDataset, collate_fn
│   ├── trainers/
│   │   └── pointer_trainer.py # Training loop
│   ├── generate_pointer_sequences.py  # Create training data
│   └── config.py              # Configuration
│
├── super_editor/              # Multi-component editor
│   ├── models/                # Encoder, decoder, edit classifier
│   ├── trainers/              # Training utilities
│   └── config.py
│
├── mel_to_mel_editor/         # Direct mel transformation
│   └── models/
│
├── audio_slicer/              # Audio segmentation utilities
│
├── rl_editor/                 # RL-based editing
│   ├── train.py               # PPO training loop
│   ├── agent.py               # Policy/Value networks
│   ├── environment.py         # Gymnasium environment
│   ├── actions.py             # Factored action space (20×5×5)
│   ├── config.py              # Hyperparameters
│   ├── features.py            # Audio feature extraction
│   ├── supervised_trainer.py  # Supervised reconstruction
│   └── infer.py               # Inference
│
├── vocoder/                   # Mel-to-audio conversion
│   ├── BigVGAN/               # (clone externally)
│   └── hifi-gan/              # (clone externally)
│
├── scripts/                   # Utilities
│   ├── generate_synthetic_pairs.py
│   ├── precache_labels.py
│   ├── precache_stems.py
│   ├── regenerate_cache.py
│   └── train_super_editor.py
│
├── training_data/             # Audio pairs for training
├── models/                    # Saved checkpoints
├── logs/                      # TensorBoard logs
├── test_audio/                # Test audio files
├── lr_finder/                 # Learning rate finder outputs
└── CLAUDE.md                  # Development guidelines
```

## Alternative Approaches

### Supervised Mel Reconstruction

Direct reconstruction using multi-scale perceptual losses:

```bash
python -m rl_editor.supervised_trainer \
    --data-dir training_data \
    --save-dir models/supervised \
    --epochs 100
```

### RL with Factored Actions

Sequential decision making with PPO + Behavioral Cloning:

```bash
# Generate BC dataset
python -m scripts.infer_rich_bc_labels --data_dir training_data --out bc_rich.npz

# Train with BC + PPO
python -m rl_editor.train \
    --save-dir models/rl_model \
    --bc-mixed bc_rich.npz \
    --bc-weight 0.3 \
    --subprocess
```

The factored action space uses 3 heads instead of 500 discrete actions:
- **Type** (20): KEEP, CUT, LOOP, FADE, GAIN, PITCH, SPEED...
- **Size** (5): BEAT, BAR, TWO_BARS, PHRASE, TWO_PHRASES
- **Amount** (5): NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE

## Configuration

### Pointer Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_mels` | 128 | Mel spectrogram bins |
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_encoder_layers` | 4 | Encoder layers |
| `n_decoder_layers` | 4 | Decoder layers |
| `frames_per_beat` | 43 | ~86ms at 22050Hz, hop=256 |
| `frames_per_bar` | 172 | 4 beats per bar |

### Audio Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample rate | 22050 Hz | Standard for music |
| n_mels | 128 | Mel frequency bins |
| n_fft | 2048 | FFT window size |
| hop_length | 256 | ~11.6ms per frame |

### Loss Weights (Pointer Network)

| Loss | Weight | Purpose |
|------|--------|---------|
| Frame Pointer CE | 1.0 | Main frame-level prediction |
| Bar Pointer CE | 0.3 | Coarse bar-level guidance |
| Beat Pointer CE | 0.3 | Medium beat-level guidance |
| Stop BCE | 0.5 | End-of-sequence prediction |
| KL Divergence | 0.01 | VAE regularization |
| Length MSE | 0.1 | Output length prediction |

### Metric Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Pointer Accuracy | >80% | Exact frame match |
| Within-5 Accuracy | >95% | Pointer within 5 frames |
| Length MAE | <50 frames | Mean absolute length error |
| Stop Precision | >90% | Correct stop predictions |

## Training Tips

1. **Start with mini config**: Use `d_model=128`, 2 layers for fast iteration
2. **Monitor gradient norms**: Should stay <10 after clipping; >100 indicates instability
3. **Check alignment quality**: Pointer sequences should have >95% valid alignments
4. **Use mixed precision**: AMP gives ~2x speedup with minimal quality loss
5. **Lower LR if unstable**: Start with 5e-5 if gradients explode, increase gradually

## Monitoring

```bash
tensorboard --logdir logs
```

### Key Metrics (Pointer Network)
- `pointer_loss` - Frame-level pointer accuracy
- `bar_pointer_loss` - Bar-level prediction
- `beat_pointer_loss` - Beat-level prediction
- `val_accuracy` - Validation pointer accuracy

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended

### Hardware by Approach

| Approach | VRAM | RAM | Training Time |
|----------|------|-----|---------------|
| Pointer Network (mini) | 4GB | 8GB | ~2 hours (50 epochs) |
| Pointer Network (full) | 8GB | 16GB | ~8 hours (100 epochs) |
| Super Editor Phase 1 | 8GB | 8GB | ~6-12 hours |
| Super Editor Phase 2 | 4GB | 8GB | ~2-4 hours |
| RL (PPO) | 8GB | 16GB | ~24+ hours |

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Gradients exploding (>100 norm) | LR too high or unstable data | Lower LR to 5e-5, check data alignment |
| Loss stuck at high value | Model not learning | Verify data pipeline, try smaller model first |
| OOM errors | Batch/sequence too large | Reduce batch size, enable chunking |
| All pointers same value | Action collapse | Increase label smoothing, check loss weights |
| Training very slow | Variable sequence lengths | Use fixed-length collation for torch.compile |
| NaN in loss | Numerical instability | Enable AMP, add gradient clipping |

## Lessons Learned

Approaches that **didn't work** (documented in `audio_slicer/`):

| Approach | Why It Failed |
|----------|---------------|
| FaceSwap-style dual autoencoder | Raw/edited audio too similar, couldn't discriminate |
| Binary classifier (edited=good) | Good content exists in both datasets |
| Direct mel generation | Quality degradation, can't preserve original audio |
| Pure RL (no BC) | 500 action combinations too large for exploration |

**Key insight**: Editing is mostly **selection/reordering**, not generation. Pointer networks preserve quality by copying frames.

## Roadmap

- [x] Hierarchical pointers (bar → beat → frame)
- [x] Sparse attention for long sequences
- [x] OneCycleLR scheduler
- [ ] Edit operation tokens (COPY, LOOP, CUT labels)
- [ ] Multi-track stem coherence

## Development

See `CLAUDE.md` for detailed development guidelines.

```bash
# Run tests
pytest rl_editor/tests/

# Check pointer network compiles
python -c "from pointer_network import PointerNetwork; print('OK')"
```

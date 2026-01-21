# Audio Editor

A deep learning system for automated music editing. The agent learns to make intelligent editing decisions by training on paired raw/edited audio examples.

## Overview

This project explores multiple approaches to learning audio editing from examples:

| Approach | Status | Description |
|----------|--------|-------------|
| **Pointer Network** | **Current Focus** | Edit operations + multi-stem + stable training |
| Supervised Mel Reconstruction | Implemented | Direct mel spectrogram reconstruction |
| RL with Factored Actions | Implemented | Sequential decision making with PPO |
| Mel-to-Mel Editor | Experimental | Transformer-based mel transformation |

## Pointer Network (Current Focus)

Instead of generating new audio, the pointer network learns to **copy and reorder frames** from the original recording. This preserves audio quality while learning editing patterns.

```
Raw Mel (T_raw frames)
        |
        v
+---------------------------+
|   Multi-Scale Encoder     |  <- Frame/Beat/Bar level encoding
|   + Music-Aware PE        |  <- Beat, bar, phrase structure
|   + Edit Style VAE        |  <- Latent space for edit diversity
|   + Stem Encoder (opt)    |  <- Multi-track: drums, bass, vocals, other
+---------------------------+
        |
        v
+---------------------------+
|  Hierarchical Decoder     |  <- Bar -> Beat -> Frame prediction
|   + Pre-LayerNorm         |  <- Stable training (no NaN)
|   + Sparse Attention      |  <- O(n) not O(n^2)
|   + KV Caching            |  <- Fast inference
+---------------------------+
        |
        v
+---------------------------+
|  Edit Op Head             |  <- COPY, LOOP, SKIP, FADE, STOP
|  Pointer Head             |  <- Frame indices into raw
+---------------------------+
```

### Why Pointers?

- **Preserves quality**: Copies frames, never generates
- **Learns patterns**: Cuts, loops, reordering from examples
- **Hierarchical**: Coarse-to-fine (bar -> beat -> frame) for musical coherence
- **Variable length**: STOP token handles different output lengths
- **Edit operations**: Explicit labels for what each edit does

### Special Tokens & Edit Operations

| Token | Value | Purpose |
|-------|-------|---------|
| `STOP_TOKEN` | -2 | End of output sequence |
| `PAD_TOKEN` | -1 | Padding for batching |

| EditOp | Value | Purpose |
|--------|-------|---------|
| `COPY` | 0 | Copy frame at pointer position |
| `LOOP_START` | 1 | Mark start of loop region |
| `LOOP_END` | 2 | End loop, jump back to LOOP_START |
| `SKIP` | 3 | Skip N frames (cut) |
| `FADE_IN` | 4 | Apply fade in |
| `FADE_OUT` | 5 | Apply fade out |
| `STOP` | 6 | End of sequence |

### Quick Start (Pointer Network)

```bash
# 1. Find optimal learning rate
python -m pointer_network.lr_finder --pointer-dir training_data/pointer_sequences

# 2. Generate pointer sequences from paired audio
python -m pointer_network.generate_pointer_sequences

# 3. (Optional) Precache stems for multi-track mode
# 3a. Extract stems from audio (raw audio waveforms)
python -m scripts.precache_stems

# 3b. Convert stems to mel spectrograms (required for model)
python scripts/convert_stems_to_mel.py

# 4. Train the pointer network
python -m pointer_network.trainers.pointer_trainer \
    --cache-dir cache \
    --pointer-dir training_data/pointer_sequences \
    --save-dir models/pointer_network \
    --epochs 100
```

### Inference Example

```python
from pointer_network import PointerNetwork, EditOp
import torch

# Load trained model
model = PointerNetwork(
    d_model=256,
    n_heads=8,
    use_pre_norm=True,   # Pre-LayerNorm for stability
    use_edit_ops=True,   # Edit operation tokens
).cuda()
checkpoint = torch.load('models/pointer_network/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference on raw mel spectrogram
with torch.no_grad():
    result = model.generate(raw_mel.unsqueeze(0).cuda(), max_length=10000)
    pointers = result['pointers'][0]  # Frame indices into raw audio

# Reconstruct edited audio by selecting frames
edited_mel = raw_mel[:, pointers[pointers >= 0]]

# EditOp usage
print(EditOp.names())  # ['COPY', 'LOOP_START', 'LOOP_END', 'SKIP', 'FADE_IN', 'FADE_OUT', 'STOP']
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

Files are matched by name prefix (e.g., `song1_raw.wav` <-> `song1_edit.wav`).

## Project Structure

```
editorbot/
├── pointer_network/           # Pointer-based editing (CURRENT FOCUS)
│   ├── models/
│   │   └── pointer_network.py # Main model with all features
│   ├── data/
│   │   └── dataset.py         # PointerDataset, collate_fn
│   ├── trainers/
│   │   └── pointer_trainer.py # Training loop with bfloat16
│   ├── generate_pointer_sequences.py  # Create training data
│   ├── lr_finder.py           # Learning rate finder
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
│   ├── actions.py             # Factored action space (20x5x5)
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

### Pointer Network Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_mels` | 128 | Mel spectrogram bins |
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_encoder_layers` | 4 | Encoder layers |
| `n_decoder_layers` | 4 | Decoder layers |
| `frames_per_beat` | 43 | ~86ms at 22050Hz, hop=256 |
| `frames_per_bar` | 172 | 4 beats per bar |
| `use_pre_norm` | True | Pre-LayerNorm (stable training) |
| `use_edit_ops` | True | Edit operation tokens |
| `use_stems` | False | Multi-stem encoding |
| `n_stems` | 4 | Number of stems |
| `label_smoothing` | 0.1 | Label smoothing for loss |

### Pointer Network Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-5 | Base learning rate |
| `scheduler_type` | warmup_cosine | Scheduler: warmup, warmup_cosine, cosine, onecycle |
| `warmup_steps` | 1000 | Linear warmup steps |
| `use_bfloat16` | True | bfloat16 mixed precision (stable) |
| `gradient_clip` | 1.0 | Gradient clipping |
| `weight_decay` | 0.01 | AdamW weight decay |

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
| Bar Pointer CE | 0.2 | Coarse bar-level guidance |
| Beat Pointer CE | 0.3 | Medium beat-level guidance |
| Edit Op CE | 1.0 | Edit operation prediction |
| Stop BCE | 0.5 | End-of-sequence prediction |
| KL Divergence | 0.1 | VAE regularization |
| Length MSE | 0.01 | Output length prediction |

### Metric Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Pointer Accuracy | >80% | Exact frame match |
| Within-5 Accuracy | >95% | Pointer within 5 frames |
| Length MAE | <50 frames | Mean absolute length error |
| Stop Precision | >90% | Correct stop predictions |

## Training Tips

### Stability Features (Prevent NaN)

1. **Pre-LayerNorm** (default): Normalizes before attention, much more stable
2. **bfloat16** (default): Same exponent range as float32, no GradScaler needed
3. **Warmup+Cosine scheduler** (default): No LR spikes like OneCycleLR
4. **Label smoothing**: 0.1 default prevents overconfident predictions
5. **Xavier init with gain=0.1**: Small initial weights for stability

### General Tips

1. **Run LR finder first**: `python -m pointer_network.lr_finder`
2. **Start with mini config**: Use `d_model=128`, 2 layers for fast iteration
3. **Monitor gradient norms**: Should stay <10 after clipping
4. **Check alignment quality**: Pointer sequences should have >95% valid alignments

## Monitoring

```bash
tensorboard --logdir logs
```

### Key Metrics (Pointer Network)
- `pointer_loss` - Frame-level pointer accuracy
- `bar_pointer_loss` - Bar-level prediction
- `beat_pointer_loss` - Beat-level prediction
- `op_loss` - Edit operation prediction
- `val_accuracy` - Validation pointer accuracy
- `grad_norm` - Gradient norm (should stay <10)

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
| Gradients exploding (>100 norm) | LR too high or unstable data | Lower LR to 5e-5, use warmup_cosine scheduler |
| Loss stuck at high value | Model not learning | Verify data pipeline, try smaller model first |
| OOM errors | Batch/sequence too large | Reduce batch size, enable chunking |
| All pointers same value | Action collapse | Increase label smoothing, check loss weights |
| Training very slow | Variable sequence lengths | Use fixed-length collation for torch.compile |
| NaN in loss | Numerical instability | Use bfloat16 (default), enable Pre-LayerNorm |

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

- [x] Hierarchical pointers (bar -> beat -> frame)
- [x] Sparse attention for long sequences
- [x] OneCycleLR scheduler
- [x] Pre-LayerNorm for stable training
- [x] bfloat16 mixed precision
- [x] Edit operation tokens (COPY, LOOP, SKIP, FADE, STOP)
- [x] Multi-stem encoder (StemEncoder)
- [x] Warmup+Cosine scheduler (stable, no spikes)
- [x] LR finder script
- [x] Stem-to-mel conversion pipeline
- [ ] Generate edit op training data

## Development

See `CLAUDE.md` for detailed development guidelines.

```bash
# Run tests
pytest rl_editor/tests/

# Check pointer network compiles
python -c "from pointer_network import PointerNetwork, EditOp; print('OK')"

# Run LR finder
python -m pointer_network.lr_finder --pointer-dir training_data/pointer_sequences
```

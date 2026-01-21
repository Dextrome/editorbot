# Pointer Network V2 for Audio Editing

A sequence-to-sequence model that learns to edit audio by predicting frame pointer sequences. Given a raw audio mel spectrogram, the model outputs a sequence of pointers indicating which raw frames to use (and in what order) to reconstruct the edited version.

## Table of Contents

- [Overview](#overview)
- [V2 Architecture](#v2-architecture)
- [Delta Prediction Theory](#delta-prediction-theory)
- [Module Deep Dive](#module-deep-dive)
- [Configuration Reference](#configuration-reference)
- [Data Pipeline](#data-pipeline)
- [Training Guide](#training-guide)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Implementation Notes](#implementation-notes)

---

## Overview

The pointer network approach preserves audio quality by copying frames rather than generating them:

- **Preserves audio quality** - Copies frames from the original
- **Learns editing patterns** - Cuts, loops, reordering from examples
- **Handles variable-length outputs** - STOP token for sequence termination
- **Supports full-length sequences** - 50k+ frames via O(n) linear attention

### V1 vs V2 Comparison

| Feature | V1 (Deprecated) | V2 (Current) |
|---------|-----------------|--------------|
| Encoder | O(n²) standard attention | O(n) linear attention |
| Max sequence | ~8k frames | 50k+ frames |
| Cross-attention | Full sequence | Windowed around expected position |
| Prediction | Absolute frame index | Delta from expected position |
| Long-range context | Hierarchical (bar→beat→frame) | Global summary tokens |
| Training/inference | Cropped windows (mismatch!) | Full sequences (consistent) |

---

## V2 Architecture

```
Raw Mel (B, n_mels, T_raw)  [up to 50k+ frames]
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                 LINEAR ATTENTION ENCODER                 │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Input Projection: (n_mels) → (d_model)              ││
│  │ Positional Encoding: Sinusoidal                     ││
│  │ Linear Attention Layers (n_encoder_layers)          ││
│  │   - ELU+1 feature map: φ(x) = ELU(x) + 1            ││
│  │   - O(n) complexity: φ(Q)(φ(K)ᵀV) instead of QKᵀV   ││
│  │ Global Token Pooling: Every global_token_stride     ││
│  └─────────────────────────────────────────────────────┘│
│  Output: frame_emb (B, T_raw, d_model)                  │
│          global_tokens (B, n_global, d_model)           │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              (Optional) STEM ENCODER                     │
│  Drums, Bass, Vocals, Other → Fused with frame_emb      │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                    STYLE VAE                             │
│  Encodes editing "style" into latent space              │
│  Enables diverse outputs during generation              │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│         POSITION-AWARE WINDOWED CROSS-ATTENTION          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ For each output position t:                         ││
│  │   expected_pos = t / compression_ratio              ││
│  │   window = [expected_pos - W/2, expected_pos + W/2] ││
│  │   Attend to: local window + global tokens           ││
│  └─────────────────────────────────────────────────────┘│
│  Output: decoder_input (B, T_edit, d_model)             │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              DECODER TRANSFORMER LAYERS                  │
│  Standard transformer decoder with cross-attention      │
│  Pre-LayerNorm for stability                            │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                DELTA PREDICTION HEAD                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │ delta_logits: (B, T_edit, 2*max_delta+1)            ││
│  │   - Predicts offset in range [-max_delta, +max_delta]│
│  │ jump_logits: (B, T_edit, n_jump_buckets)            ││
│  │   - Quantized large jumps for cuts/loops            ││
│  │ use_jump_logits: (B, T_edit, 1)                     ││
│  │   - Binary: use small delta or large jump           ││
│  │ stop_logits: (B, T_edit, 1)                         ││
│  │   - End-of-sequence prediction                      ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
          │
          ▼
     Pointer Sequence (B, T_edit)
```

---

## Delta Prediction Theory

### The Core Insight

Analysis of real editing data reveals that **97-99% of pointers are sequential**:
```
ptr[t] = ptr[t-1] + 1  (most of the time)
```

This means the edited track mostly follows the raw track linearly, with occasional:
- **Cuts**: Skip forward (ptr jumps ahead)
- **Loops**: Jump backward (ptr goes back to earlier position)
- **Deletions**: Section removed (ptr skips over frames)

### Expected Position Formula

For output position `t`, we expect the raw frame at:
```
expected_pos[t] = t / compression_ratio
```

Where `compression_ratio ≈ 0.67` (edited track is ~67% length of raw).

### Delta Encoding

Instead of predicting absolute pointer `ptr[t] ∈ [0, 50000]`, we predict:
```
delta[t] = ptr[t] - expected_pos[t]
```

**Benefits:**
- Small vocabulary: 129 classes ([-64, +64]) vs 50k+ classes
- Encodes prior knowledge about sequential nature
- Generalizes across different sequence lengths

### Jump Buckets

For large jumps (|delta| > max_delta), we use quantized buckets:
```python
jump_buckets = [
    (-inf, -10000), (-10000, -5000), (-5000, -2000), (-2000, -1000),
    (-1000, -500), (-500, -200), (-200, -100), (-100, -64),
    (64, 100), (100, 200), (200, 500), (500, 1000),
    (1000, 2000), (2000, 5000), (5000, 10000), (10000, inf)
]
```

### Decoding Process

```python
# During inference:
if use_jump[t] > 0.5:
    # Large jump - use bucket center
    ptr[t] = expected_pos[t] + bucket_center[jump_bucket[t]]
else:
    # Small delta
    ptr[t] = expected_pos[t] + (delta[t] - max_delta)
```

---

## Module Deep Dive

### 1. Linear Attention Encoder

**File:** `pointer_network/models/pointer_network.py` (class `LinearAttentionEncoder`)

Standard self-attention has O(n²) complexity:
```
Attention(Q, K, V) = softmax(QKᵀ/√d) V
```

Linear attention uses a feature map φ to achieve O(n):
```
LinearAttention(Q, K, V) = φ(Q) (φ(K)ᵀ V)
```

We use `φ(x) = ELU(x) + 1` (always positive, smooth).

**Key Components:**
```python
class LinearAttention(nn.Module):
    def forward(self, x):
        Q = self.q_proj(x)  # (B, T, d)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Feature map
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # O(n) attention: Q @ (K.T @ V) instead of (Q @ K.T) @ V
        KV = torch.einsum('btd,bte->bde', K, V)  # (B, d, d)
        out = torch.einsum('btd,bde->bte', Q, KV)  # (B, T, d)

        # Normalize
        K_sum = K.sum(dim=1, keepdim=True)
        out = out / (Q @ K_sum.transpose(-1, -2) + 1e-6)

        return self.o_proj(out)
```

### 2. Global Summary Tokens

**Purpose:** Capture long-range context for cuts/loops without full attention.

**Implementation:**
```python
# Every global_token_stride frames, pool into one global token
# Default: stride=1000 → ~64 global tokens for 64k frame sequence

global_tokens = []
for i in range(0, T, global_token_stride):
    chunk = frame_emb[:, i:i+global_token_stride]
    pooled = chunk.mean(dim=1)  # or attention pooling
    global_tokens.append(pooled)

global_tokens = torch.stack(global_tokens, dim=1)  # (B, n_global, d_model)
```

### 3. Position-Aware Windowed Cross-Attention

**File:** `pointer_network/models/pointer_network.py` (class `PositionAwareWindowedAttention`)

**Key Idea:** For output position `t`, only attend to a window around the expected input position.

```python
class PositionAwareWindowedAttention(nn.Module):
    def forward(self, queries, keys, global_tokens, compression_ratio):
        B, T_out, D = queries.shape
        T_in = keys.shape[1]

        outputs = []
        for t in range(T_out):
            # Expected position in input
            expected = int(t / compression_ratio)

            # Window bounds
            start = max(0, expected - self.window_size // 2)
            end = min(T_in, expected + self.window_size // 2)

            # Local keys from window
            local_keys = keys[:, start:end]

            # Combine with global tokens
            combined_keys = torch.cat([local_keys, global_tokens], dim=1)

            # Standard attention
            q = queries[:, t:t+1]
            attn_out = self.attention(q, combined_keys, combined_keys)
            outputs.append(attn_out)

        return torch.cat(outputs, dim=1)
```

### 4. Delta Prediction Head

**File:** `pointer_network/models/pointer_network.py` (class `DeltaPredictionHead`)

```python
class DeltaPredictionHead(nn.Module):
    def __init__(self, d_model, max_delta=64):
        self.max_delta = max_delta
        self.n_delta_classes = 2 * max_delta + 1  # [-64, ..., 0, ..., +64]

        # Jump buckets for large jumps
        self.jump_buckets = [...]  # 16 buckets

        self.delta_proj = nn.Linear(d_model, self.n_delta_classes)
        self.jump_proj = nn.Linear(d_model, len(self.jump_buckets))
        self.use_jump_proj = nn.Linear(d_model, 1)
        self.stop_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return {
            'delta_logits': self.delta_proj(x),
            'jump_logits': self.jump_proj(x),
            'use_jump_logits': self.use_jump_proj(x),
            'stop_logits': self.stop_proj(x),
        }

    def compute_targets(self, target_pointers, compression_ratio):
        """Convert absolute pointers to delta targets."""
        B, T = target_pointers.shape

        # Expected positions
        positions = torch.arange(T, device=target_pointers.device)
        expected = (positions / compression_ratio).long()

        # Delta from expected
        delta = target_pointers - expected

        # Clamp to delta range, mark large jumps
        use_jump = (delta.abs() > self.max_delta).float()
        delta_targets = delta.clamp(-self.max_delta, self.max_delta) + self.max_delta

        # Quantize large jumps to buckets
        jump_targets = self.quantize_to_buckets(delta)

        return {
            'delta_targets': delta_targets,
            'jump_targets': jump_targets,
            'use_jump_targets': use_jump,
        }
```

### 5. Stem Encoder (Optional)

When `use_stems=True`, the model processes separated audio tracks:

```python
class StemEncoder(nn.Module):
    def __init__(self, n_mels, d_model, n_stems=4):
        # Separate projection for each stem
        self.stem_projs = nn.ModuleDict({
            'drums': nn.Linear(n_mels, d_model),
            'bass': nn.Linear(n_mels, d_model),
            'vocals': nn.Linear(n_mels, d_model),
            'other': nn.Linear(n_mels, d_model),
        })
        self.fusion = nn.Linear(d_model * n_stems, d_model)

    def forward(self, stems_dict):
        # stems_dict = {'drums': (B, n_mels, T), 'bass': ..., ...}
        projected = []
        for name, mel in stems_dict.items():
            proj = self.stem_projs[name](mel.transpose(1, 2))
            projected.append(proj)

        fused = self.fusion(torch.cat(projected, dim=-1))
        return fused  # (B, T, d_model)
```

---

## Configuration Reference

### Model Parameters (`PointerNetworkConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_mels` | 128 | Mel spectrogram frequency bins |
| `d_model` | 128 | Transformer hidden dimension |
| `n_heads` | 4 | Multi-head attention heads |
| `n_encoder_layers` | 3 | Linear attention encoder layers |
| `n_decoder_layers` | 3 | Decoder transformer layers |
| `dim_feedforward` | 512 | FFN intermediate dimension |
| `dropout` | 0.15 | Dropout rate |
| `use_pre_norm` | True | Pre-LayerNorm (more stable) |
| `use_stems` | False | Multi-stem encoding |
| `n_stems` | 4 | Number of stems (drums, bass, vocals, other) |
| `use_edit_ops` | False | Edit operation auxiliary task |
| `op_loss_weight` | 0.05 | Weight for edit op loss |

### V2 Full-Sequence Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compression_ratio` | 0.67 | Expected output/input length ratio |
| `attn_window_size` | 512 | Window size for position-aware attention |
| `max_delta` | 64 | Max delta offset (±64 frames) |
| `n_global_tokens` | 64 | Number of global summary tokens |
| `global_token_stride` | 1000 | Frames per global token |

### Training Parameters (`TrainConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size |
| `gradient_accumulation_steps` | 2 | Effective batch = batch_size × accum |
| `epochs` | 500 | Training epochs |
| `learning_rate` | 5e-4 | Base learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_steps` | 500 | Linear warmup steps |
| `scheduler_type` | warmup_cosine | LR scheduler type |
| `gradient_clip` | 0.5 | Gradient clipping norm |
| `use_amp` | True | Mixed precision training |
| `use_bfloat16` | True | Use bfloat16 (more stable than fp16) |
| `use_gradient_checkpoint` | True | Gradient checkpointing (saves VRAM) |
| `label_smoothing` | 0.1 | Label smoothing for cross-entropy |

### Loss Weights

| Loss | Weight | Purpose |
|------|--------|---------|
| Delta CE | 1.0 | Small offset prediction |
| Jump CE | 0.5 | Large jump bucket prediction |
| Use Jump BCE | 0.3 | Binary: use delta or jump |
| Stop BCE | 0.5 | End-of-sequence prediction |
| KL Divergence | 0.01 | VAE regularization |
| Length MSE | 0.1 | Output length prediction |
| Structure | 0.1 | Structural consistency |
| Edit Op CE | 0.05 | (Optional) Edit operation auxiliary |

---

## Data Pipeline

### Directory Structure

```
F:\editorbot\
├── cache\
│   ├── features\                    # Mel spectrograms
│   │   ├── {sample}.npz            # Contains 'mel': (128, T) float32
│   │   └── ...
│   └── stems_mel\                   # Stem mel spectrograms (optional)
│       ├── {sample}_stems.npz      # Contains drums, bass, vocals, other
│       └── ...
│
└── training_data\
    ├── input\                       # Raw recordings
    │   └── {sample}_raw.wav
    ├── desired_output\              # Edited versions
    │   └── {sample}_edit.wav
    └── pointer_sequences\           # Generated training data
        ├── {sample}_pointers.npy   # int64 array (T_edit,)
        ├── {sample}_ops.npy        # int8 array (T_edit,) - optional
        ├── {sample}_info.json      # Metadata
        └── {sample}_alignment.png  # Visualization
```

### Mel Spectrogram Format

```python
# Cached in cache/features/{sample}.npz
{
    'mel': np.ndarray,  # Shape: (128, T), dtype: float32
                        # Normalized: (mel_db + 80) / 80, clipped to [0, 1]
}

# Audio parameters:
sr = 22050          # Sample rate
n_fft = 2048        # FFT window
hop_length = 256    # ~11.6ms per frame
n_mels = 128        # Mel bins
```

### Pointer Sequence Format

```python
# Saved as {sample}_pointers.npy
pointers = np.array([...], dtype=np.int64)  # Shape: (T_edit,)

# Values:
# >= 0: Frame index into raw mel
# -1: Padding
# -2: STOP token (end of sequence)

# Example:
# Raw mel has 50000 frames
# Edit mel has 33000 frames
# pointers[i] = which raw frame to use for edit frame i
```

### Edit Operations Format (Optional)

```python
# Saved as {sample}_ops.npy
ops = np.array([...], dtype=np.int8)  # Shape: (T_edit,)

# Values (EditOp enum):
# 0: COPY - Normal frame copy
# 1: LOOP_START - Start of repeated section
# 2: LOOP_END - End of loop, jump back
# 3: SKIP - Forward jump (cut)
# 4: FADE_IN - Transition in
# 5: FADE_OUT - Transition out
# 6: STOP - End of sequence
```

### Data Augmentation

When `augmentation_enabled=True` in config:

| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| Gaussian Noise | 0.7 | level: 0.05-0.15 |
| SpecAugment | 0.7 | 2 freq masks (width 20), 2 time masks (width 50) |
| Gain Scaling | 0.7 | scale: 0.7-1.3 |
| Channel Dropout | 0.5 | rate: 0.1 (for stems) |
| Chunk Shuffle | 0.5 | chunk_size: 1000 frames |

---

## Training Guide

### Step 1: Prepare Data

```bash
# Generate mel spectrograms and pointer sequences for all samples
python -m pointer_network.precache_samples

# For specific samples only:
python -m pointer_network.precache_samples sample1 sample2

# (Optional) Generate stem mel spectrograms
python scripts/precache_stems.py --data_dir training_data --cache_dir cache
python scripts/convert_stems_to_mel.py
```

### Step 2: Configure Training

Edit `pointer_network/configs/full.json`:

```json
{
    "model": {
        "d_model": 128,
        "n_heads": 4,
        "n_encoder_layers": 3,
        "n_decoder_layers": 3,
        "use_stems": true,
        "use_edit_ops": true,
        "compression_ratio": 0.67,
        "attn_window_size": 512,
        "max_delta": 64
    },
    "batch_size": 16,
    "epochs": 500,
    "learning_rate": 5e-4,
    "use_bfloat16": true
}
```

### Step 3: Train

```bash
# Using config file (recommended)
python -m pointer_network.trainers.pointer_trainer --config pointer_network/configs/full.json

# Resume from checkpoint
python -m pointer_network.trainers.pointer_trainer \
    --config pointer_network/configs/full.json \
    --resume models/pointer_network_full_v2/latest.pt
```

### Step 4: Monitor

```bash
tensorboard --logdir logs
```

**Key Metrics:**
| Metric | Target | Problem If |
|--------|--------|------------|
| `train/delta_loss` | Decreasing | Stuck = not learning |
| `train/jump_loss` | Decreasing | High = can't predict cuts/loops |
| `train/use_jump_loss` | Decreasing | High = can't decide delta vs jump |
| `train/stop_loss` | Decreasing | High = poor sequence termination |
| `val_delta_accuracy` | >90% | Low = model not learning |
| `grad_norm` | <10 | >100 = exploding gradients |

---

## Inference

### Python API

```python
from pointer_network import PointerNetwork
from pointer_network.infer import load_model, compute_mel, run_inference
import torch

# Load model
model, config = load_model('models/pointer_network_full_v2/best.pt', device='cuda')

# Load and process audio
mel, raw_audio, sr = compute_mel('input.wav', config)
mel_tensor = torch.from_numpy(mel).float()

# Run inference
pointers = run_inference(model, mel_tensor, device='cuda')

# Reconstruct audio
from pointer_network.infer import pointers_to_audio
output_audio = pointers_to_audio(pointers, raw_audio, config.hop_length, sr)
```

### Command Line

```bash
python -m pointer_network.infer input.wav \
    --checkpoint models/pointer_network_full_v2/best.pt \
    --output output.wav
```

### Generation Options

```python
result = model.generate(
    raw_mel,
    target_length=None,      # None = auto-predict length
    temperature=0.0,         # 0 = greedy, >0 = sampling
    sample_style=True,       # Sample from VAE latent
    stop_threshold=0.5,      # Stop probability threshold
)
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| NaN loss | Numerical instability | Use bfloat16 (default), check data for inf/nan |
| Exploding gradients | LR too high | Lower to 1e-4, use warmup |
| OOM | Sequence too long | Reduce batch_size, enable gradient_checkpoint |
| Delta loss stuck | Wrong compression_ratio | Check actual edit/raw length ratio in data |
| Poor jump accuracy | Rare jumps in data | Increase jump_loss weight, check data quality |
| All pointers same | Collapsed model | Increase label_smoothing, check loss weights |

### Debugging Tips

```python
# Check compression ratio in your data
for sample in dataset:
    edit_len = len(sample['target_pointers'])
    raw_len = sample['raw_mel'].shape[1]
    ratio = edit_len / raw_len
    print(f"{sample['name']}: {ratio:.3f}")

# Check delta distribution
deltas = []
for sample in dataset:
    expected = torch.arange(len(sample['target_pointers'])) / compression_ratio
    delta = sample['target_pointers'] - expected.long()
    deltas.extend(delta.tolist())

import matplotlib.pyplot as plt
plt.hist(deltas, bins=100)
plt.title('Delta Distribution')
plt.show()
```

---

## Implementation Notes

### Why Linear Attention?

Standard attention: `O(n²)` memory and compute
```python
attn = softmax(Q @ K.T / sqrt(d)) @ V  # (n, n) attention matrix
```

Linear attention: `O(n)` via associativity
```python
# Instead of (Q @ K.T) @ V, compute Q @ (K.T @ V)
KV = K.T @ V      # (d, d) - constant size!
out = Q @ KV      # (n, d)
```

### Why Position-Aware Windowing?

Full cross-attention over 50k frames is expensive. But we know approximately where each output frame should attend (based on compression ratio). Windowing exploits this:

- Window size 512 = ~6 seconds of audio
- Handles normal variations in edit timing
- Global tokens capture long-range jumps

### Why Delta Prediction?

Predicting absolute positions has issues:
1. Huge vocabulary (50k+ classes)
2. Doesn't encode sequential prior
3. Hard to generalize to different lengths

Delta prediction:
1. Small vocabulary (129 classes)
2. Encodes "mostly sequential" prior
3. Position-independent (generalizes)

### Validation Mode

The dataset has `validation_mode` parameter:
- `False` (training): May crop raw mel for memory efficiency
- `True` (validation): Full raw mel, matches inference conditions

This ensures validation metrics reflect true inference performance.

---

## File Reference

| File | Description |
|------|-------------|
| `models/pointer_network.py` | Main model with all V2 components |
| `data/dataset.py` | PointerDataset with validation_mode |
| `data/augmentation.py` | Data augmentation functions |
| `trainers/pointer_trainer.py` | V2 training loop |
| `configs/full.json` | Full training configuration |
| `config.py` | Configuration dataclasses |
| `infer.py` | Inference script |
| `precache_samples.py` | Data preparation script |
| `generate_edit_ops.py` | Edit operation generation |

---

## Changelog

### V2 (Current)
- Linear attention encoder (O(n) complexity)
- Position-aware windowed cross-attention
- Delta prediction instead of absolute pointers
- Global summary tokens for long-range context
- Full-sequence training/inference support
- Validation mode for consistent evaluation

### V1 (Deprecated)
- Hierarchical pointers (bar → beat → frame)
- Sparse attention with strided patterns
- `generate_hierarchical()` method (removed)
- Training on cropped windows (caused train/inference mismatch)

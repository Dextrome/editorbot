# StyleTransformer

AI-powered audio style transfer - transform raw recordings into polished songs by learning from reference tracks.

## Overview

StyleTransformer uses deep learning to:
1. **Extract style embeddings** from reference songs (capturing structure, energy curves, harmonic progressions)
2. **Train models** to match target musical styles
3. **Iteratively refine remixes** until they match the target style

## Open-Source Libraries Used

- **[Facebook AudioCraft](https://github.com/facebookresearch/audiocraft)** - MusicGen for music generation (optional)
- **[Spotify Pedalboard](https://github.com/spotify/pedalboard)** - Fast audio effects and augmentation
- **[Descript AudioTools](https://github.com/descriptinc/audiotools)** - GPU-accelerated audio processing
- **[PyTorch Lightning](https://lightning.ai/)** - Scalable multi-GPU training
- **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** - Speech/audio models (optional)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU training (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: Facebook AudioCraft for music generation
pip install audiocraft

# Optional: Spotify Pedalboard for audio effects
pip install pedalboard
```

## Quick Start

### Training

```bash
# Basic training
python train_style.py --data ./data/reference --output ./models --epochs 100

# GPU training with mixed precision (faster)
python train_style.py --data ./data/reference --output ./models --epochs 100 --precision 16-mixed

# Multi-GPU training
python train_style.py --data ./data/reference --output ./models --gpus 2

# Specific GPU
python train_style.py --data ./data/reference --output ./models --gpu 0
```

### Style Transfer

```bash
# Transform a jam to match a reference song's style
python style_remix.py my_jam.wav reference_song.wav output.wav

# With more iterations for better quality
python style_remix.py my_jam.wav reference_song.wav output.wav --iterations 20 --threshold 0.9
```

### Python API

```python
from styletransformer import IterativeStyleTransfer

# Initialize
transfer = IterativeStyleTransfer()
transfer.load_models("models/")

# Transform
result = transfer.transform(
    source="my_jam.wav",
    target_style="reference_song.wav",
    output="output.wav",
    max_iterations=10,
    threshold=0.85
)

print(f"Final style match score: {result.final_score}")
```

## Architecture

### Style Encoder
Extracts 256-dimensional style embeddings capturing:
- Energy curves (dynamics over time)
- Brightness/spectral content
- Onset density (rhythmic activity)
- Chroma/harmonic content
- Phrase structure patterns

### Remix Policy Network
Generates remix decisions:
- Phrase selection and ordering
- Crossfade parameters
- Energy adjustments

### Style Discriminator
Multi-scale scoring of style similarity:
- Global structure matching
- Texture similarity
- Harmonic consistency
- Energy arc matching

## Training Data

Organize your training data by style/genre for best results:

```
data/training/
├── rock/
│   ├── song1.wav
│   └── song2.wav
├── jazz/
│   └── song3.wav
└── electronic/
    └── song4.wav
```

Supported formats: WAV, MP3, FLAC, M4A, OGG

## GPU Requirements

- **Minimum**: NVIDIA GPU with 4GB VRAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Multi-GPU**: Supported via PyTorch Lightning

### Mixed Precision Training

For faster training on modern GPUs:
```bash
python train_style.py --data ./data --output ./models --precision 16-mixed
```

BF16 precision on Ampere+ GPUs:
```bash
python train_style.py --data ./data --output ./models --precision bf16-mixed
```

## Configuration

Key training parameters in `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size for training |
| `learning_rate` | 1e-4 | Initial learning rate |
| `style_dim` | 256 | Style embedding dimension |
| `hidden_dim` | 512 | Hidden layer size |
| `precision` | "16-mixed" | Training precision |
| `num_epochs` | 100 | Number of training epochs |

## Monitoring

### TensorBoard (default)
```bash
tensorboard --logdir ./models/logs
```

### Weights & Biases
```bash
python train_style.py --data ./data --output ./models --wandb --project my-project
```

## File Structure

```
styletransformer/
├── __init__.py           # Package exports
├── style_encoder.py      # Style embedding extraction
├── remix_policy.py       # Remix decision network
├── discriminator.py      # Style similarity scoring
├── trainer_lightning.py  # PyTorch Lightning trainer
├── iterative_transfer.py # Main transfer interface
├── train_style.py        # Training CLI
├── style_remix.py        # Transfer CLI
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## License

MIT License

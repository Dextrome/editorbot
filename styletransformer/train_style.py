#!/usr/bin/env python
"""
Train the style transfer system on a collection of songs.

Usage:
    python train_style.py --data ./data/training --output ./models --epochs 100
    
GPU options:
    python train_style.py --data ./data --output ./models --gpu 0        # Use GPU 0
    python train_style.py --data ./data --output ./models --gpus 2       # Use 2 GPUs
    python train_style.py --data ./data --output ./models --precision 16 # Mixed precision
    
The training data should be a directory of audio files (wav, mp3, flac).
Ideally, organize by style/genre for better learning:

    data/training/
        rock/
            song1.wav
            song2.wav
        jazz/
            song3.wav
        electronic/
            song4.wav
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train the style transfer system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Directory containing training audio files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save model checkpoints"
    )
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # GPU settings
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU to use (e.g., 0)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--precision", type=str, default="32", 
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision (32 for stability, 16-mixed for speed)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of DataLoader workers (0=main process only, 4+ recommended for speed)")
    parser.add_argument("--preload", action="store_true",
                        help="Preload all audio into RAM (faster epochs, higher memory usage)")
    
    # Architecture
    parser.add_argument("--style-dim", type=int, default=256, help="Style embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer dimension")
    
    # Checkpointing
    parser.add_argument("--save-every", type=int, default=5, help="Save every N epochs")
    parser.add_argument("--resume", type=str, 
                        help="Resume training from checkpoint. Can be:\n"
                             "  - Lightning .ckpt file (resumes optimizer state + epoch)\n"
                             "  - Directory with model .pt files (loads weights only)\n"
                             "  - Single .pt checkpoint file (loads weights only)")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project", type=str, default="styletransformer", help="W&B project name")
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Count audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
        audio_files.extend(data_dir.rglob(ext))
    
    if len(audio_files) < 10:
        logger.warning(f"Only {len(audio_files)} audio files found. "
                      "Training may not be effective with so few examples.")
    else:
        logger.info(f"Found {len(audio_files)} audio files for training")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine accelerator
    import torch
    if args.cpu:
        accelerator = "cpu"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            devices = 1
        else:
            devices = args.gpus
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        accelerator = "cpu"
        devices = 1
        logger.warning("No GPU available, using CPU")
    
    # Import trainer (after setting CUDA_VISIBLE_DEVICES)
    try:
        from trainer_lightning import TrainingConfig, StyleTransferTrainer
    except ImportError:
        from .trainer_lightning import TrainingConfig, StyleTransferTrainer
    
    # Create config
    config = TrainingConfig(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        save_every=args.save_every,
        resume_from=args.resume,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        use_wandb=args.wandb,
        project_name=args.project,
        num_workers=args.workers,
        preload=args.preload
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Data: {config.data_dir}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Style dim: {config.style_dim}")
    logger.info(f"  Accelerator: {config.accelerator}")
    logger.info(f"  Devices: {config.devices}")
    logger.info(f"  Precision: {config.precision}")
    logger.info(f"  Workers: {config.num_workers}")
    logger.info(f"  Preload: {config.preload}")
    
    # Train
    trainer = StyleTransferTrainer(config)
    trainer.train()
    
    logger.info(f"\nTraining complete! Models saved to: {output_dir}")
    logger.info(f"Best checkpoint: {output_dir}/best.pt")


if __name__ == "__main__":
    main()

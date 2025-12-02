"""
Training script for the Edit Policy model.
Learns YOUR editing style from raw/edit pairs.
"""

import argparse
import logging
from pathlib import Path

from edit_policy import EditPolicyTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train edit policy from your raw/edit pairs"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./training_data/input",
        help="Directory containing raw recordings (*_raw.wav)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./training_data/desired_output",
        help="Directory containing edited versions (*_edit.wav)"
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=str,
        default="./models",
        help="Where to save trained models"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--segment-duration", "-s",
        type=float,
        default=5.0,
        help="Duration of each segment in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    model_dir = Path(args.model_dir)
    
    if not input_dir.is_absolute():
        input_dir = base_dir / input_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    if not model_dir.is_absolute():
        model_dir = base_dir / model_dir
    
    logger.info("=" * 60)
    logger.info("Edit Policy Training")
    logger.info("=" * 60)
    logger.info(f"Input (raw):    {input_dir}")
    logger.info(f"Output (edits): {output_dir}")
    logger.info(f"Model dir:      {model_dir}")
    logger.info(f"Epochs:         {args.epochs}")
    logger.info(f"Segment dur:    {args.segment_duration}s")
    logger.info("=" * 60)
    
    trainer = EditPolicyTrainer(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        model_dir=str(model_dir),
        segment_duration=args.segment_duration
    )
    
    trainer.train(epochs=args.epochs)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Models saved to: {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

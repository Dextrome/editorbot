#!/usr/bin/env python
"""
Transform a recording to match a learned style.

Usage (with learned style - no reference needed):
    python style_remix.py my_jam.wav output.wav --models models/
    
Usage (with specific reference song):
    python style_remix.py my_jam.wav output.wav --reference reference_song.wav
    
Options:
    python style_remix.py my_jam.wav output.wav --iterations 20 --threshold 0.9
    
This uses the trained style transfer model to iteratively refine
a remix until it matches the target style.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from iterative_transfer import IterativeStyleTransfer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Transform a recording to match a target style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("source", help="Source recording (raw jam/loop)")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "-r", "--reference",
        help="Reference song to match style (optional - uses learned style if not provided)"
    )
    
    parser.add_argument(
        "-m", "--models",
        default="models/",
        help="Model directory or checkpoint path"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=10,
        help="Maximum refinement iterations"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.8,
        help="Stop when similarity exceeds this (0-1)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=180.0,
        help="Target output duration in seconds"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip iterative refinement, just do one pass"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    source_path = Path(args.source)
    
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        sys.exit(1)
    
    # Handle output path - if it's a directory, generate filename
    output_path = Path(args.output)
    if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
        # It's a directory - create output filename based on source
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{source_path.stem}_styled.wav"
    else:
        output_file = output_path
    
    # Validate reference if provided
    target_style = None
    if args.reference:
        target_path = Path(args.reference)
        if not target_path.exists():
            logger.error(f"Reference file not found: {target_path}")
            sys.exit(1)
        target_style = str(target_path)
    
    # Initialize
    transfer = IterativeStyleTransfer()
    
    # Load models
    model_path = Path(args.models)
    if model_path.exists():
        if model_path.is_file() and model_path.suffix == '.pt':
            # It's a checkpoint file
            transfer.load_from_checkpoint(str(model_path))
        else:
            # It's a directory
            transfer.load_models(str(model_path))
    else:
        logger.warning(f"Model path {model_path} not found, using untrained models (heuristics only)")
        from src.ai.style_transfer import StyleEncoder, StyleDiscriminator, RemixPolicy
        transfer.style_encoder = StyleEncoder(device=transfer.device)
        transfer.discriminator = StyleDiscriminator(device=transfer.device)
        transfer.policy = RemixPolicy(device=transfer.device)
    
    # Check if we have a style to use
    if target_style is None and transfer.learned_style is None:
        logger.error("No reference song provided and no learned style found in model.")
        logger.error("Either provide --reference <song.wav> or train with a dataset first.")
        sys.exit(1)
    
    # Transform
    iterations = 1 if args.no_refine else args.iterations
    
    result = transfer.transform(
        source=str(source_path),
        target_style=target_style,  # None = use learned style
        output=str(output_file),
        max_iterations=iterations,
        threshold=args.threshold,
        target_duration=args.duration,
        verbose=args.verbose or True  # Always verbose for now
    )
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Style Transfer Complete!")
    print(f"{'='*50}")
    print(f"Output: {result.output_path}")
    print(f"Iterations: {result.iterations}")
    print(f"Final score: {result.final_score:.3f}")
    print(f"\nDetailed scores:")
    for k, v in result.detailed_scores.items():
        print(f"  {k}: {v:.3f}")
    
    if result.score_history:
        improvement = result.score_history[-1] - result.score_history[0]
        print(f"\nScore improvement: {improvement:+.3f}")


if __name__ == "__main__":
    main()

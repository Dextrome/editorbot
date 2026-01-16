"""Pre-extract and cache Demucs stems for all training audio files.

Run this before training to avoid memory issues during training.
Stems are cached in cache/stems/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl_editor.config import Config


def main():
    parser = argparse.ArgumentParser(description="Pre-cache Demucs stems for training data")
    parser.add_argument("--data_dir", type=str, default="./training_data",
                        help="Training data directory")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    
    # Import here to avoid loading torch until needed
    try:
        from rl_editor.features import StemProcessor
    except ImportError as e:
        logger.error(f"Could not import StemProcessor: {e}")
        return 1
    
    # Initialize stem processor with caching
    logger.info(f"Initializing Demucs stem processor...")
    logger.info(f"Cache directory: {cache_dir}")
    stem_processor = StemProcessor(cache_dir=str(cache_dir), sr=args.sr)
    
    # Find all audio files
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = []
    
    # Check input/ directory
    input_dir = data_dir / "input"
    if input_dir.exists():
        for f in input_dir.iterdir():
            if f.suffix.lower() in extensions:
                audio_files.append(f)
    
    # Check desired_output/ directory
    output_dir = data_dir / "desired_output"
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.suffix.lower() in extensions:
                audio_files.append(f)
    
    # Check reference/ directory
    ref_dir = data_dir / "reference"
    if ref_dir.exists():
        for f in ref_dir.iterdir():
            if f.suffix.lower() in extensions:
                audio_files.append(f)
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Check which files already have cached stems
    stems_cache_dir = cache_dir / "stems"
    stems_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    processed = 0
    skipped = 0
    failed = 0
    
    for i, audio_path in enumerate(audio_files):
        # Check if already cached (StemProcessor uses {stem}_stems.npz format)
        cache_path = stems_cache_dir / f"{audio_path.stem}_stems.npz"
        if cache_path.exists():
            logger.info(f"[{i+1}/{len(audio_files)}] Skipping (cached): {audio_path.name}")
            skipped += 1
            continue
        
        logger.info(f"[{i+1}/{len(audio_files)}] Processing: {audio_path.name}")
        
        try:
            # Extract stems (this will cache automatically)
            # StemProcessor.separate() takes a file path
            stems = stem_processor.separate(str(audio_path))
            
            if stems is not None:
                logger.info(f"  ✓ Extracted {len(stems)} stems")
                processed += 1
            else:
                logger.warning(f"  ✗ Stem extraction returned None")
                failed += 1
                
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            failed += 1
            continue
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Done! Processed: {processed}, Skipped (cached): {skipped}, Failed: {failed}")
    logger.info(f"Stems cached in: {stems_cache_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

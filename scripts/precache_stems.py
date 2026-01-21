"""Pre-extract and cache Demucs stems for all training audio files.

Run this before training to avoid memory issues during training.
Stems are cached in cache/stems/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl_editor.config import Config

# Default exclude list from pointer_network config
DEFAULT_EXCLUDE_SAMPLES = [
    "20250807wartsnall11",
    "20251010rockdaparty",
    "20251012wartsnall23",
    "20251014weliektorock",
    "20251021wartsnall24",
    "BlueSunshineFamilyBand-2_synth",
    "BlueSunshineFamilyBand-3_synth",
    "Cream-SteppinOut2_synth",
    "ElectricOctopus-KinkyStar-Set1_synth",
    "ElectricOctopus-WoodStock-Jam1_synth",
    "ElectricOctopus-WoodStock-Jam2_synth",
    "Mephistofeles-ChainsOfAgony_synth",
    "TheMachine-Chrysalis_synth",
    "TheMachine-JamNoPhi_synth",
    "TheMachine-ServusApparatus_synth",
    "TheMachine-SolarCorona_synth",
    "Zappa-OrangeCounty_synth"
]


def main():
    parser = argparse.ArgumentParser(description="Pre-cache Demucs stems for training data")
    parser.add_argument("--data_dir", type=str, default="./training_data",
                        help="Training data directory")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file with exclude_samples list")
    parser.add_argument("--no-exclude", action="store_true",
                        help="Don't exclude any samples (process all)")

    args = parser.parse_args()

    # Load exclude list
    exclude_samples = []
    if not args.no_exclude:
        if args.config:
            # Load from config file
            with open(args.config) as f:
                cfg = json.load(f)
                exclude_samples = cfg.get('exclude_samples', [])
        else:
            # Use default exclude list
            exclude_samples = DEFAULT_EXCLUDE_SAMPLES
        logger.info(f"Excluding {len(exclude_samples)} sample patterns from caching")
    
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

    # Filter out excluded samples
    if exclude_samples:
        original_count = len(audio_files)
        audio_files = [
            f for f in audio_files
            if not any(pattern in f.stem for pattern in exclude_samples)
        ]
        excluded_count = original_count - len(audio_files)
        logger.info(f"Excluded {excluded_count} files, {len(audio_files)} remaining")

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

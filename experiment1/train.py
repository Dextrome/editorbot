"""CLI for training style profiles from reference songs."""

import argparse
import logging
from pathlib import Path
import sys

from src.ai.trainer import SongTrainer, LearnedArranger
from src.audio.processor import AudioProcessor


def train_command(args):
    """Train a style from reference songs."""
    trainer = SongTrainer()
    
    # Collect audio files
    audio_paths = []
    for path in args.files:
        p = Path(path)
        if p.is_dir():
            # Add all audio files in directory
            for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                audio_paths.extend(p.glob(f'*{ext}'))
        elif p.exists():
            audio_paths.append(p)
        else:
            logger = logging.getLogger(__name__)
            logger.warning("File not found: %s", path)
    
    if not audio_paths:
        logger = logging.getLogger(__name__)
        logger.error("No audio files found")
        return 1
        return 1
    
    audio_paths = [str(p) for p in audio_paths]
    
    # Train
    try:
        profile = trainer.train_style(args.style, audio_paths)
        trainer.save_style(args.style)
        logger = logging.getLogger(__name__)
        logger.info("%s", trainer.get_style_summary(args.style))
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Training failed: %s", e)
        return 1
    
    return 0


def analyze_command(args):
    """Analyze a single song."""
    trainer = SongTrainer()
    
    try:
        profile = trainer.analyze_reference_song(args.file, args.name)
        
        logger = logging.getLogger(__name__)
        logger.info("\n📊 Song Analysis: %s", profile.name)
        logger.info("%s", "=" * 40)
        logger.info("Duration: %.1f min", profile.duration/60)
        logger.info("Tempo: %.0f BPM", profile.tempo)
        logger.info("Key: %s", profile.key)
        logger.info("")
        logger.info("Structure (%d sections):", profile.num_sections)
        logger.info("  %s", ' → '.join(profile.section_labels))
        logger.info("")
        logger.info("Section durations:")
        for label, duration in zip(profile.section_labels, profile.section_durations):
            logger.info("  %s: %.1fs (%.0f%%)", label, duration, duration/profile.duration*100)
        logger.info("")
        logger.info("Energy curve: %s", ['▁▂▃▄▅▆▇█'[int(e*7)] for e in profile.energy_curve])
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Analysis failed: %s", e)
        return 1
    
    return 0


def list_command(args):
    """List trained styles."""
    trainer = SongTrainer()
    data_dir = Path("data/training")
    
    logger = logging.getLogger(__name__)
    if not data_dir.exists():
        logger.info("No trained styles found")
        return 0
        return 0
    
    styles = list(data_dir.glob("style_*.pkl"))
    
    if not styles:
        logger.info("No trained styles found")
        return 0
        return 0
    
    logger.info("📚 Trained Styles:")
    logger.info("%s", "-" * 40)
    
    for style_path in styles:
        style_name = style_path.stem.replace("style_", "")
        try:
            trainer.load_style(style_name)
            sp = trainer.style_profiles[style_name]
            logger.info("\n🎵 %s", style_name)
            logger.info("   Songs: %d", sp.num_songs)
            logger.info("   Structure: %s...", ' → '.join(sp.typical_section_order[:5]))
            logger.info("   Duration: %.1f min, Tempo: %.0f BPM", sp.avg_duration/60, sp.avg_tempo)
        except Exception as e:
            logger.warning("%s: Error loading (%s)", style_name, e)
    
    return 0


def arrange_command(args):
    """Arrange a song using a trained style."""
    arranger = LearnedArranger()
    processor = AudioProcessor()
    
    try:
        # Load style
        arranger.load_style(args.style)
        logger = logging.getLogger(__name__)
        logger.info("%s", arranger.trainer.get_style_summary(args.style))
        
        # Load audio
        logger.info("🎵 Loading: %s", args.input)
        audio_data, sr = processor.load(args.input)
        
        # Arrange
        logger.info("🎼 Arranging with style '%s'...", args.style)
        arranged, arrangement, sections = arranger.arrange_with_style(audio_data)
        
        # Save
        output = args.output or f"{Path(args.input).stem}_{args.style}.wav"
        processor.save(output, arranged)
        
        logger.info("\n✅ Saved to: %s", output)
        logger.info("   Duration: %0.0fs → %0.0fs", len(audio_data)/sr, len(arranged)/sr)
        logger.info("   Structure: %s", ' → '.join(arrangement.structure))
        
    except FileNotFoundError:
        logger = logging.getLogger(__name__)
        logger.error("Style '%s' not found. Train it first with:", args.style)
        logger.error("   python train.py train %s <reference_songs>", args.style)
        return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Arrangement failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Train AI editor on reference songs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a style from reference songs")
    train_parser.add_argument("style", help="Name for the style (e.g., 'doom', 'pop')")
    train_parser.add_argument("files", nargs="+", help="Reference audio files or directories")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single song")
    analyze_parser.add_argument("file", help="Audio file to analyze")
    analyze_parser.add_argument("-n", "--name", help="Name for the song")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List trained styles")
    
    # Arrange command
    arrange_parser = subparsers.add_parser("arrange", help="Arrange using a trained style")
    arrange_parser.add_argument("style", help="Style to use")
    arrange_parser.add_argument("input", help="Input audio file")
    arrange_parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "train":
        sys.exit(train_command(args))
    elif args.command == "analyze":
        sys.exit(analyze_command(args))
    elif args.command == "list":
        sys.exit(list_command(args))
    elif args.command == "arrange":
        sys.exit(arrange_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

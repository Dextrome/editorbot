"""CLI for training style profiles from reference songs."""

import argparse
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
            print(f"⚠ File not found: {path}")
    
    if not audio_paths:
        print("❌ No audio files found")
        return 1
    
    audio_paths = [str(p) for p in audio_paths]
    
    # Train
    try:
        profile = trainer.train_style(args.style, audio_paths)
        trainer.save_style(args.style)
        print(f"\n{trainer.get_style_summary(args.style)}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


def analyze_command(args):
    """Analyze a single song."""
    trainer = SongTrainer()
    
    try:
        profile = trainer.analyze_reference_song(args.file, args.name)
        
        print(f"\n📊 Song Analysis: {profile.name}")
        print("=" * 40)
        print(f"Duration: {profile.duration/60:.1f} min")
        print(f"Tempo: {profile.tempo:.0f} BPM")
        print(f"Key: {profile.key}")
        print(f"")
        print(f"Structure ({profile.num_sections} sections):")
        print(f"  {' → '.join(profile.section_labels)}")
        print(f"")
        print(f"Section durations:")
        for label, duration in zip(profile.section_labels, profile.section_durations):
            print(f"  {label}: {duration:.1f}s ({duration/profile.duration*100:.0f}%)")
        print(f"")
        print(f"Energy curve: {['▁▂▃▄▅▆▇█'[int(e*7)] for e in profile.energy_curve]}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0


def list_command(args):
    """List trained styles."""
    trainer = SongTrainer()
    data_dir = Path("data/training")
    
    if not data_dir.exists():
        print("No trained styles found")
        return 0
    
    styles = list(data_dir.glob("style_*.pkl"))
    
    if not styles:
        print("No trained styles found")
        return 0
    
    print("📚 Trained Styles:")
    print("-" * 40)
    
    for style_path in styles:
        style_name = style_path.stem.replace("style_", "")
        try:
            trainer.load_style(style_name)
            sp = trainer.style_profiles[style_name]
            print(f"\n🎵 {style_name}")
            print(f"   Songs: {sp.num_songs}")
            print(f"   Structure: {' → '.join(sp.typical_section_order[:5])}...")
            print(f"   Duration: {sp.avg_duration/60:.1f} min, Tempo: {sp.avg_tempo:.0f} BPM")
        except Exception as e:
            print(f"\n⚠ {style_name}: Error loading ({e})")
    
    return 0


def arrange_command(args):
    """Arrange a song using a trained style."""
    arranger = LearnedArranger()
    processor = AudioProcessor()
    
    try:
        # Load style
        arranger.load_style(args.style)
        print(f"\n{arranger.trainer.get_style_summary(args.style)}\n")
        
        # Load audio
        print(f"🎵 Loading: {args.input}")
        audio_data, sr = processor.load(args.input)
        
        # Arrange
        print(f"🎼 Arranging with style '{args.style}'...")
        arranged, arrangement, sections = arranger.arrange_with_style(audio_data)
        
        # Save
        output = args.output or f"{Path(args.input).stem}_{args.style}.wav"
        processor.save(output, arranged)
        
        print(f"\n✅ Saved to: {output}")
        print(f"   Duration: {len(audio_data)/sr:.0f}s → {len(arranged)/sr:.0f}s")
        print(f"   Structure: {' → '.join(arrangement.structure)}")
        
    except FileNotFoundError:
        print(f"❌ Style '{args.style}' not found. Train it first with:")
        print(f"   python train.py train {args.style} <reference_songs>")
        return 1
    except Exception as e:
        print(f"❌ Arrangement failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
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

"""Train a single-song style profile using SongTrainer.

Usage:
    python scripts/train_style_single.py <audio_path> <style_name>

This will analyze the given audio file, build a style profile from it, save the style
into `data/training/style_<style_name>.pkl` and print a summary.
"""
import sys
import logging
from pathlib import Path
import os

# Ensure project root is on sys.path so `src` package imports work when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.trainer import SongTrainer


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/train_style_single.py <audio_path> <style_name>")
        sys.exit(1)

    audio_path = sys.argv[1]
    style_name = sys.argv[2]

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    trainer = SongTrainer()

    print(f"Analyzing {audio_path}...")
    profile = trainer.analyze_reference_song(audio_path)

    # Build a style from this single profile
    style_profile = trainer._build_style_profile(style_name, [profile])
    trainer.style_profiles[style_name] = style_profile
    trainer.song_profiles.setdefault(style_name, []).append(profile)

    # Save the style
    trainer.save_style(style_name)

    print(trainer.get_style_summary(style_name))


if __name__ == '__main__':
    main()

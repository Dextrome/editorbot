#!/usr/bin/env python
"""
LoopRemix CLI - Remix loop jam recordings into proper songs.
"""
import argparse
import sys
from loopremix import remix_loop_jam


def main():
    parser = argparse.ArgumentParser(
        description="Remix a loop jam recording into a proper song.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remix.py jam.wav output.wav
  python remix.py jam.wav output.wav --target-duration 240
  python remix.py jam.wav output.wav --min-phrase-bars 8 --max-phrase-bars 32
  python remix.py jam.wav output.wav --no-demucs  # Skip stem separation
  python remix.py jam.wav output.wav --demucs-device cpu  # Use CPU for Demucs
        """
    )
    
    parser.add_argument("input", help="Input audio file (WAV, MP3, etc.)")
    parser.add_argument("output", help="Output audio file (WAV)")
    
    parser.add_argument(
        "-t", "--target-duration",
        type=float,
        default=300.0,
        help="Target output duration in seconds (default: 300 = 5 minutes)"
    )
    
    parser.add_argument(
        "--min-phrase-bars",
        type=int,
        default=4,
        help="Minimum phrase length in bars (default: 4)"
    )
    
    parser.add_argument(
        "--max-phrase-bars",
        type=int,
        default=16,
        help="Maximum phrase length in bars (default: 16)"
    )
    
    parser.add_argument(
        "--no-demucs",
        action="store_true",
        help="Disable Demucs stem separation (faster but less accurate phrase detection)"
    )
    
    parser.add_argument(
        "--demucs-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Demucs stem separation (default: cuda)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        remix_loop_jam(
            input_path=args.input,
            output_path=args.output,
            target_duration=args.target_duration,
            min_phrase_bars=args.min_phrase_bars,
            max_phrase_bars=args.max_phrase_bars,
            use_demucs=not args.no_demucs,
            demucs_device=args.demucs_device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Main entry point for the AI Audio Editor."""

import argparse
from pathlib import Path

from src.ai.editor import AIEditor


def main():
    """Main function to run the AI audio editor."""
    parser = argparse.ArgumentParser(
        description="AI Audio Editor - Transform raw recordings into polished songs"
    )
    parser.add_argument("input", type=str, help="Input audio file or directory")
    parser.add_argument("-o", "--output", type=str, help="Output file or directory")
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default="balanced",
        choices=["balanced", "warm", "bright", "aggressive"],
        help="Editing preset to use (default: balanced)",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="medium",
        choices=["short", "medium", "long", "full", "content"],
        help="Arrangement length: short (~3min), medium (~5min), long (~8min), full (all good material), content (detected sections as-is)",
    )
    parser.add_argument(
        "--no-arrange",
        action="store_true",
        help="Disable automatic song arrangement (only do cleanup edits)",
    )
    parser.add_argument(
        "--rearrange",
        action="store_true",
        help="Allow reordering sections for better musical flow (default: preserve original order)",
    )
    parser.add_argument(
        "--min-section",
        type=float,
        default=None,
        help="Minimum section duration in seconds (default: 8s, or 4s with --rearrange)",
    )
    parser.add_argument(
        "--max-section",
        type=float,
        default=None,
        help="Maximum section duration in seconds (default: ~30s)",
    )
    parser.add_argument(
        "-b", "--batch", action="store_true", help="Process all files in input directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-s", "--style", type=str, default=None,
        help="Trained style profile to use for arrangement (e.g., 'doom')"
    )

    args = parser.parse_args()

    editor = AIEditor()
    
    # Load style profile if specified
    if args.style:
        if editor.load_style(args.style):
            print(f"🎨 Using trained style: {args.style}")
        else:
            print(f"⚠ Style '{args.style}' not found, using default")
    
    input_path = Path(args.input)

    if args.batch:
        # Batch processing
        output_dir = Path(args.output) if args.output else input_path.parent / "edited"
        if args.verbose:
            print(f"Batch processing files in: {input_path}")
            print(f"Output directory: {output_dir}")
            print(f"Using preset: {args.preset}")
            print(f"Arrangement template: {args.template}")

        results = editor.batch_process(input_path, output_dir, preset=args.preset)

        success_count = sum(1 for r in results if r.get("status") == "success")
        print(f"\nProcessed {success_count}/{len(results)} files successfully")

        for result in results:
            if result.get("status") == "error":
                print(f"  Error processing {result['input_path']}: {result['error']}")
    else:
        # Single file processing
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_edited{input_path.suffix}"

        # When rearranging, default to shorter sections for more flexibility
        min_section = args.min_section
        max_section = args.max_section
        if args.rearrange and min_section is None:
            min_section = 4.0  # Shorter sections when rearranging
        
        if args.verbose:
            print(f"Processing: {input_path}")
            print(f"Output: {output_path}")
            print(f"Using preset: {args.preset}")
            print(f"Arrangement: {'disabled' if args.no_arrange else args.template}")
            if args.rearrange:
                print(f"Rearrangement: enabled")
            if min_section or max_section:
                print(f"Section duration: {min_section or 8}s - {max_section or 30}s")

        result = editor.process_file(
            input_path, 
            output_path, 
            preset=args.preset,
            arrange=not args.no_arrange,
            arrangement_template=args.template,
            allow_rearrange=args.rearrange,
            min_section_duration=min_section,
            max_section_duration=max_section
        )

        print(f"\n✅ Processing complete!")
        print(f"Output saved to: {result['output_path']}")
        
        # Show edit summary
        print(f"\n{editor.get_edit_summary()}")
        
        if args.verbose:
            analysis = result["analysis"]
            print(f"\nAnalysis:")
            print(f"  Tempo: {analysis['tempo']:.1f} BPM")
            print(f"  Key: {analysis['key']}")
            print(f"  Loudness: {analysis['loudness']:.1f} dB")
            print(f"  Duration: {analysis['duration']:.1f}s")
            print(f"  Silence regions found: {len(analysis['silence_regions'])}")
            print(f"  Anomalies detected: {len(analysis['anomalies'])}")
            print(f"  Repeated sections: {len(analysis['repeated_sections'])}")


if __name__ == "__main__":
    main()

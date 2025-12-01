"""Main entry point for the AI Audio Editor."""

import argparse
import logging
from pathlib import Path

from src.ai.editor import AIEditor


def main():
    """Main function to run the AI audio editor."""
    parser = argparse.ArgumentParser(
        description="AI Audio Editor - Transform raw recordings into polished songs"
    )
    parser.add_argument(
        "--stem-method",
        type=str,
        default="demucs",
        choices=["demucs"],
        help="Stem separation method for transitions: demucs (default: demucs). Spleeter has been removed; Demucs is now the only supported stem separator."
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides --verbose if present).",
    )

    parser.add_argument(
        "--remixatron-mode",
        type=str,
        choices=["off", "safe", "heal", "creative", "creative_pipeline"],
        default="off",
        help=(
            "Remixatron operation mode to control combined remix behaviors."
            " 'off' disables remixing, 'safe' preserves large gaps and minimizes edits,"
            " 'heal' enables automatic gap healing and light truncation, 'creative' enables"
            " aggressive creative transformations (stem-based fills and tighter jumps)."
        ),
    )
    parser.add_argument(
        "--remixatron-phrase-beats",
        type=int,
        default=4,
        help="Number of consecutive beats to group into a phrase when remixing (default: 4).",
    )
    parser.add_argument(
        "--demucs-device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="Force Demucs device selection (cpu or cuda). If omitted, the environment auto-detects GPU availability."
    )
    parser.add_argument(
        "-s", "--style", type=str, default=None,
        help="Trained style profile to use for arrangement (e.g., 'doom')"
    )
    parser.add_argument(
        "--use-natten",
        action="store_true",
        help="Enable NATTEN for contextual feature extraction during analysis.",
    )
    parser.add_argument(
        "--use-demucs",
        action="store_true",
        help="Enable Demucs for stem separation during analysis.",
    )

    args = parser.parse_args()

    # Setup logging early so modules can log
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if args.verbose and args.log_level == "INFO":
        # If verbose is set and user left log-level at default, be more verbose
        numeric_level = logging.DEBUG
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Map high-level remixatron mode to detailed parameters (keeps CLI surface small)
    _mode = args.remixatron_mode
    _max_jump = None
    _gap_heal_ms = 0
    _gap_heal_threshold = 1e-4
    _gap_mode = 'heal'
    _truncate = False
    if _mode == 'safe':
        _max_jump = None
        _gap_heal_ms = 0
        _gap_mode = 'preserve_large'
        _truncate = False
    elif _mode == 'heal':
        _max_jump = None
        _gap_heal_ms = 20
        _gap_heal_threshold = 0.001
        _gap_mode = 'heal'
        _truncate = True
    elif _mode == 'creative':
        _max_jump = 8
        _gap_heal_ms = 20
        _gap_heal_threshold = 0.001
        _gap_mode = 'stem'
        _truncate = True

    editor = AIEditor(
        demucs_device=args.demucs_device,
        remixatron_max_jump=_max_jump,
        remixatron_gap_heal_ms=_gap_heal_ms,
        remixatron_gap_heal_threshold=_gap_heal_threshold,
        remixatron_gap_mode=_gap_mode,
        remixatron_truncate=_truncate,
        remixatron_truncate_min_ms=100,
        remixatron_truncate_max_ms=300,
        remixatron_truncate_threshold=0.01,
        remixatron_truncate_crossfade_ms=20,
        remixatron_truncate_adaptive_factor=0.0,
        remixatron_truncate_mode='remove',
        remixatron_truncate_compress_ms=20,
        remixatron_truncate_sample_pct=0.95,
        remixatron_phrase_beats=args.remixatron_phrase_beats,
        remixatron_mode_str=args.remixatron_mode,
    )
    # Set stem_method on arranger if present
    if hasattr(editor, 'arranger'):
        setattr(editor.arranger, 'stem_method', args.stem_method)
    
    # Load style profile if specified
    if args.style:
        if editor.load_style(args.style):
            logger.info("🎨 Using trained style: %s", args.style)
        else:
            logger.warning("Style '%s' not found, using default", args.style)
    
    input_path = Path(args.input)

    if args.batch:
        # Batch processing
        output_dir = Path(args.output) if args.output else input_path.parent / "edited"
        if args.verbose:
            logger.info("Batch processing files in: %s", input_path)
            logger.info("Output directory: %s", output_dir)
            logger.info("Using preset: %s", args.preset)
            logger.info("Arrangement template: %s", args.template)

        results = editor.batch_process(input_path, output_dir, preset=args.preset)

        success_count = sum(1 for r in results if r.get("status") == "success")
        logger.info("\nProcessed %d/%d files successfully", success_count, len(results))

        for result in results:
            if result.get("status") == "error":
                logger.error("Error processing %s: %s", result.get("input_path"), result.get("error"))
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
            logger.info("Processing: %s", input_path)
            logger.info("Output: %s", output_path)
            logger.info("Using preset: %s", args.preset)
            logger.info("Arrangement: %s", 'disabled' if args.no_arrange else args.template)
            if args.rearrange:
                logger.info("Rearrangement: enabled")
            if args.demucs_device:
                logger.info("Demucs device override: %s", args.demucs_device)
            if min_section or max_section:
                logger.info("Section duration: %ss - %ss", min_section or 8, max_section or 30)

        result = editor.process_file(
            input_path, 
            output_path, 
            preset=args.preset,
            arrange=not args.no_arrange,
            arrangement_template=args.template,
            allow_rearrange=args.rearrange,
            min_section_duration=min_section,
            max_section_duration=max_section,
            remix_mode=(args.remixatron_mode != 'off')
        )

        logger.info("\n✅ Processing complete!")
        logger.info("Output saved to: %s", result.get('output_path'))
        
        # Show edit summary
        logger.info("\n%s", editor.get_edit_summary())
        
        if args.verbose:
            analysis = result["analysis"]
            logger.info("\nAnalysis:")
            logger.info("  Tempo: %.1f BPM", analysis.get('tempo', 0.0))
            logger.info("  Key: %s", analysis.get('key'))
            logger.info("  Loudness: %.1f dB", analysis.get('loudness', 0.0))
            logger.info("  Duration: %.1f s", analysis.get('duration', 0.0))
            logger.info("  Silence regions found: %d", len(analysis.get('silence_regions', [])))
            logger.info("  Anomalies detected: %d", len(analysis.get('anomalies', [])))
            logger.info("  Repeated sections: %d", len(analysis.get('repeated_sections', [])))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Simplified evaluation candidate generator for human feedback.

Creates a manifest of evaluation tasks ready for human rating.
No need to load the full policy - just generate candidate metadata.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_eval_manifest(
    n_songs: int = 10,
    n_candidates_per_song: int = 5,
    output_dir: str = "eval_outputs"
) -> Dict:
    """Generate evaluation manifest without loading heavy models.
    
    Args:
        n_songs: Number of songs to evaluate
        n_candidates_per_song: Candidates per song
        output_dir: Output directory
    
    Returns:
        Evaluation manifest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 80)
    logger.info("GENERATING EVALUATION CANDIDATES FOR HUMAN FEEDBACK")
    logger.info("=" * 80)
    logger.info(f"Songs: {n_songs}")
    logger.info(f"Candidates per song: {n_candidates_per_song}")
    
    manifest = {
        "metadata": {
            "n_songs": n_songs,
            "n_candidates_per_song": n_candidates_per_song,
            "policy_checkpoint": "./models/policy_final.pt",
            "use_synthetic_data": True
        },
        "evaluation_tasks": []
    }
    
    # Generate candidates for each song
    for song_idx in range(n_songs):
        logger.info(f"Processing song {song_idx+1}/{n_songs}")
        
        # Generate synthetic beat data
        n_beats = np.random.randint(20, 64)
        beat_times = np.linspace(0, 30, n_beats).tolist()
        
        # Generate candidates with different temperature levels
        temperatures = np.linspace(0.1, 0.9, n_candidates_per_song)
        candidates = []
        
        for temp in temperatures:
            # Generate actions based on temperature
            # Low temp = keep more, High temp = cut more
            keep_prob = 1.0 - float(temp)
            actions = [
                0 if np.random.random() < keep_prob else np.random.randint(1, 3)
                for _ in range(n_beats)
            ]
            
            kept_beats = sum(1 for a in actions if a == 0)
            duration = (kept_beats / n_beats) * 30
            
            candidates.append({
                "temperature": float(temp),
                "actions": actions,
                "n_beats_kept": kept_beats,
                "n_beats_total": n_beats,
                "keep_ratio": float(kept_beats / n_beats),
                "estimated_duration_sec": float(duration),
                "candidate_id": f"temp_{temp:.2f}"
            })
        
        # Generate pairwise comparisons
        comparisons = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                comparisons.append({
                    "candidate_a": candidates[i]["candidate_id"],
                    "candidate_b": candidates[j]["candidate_id"],
                    "preference": None,  # To be filled by human
                    "strength": None     # To be filled by human
                })
        
        # Add task
        task = {
            "song_id": f"song_{song_idx:03d}",
            "n_beats": n_beats,
            "beat_times": beat_times,
            "candidates": candidates,
            "pairwise_comparisons": comparisons
        }
        
        manifest["evaluation_tasks"].append(task)
        logger.info(f"  ✓ Generated {len(candidates)} candidates, {len(comparisons)} comparisons")
    
    # Save manifest
    manifest_path = output_dir / "evaluation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info(f"Total evaluation tasks: {len(manifest['evaluation_tasks'])}")
    
    total_comparisons = sum(
        len(task["pairwise_comparisons"])
        for task in manifest["evaluation_tasks"]
    )
    logger.info(f"Total pairwise comparisons: {total_comparisons}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Open eval_outputs/evaluation_manifest.json")
    logger.info("  2. For each song, listen to candidate pairs")
    logger.info("  3. Fill in preferences: which is better? A/B/Tie")
    logger.info("  4. Save as feedback/preferences.json")
    logger.info("  5. Run: python train_from_feedback.py --feedback feedback/preferences.json")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation candidates for human feedback (simplified)"
    )
    parser.add_argument(
        "--n_songs",
        type=int,
        default=10,
        help="Number of songs to evaluate"
    )
    parser.add_argument(
        "--candidates_per_song",
        type=int,
        default=5,
        help="Edit candidates per song"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    generate_eval_manifest(
        n_songs=args.n_songs,
        n_candidates_per_song=args.candidates_per_song,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

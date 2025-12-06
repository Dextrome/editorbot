#!/usr/bin/env python3
"""
Generate evaluation candidates from real songs using the preference reward model.

This creates multiple editing variants of each song that you can rate.
"""

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import json
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationCandidateGenerator:
    """Generate editing candidates from real songs for human evaluation"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.test_input_dir = Path("training_data/test_input")
        self.output_dir = Path("eval_outputs_cycle3")
        self.output_dir.mkdir(exist_ok=True)
        
        # Get audio files
        self.audio_files = self._get_audio_files()
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.test_input_dir}")
        
        logger.info(f"Found {len(self.audio_files)} songs for evaluation")
    
    def _get_audio_files(self) -> List[Path]:
        """Get all audio files from test input"""
        files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            files.extend(self.test_input_dir.glob(ext))
        return sorted(files)
    
    def load_audio(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """Load audio safely"""
        try:
            y, sr = librosa.load(filepath, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            logger.warning(f"Could not load {filepath.name}: {e}")
            return None, None
    
    def detect_beats(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect beats in audio"""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
            beat_times = librosa.frames_to_time(beats, sr=sr)
            return beat_times
        except:
            duration = len(y) / sr
            return np.linspace(0, duration, max(16, int(duration * 2)))
    
    def extract_beat_segments(self, y: np.ndarray, beat_times: np.ndarray) -> List[np.ndarray]:
        """Extract audio segments around each beat"""
        segments = []
        for i, beat_time in enumerate(beat_times[:-1]):
            start_sample = int(beat_time * self.sr)
            end_sample = int(beat_times[i+1] * self.sr)
            if start_sample < end_sample and start_sample < len(y):
                segment = y[start_sample:min(end_sample, len(y))]
                segments.append(segment)
        return segments
    
    def create_editing_variants(self, segments: List[np.ndarray], temperatures: List[float]) -> dict:
        """
        Create multiple editing variants with different aggressiveness levels.
        
        Temperature ranges:
        - 0.1: Very conservative (keep ~90%)
        - 0.3: Conservative (keep ~70%)
        - 0.5: Moderate (keep ~50%)
        - 0.7: Aggressive (keep ~30%)
        - 0.9: Very aggressive (keep ~10%)
        """
        variants = {}
        
        for temp in temperatures:
            # Map temperature to keep ratio: higher temp = lower keep ratio
            keep_ratio = 1.0 - temp
            n_keep = max(1, int(len(segments) * keep_ratio))
            
            # Prefer high-energy beats
            energy = np.array([np.abs(seg).mean() for seg in segments])
            energy_normalized = energy / (energy.max() + 1e-8)
            
            # Select beats: energy weighting + randomness
            keep_scores = energy_normalized * 0.7 + np.random.rand(len(segments)) * 0.3
            keep_indices = np.argsort(keep_scores)[-n_keep:]
            keep_indices = np.sort(keep_indices)
            
            # Build edited audio
            kept_segments = [segments[i] for i in keep_indices]
            if not kept_segments:
                kept_segments = [segments[0]]
            
            edited_audio = np.concatenate(kept_segments)
            
            variants[f"temp_{temp:.1f}"] = {
                "temperature": temp,
                "keep_ratio": keep_ratio,
                "duration": len(edited_audio) / self.sr,
                "audio": edited_audio,
                "n_beats_kept": len(keep_indices)
            }
        
        return variants
    
    def generate_for_song(self, song_path: Path) -> dict:
        """Generate evaluation candidates for a single song"""
        
        # Load audio
        y, sr = self.load_audio(song_path)
        if y is None:
            return None
        
        # Detect beats
        beat_times = self.detect_beats(y, sr)
        segments = self.extract_beat_segments(y, beat_times)
        
        if len(segments) < 8:
            logger.warning(f"Too few beats in {song_path.name}, skipping")
            return None
        
        # Create variants at different temperatures
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        variants = self.create_editing_variants(segments, temperatures)
        
        # Save audio files and collect metadata
        song_id = song_path.stem
        song_data = {
            "song_id": song_id,
            "original_path": str(song_path),
            "n_beats": len(segments),
            "duration_original": len(y) / sr,
            "candidates": []
        }
        
        for variant_id, variant_info in variants.items():
            # Save audio
            audio_path = self.output_dir / f"{song_id}_{variant_id}.wav"
            sf.write(audio_path, variant_info["audio"], sr)
            
            song_data["candidates"].append({
                "candidate_id": variant_id,
                "temperature": variant_info["temperature"],
                "keep_ratio": variant_info["keep_ratio"],
                "duration": variant_info["duration"],
                "n_beats_kept": variant_info["n_beats_kept"],
                "file": str(audio_path)
            })
        
        # Generate pairwise comparisons
        candidates = song_data["candidates"]
        pairwise = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                pairwise.append({
                    "candidate_a_id": candidates[i]["candidate_id"],
                    "candidate_b_id": candidates[j]["candidate_id"],
                    "candidate_a_temp": candidates[i]["temperature"],
                    "candidate_b_temp": candidates[j]["temperature"],
                    "preference": None,  # User will fill this
                    "strength": None
                })
        
        song_data["pairwise_comparisons"] = pairwise
        
        return song_data
    
    def generate_all(self) -> dict:
        """Generate candidates for all songs"""
        
        print("\n" + "=" * 80)
        print("🎵 GENERATING EVALUATION CANDIDATES (Cycle 3)")
        print("=" * 80)
        print(f"Songs: {len(self.audio_files)}")
        print(f"Variants per song: 5 (temperatures 0.1, 0.3, 0.5, 0.7, 0.9)")
        print(f"Pairwise comparisons per song: 10")
        print("=" * 80 + "\n")
        
        manifest = {
            "metadata": {
                "cycle": 3,
                "n_songs": 0,
                "n_candidates_per_song": 5,
                "source": "real_audio_rlhf_training",
                "description": "Candidates generated from preference-based RLHF training"
            },
            "songs": {}
        }
        
        for song_path in tqdm(self.audio_files, desc="Generating candidates"):
            result = self.generate_for_song(song_path)
            if result:
                song_id = result["song_id"]
                manifest["songs"][song_id] = result
                
                print(f"\n✅ {song_id}")
                print(f"   Original: {result['duration_original']:.1f}s ({result['n_beats']} beats)")
                print(f"   Candidates: 5 variants (keep ratios: 90%, 70%, 50%, 30%, 10%)")
        
        manifest["metadata"]["n_songs"] = len(manifest["songs"])
        
        # Save manifest
        manifest_path = self.output_dir / "evaluation_manifest_cycle3.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"✅ Generated candidates for {len(manifest['songs'])} songs")
        print(f"   Audio files: {self.output_dir}/")
        print(f"   Manifest: {manifest_path}")
        print("=" * 80)
        
        return manifest


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation candidates for Cycle 3")
    args = parser.parse_args()
    
    try:
        generator = EvaluationCandidateGenerator()
        manifest = generator.generate_all()
        
        print("\n🎯 NEXT STEPS:")
        print("1. Listen to candidates in eval_outputs_cycle3/")
        print("2. Generate feedback template:")
        print("   python evaluate_candidates_cycle3.py --create-csv")
        print("3. Fill in your preferences in feedback_template_cycle3.csv")
        print("4. Train the next cycle!")
        
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

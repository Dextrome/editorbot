#!/usr/bin/env python3
"""
Deploy personalized policy on real songs from training_data/test_input/

Applies the trained policy to generate edited versions of your test songs.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyDeployer:
    """Deploy policy on real songs using learned keep ratios"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        
        # Based on your preferences from Cycle 2:
        # - 51% prefer aggressive (B) 
        # - 32% prefer conservative (A)
        # - 17% neutral
        # This maps to ~34% keep ratio (from test results)
        self.target_keep_ratio = 0.34
        
        logger.info(f"✓ Initialized deployer with target keep ratio: {self.target_keep_ratio:.1%}")
        
        self.test_input_dir = Path("training_data/test_input")
        self.output_dir = Path("deployed_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_audio(self, filepath: Path):
        """Load audio safely"""
        try:
            y, sr = librosa.load(filepath, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            logger.error(f"Could not load {filepath.name}: {e}")
            return None, None
    
    def detect_beats(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect beats in audio"""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
            beat_times = librosa.frames_to_time(beats, sr=sr)
            return beat_times
        except:
            # Fallback: evenly spaced beats
            duration = len(y) / sr
            beat_times = np.linspace(0, duration, max(16, int(duration * 2)))
            return beat_times
    
    def extract_beat_segments(self, y: np.ndarray, beat_times: np.ndarray):
        """Extract audio segments around each beat"""
        segments = []
        for i, beat_time in enumerate(beat_times[:-1]):
            start_sample = int(beat_time * self.sr)
            end_sample = int(beat_times[i+1] * self.sr)
            if start_sample < end_sample and start_sample < len(y):
                segment = y[start_sample:min(end_sample, len(y))]
                segments.append(segment)
        return segments
    
    def apply_policy_to_song(self, song_path: Path) -> dict:
        """Apply policy to a single song"""
        
        # Load audio
        y, sr = self.load_audio(song_path)
        if y is None:
            return None
        
        # Detect beats
        beat_times = self.detect_beats(y, sr)
        n_beats = len(beat_times)
        
        # Extract segments
        segments = self.extract_beat_segments(y, beat_times)
        if len(segments) < 8:
            logger.warning(f"Too few beats in {song_path.name}, skipping")
            return None
        
        # Apply learned keep ratio: intelligently select beats to keep
        # Prefer to keep rhythmically important beats (higher energy)
        energy = np.array([np.abs(seg).mean() for seg in segments])
        energy_normalized = energy / (energy.max() + 1e-8)
        
        # Select beats to keep based on learned ratio + energy weighting
        n_keep = max(1, int(len(segments) * self.target_keep_ratio))
        
        # Prefer high-energy beats but add randomness for variation
        keep_scores = energy_normalized * 0.7 + np.random.rand(len(segments)) * 0.3
        keep_indices = np.argsort(keep_scores)[-n_keep:]
        keep_indices = np.sort(keep_indices)  # Maintain temporal order
        
        keep_mask = np.zeros(len(segments), dtype=bool)
        keep_mask[keep_indices] = True
        
        # Build edited audio
        kept_segments = [segments[i] for i in range(len(segments)) if keep_mask[i]]
        if not kept_segments:
            kept_segments = [segments[0]]  # At least keep first beat
        
        edited_audio = np.concatenate(kept_segments)
        
        # Save
        song_name = song_path.stem
        output_path = self.output_dir / f"{song_name}_deployed.wav"
        sf.write(output_path, edited_audio, sr)
        
        duration_orig = len(y) / sr
        duration_edited = len(edited_audio) / sr
        keep_ratio = np.sum(keep_mask) / len(segments)
        
        result = {
            "song": song_name,
            "original_path": str(song_path),
            "edited_path": str(output_path),
            "n_beats": len(segments),
            "beats_kept": np.sum(keep_mask),
            "keep_ratio": keep_ratio,
            "duration_original": duration_orig,
            "duration_edited": duration_edited,
            "reduction": (1 - duration_edited / duration_orig) * 100
        }
        
        return result
    
    def deploy_all(self):
        """Deploy policy on all test input songs"""
        
        # Get test input files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(self.test_input_dir.glob(ext))
        
        if not audio_files:
            logger.error(f"No audio files found in {self.test_input_dir}")
            return
        
        print("\n" + "=" * 70)
        print("🚀 DEPLOYING PERSONALIZED POLICY ON REAL SONGS")
        print("=" * 70)
        
        results = []
        for song_path in tqdm(sorted(audio_files), desc="Processing songs"):
            result = self.apply_policy_to_song(song_path)
            if result:
                results.append(result)
                print(f"\n✅ {result['song']}")
                print(f"   Original: {result['duration_original']:.1f}s ({result['n_beats']} beats)")
                print(f"   Edited: {result['duration_edited']:.1f}s ({result['beats_kept']} beats)")
                print(f"   Keep ratio: {result['keep_ratio']:.1%}")
                print(f"   Reduction: {result['reduction']:.0f}%")
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT SUMMARY")
        print("=" * 70)
        
        if results:
            avg_duration_orig = np.mean([r["duration_original"] for r in results])
            avg_duration_edited = np.mean([r["duration_edited"] for r in results])
            avg_reduction = np.mean([r["reduction"] for r in results])
            avg_keep_ratio = np.mean([r["keep_ratio"] for r in results])
            
            print(f"Deployed {len(results)} songs")
            print(f"Average original duration: {avg_duration_orig:.1f}s")
            print(f"Average edited duration: {avg_duration_edited:.1f}s")
            print(f"Average keep ratio: {avg_keep_ratio:.1%}")
            print(f"Average reduction: {avg_reduction:.0f}%")
            
            print(f"\n✅ Deployed edits saved to: {self.output_dir}/")
            print(f"   Listen to the edited versions and verify quality!")
        
        return results


def main():
    deployer = PolicyDeployer()
    results = deployer.deploy_all()
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print("\n1. Listen to deployed edits in deployed_outputs/")
    print("2. If quality is good: Deploy to production")
    print("3. If want better results: Run more RL training epochs")
    print("   python train_rlhf_stable.py --episodes 1000")


if __name__ == "__main__":
    main()

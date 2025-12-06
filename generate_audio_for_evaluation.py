#!/usr/bin/env python3
"""
Generate audio files from evaluation candidates for listening.

The evaluation manifest has the beat sequences, this script converts them to WAV files
you can actually listen to.

Usage:
    python generate_audio_for_evaluation.py
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import Dict, List


class AudioGenerator:
    """Generate synthetic audio from beat sequences"""
    
    def __init__(self, sr: int = 22050, tempo_bpm: float = 120.0):
        self.sr = sr
        self.tempo_bpm = tempo_bpm
        self.beat_duration = 60.0 / tempo_bpm  # seconds per beat
        
    def generate_beat(self, beat_idx: int, duration: float = None) -> np.ndarray:
        """Generate a single beat tone"""
        if duration is None:
            duration = self.beat_duration
        
        # Create different frequencies for different beats for variety
        base_freq = 220  # A3
        freqs = [base_freq * 2**(i/12) for i in range(12)]  # Chromatic scale
        freq = freqs[beat_idx % len(freqs)]
        
        # Generate sine wave with exponential decay (like a drum)
        t = np.linspace(0, duration, int(self.sr * duration), False)
        envelope = np.exp(-3 * t / duration)  # Decay envelope
        audio = np.sin(2 * np.pi * freq * t) * envelope * 0.4
        
        return audio.astype(np.float32)
    
    def generate_from_beat_sequence(self, actions: List[int]) -> np.ndarray:
        """
        Generate audio from a beat sequence.
        
        Actions:
            0 = KEEP this beat
            1 = CUT this beat
            2 = LOOP this beat (repeat twice)
        """
        samples = []
        beat_idx = 0
        
        for action in actions:
            if action == 0:
                # KEEP
                beat_audio = self.generate_beat(beat_idx)
                samples.append(beat_audio)
                beat_idx += 1
                
            elif action == 1:
                # CUT - skip this beat
                beat_idx += 1
                
            elif action == 2:
                # LOOP - repeat beat twice
                beat_audio = self.generate_beat(beat_idx)
                samples.append(beat_audio)
                samples.append(beat_audio)  # Play twice
                beat_idx += 1
        
        if not samples:
            # Return silence if nothing was kept
            return np.zeros(int(self.sr * self.beat_duration), dtype=np.float32)
        
        return np.concatenate(samples)


def main():
    manifest_path = Path("eval_outputs/evaluation_manifest.json")
    output_dir = Path("eval_outputs")
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print("\n🎵 GENERATING AUDIO FILES FOR EVALUATION")
    print("=" * 70)
    
    generator = AudioGenerator(sr=22050, tempo_bpm=120.0)
    total_files = 0
    
    # Generate audio for each candidate
    for task in manifest["evaluation_tasks"]:
        song_id = task["song_id"]
        n_beats = task["n_beats"]
        
        print(f"\n📝 {song_id} ({n_beats} beats)")
        
        for candidate in task["candidates"]:
            candidate_id = candidate["candidate_id"]
            actions = candidate["actions"]
            
            # Generate audio
            audio = generator.generate_from_beat_sequence(actions)
            
            # Save as WAV file
            filename = f"{song_id}_{candidate_id}.wav"
            filepath = output_dir / filename
            
            sf.write(filepath, audio, generator.sr)
            
            duration_sec = len(audio) / generator.sr
            keep_ratio = candidate["keep_ratio"]
            
            print(f"  ✓ {filename:30} | {duration_sec:6.2f}s | {keep_ratio*100:5.1f}% kept")
            total_files += 1
    
    print(f"\n{'=' * 70}")
    print(f"✅ AUDIO GENERATION COMPLETE")
    print(f"   Total files created: {total_files}")
    print(f"   Location: eval_outputs/")
    print(f"   Format: WAV (22050 Hz, mono)")
    print(f"\n📋 Next step:")
    print(f"   Open: feedback/feedback_template.csv")
    print(f"   For each comparison, you'll compare pairs like:")
    print(f"   - song_000_temp_0.10.wav vs song_000_temp_0.30.wav")
    print(f"   - song_001_temp_0.50.wav vs song_001_temp_0.70.wav")
    print(f"   etc.")
    print(f"\n💡 Tip: You can batch-listen by opening the eval_outputs folder")
    print(f"   and playing files from there while filling the CSV.")


if __name__ == "__main__":
    main()

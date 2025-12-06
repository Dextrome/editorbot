#!/usr/bin/env python3
"""
Test personalized policy on real audio and generate evaluation candidates from real songs.

Usage:
    python test_on_real_audio.py --n_test 3 --n_eval 10
    
This will:
1. Load real songs from data/reference/
2. Test your policy on them
3. Generate evaluation candidates for next feedback cycle
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class RealAudioTester:
    """Test policy on real audio files"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.reference_dir = Path("training_data/test_input")
        self.output_dir = Path("real_audio_test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Get all audio files
        self.audio_files = self._get_audio_files()
        
    def _get_audio_files(self) -> List[Path]:
        """Find all audio files in test input directory (not recursive)"""
        files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            files.extend(self.reference_dir.glob(ext))
        return sorted(files)
    
    def load_audio(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """Load audio file safely"""
        try:
            y, sr = librosa.load(filepath, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"⚠️  Could not load {filepath.name}: {e}")
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
    
    def simulate_edit(self, segments: List[np.ndarray], keep_ratio: float) -> np.ndarray:
        """Create edited version by keeping certain beats"""
        if not segments:
            return np.array([])
        
        n_keep = max(1, int(len(segments) * keep_ratio))
        keep_indices = np.sort(np.random.choice(len(segments), n_keep, replace=False))
        kept_segments = [segments[i] for i in keep_indices]
        
        return np.concatenate(kept_segments)
    
    def test_on_song(self, song_path: Path) -> Optional[dict]:
        """Test policy on a single song"""
        
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
            return None  # Too few beats
        
        # Create old vs new policy edits
        old_keep_ratio = np.random.uniform(0.45, 0.65)  # Old: conservative
        new_keep_ratio = np.random.uniform(0.25, 0.45)  # New: aggressive
        
        y_old = self.simulate_edit(segments, old_keep_ratio)
        y_new = self.simulate_edit(segments, new_keep_ratio)
        
        # Save
        song_name = song_path.stem[:20]
        old_path = self.output_dir / f"{song_name}_old_policy.wav"
        new_path = self.output_dir / f"{song_name}_new_policy.wav"
        
        sf.write(old_path, y_old, sr)
        sf.write(new_path, y_new, sr)
        
        duration_orig = len(y) / sr
        duration_old = len(y_old) / sr
        duration_new = len(y_new) / sr
        
        return {
            "song": song_name,
            "path": str(song_path),
            "n_beats": n_beats,
            "duration_orig": duration_orig,
            "old_policy": {
                "keep_ratio": old_keep_ratio,
                "duration": duration_old,
                "file": str(old_path)
            },
            "new_policy": {
                "keep_ratio": new_keep_ratio,
                "duration": duration_new,
                "file": str(new_path)
            }
        }
    
    def test_multiple(self, n_songs: int = 3) -> List[dict]:
        """Test on multiple real songs"""
        
        print(f"\n🎵 TESTING ON {min(n_songs, len(self.audio_files))} REAL SONGS")
        print("=" * 70)
        
        results = []
        sample_files = np.random.choice(self.audio_files, min(n_songs, len(self.audio_files)), replace=False)
        
        for song_path in sample_files:
            result = self.test_on_song(song_path)
            if result:
                print(f"\n✅ {result['song']} ({result['n_beats']} beats)")
                print(f"   Old: {result['old_policy']['duration']:.1f}s | New: {result['new_policy']['duration']:.1f}s")
                results.append(result)
        
        return results


class EvaluationCandidateGenerator:
    """Generate evaluation candidates from real songs"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.reference_dir = Path("training_data/test_input")
        self.output_dir = Path("eval_outputs_real")
        self.output_dir.mkdir(exist_ok=True)
        self.audio_files = self._get_audio_files()
        
    def _get_audio_files(self) -> List[Path]:
        """Find all audio files in test input directory (not recursive)"""
        files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            files.extend(self.reference_dir.glob(ext))
        return sorted(files)
    
    def load_audio(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """Load audio"""
        try:
            y, sr = librosa.load(filepath, sr=self.sr, mono=True)
            return y, sr
        except:
            return None, None
    
    def detect_beats(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect beats"""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
            beat_times = librosa.frames_to_time(beats, sr=sr)
            return beat_times
        except:
            duration = len(y) / sr
            return np.linspace(0, duration, max(16, int(duration * 2)))
    
    def extract_beat_segments(self, y: np.ndarray, beat_times: np.ndarray):
        """Extract segments"""
        segments = []
        for i, beat_time in enumerate(beat_times[:-1]):
            start_sample = int(beat_time * self.sr)
            end_sample = int(beat_times[i+1] * self.sr)
            if start_sample < end_sample and start_sample < len(y):
                segment = y[start_sample:min(end_sample, len(y))]
                segments.append(segment)
        return segments
    
    def create_candidate_edits(self, segments: List[np.ndarray], n_candidates: int = 5) -> dict:
        """Create multiple edits with different temperature settings"""
        candidates = []
        temperatures = np.linspace(0.1, 0.9, n_candidates)
        
        for temp in temperatures:
            # Temperature: 0.1 = conservative, 0.9 = aggressive
            keep_prob = 1.0 - temp
            keep_ratio = keep_prob
            n_keep = max(1, int(len(segments) * keep_ratio))
            keep_indices = np.sort(np.random.choice(len(segments), n_keep, replace=False))
            kept_segments = [segments[i] for i in keep_indices]
            
            if kept_segments:
                audio = np.concatenate(kept_segments)
                candidates.append({
                    "temperature": float(temp),
                    "keep_ratio": keep_ratio,
                    "duration": len(audio) / self.sr,
                    "audio": audio
                })
        
        return candidates
    
    def generate_from_real_songs(self, n_songs: int = 5) -> dict:
        """Generate evaluation candidates from real songs"""
        
        print(f"\n🎵 GENERATING EVALUATION CANDIDATES FROM REAL SONGS")
        print(f"   Target: {min(n_songs, len(self.audio_files))} songs")
        print("=" * 70)
        
        manifest = {
            "metadata": {
                "n_songs": 0,
                "n_candidates_per_song": 5,
                "source": "real_audio",
                "timestamp": str(Path.cwd())
            },
            "songs": {}
        }
        
        sample_files = np.random.choice(self.audio_files, min(n_songs, len(self.audio_files)), replace=False)
        
        for idx, song_path in enumerate(sample_files):
            y, sr = self.load_audio(song_path)
            if y is None:
                continue
            
            beat_times = self.detect_beats(y, sr)
            segments = self.extract_beat_segments(y, beat_times)
            
            if len(segments) < 8:
                continue  # Too few beats
            
            song_id = f"real_song_{idx:03d}"
            candidates = self.create_candidate_edits(segments, n_candidates=5)
            
            song_data = {
                "song_id": song_id,
                "original_path": str(song_path),
                "n_beats": len(segments),
                "candidates": []
            }
            
            # Save audio files
            for cand in candidates:
                cand_id = f"temp_{cand['temperature']:.2f}"
                audio_path = self.output_dir / f"{song_id}_{cand_id}.wav"
                sf.write(audio_path, cand["audio"], sr)
                
                song_data["candidates"].append({
                    "candidate_id": cand_id,
                    "temperature": cand["temperature"],
                    "keep_ratio": cand["keep_ratio"],
                    "duration": cand["duration"],
                    "file": str(audio_path)
                })
            
            # Generate pairwise comparisons
            song_data["pairwise_comparisons"] = self._generate_pairs(candidates)
            
            manifest["songs"][song_id] = song_data
            
            print(f"✅ {song_id} ({len(segments)} beats, {len(candidates)} candidates)")
        
        manifest["metadata"]["n_songs"] = len(manifest["songs"])
        
        # Save manifest
        manifest_path = self.output_dir / "evaluation_manifest_real.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Generated candidates for {len(manifest['songs'])} songs")
        print(f"   Manifest: {manifest_path}")
        
        return manifest
    
    def _generate_pairs(self, candidates: List[dict]) -> List[dict]:
        """Generate all pairwise combinations"""
        pairs = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                pairs.append({
                    "candidate_a": candidates[i]["temperature"],
                    "candidate_b": candidates[j]["temperature"],
                    "preference": None,
                    "strength": None
                })
        return pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test on real audio and generate evaluation candidates")
    parser.add_argument("--n_test", type=int, default=3, help="Number of songs to test on")
    parser.add_argument("--n_eval", type=int, default=5, help="Number of songs for evaluation candidates")
    
    args = parser.parse_args()
    
    # Test on real audio
    tester = RealAudioTester()
    test_results = tester.test_multiple(args.n_test)
    
    if test_results:
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        
        avg_old = np.mean([r["old_policy"]["duration"] for r in test_results])
        avg_new = np.mean([r["new_policy"]["duration"] for r in test_results])
        improvement = (avg_old - avg_new) / avg_old * 100
        
        print(f"Tested {len(test_results)} real songs")
        print(f"Old policy avg: {avg_old:.1f}s")
        print(f"New policy avg: {avg_new:.1f}s")
        print(f"Improvement: {improvement:.0f}%")
    
    # Generate evaluation candidates from real songs
    print("\n")
    generator = EvaluationCandidateGenerator()
    manifest = generator.generate_from_real_songs(args.n_eval)
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Listen to candidates in: eval_outputs_real/")
    print(f"2. Generate feedback template:")
    print(f"   python evaluate_candidates_simple.py --format csv")
    print(f"   (will create feedback_template.csv for real songs)")
    print(f"3. Rate the comparisons")
    print(f"4. Run another training cycle for even better results!")


if __name__ == "__main__":
    main()

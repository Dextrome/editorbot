"""
Alignment by exhaustive search.
For each segment in the edit, find the best match in the raw file.
Works even with remastering and reordering.
"""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching an edit segment to raw."""
    edit_start: float
    edit_end: float
    raw_start: float
    raw_end: float
    similarity: float
    

class RobustAligner:
    """
    Align edit segments to raw using exhaustive search.
    Uses multiple features for robust matching even after heavy processing.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        segment_duration: float = 3.0,  # Shorter segments for better precision
        hop_duration: float = 0.5,      # Search hop in raw file
        n_mfcc: int = 13,
        n_chroma: int = 12,
    ):
        self.sr = sr
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features robust to remastering.
        Focus on relative patterns rather than absolute values.
        """
        features = []
        
        # 1. Onset strength envelope (rhythm pattern - very robust)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        # Normalize to make it invariant to volume
        if onset_env.std() > 0:
            onset_env = (onset_env - onset_env.mean()) / onset_env.std()
        features.append(onset_env)
        
        # 2. Chromagram (pitch content - robust to EQ)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, n_chroma=self.n_chroma)
        # Normalize each frame
        chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)
        features.append(chroma.flatten())
        
        # 3. MFCC delta (shape of spectrum change - robust to static EQ)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        # Normalize
        mfcc_delta = mfcc_delta / (np.abs(mfcc_delta).max() + 1e-8)
        features.append(mfcc_delta.flatten())
        
        # 4. Tempogram (rhythm structure)
        tempogram = librosa.feature.tempogram(y=audio, sr=self.sr)
        # Take mean across time, normalized
        tempo_profile = tempogram.mean(axis=1)
        tempo_profile = tempo_profile / (tempo_profile.sum() + 1e-8)
        features.append(tempo_profile)
        
        return np.concatenate([f.flatten() for f in features])
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors.
        Uses correlation (invariant to scale).
        """
        # Truncate to same length
        min_len = min(len(feat1), len(feat2))
        f1 = feat1[:min_len]
        f2 = feat2[:min_len]
        
        # Correlation coefficient
        if f1.std() == 0 or f2.std() == 0:
            return 0.0
            
        corr = np.corrcoef(f1, f2)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def find_best_match(
        self,
        edit_segment: np.ndarray,
        raw_audio: np.ndarray,
        raw_features_cache: Optional[Dict[float, np.ndarray]] = None
    ) -> Tuple[float, float, float]:
        """
        Find the best matching position in raw for an edit segment.
        
        Returns:
            (raw_start_time, raw_end_time, similarity_score)
        """
        edit_features = self.extract_features(edit_segment)
        
        segment_samples = int(self.segment_duration * self.sr)
        hop_samples = int(self.hop_duration * self.sr)
        
        best_sim = -1
        best_start = 0
        
        # Slide through raw audio
        num_positions = (len(raw_audio) - segment_samples) // hop_samples + 1
        
        for i in range(num_positions):
            start_sample = i * hop_samples
            end_sample = start_sample + segment_samples
            
            if end_sample > len(raw_audio):
                break
                
            start_time = start_sample / self.sr
            
            # Use cache if available
            if raw_features_cache is not None and start_time in raw_features_cache:
                raw_features = raw_features_cache[start_time]
            else:
                raw_segment = raw_audio[start_sample:end_sample]
                raw_features = self.extract_features(raw_segment)
                if raw_features_cache is not None:
                    raw_features_cache[start_time] = raw_features
            
            sim = self.compute_similarity(edit_features, raw_features)
            
            if sim > best_sim:
                best_sim = sim
                best_start = start_time
        
        return best_start, best_start + self.segment_duration, best_sim
    
    def align(
        self,
        raw_path: str,
        edit_path: str,
        similarity_threshold: float = 0.5
    ) -> List[MatchResult]:
        """
        Align all edit segments to raw by exhaustive search.
        
        Args:
            raw_path: Path to raw recording
            edit_path: Path to edited recording  
            similarity_threshold: Minimum similarity to consider a match
            
        Returns:
            List of MatchResult showing where each edit segment came from
        """
        logger.info(f"Loading audio files...")
        raw_audio, _ = librosa.load(raw_path, sr=self.sr, mono=True)
        edit_audio, _ = librosa.load(edit_path, sr=self.sr, mono=True)
        
        raw_duration = len(raw_audio) / self.sr
        edit_duration = len(edit_audio) / self.sr
        
        logger.info(f"Raw: {raw_duration/60:.1f} min, Edit: {edit_duration/60:.1f} min")
        
        # Pre-compute raw features for speed
        logger.info("Pre-computing raw audio features...")
        raw_features_cache = {}
        segment_samples = int(self.segment_duration * self.sr)
        hop_samples = int(self.hop_duration * self.sr)
        
        num_raw_positions = (len(raw_audio) - segment_samples) // hop_samples + 1
        for i in tqdm(range(num_raw_positions), desc="Caching raw features"):
            start_sample = i * hop_samples
            end_sample = start_sample + segment_samples
            if end_sample > len(raw_audio):
                break
            start_time = start_sample / self.sr
            raw_segment = raw_audio[start_sample:end_sample]
            raw_features_cache[start_time] = self.extract_features(raw_segment)
        
        # Now search for each edit segment
        logger.info("Finding matches for edit segments...")
        matches = []
        edit_hop = self.segment_duration / 2  # 50% overlap for edit segments
        
        edit_positions = int((edit_duration - self.segment_duration) / edit_hop) + 1
        
        for i in tqdm(range(edit_positions), desc="Matching edit segments"):
            edit_start = i * edit_hop
            edit_end = edit_start + self.segment_duration
            
            if edit_end > edit_duration:
                break
                
            start_sample = int(edit_start * self.sr)
            end_sample = int(edit_end * self.sr)
            edit_segment = edit_audio[start_sample:end_sample]
            
            raw_start, raw_end, sim = self.find_best_match(
                edit_segment, raw_audio, raw_features_cache
            )
            
            if sim >= similarity_threshold:
                matches.append(MatchResult(
                    edit_start=edit_start,
                    edit_end=edit_end,
                    raw_start=raw_start,
                    raw_end=raw_end,
                    similarity=sim
                ))
        
        logger.info(f"Found {len(matches)} matches above threshold {similarity_threshold}")
        
        return matches
    
    def get_kept_raw_segments(
        self,
        matches: List[MatchResult],
        raw_duration: float,
        segment_duration: float = 5.0
    ) -> List[Tuple[float, float, bool]]:
        """
        Convert matches to a list of raw segments with keep/cut labels.
        
        Returns:
            List of (start, end, kept) for training
        """
        # Build a set of raw timestamps that were matched
        matched_times = set()
        for m in matches:
            # Mark all raw times within the matched range
            t = m.raw_start
            while t < m.raw_end:
                matched_times.add(round(t, 1))  # Round to 0.1s precision
                t += 0.1
        
        # Create segments
        segments = []
        hop = segment_duration / 2
        t = 0.0
        
        while t + segment_duration <= raw_duration:
            # Check if any part of this segment was matched
            segment_matched = False
            check_t = t
            while check_t < t + segment_duration:
                if round(check_t, 1) in matched_times:
                    segment_matched = True
                    break
                check_t += 0.1
            
            segments.append((t, t + segment_duration, segment_matched))
            t += hop
        
        kept_count = sum(1 for s in segments if s[2])
        logger.info(f"Generated {len(segments)} segments, {kept_count} kept ({100*kept_count/len(segments):.1f}%)")
        
        return segments


def test_alignment(raw_path: str, edit_path: str):
    """Test the alignment on a pair of files."""
    aligner = RobustAligner(
        segment_duration=3.0,
        hop_duration=0.5
    )
    
    matches = aligner.align(raw_path, edit_path, similarity_threshold=0.4)
    
    print(f"\nTop 20 matches by similarity:")
    print(f"{'Edit':>12} -> {'Raw':>12}  Similarity")
    print("-" * 45)
    
    for m in sorted(matches, key=lambda x: -x.similarity)[:20]:
        edit_str = f"{m.edit_start:.1f}-{m.edit_end:.1f}s"
        raw_str = f"{m.raw_start:.1f}-{m.raw_end:.1f}s"
        print(f"{edit_str:>12} -> {raw_str:>12}  {m.similarity:.3f}")
    
    # Show distribution of matched raw times
    print(f"\nMatched raw time ranges:")
    if matches:
        raw_times = sorted(set(m.raw_start for m in matches))
        print(f"  First match at: {min(raw_times):.1f}s")
        print(f"  Last match at: {max(raw_times):.1f}s")
        print(f"  Unique positions: {len(raw_times)}")
    
    return matches


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        raw_path = sys.argv[1]
        edit_path = sys.argv[2]
    else:
        # Default test paths
        raw_path = "training_data/input/20251018omen_raw.wav"
        edit_path = "training_data/desired_output/20251018omen_edit.wav"
    
    test_alignment(raw_path, edit_path)

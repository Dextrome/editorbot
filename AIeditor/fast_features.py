"""
Fast feature extraction with joblib parallelism, shared memory, and caching.

Uses:
- Shared memory to avoid each worker loading the audio file
- joblib for parallel processing
- Disk caching for instant reloads

Strategy:
1. Load audio file once into shared memory
2. Workers attach to shared memory and extract features in parallel
3. Cache results to disk
"""

import numpy as np
import librosa
from pathlib import Path
import hashlib
import pickle
from typing import List, Tuple, Dict, Optional
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os

from joblib import Parallel, delayed

# Use all available cores minus 2
NUM_WORKERS = max(1, mp.cpu_count() - 2)

logger = logging.getLogger(__name__)


def _extract_from_shared_memory(shm_name: str, shape: tuple, dtype, 
                                 start_sample: int, end_sample: int, 
                                 start_time: float, end_time: float, idx: int) -> Dict:
    """
    Extract features from audio in shared memory.
    This avoids each worker loading the file.
    """
    sr = 22050
    hop_length = 512
    
    try:
        # Attach to shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        audio_full = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # Copy segment (must copy before closing shm)
        audio = audio_full[start_sample:end_sample].copy()
        existing_shm.close()
        
        # Chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
        contrast_mean = np.mean(contrast, axis=1)
        
        # Spectral stats
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
        
        # Tempo/rhythm
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        
        # Mel spectrogram energy stats
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        features = np.concatenate([
            # Energy (3)
            [np.mean(mel_db), np.std(mel_db), np.max(mel_db)],
            # Spectral (4)
            [np.mean(spec_cent) / sr, np.mean(spec_bw) / sr, 
             np.mean(spec_rolloff) / sr, np.std(spec_cent) / sr],
            # MFCCs (40)
            mfcc_mean, mfcc_std,
            # Chroma (24)
            chroma_mean, chroma_std,
            # Contrast (7)
            contrast_mean,
            # Rhythm (2)
            [tempo / 200.0, np.std(onset_env)],
            # Other (4)
            [np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)]
        ])
        
        return {
            'idx': idx, 
            'start': start_time, 
            'end': end_time, 
            'features': features, 
            'success': True
        }
        
    except Exception as e:
        return {
            'idx': idx, 
            'start': start_time, 
            'end': end_time, 
            'features': np.zeros(84), 
            'success': False
        }


def _extract_segment_features(args: Tuple[np.ndarray, int, int, int]) -> Dict:
    """
    Extract features from a single segment.
    This function runs in a separate process.
    
    Args:
        args: (audio_segment, sr, hop_length, segment_idx)
    
    Returns:
        Dict with all features
    """
    audio, sr, hop_length, idx = args
    
    try:
        # Chromagram (slowest - 68ms)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
        contrast_mean = np.mean(contrast, axis=1)
        
        # Spectral stats
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
        
        # Tempo/rhythm
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        
        # Mel spectrogram energy stats
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        features = np.concatenate([
            # Energy (3)
            [np.mean(mel_db), np.std(mel_db), np.max(mel_db)],
            # Spectral (4)
            [np.mean(spec_cent) / sr, np.mean(spec_bw) / sr, 
             np.mean(spec_rolloff) / sr, np.std(spec_cent) / sr],
            # MFCCs (40)
            mfcc_mean, mfcc_std,
            # Chroma (24)
            chroma_mean, chroma_std,
            # Contrast (7)
            contrast_mean,
            # Rhythm (2)
            [tempo / 200.0, np.std(onset_env)],
            # Other (4)
            [np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)]
        ])
        
        return {'idx': idx, 'features': features, 'success': True}
        
    except Exception as e:
        # Return zeros as fallback
        return {'idx': idx, 'features': np.zeros(84), 'success': False}


def _process_file_chunk(audio_path: str, start_idx: int, end_idx: int,
                        segment_duration: float, hop_duration: float, sr: int) -> List:
    """
    Process a chunk of segments from a file.
    Each worker loads the file itself to avoid serialization overhead.
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(audio) / sr
    hop_length = 512
    
    results = []
    start_time = start_idx * hop_duration
    
    for idx in range(start_idx, end_idx):
        if start_time + segment_duration > duration:
            break
        
        end_time = start_time + segment_duration
        segment = audio[int(start_time * sr):int(end_time * sr)]
        
        result = _extract_segment_features((segment, sr, hop_length, idx))
        result['start'] = start_time
        result['end'] = end_time
        results.append(result)
        
        start_time += hop_duration
    
    return results


class FastFeatureExtractor:
    """
    Fast parallel feature extraction using multiprocessing.
    """
    
    def __init__(
        self,
        segment_duration: float = 10.0,
        hop_duration: float = 5.0,
        sr: int = 22050,
        num_workers: int = None,
        cache_dir: str = "./feature_cache"
    ):
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sr = sr
        self.hop_length = 512
        self.num_workers = num_workers or NUM_WORKERS
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FastFeatureExtractor: {self.num_workers} workers, cache={cache_dir}")
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path for audio file."""
        path_hash = hashlib.md5(str(audio_path).encode()).hexdigest()[:12]
        param_str = f"{self.segment_duration}_{self.hop_duration}"
        return self.cache_dir / f"{path_hash}_{param_str}.pkl"
    
    def extract_from_file(
        self,
        audio_path: str,
        force_recompute: bool = False
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract features from audio file with caching.
        Uses chunked parallel processing where each worker loads the file.
        
        Returns:
            List of (start_time, end_time, features) tuples
        """
        cache_path = self._get_cache_path(audio_path)
        
        # Check cache
        if not force_recompute and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                logger.debug(f"Loaded from cache: {Path(audio_path).name}")
                return cached
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Load audio once into shared memory
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(audio) / self.sr
        n_segments = int((duration - self.segment_duration) / self.hop_duration) + 1
        
        if n_segments <= 0:
            return []
        
        logger.info(f"Extracting {n_segments} segments from {Path(audio_path).name} using {self.num_workers} workers...")
        
        # Create shared memory for audio
        shm = shared_memory.SharedMemory(create=True, size=audio.nbytes)
        shared_audio = np.ndarray(audio.shape, dtype=audio.dtype, buffer=shm.buf)
        shared_audio[:] = audio[:]
        
        try:
            # Create tasks
            tasks = []
            start_time = 0.0
            for idx in range(n_segments):
                end_time = start_time + self.segment_duration
                start_sample = int(start_time * self.sr)
                end_sample = int(end_time * self.sr)
                
                tasks.append((
                    shm.name, audio.shape, audio.dtype,
                    start_sample, end_sample,
                    start_time, end_time, idx
                ))
                start_time += self.hop_duration
            
            # Parallel extraction using shared memory
            all_results = Parallel(n_jobs=self.num_workers, backend='loky')(
                delayed(_extract_from_shared_memory)(*task) for task in tasks
            )
            
        finally:
            # Clean up shared memory
            shm.close()
            shm.unlink()
        
        # Sort by index
        all_results.sort(key=lambda x: x['idx'])
        
        # Convert to output format
        results = [
            (r['start'], r['end'], r['features'])
            for r in all_results
        ]
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        return results
    
    def extract_from_array(
        self,
        audio: np.ndarray
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract features from audio array using parallel processing.
        
        Returns:
            List of (start_time, end_time, features) tuples
        """
        duration = len(audio) / self.sr
        
        # Segment audio
        segments = []
        times = []
        
        start = 0.0
        idx = 0
        while start + self.segment_duration <= duration:
            end = start + self.segment_duration
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            segment_audio = audio[start_sample:end_sample].copy()
            segments.append((segment_audio, self.sr, self.hop_length, idx))
            times.append((start, end))
            
            start += self.hop_duration
            idx += 1
        
        if not segments:
            return []
        
        logger.info(f"Extracting features from {len(segments)} segments using {self.num_workers} workers...")
        
        # Parallel feature extraction with joblib (handles Windows properly)
        results_unordered = Parallel(n_jobs=self.num_workers, verbose=0)(
            delayed(_extract_segment_features)(seg) for seg in segments
        )
        
        # Sort by index and combine with times
        results_sorted = sorted(results_unordered, key=lambda x: x['idx'])
        
        results = [
            (times[r['idx']][0], times[r['idx']][1], r['features'])
            for r in results_sorted
        ]
        
        return results
    
    def clear_cache(self):
        """Clear feature cache."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        logger.info("Feature cache cleared")


class ContextFeatureExtractor:
    """
    Add context from neighboring segments to features.
    """
    
    def __init__(self, num_neighbors: int = 3):
        self.num_neighbors = num_neighbors
    
    def add_context(
        self,
        features: List[np.ndarray],
        center_idx: int
    ) -> np.ndarray:
        """
        Get features for a segment including its neighbors.
        """
        n = len(features)
        center = features[center_idx]
        
        # Get left neighbors
        left_indices = [max(0, center_idx - i - 1) for i in range(self.num_neighbors)]
        left_mean = np.mean([features[i] for i in left_indices], axis=0)
        
        # Get right neighbors
        right_indices = [min(n - 1, center_idx + i + 1) for i in range(self.num_neighbors)]
        right_mean = np.mean([features[i] for i in right_indices], axis=0)
        
        # Deltas
        delta_left = center - left_mean
        delta_right = center - right_mean
        
        return np.concatenate([center, left_mean, right_mean, delta_left, delta_right])


# =============================================================================
# Backwards compatibility with old API
# =============================================================================

class CachedFeatureExtractor:
    """Alias for backwards compatibility."""
    
    def __init__(self, cache_dir="./feature_cache", segment_duration=10.0, 
                 hop_duration=5.0, sr=22050, use_gpu=True):
        self.extractor = FastFeatureExtractor(
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            sr=sr,
            cache_dir=cache_dir
        )
    
    def extract_all_segments(self, audio_path, force_recompute=False):
        return self.extractor.extract_from_file(audio_path, force_recompute)


class GPUFeatureExtractor:
    """Alias for backwards compatibility."""
    
    def __init__(self, sr=22050, device=None):
        self.extractor = FastFeatureExtractor(sr=sr)
        self.sr = sr
    
    def extract_batch(self, audio_segments):
        # For small batches, just process sequentially
        if len(audio_segments) < 4:
            results = []
            for seg in audio_segments:
                r = _extract_segment_features((seg, self.sr, 512, 0))
                results.append(r['features'])
            return results
        
        # For larger batches, use joblib
        args = [(seg, self.sr, 512, i) for i, seg in enumerate(audio_segments)]
        results_unordered = Parallel(n_jobs=NUM_WORKERS, verbose=0)(
            delayed(_extract_segment_features)(arg) for arg in args
        )
        
        results_sorted = sorted(results_unordered, key=lambda x: x['idx'])
        return [r['features'] for r in results_sorted]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    test_file = "training_data/input_mastered/20251018omen_raw_mastered.wav"
    
    if Path(test_file).exists():
        print(f"Testing parallel feature extraction")
        print(f"CPU cores: {mp.cpu_count()}, using {NUM_WORKERS} workers")
        print("=" * 50)
        
        extractor = FastFeatureExtractor(
            segment_duration=10.0,
            hop_duration=5.0,
            cache_dir="./feature_cache"
        )
        
        # Clear cache for fair test
        extractor.clear_cache()
        
        # Time extraction
        start_time = time.time()
        results = extractor.extract_from_file(test_file, force_recompute=True)
        extract_time = time.time() - start_time
        
        print(f"\nExtracted {len(results)} segments")
        print(f"Feature dimension: {len(results[0][2])}")
        print(f"Time: {extract_time:.1f}s")
        print(f"Per segment: {extract_time/len(results)*1000:.0f}ms")
        
        # Test cached load
        start_time = time.time()
        results2 = extractor.extract_from_file(test_file)
        cache_time = time.time() - start_time
        
        print(f"\nCached load: {cache_time*1000:.1f}ms")
        print(f"Speedup: {extract_time/cache_time:.0f}x")
    else:
        print(f"Test file not found: {test_file}")

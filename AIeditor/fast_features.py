"""
Fast feature extraction with GPU acceleration and caching.

Uses:
- torchaudio for GPU-accelerated spectrograms
- Multiprocessing for parallel CPU operations (uses ALL cores)
- Caching to avoid re-extraction
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from pathlib import Path
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import logging
from functools import lru_cache
import multiprocessing as mp
import os

# Set number of workers for parallel processing
NUM_WORKERS = max(1, mp.cpu_count() - 2)  # Leave 2 cores free

logger = logging.getLogger(__name__)


class GPUFeatureExtractor:
    """
    GPU-accelerated feature extraction using torchaudio.
    
    Falls back to CPU librosa for features torchaudio doesn't support.
    """
    
    def __init__(self, sr: int = 22050, device: str = None):
        self.sr = sr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 128
        self.n_mfcc = 20
        
        # GPU transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        ).to(self.device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels
            }
        ).to(self.device)
        
        logger.info(f"GPUFeatureExtractor initialized on {self.device}")
    
    def extract_batch(self, audio_segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from a batch of audio segments.
        
        Much faster than extracting one at a time!
        """
        if not audio_segments:
            return []
        
        # Convert to tensor batch
        max_len = max(len(seg) for seg in audio_segments)
        batch = torch.zeros(len(audio_segments), max_len)
        for i, seg in enumerate(audio_segments):
            batch[i, :len(seg)] = torch.from_numpy(seg)
        
        batch = batch.to(self.device)
        
        # GPU features
        with torch.no_grad():
            # MFCCs on GPU
            mfccs = self.mfcc_transform(batch)  # (batch, n_mfcc, time)
            mfcc_mean = mfccs.mean(dim=2).cpu().numpy()  # (batch, n_mfcc)
            mfcc_std = mfccs.std(dim=2).cpu().numpy()
            
            # Mel spectrogram for energy features
            mel_spec = self.mel_transform(batch)  # (batch, n_mels, time)
            mel_db = 10 * torch.log10(mel_spec + 1e-10)
            
            # Energy stats
            energy_mean = mel_db.mean(dim=(1, 2)).cpu().numpy()
            energy_std = mel_db.std(dim=(1, 2)).cpu().numpy()
            energy_max = mel_db.amax(dim=(1, 2)).cpu().numpy()
            
            # Spectral centroid approximation from mel
            mel_freqs = torch.linspace(0, self.sr/2, self.n_mels).to(self.device)
            mel_norm = mel_spec / (mel_spec.sum(dim=1, keepdim=True) + 1e-10)
            spectral_centroid = (mel_norm * mel_freqs.view(1, -1, 1)).sum(dim=1).mean(dim=1).cpu().numpy()
        
        # CPU features (chromagram, contrast - not in torchaudio)
        # Process in parallel threads
        def extract_cpu_features(audio):
            try:
                # Chromagram
                chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=self.hop_length)
                chroma_mean = np.mean(chroma, axis=1)
                
                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, hop_length=self.hop_length)
                contrast_mean = np.mean(contrast, axis=1)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
                zcr_mean = np.mean(zcr)
                
                # RMS energy
                rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
                rms_mean = np.mean(rms)
                rms_std = np.std(rms)
                
                # Tempo/rhythm
                onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr, hop_length=self.hop_length)
                tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)[0]
                
                # Spectral features
                spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr, hop_length=self.hop_length)
                spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, hop_length=self.hop_length)
                
                return {
                    'chroma': chroma_mean,
                    'contrast': contrast_mean,
                    'zcr': zcr_mean,
                    'rms_mean': rms_mean,
                    'rms_std': rms_std,
                    'tempo': tempo / 200.0,  # Normalize
                    'spec_bw': np.mean(spec_bw) / self.sr,
                    'spec_rolloff': np.mean(spec_rolloff) / self.sr
                }
            except Exception as e:
                logger.warning(f"CPU feature extraction failed: {e}")
                return None
        
        # Parallel CPU extraction
        with ThreadPoolExecutor(max_workers=4) as executor:
            cpu_results = list(executor.map(extract_cpu_features, audio_segments))
        
        # Combine all features
        all_features = []
        for i in range(len(audio_segments)):
            cpu_feat = cpu_results[i]
            if cpu_feat is None:
                # Fallback to zeros
                cpu_feat = {
                    'chroma': np.zeros(12),
                    'contrast': np.zeros(7),
                    'zcr': 0, 'rms_mean': 0, 'rms_std': 0,
                    'tempo': 0, 'spec_bw': 0, 'spec_rolloff': 0
                }
            
            features = np.concatenate([
                # GPU features
                [energy_mean[i], energy_std[i], energy_max[i]],
                [spectral_centroid[i] / self.sr],  # Normalized
                mfcc_mean[i],
                mfcc_std[i],
                # CPU features
                cpu_feat['chroma'],
                cpu_feat['contrast'],
                [cpu_feat['zcr'], cpu_feat['rms_mean'], cpu_feat['rms_std']],
                [cpu_feat['tempo'], cpu_feat['spec_bw'], cpu_feat['spec_rolloff']]
            ])
            all_features.append(features)
        
        return all_features
    
    def extract_single(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from a single segment."""
        return self.extract_batch([audio])[0]


class CachedFeatureExtractor:
    """
    Feature extractor with disk caching.
    
    Caches extracted features to avoid re-computation.
    """
    
    def __init__(
        self,
        cache_dir: str = "./feature_cache",
        segment_duration: float = 10.0,
        hop_duration: float = 5.0,
        sr: int = 22050,
        use_gpu: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sr = sr
        
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        self.gpu_extractor = GPUFeatureExtractor(sr=sr, device=self.device)
        
        logger.info(f"CachedFeatureExtractor: cache={cache_dir}, device={self.device}")
    
    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key from file path and params."""
        path_hash = hashlib.md5(str(audio_path).encode()).hexdigest()[:12]
        param_str = f"{self.segment_duration}_{self.hop_duration}"
        return f"{path_hash}_{param_str}"
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path."""
        key = self._get_cache_key(audio_path)
        return self.cache_dir / f"{key}.pkl"
    
    def extract_all_segments(
        self,
        audio_path: str,
        force_recompute: bool = False
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract features for all segments in audio file.
        
        Uses cache if available.
        
        Returns:
            List of (start_time, end_time, features) tuples
        """
        cache_path = self._get_cache_path(audio_path)
        
        # Check cache
        if not force_recompute and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                logger.debug(f"Loaded from cache: {audio_path}")
                return cached
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(audio) / self.sr
        
        # Segment audio
        segments_audio = []
        segments_times = []
        
        start = 0.0
        while start + self.segment_duration <= duration:
            end = start + self.segment_duration
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            segments_audio.append(audio[start_sample:end_sample])
            segments_times.append((start, end))
            start += self.hop_duration
        
        # Batch extract features
        if segments_audio:
            logger.info(f"Extracting {len(segments_audio)} segments from {Path(audio_path).name}...")
            features_list = self.gpu_extractor.extract_batch(segments_audio)
        else:
            features_list = []
        
        # Combine
        results = [
            (start, end, features)
            for (start, end), features in zip(segments_times, features_list)
        ]
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            logger.debug(f"Cached: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        return results
    
    def clear_cache(self):
        """Clear all cached features."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        logger.info("Feature cache cleared")


def parallel_extract_features(
    audio_paths: List[str],
    segment_duration: float = 10.0,
    hop_duration: float = 5.0,
    cache_dir: str = "./feature_cache",
    max_workers: int = 4
) -> Dict[str, List[Tuple[float, float, np.ndarray]]]:
    """
    Extract features from multiple files in parallel.
    
    Uses a mix of GPU (for spectrograms) and multiprocessing (for files).
    """
    results = {}
    
    # Create extractor (will use GPU if available)
    extractor = CachedFeatureExtractor(
        cache_dir=cache_dir,
        segment_duration=segment_duration,
        hop_duration=hop_duration
    )
    
    # Process files (cache handles parallelism at segment level)
    for path in audio_paths:
        results[path] = extractor.extract_all_segments(path)
    
    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample file
    test_file = "training_data/input_mastered/20251018omen_raw_mastered.wav"
    
    if Path(test_file).exists():
        print(f"Testing GPU feature extraction on: {test_file}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print()
        
        # Test cached extractor
        extractor = CachedFeatureExtractor(
            cache_dir="./feature_cache",
            segment_duration=10.0,
            hop_duration=5.0
        )
        
        # First run (no cache)
        extractor.clear_cache()
        start_time = time.time()
        results = extractor.extract_all_segments(test_file, force_recompute=True)
        first_time = time.time() - start_time
        
        print(f"First extraction: {first_time:.2f}s for {len(results)} segments")
        print(f"Feature dimension: {len(results[0][2])}")
        
        # Second run (cached)
        start_time = time.time()
        results2 = extractor.extract_all_segments(test_file)
        cached_time = time.time() - start_time
        
        print(f"Cached load: {cached_time:.4f}s")
        print(f"Speedup: {first_time/cached_time:.1f}x")
    else:
        print(f"Test file not found: {test_file}")

"""Enhanced audio feature extraction for RL-based audio editor.

Provides rich spectral, rhythmic, and stem-based features for better
state representation and learning.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    
    # Basic spectral (always computed)
    use_onset_strength: bool = True
    use_rms: bool = True
    use_spectral_centroid: bool = True
    use_zcr: bool = True
    
    # Extended spectral features
    use_spectral_rolloff: bool = True
    use_spectral_bandwidth: bool = True
    use_spectral_flatness: bool = True
    use_spectral_contrast: bool = True
    n_contrast_bands: int = 6  # Number of spectral contrast bands
    
    # Timbral features
    use_mfcc: bool = True
    n_mfcc: int = 13  # Use coefficients 1-13 (skip 0 which is energy)
    use_mfcc_delta: bool = True  # First derivative
    
    # Harmonic features
    use_chroma: bool = True
    n_chroma: int = 12
    
    # Rhythmic features
    use_tempo_features: bool = True
    use_beat_phase: bool = True  # Position in bar (1-4 typically)
    
    # Temporal context (delta features)
    use_delta_features: bool = True
    
    # Stem features (requires Demucs)
    use_stem_features: bool = False
    stem_names: Tuple[str, ...] = ("drums", "bass", "vocals", "other")


class BeatFeatureExtractor:
    """Extract rich features per beat from audio."""
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        config: Optional[FeatureConfig] = None,
    ):
        """Initialize feature extractor.
        
        Args:
            sr: Sample rate
            hop_length: Hop length for STFT
            n_fft: FFT size
            n_mels: Number of mel bands
            config: Feature configuration
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.config = config or FeatureConfig()
        
    def extract_features(
        self,
        y: np.ndarray,
        beat_frames: np.ndarray,
        beat_times: np.ndarray,
        tempo: float,
    ) -> np.ndarray:
        """Extract all configured features for each beat.
        
        Args:
            y: Audio signal (mono)
            beat_frames: Beat positions in frames
            beat_times: Beat positions in seconds
            tempo: Estimated tempo in BPM
            
        Returns:
            Feature array of shape (n_beats, feature_dim)
        """
        n_beats = len(beat_frames)
        if n_beats == 0:
            return np.zeros((0, self.get_feature_dim()))
        
        # Compute frame-level features first
        frame_features = self._compute_frame_features(y)
        
        # Aggregate to beat level
        beat_features = self._aggregate_to_beats(
            frame_features, beat_frames, len(y) // self.hop_length + 1
        )
        
        # Add delta features BEFORE rhythmic (temporal derivatives of spectral features)
        if self.config.use_delta_features:
            delta = self._compute_delta_features(beat_features)
            beat_features = np.concatenate([beat_features, delta], axis=1)
        
        # Add rhythmic features AFTER delta (beat position, tempo features)
        if self.config.use_tempo_features or self.config.use_beat_phase:
            rhythmic = self._compute_rhythmic_features(
                beat_times, tempo, n_beats
            )
            beat_features = np.concatenate([beat_features, rhythmic], axis=1)
        
        return beat_features.astype(np.float32)
    
    def _compute_frame_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute frame-level features."""
        features = {}
        
        # Pre-compute STFT and power spectrum (reused by multiple features)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        S_power = S ** 2
        
        # === Basic spectral ===
        if self.config.use_onset_strength:
            features["onset"] = librosa.onset.onset_strength(
                y=y, sr=self.sr, hop_length=self.hop_length
            )
        
        if self.config.use_rms:
            features["rms"] = librosa.feature.rms(
                S=S, hop_length=self.hop_length
            )[0]
        
        if self.config.use_spectral_centroid:
            features["centroid"] = librosa.feature.spectral_centroid(
                S=S, sr=self.sr
            )[0]
        
        if self.config.use_zcr:
            features["zcr"] = librosa.feature.zero_crossing_rate(
                y, hop_length=self.hop_length
            )[0]
        
        # === Extended spectral ===
        if self.config.use_spectral_rolloff:
            features["rolloff"] = librosa.feature.spectral_rolloff(
                S=S, sr=self.sr
            )[0]
        
        if self.config.use_spectral_bandwidth:
            features["bandwidth"] = librosa.feature.spectral_bandwidth(
                S=S, sr=self.sr
            )[0]
        
        if self.config.use_spectral_flatness:
            features["flatness"] = librosa.feature.spectral_flatness(
                S=S_power
            )[0]
        
        if self.config.use_spectral_contrast:
            contrast = librosa.feature.spectral_contrast(
                S=S, sr=self.sr, n_bands=self.config.n_contrast_bands
            )
            # Shape: (n_bands + 1, n_frames) - store each band separately
            for i in range(contrast.shape[0]):
                features[f"contrast_{i}"] = contrast[i]
        
        # === Timbral (MFCCs) ===
        if self.config.use_mfcc:
            mfcc = librosa.feature.mfcc(
                S=librosa.power_to_db(S_power),
                sr=self.sr,
                n_mfcc=self.config.n_mfcc + 1  # +1 to skip coef 0
            )
            # Skip coefficient 0 (energy), use 1 to n_mfcc
            for i in range(1, self.config.n_mfcc + 1):
                features[f"mfcc_{i}"] = mfcc[i]
            
            if self.config.use_mfcc_delta:
                mfcc_delta = librosa.feature.delta(mfcc)
                for i in range(1, self.config.n_mfcc + 1):
                    features[f"mfcc_delta_{i}"] = mfcc_delta[i]
        
        # === Harmonic (Chroma) ===
        if self.config.use_chroma:
            chroma = librosa.feature.chroma_stft(
                S=S_power, sr=self.sr, n_chroma=self.config.n_chroma
            )
            for i in range(self.config.n_chroma):
                features[f"chroma_{i}"] = chroma[i]
        
        return features
    
    def _aggregate_to_beats(
        self,
        frame_features: Dict[str, np.ndarray],
        beat_frames: np.ndarray,
        n_frames: int,
    ) -> np.ndarray:
        """Aggregate frame-level features to beat-level using mean pooling.
        
        Args:
            frame_features: Dict of frame-level feature arrays
            beat_frames: Beat positions in frames
            n_frames: Total number of frames
            
        Returns:
            Beat-level features (n_beats, n_features)
        """
        n_beats = len(beat_frames)
        
        # Create beat boundaries
        boundaries = np.concatenate([[0], beat_frames, [n_frames]])
        
        # Aggregate each feature
        aggregated = []
        for name, feat in sorted(frame_features.items()):
            beat_feat = np.zeros(n_beats)
            for i in range(n_beats):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                if start >= len(feat):
                    start = len(feat) - 1
                if end > len(feat):
                    end = len(feat)
                if start >= end:
                    end = start + 1
                if end > len(feat):
                    beat_feat[i] = feat[-1] if len(feat) > 0 else 0.0
                else:
                    beat_feat[i] = np.mean(feat[start:end])
            aggregated.append(beat_feat)
        
        return np.stack(aggregated, axis=1) if aggregated else np.zeros((n_beats, 0))
    
    def _compute_rhythmic_features(
        self,
        beat_times: np.ndarray,
        tempo: float,
        n_beats: int,
    ) -> np.ndarray:
        """Compute rhythmic/positional features.
        
        Features:
        - Beat phase (0-1 position in assumed 4/4 bar)
        - Local tempo deviation
        - Inter-beat interval ratio
        """
        features = []
        
        if self.config.use_beat_phase:
            # Estimate bar position (assume 4/4 time)
            bar_length = 4
            phase = np.arange(n_beats) % bar_length / bar_length
            # Also add sine/cosine encoding for continuity
            phase_sin = np.sin(2 * np.pi * phase)
            phase_cos = np.cos(2 * np.pi * phase)
            features.extend([phase, phase_sin, phase_cos])
        
        if self.config.use_tempo_features:
            # Inter-beat intervals
            ibi = np.diff(beat_times, prepend=beat_times[0] if n_beats > 0 else 0)
            ibi = np.clip(ibi, 0.1, 2.0)  # Reasonable bounds
            
            # Expected interval from tempo
            expected_ibi = 60.0 / max(tempo, 60)
            
            # Tempo deviation (how much this beat deviates from expected)
            tempo_dev = (ibi - expected_ibi) / expected_ibi
            tempo_dev = np.clip(tempo_dev, -1.0, 1.0)
            
            # IBI ratio (current / previous) - rhythm stability
            ibi_ratio = np.ones(n_beats)
            if n_beats > 1:
                ibi_ratio[1:] = ibi[1:] / np.maximum(ibi[:-1], 0.1)
            ibi_ratio = np.clip(ibi_ratio, 0.5, 2.0)
            
            features.extend([tempo_dev, ibi_ratio])
        
        if features:
            return np.stack(features, axis=1)
        return np.zeros((n_beats, 0))
    
    def _compute_delta_features(
        self,
        beat_features: np.ndarray,
    ) -> np.ndarray:
        """Compute temporal delta (derivative) features.
        
        Delta = current - previous (first beat uses 0)
        """
        if len(beat_features) <= 1:
            return np.zeros_like(beat_features)
        
        delta = np.zeros_like(beat_features)
        delta[1:] = beat_features[1:] - beat_features[:-1]
        
        # Normalize deltas
        std = np.std(delta, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        delta = delta / std
        delta = np.clip(delta, -3.0, 3.0)
        
        return delta
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension."""
        dim = 0
        
        # Basic spectral (4)
        if self.config.use_onset_strength:
            dim += 1
        if self.config.use_rms:
            dim += 1
        if self.config.use_spectral_centroid:
            dim += 1
        if self.config.use_zcr:
            dim += 1
        
        # Extended spectral (4 + contrast bands)
        if self.config.use_spectral_rolloff:
            dim += 1
        if self.config.use_spectral_bandwidth:
            dim += 1
        if self.config.use_spectral_flatness:
            dim += 1
        if self.config.use_spectral_contrast:
            dim += self.config.n_contrast_bands + 1
        
        # MFCCs
        if self.config.use_mfcc:
            dim += self.config.n_mfcc  # Coefficients 1-13
            if self.config.use_mfcc_delta:
                dim += self.config.n_mfcc  # Delta coefficients
        
        # Chroma
        if self.config.use_chroma:
            dim += self.config.n_chroma
        
        # Rhythmic
        if self.config.use_beat_phase:
            dim += 3  # phase, sin, cos
        if self.config.use_tempo_features:
            dim += 2  # tempo_dev, ibi_ratio
        
        # Delta features (doubles the non-delta features)
        if self.config.use_delta_features:
            # Delta of everything computed before rhythmic features
            base_dim = dim
            if self.config.use_beat_phase:
                base_dim -= 3
            if self.config.use_tempo_features:
                base_dim -= 2
            dim += base_dim
        
        return dim
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        names = []
        
        if self.config.use_onset_strength:
            names.append("onset")
        if self.config.use_rms:
            names.append("rms")
        if self.config.use_spectral_centroid:
            names.append("centroid")
        if self.config.use_zcr:
            names.append("zcr")
        if self.config.use_spectral_rolloff:
            names.append("rolloff")
        if self.config.use_spectral_bandwidth:
            names.append("bandwidth")
        if self.config.use_spectral_flatness:
            names.append("flatness")
        if self.config.use_spectral_contrast:
            for i in range(self.config.n_contrast_bands + 1):
                names.append(f"contrast_{i}")
        if self.config.use_mfcc:
            for i in range(1, self.config.n_mfcc + 1):
                names.append(f"mfcc_{i}")
            if self.config.use_mfcc_delta:
                for i in range(1, self.config.n_mfcc + 1):
                    names.append(f"mfcc_delta_{i}")
        if self.config.use_chroma:
            for i in range(self.config.n_chroma):
                names.append(f"chroma_{i}")
        if self.config.use_beat_phase:
            names.extend(["beat_phase", "beat_phase_sin", "beat_phase_cos"])
        if self.config.use_tempo_features:
            names.extend(["tempo_dev", "ibi_ratio"])
        if self.config.use_delta_features:
            base_names = names.copy()
            if self.config.use_beat_phase:
                base_names = base_names[:-3]
            if self.config.use_tempo_features:
                base_names = base_names[:-2]
            for name in base_names:
                names.append(f"delta_{name}")
        
        return names


class StemFeatureExtractor:
    """Extract features from separated stems (drums, bass, vocals, other)."""
    
    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        stem_names: Tuple[str, ...] = ("drums", "bass", "vocals", "other"),
    ):
        """Initialize stem feature extractor.
        
        Args:
            sr: Sample rate
            hop_length: Hop length
            stem_names: Names of stems to process
        """
        self.sr = sr
        self.hop_length = hop_length
        self.stem_names = stem_names
    
    def extract_stem_features(
        self,
        stems: Dict[str, np.ndarray],
        beat_frames: np.ndarray,
        n_frames: int,
    ) -> np.ndarray:
        """Extract per-beat features from stems.
        
        For each stem, compute:
        - Energy (RMS)
        - Spectral centroid
        - Energy ratio vs total
        
        Args:
            stems: Dict mapping stem name to audio array
            beat_frames: Beat positions in frames
            n_frames: Total number of frames
            
        Returns:
            Features array of shape (n_beats, n_stem_features)
        """
        n_beats = len(beat_frames)
        if n_beats == 0:
            return np.zeros((0, len(self.stem_names) * 3))
        
        # Compute total energy for ratio
        total_energy = np.zeros(n_frames)
        stem_energies = {}
        stem_centroids = {}
        
        for stem_name in self.stem_names:
            if stem_name not in stems:
                stem_energies[stem_name] = np.zeros(n_frames)
                stem_centroids[stem_name] = np.zeros(n_frames)
                continue
            
            y_stem = stems[stem_name]
            rms = librosa.feature.rms(y=y_stem, hop_length=self.hop_length)[0]
            centroid = librosa.feature.spectral_centroid(
                y=y_stem, sr=self.sr, hop_length=self.hop_length
            )[0]
            
            # Pad/truncate to n_frames
            if len(rms) < n_frames:
                rms = np.pad(rms, (0, n_frames - len(rms)))
                centroid = np.pad(centroid, (0, n_frames - len(centroid)))
            elif len(rms) > n_frames:
                rms = rms[:n_frames]
                centroid = centroid[:n_frames]
            
            stem_energies[stem_name] = rms
            stem_centroids[stem_name] = centroid
            total_energy += rms ** 2
        
        total_energy = np.sqrt(np.maximum(total_energy, 1e-10))
        
        # Aggregate to beats
        boundaries = np.concatenate([[0], beat_frames, [n_frames]])
        features = []
        
        for stem_name in self.stem_names:
            energy = stem_energies[stem_name]
            centroid = stem_centroids[stem_name]
            
            beat_energy = np.zeros(n_beats)
            beat_centroid = np.zeros(n_beats)
            beat_ratio = np.zeros(n_beats)
            
            for i in range(n_beats):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                start = min(start, len(energy) - 1)
                end = min(end, len(energy))
                if start >= end:
                    end = start + 1
                if end > len(energy):
                    continue
                
                beat_energy[i] = np.mean(energy[start:end])
                beat_centroid[i] = np.mean(centroid[start:end])
                total_seg = np.mean(total_energy[start:end])
                beat_ratio[i] = beat_energy[i] / max(total_seg, 1e-10)
            
            features.extend([beat_energy, beat_centroid, beat_ratio])
        
        return np.stack(features, axis=1).astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """Get stem feature dimension."""
        return len(self.stem_names) * 3  # energy, centroid, ratio per stem
    
    def get_feature_names(self) -> List[str]:
        """Get stem feature names."""
        names = []
        for stem in self.stem_names:
            names.extend([
                f"{stem}_energy",
                f"{stem}_centroid",
                f"{stem}_ratio",
            ])
        return names


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to zero mean and unit variance.
    
    Args:
        features: Feature array (n_samples, n_features)
        mean: Precomputed mean (for test data)
        std: Precomputed std (for test data)
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    std = np.where(std < eps, 1.0, std)
    normalized = (features - mean) / std
    
    # Clip extreme values
    normalized = np.clip(normalized, -5.0, 5.0)
    
    return normalized, mean, std


def get_basic_feature_config() -> FeatureConfig:
    """Get config for basic features (fast, backward compatible)."""
    return FeatureConfig(
        use_onset_strength=True,
        use_rms=True,
        use_spectral_centroid=True,
        use_zcr=True,
        use_spectral_rolloff=False,
        use_spectral_bandwidth=False,
        use_spectral_flatness=False,
        use_spectral_contrast=False,
        use_mfcc=False,
        use_mfcc_delta=False,
        use_chroma=False,
        use_tempo_features=False,
        use_beat_phase=False,
        use_delta_features=False,
        use_stem_features=False,
    )


def get_enhanced_feature_config() -> FeatureConfig:
    """Get config for enhanced features (richer, slower)."""
    return FeatureConfig(
        use_onset_strength=True,
        use_rms=True,
        use_spectral_centroid=True,
        use_zcr=True,
        use_spectral_rolloff=True,
        use_spectral_bandwidth=True,
        use_spectral_flatness=True,
        use_spectral_contrast=True,
        n_contrast_bands=6,
        use_mfcc=True,
        n_mfcc=13,
        use_mfcc_delta=True,
        use_chroma=True,
        n_chroma=12,
        use_tempo_features=True,
        use_beat_phase=True,
        use_delta_features=True,
        use_stem_features=False,  # Still needs explicit stems
    )


def get_full_feature_config() -> FeatureConfig:
    """Get config for full features including stems."""
    config = get_enhanced_feature_config()
    config.use_stem_features = True
    return config


class StemProcessor:
    """Process audio stems using Demucs with caching support.
    
    Stems are separated once and cached to disk for fast subsequent access.
    This makes stem features practical for training.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        sr: int = 22050,
        model: str = "htdemucs",
        device: Optional[str] = None,
    ):
        """Initialize stem processor.
        
        Args:
            cache_dir: Directory to cache separated stems
            sr: Target sample rate for output stems
            model: Demucs model name (htdemucs, htdemucs_ft, etc.)
            device: Device for Demucs ('cuda' or 'cpu', auto-detected if None)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sr = sr
        self.model = model
        self.device = device
        self.stem_names = ("drums", "bass", "vocals", "other")
        
        # Lazy load demucs
        self._separator = None
        
    def _get_separator(self):
        """Lazy-load Demucs separator."""
        if self._separator is None:
            try:
                from shared.demucs_wrapper import DemucsSeparator
                self._separator = DemucsSeparator(model=self.model)
                logger.info(f"Initialized Demucs separator: {self.model}")
            except ImportError:
                logger.warning("Demucs not available - stem features will be zeros")
                return None
        return self._separator
    
    def _get_cache_path(self, audio_path: str) -> Optional[Path]:
        """Get cache path for stems."""
        if self.cache_dir is None:
            return None
        
        # Create unique cache key from file path
        audio_path = Path(audio_path)
        cache_name = f"{audio_path.stem}_stems.npz"
        return self.cache_dir / "stems" / cache_name
    
    def _load_cached_stems(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load cached stems if available."""
        cache_path = self._get_cache_path(audio_path)
        if cache_path is None or not cache_path.exists():
            return None
        
        try:
            data = np.load(cache_path)
            stems = {name: data[name] for name in self.stem_names if name in data}
            logger.debug(f"Loaded cached stems from {cache_path}")
            return stems
        except Exception as e:
            logger.warning(f"Failed to load cached stems: {e}")
            return None
    
    def _save_cached_stems(self, audio_path: str, stems: Dict[str, np.ndarray]) -> None:
        """Save stems to cache."""
        cache_path = self._get_cache_path(audio_path)
        if cache_path is None:
            return
        
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **stems)
            logger.debug(f"Saved stems to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache stems: {e}")
    
    def separate(
        self,
        audio_path: str,
        force_recompute: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Separate audio into stems.
        
        Args:
            audio_path: Path to audio file
            force_recompute: If True, ignore cache and recompute
            
        Returns:
            Dict mapping stem name to audio array, or None if separation fails
        """
        # Check cache first
        if not force_recompute:
            cached = self._load_cached_stems(audio_path)
            if cached is not None:
                return cached
        
        # Get separator
        separator = self._get_separator()
        if separator is None:
            return None
        
        try:
            logger.info(f"Running Demucs separation on {audio_path}...")
            stems = separator.separate(
                audio_path,
                resample_to=self.sr,
                device=self.device,
            )
            
            # Remove the _sr key if present
            stems.pop('_sr', None)
            
            # Convert to mono if stereo
            for name in list(stems.keys()):
                if stems[name].ndim > 1:
                    stems[name] = np.mean(stems[name], axis=-1)
                stems[name] = stems[name].astype(np.float32)
            
            # Cache results
            self._save_cached_stems(audio_path, stems)
            
            logger.info(f"Demucs separation complete: {list(stems.keys())}")
            return stems
            
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            return None
    
    def separate_array(
        self,
        audio: np.ndarray,
        sr: int,
        temp_name: str = "temp_audio",
    ) -> Optional[Dict[str, np.ndarray]]:
        """Separate audio array into stems.
        
        This creates a temporary file for Demucs processing.
        
        Args:
            audio: Audio array
            sr: Sample rate
            temp_name: Name for temporary file (for caching)
            
        Returns:
            Dict mapping stem name to audio array, or None if separation fails
        """
        import tempfile
        import soundfile as sf
        
        # Check if we have cached stems for this name
        if self.cache_dir:
            cache_path = self.cache_dir / "stems" / f"{temp_name}_stems.npz"
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    return {name: data[name] for name in self.stem_names if name in data}
                except Exception:
                    pass
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            sf.write(temp_path, audio, sr)
            stems = self.separate(temp_path, force_recompute=True)
            
            # Cache with the temp_name
            if stems and self.cache_dir:
                cache_path = self.cache_dir / "stems" / f"{temp_name}_stems.npz"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, **stems)
            
            return stems
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    def get_stem_features(
        self,
        stems: Dict[str, np.ndarray],
        beat_frames: np.ndarray,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Extract features from stems at beat level.
        
        Features per stem (4 stems × 3 features = 12 total):
        - Energy (RMS)
        - Spectral centroid
        - Energy ratio (stem / total)
        
        Args:
            stems: Dict mapping stem name to audio array
            beat_frames: Beat positions in frames
            hop_length: Hop length for feature extraction
            
        Returns:
            Feature array of shape (n_beats, 12)
        """
        extractor = StemFeatureExtractor(
            sr=self.sr,
            hop_length=hop_length,
            stem_names=self.stem_names,
        )
        
        n_frames = max(
            len(librosa.feature.rms(y=y, hop_length=hop_length)[0])
            for y in stems.values()
        ) if stems else 0
        
        return extractor.extract_stem_features(stems, beat_frames, n_frames)


def precache_stems_for_directory(
    audio_dir: str,
    cache_dir: str,
    sr: int = 22050,
    model: str = "htdemucs",
    extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac"),
) -> None:
    """Pre-compute and cache stems for all audio files in a directory.
    
    This is useful for preparing training data offline.
    
    Args:
        audio_dir: Directory containing audio files
        cache_dir: Directory to store cached stems
        sr: Sample rate for output stems
        model: Demucs model name
        extensions: Audio file extensions to process
    """
    from pathlib import Path
    import tqdm
    
    audio_dir = Path(audio_dir)
    processor = StemProcessor(cache_dir=cache_dir, sr=sr, model=model)
    
    # Find all audio files
    files = []
    for ext in extensions:
        files.extend(audio_dir.rglob(f"*{ext}"))
    
    logger.info(f"Pre-caching stems for {len(files)} files...")
    
    for audio_path in tqdm.tqdm(files, desc="Separating stems"):
        try:
            # This will cache automatically
            processor.separate(str(audio_path))
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
    
    logger.info("Stem pre-caching complete!")


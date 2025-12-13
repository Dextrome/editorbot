"""Inference utilities for the RL audio editor.

Provides utility functions for loading audio and creating edited output.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa

from rl_editor.config import Config
from rl_editor.state import AudioState

# Import feature extraction components used by training
try:
    from rl_editor.features import (
        BeatFeatureExtractor,
        get_enhanced_feature_config,
        get_basic_feature_config,
        StemProcessor,
    )
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    StemProcessor = None

# Import feature cache for loading pre-cached features
try:
    from rl_editor.cache import FeatureCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

logger = logging.getLogger(__name__)


def load_and_process_audio(
    audio_path: str, 
    config: Config,
    max_beats: int = 0,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, int, AudioState]:
    """Load audio and compute features matching the training pipeline.
    
    Uses the same feature extraction as training to ensure consistent
    input dimensions for the model.
    
    Args:
        audio_path: Path to audio file
        config: Configuration object (determines feature mode)
        max_beats: Maximum number of beats to process
        cache_dir: Directory to check for cached features
        
    Returns:
        Tuple of (audio_array, sample_rate, audio_state)
    """
    sr = config.audio.sample_rate
    audio_path = Path(audio_path)
    logger.info(f"Loading audio: {audio_path}")
    
    # Check if features are cached
    feature_cache = None
    cached_features = None
    if HAS_CACHE and cache_dir:
        feature_cache = FeatureCache(cache_dir=Path(cache_dir), enabled=True)
        cached_features = feature_cache.load_full(audio_path)
    
    if cached_features is not None:
        logger.info("Using cached features")
        beat_times = cached_features["beat_times"]
        beat_features = cached_features["beat_features"]
        tempo = float(cached_features["tempo"])
        
        # Still need to load audio for output generation
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    else:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        duration = len(y) / sr
        logger.info(f"Audio duration: {duration:.2f}s")
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        logger.info(f"Detected {len(beat_times)} beats at tempo {tempo:.1f} BPM")
        
        # Use enhanced feature extractor if available (matching training)
        if HAS_ENHANCED_FEATURES:
            feature_mode = getattr(config.features, 'feature_mode', 'basic') if hasattr(config, 'features') else 'basic'
            
            if feature_mode == "full":
                feat_config = get_enhanced_feature_config()
                feat_config.use_stem_features = True
            elif feature_mode == "enhanced":
                feat_config = get_enhanced_feature_config()
            else:
                feat_config = get_basic_feature_config()
            
            feature_extractor = BeatFeatureExtractor(
                sr=sr,
                hop_length=config.audio.hop_length,
                n_fft=config.audio.n_fft,
                n_mels=config.audio.n_mels,
                config=feat_config,
            )
            
            logger.info(f"Using {feature_mode} features: {feature_extractor.get_feature_dim()} dims")
            beat_features = feature_extractor.extract_features(y, beats, beat_times, tempo)
            
            # If full mode, also add stem features (requires Demucs separation)
            if feature_mode == "full" and config.data.use_stems:
                logger.info("Extracting stem features (this may take a moment)...")
                try:
                    stem_processor = StemProcessor(
                        cache_dir=Path(cache_dir) if cache_dir else None,
                        sr=sr,
                    )
                    
                    # Check for cached stems first
                    cached_stems = None
                    if feature_cache:
                        cached_stems = feature_cache.load_stems(audio_path)
                    
                    if cached_stems is not None:
                        stems = cached_stems
                        logger.info("Using cached stems")
                    else:
                        stems = stem_processor.separate(str(audio_path))
                        # Cache for future use
                        if feature_cache and stems:
                            feature_cache.save_stems(audio_path, stems)
                    
                    if stems:
                        stem_features = stem_processor.get_stem_features(
                            stems, beats, hop_length=config.audio.hop_length
                        )
                        logger.info(f"Stem features shape: {stem_features.shape}")
                        beat_features = np.concatenate([beat_features, stem_features], axis=1)
                    else:
                        logger.warning("Could not extract stems - padding with zeros")
                        # Pad with zeros for stem features (12 dims: 4 stems × 3 features)
                        stem_padding = np.zeros((len(beat_features), 12))
                        beat_features = np.concatenate([beat_features, stem_padding], axis=1)
                except Exception as e:
                    logger.warning(f"Stem extraction failed: {e} - padding with zeros")
                    stem_padding = np.zeros((len(beat_features), 12))
                    beat_features = np.concatenate([beat_features, stem_padding], axis=1)
        else:
            # Fall back to basic features
            logger.warning("Enhanced features not available - using basic features")
            beat_features = _extract_basic_features(y, sr, beats, beat_times)
    
    logger.info(f"Beat features shape: {beat_features.shape}")
    
    # Truncate to max_beats
    if max_beats > 0 and len(beat_times) > max_beats:
        logger.warning(f"Truncating from {len(beat_times)} to {max_beats} beats")
        beat_times = beat_times[:max_beats]
        beat_features = beat_features[:max_beats]
    
    # Normalize features (per-feature normalization)
    mean = beat_features.mean(axis=0, keepdims=True)
    std = beat_features.std(axis=0, keepdims=True) + 1e-8
    beat_features = (beat_features - mean) / std
    
    # Create AudioState
    audio_state = AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=float(tempo),
    )
    
    return y, sr, audio_state


def _extract_basic_features(y: np.ndarray, sr: int, beats: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Extract basic 4-dimensional beat features (fallback)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    beat_features = []
    frames = librosa.time_to_frames(beat_times, sr=sr)
    frames = np.concatenate(([0], frames, [len(onset_env)]))
    
    for i in range(len(beats)):
        start = frames[i]
        end = frames[i+1]
        if start >= len(onset_env):
            break
        if end > len(onset_env):
            end = len(onset_env)
        if start == end:
            end = start + 1
            
        b_onset = np.mean(onset_env[start:end])
        b_centroid = np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))])
        b_zcr = np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))])
        b_rms = np.mean(rms[min(start, len(rms)-1):min(end, len(rms))])
        beat_features.append([b_onset, b_centroid, b_zcr, b_rms])
    
    return np.array(beat_features) if beat_features else np.zeros((0, 4))


def apply_crossfade(seg1: np.ndarray, seg2: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """Apply crossfade between two audio segments.
    
    Args:
        seg1: First segment (fade out at end)
        seg2: Second segment (fade in at start)
        crossfade_samples: Number of samples for crossfade
        
    Returns:
        Combined audio with crossfade
    """
    if crossfade_samples <= 0:
        return np.concatenate([seg1, seg2])
    
    # Ensure we have enough samples
    actual_fade = min(crossfade_samples, len(seg1), len(seg2))
    
    if actual_fade < 10:  # Too short for meaningful crossfade
        return np.concatenate([seg1, seg2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, actual_fade)
    fade_in = np.linspace(0.0, 1.0, actual_fade)
    
    # Copy segments to avoid modifying originals
    result = np.zeros(len(seg1) + len(seg2) - actual_fade)
    result[:len(seg1) - actual_fade] = seg1[:-actual_fade]
    
    # Apply crossfade in overlap region
    overlap_start = len(seg1) - actual_fade
    result[overlap_start:overlap_start + actual_fade] = (
        seg1[-actual_fade:] * fade_out + seg2[:actual_fade] * fade_in
    )
    
    # Add rest of second segment
    result[overlap_start + actual_fade:] = seg2[actual_fade:]
    
    return result

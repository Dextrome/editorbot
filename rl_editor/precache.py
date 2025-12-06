#!/usr/bin/env python
"""Pre-cache all features for faster training.

This script processes all audio files in the training data directory and
caches beat features, mel spectrograms, stems (if enabled), and edit labels.

Usage:
    python -m rl_editor.precache --data_dir ./training_data
    python -m rl_editor.precache --data_dir ./training_data --stems  # Include Demucs stems
    python -m rl_editor.precache --data_dir ./training_data --feature_mode full  # All features

Example:
    # Basic features only (fast)
    python -m rl_editor.precache --data_dir ./training_data --feature_mode basic
    
    # Enhanced features (spectral, MFCCs, chroma, rhythmic)
    python -m rl_editor.precache --data_dir ./training_data --feature_mode enhanced
    
    # Full features including stem separation (slow, requires GPU)
    python -m rl_editor.precache --data_dir ./training_data --feature_mode full --stems
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import torch

from rl_editor.config import Config
from rl_editor.cache import FeatureCache, DEFAULT_CACHE_DIR
from rl_editor.utils import load_audio, compute_mel_spectrogram

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from rl_editor.features import (
        BeatFeatureExtractor,
        StemProcessor,
        get_basic_feature_config,
        get_enhanced_feature_config,
    )
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    logger.warning("Enhanced features module not available")

try:
    from shared.demucs_wrapper import DemucsSeparator
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False
    logger.warning("Demucs not available - stem separation disabled")


def find_audio_files(data_dir: Path, subdirs: List[str]) -> List[Path]:
    """Find all audio files in data directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []
    
    for subdir in subdirs:
        target = data_dir / subdir
        if target.exists():
            for p in target.rglob("*"):
                if p.suffix.lower() in extensions:
                    files.append(p)
    
    return sorted(set(files))


def find_pairs(data_dir: Path, raw_subdir: str, edited_subdir: str) -> List[Tuple[Path, Path]]:
    """Find matching raw/edited audio pairs."""
    input_dir = data_dir / raw_subdir
    output_dir = data_dir / edited_subdir
    
    if not input_dir.exists() or not output_dir.exists():
        return []
    
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    
    # Build mapping from base name to input file
    input_files = {}
    for p in input_dir.iterdir():
        if p.suffix.lower() in extensions:
            base_name = p.stem
            if base_name.endswith("_raw"):
                base_name = base_name[:-4]
            input_files[base_name] = p
    
    # Find matching output files
    pairs = []
    for p in output_dir.iterdir():
        if p.suffix.lower() in extensions:
            base_name = p.stem
            if base_name.endswith("_edit"):
                base_name = base_name[:-5]
            
            if base_name in input_files:
                pairs.append((input_files[base_name], p))
    
    return pairs


def cache_file_features(
    audio_path: Path,
    cache: FeatureCache,
    feature_extractor: Optional["BeatFeatureExtractor"],
    config: Config,
    stem_processor: Optional["StemProcessor"] = None,
    force: bool = False,
) -> Dict:
    """Cache features for a single audio file, including stem features if available.
    
    Args:
        audio_path: Path to audio file
        cache: Feature cache instance
        feature_extractor: Beat feature extractor
        config: Config instance
        stem_processor: Optional stem processor for Demucs features
        force: Force recomputation even if cached
    
    Returns:
        Dict with cached data summary
    """
    # Check if already cached
    if not force and cache.has_cached(audio_path, "features"):
        return {"status": "cached", "path": str(audio_path)}
    
    try:
        # Load audio
        y, sr = load_audio(str(audio_path), sr=config.audio.sample_rate)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Extract features
        if feature_extractor is not None and len(beats) > 0:
            beat_features = feature_extractor.extract_features(
                y, beats, beat_times, tempo
            )
        else:
            # Basic features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            
            frames = librosa.time_to_frames(beat_times, sr=sr)
            frames = np.concatenate(([0], frames, [len(onset_env)]))
            
            beat_features = []
            for i in range(len(beats)):
                start, end = frames[i], frames[i + 1]
                if start >= len(onset_env):
                    break
                end = min(end, len(onset_env))
                if start == end:
                    end = start + 1
                
                beat_features.append([
                    np.mean(onset_env[start:end]),
                    np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))]),
                    np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))]),
                    np.mean(rms[min(start, len(rms)-1):min(end, len(rms))]),
                ])
            
            beat_features = np.array(beat_features) if beat_features else np.zeros((0, 4))
        
        # Save features
        cache.save_features(
            audio_path,
            beat_features=beat_features,
            beat_times=beat_times,
            beats=beats,
            tempo=float(tempo) if hasattr(tempo, '__float__') else tempo,
        )
        
        # Compute and cache mel spectrogram
        mel = compute_mel_spectrogram(
            y, sr=sr,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
        )
        cache.save_mel(audio_path, mel)
        
        # Add stem features if stem processor available
        if stem_processor is not None and len(beats) > 0:
            # Try to load cached stems first
            stems = cache.load_stems(audio_path)
            if stems is None:
                # Separate stems (expensive, but will be cached)
                stems = stem_processor.separate(str(audio_path))
                if stems:
                    cache.save_stems(audio_path, stems)
            
            if stems:
                stem_features = stem_processor.get_stem_features(
                    stems, beats, hop_length=config.audio.hop_length
                )
                # Concatenate base features with stem features
                beat_features = np.concatenate([beat_features, stem_features], axis=1)
                
                # Re-save with full features
                cache.save_features(
                    audio_path,
                    beat_features=beat_features,
                    beat_times=beat_times,
                    beats=beats,
                    tempo=float(tempo) if hasattr(tempo, '__float__') else tempo,
                )
        
        return {
            "status": "computed",
            "path": str(audio_path),
            "n_beats": len(beats),
            "duration": len(y) / sr,
            "feature_dim": beat_features.shape[-1] if beat_features.ndim > 1 else 0,
        }
        
    except Exception as e:
        logger.error(f"Error caching features for {audio_path}: {e}")
        return {"status": "error", "path": str(audio_path), "error": str(e)}


def cache_stems(
    audio_path: Path,
    cache: FeatureCache,
    stem_processor: Optional["StemProcessor"],
    force: bool = False,
) -> Dict:
    """Cache Demucs stems for a single audio file.
    
    Returns:
        Dict with status
    """
    if stem_processor is None:
        return {"status": "skipped", "path": str(audio_path), "reason": "no stem processor"}
    
    # Check if already cached
    if not force and cache.has_cached(audio_path, "stems"):
        return {"status": "cached", "path": str(audio_path)}
    
    try:
        # Run Demucs
        stems = stem_processor.separate(str(audio_path))
        
        if stems:
            cache.save_stems(audio_path, stems)
            return {
                "status": "computed",
                "path": str(audio_path),
                "stems": list(stems.keys()),
            }
        else:
            return {"status": "error", "path": str(audio_path), "error": "separation failed"}
            
    except Exception as e:
        logger.error(f"Error caching stems for {audio_path}: {e}")
        return {"status": "error", "path": str(audio_path), "error": str(e)}


def cache_pair_labels(
    raw_path: Path,
    edited_path: Path,
    cache: FeatureCache,
    config: Config,
    force: bool = False,
) -> Dict:
    """Cache edit labels for a raw/edited pair.
    
    Note: This requires computing labels which depends on alignment,
    so we delegate to the data module's alignment logic.
    """
    # Check if already cached
    if not force and cache.load_labels(raw_path, edited_path) is not None:
        return {"status": "cached", "raw": str(raw_path), "edited": str(edited_path)}
    
    try:
        from rl_editor.data import PairedAudioDataset
        
        # Load cached features
        raw_features = cache.load_features(raw_path)
        edited_features = cache.load_features(edited_path)
        
        if raw_features is None or edited_features is None:
            return {
                "status": "skipped",
                "raw": str(raw_path),
                "reason": "features not cached yet",
            }
        
        # Use alignment to compute labels
        # This is a simplified DTW-based alignment
        raw_beat_times = raw_features["beat_times"]
        edited_beat_times = edited_features["beat_times"]
        
        # Simple ratio-based labeling
        n_raw = len(raw_beat_times)
        n_edited = len(edited_beat_times)
        
        if n_raw == 0:
            labels = np.array([], dtype=np.float32)
        elif n_edited == 0:
            labels = np.zeros(n_raw, dtype=np.float32)
        else:
            # Basic alignment: assume linear mapping
            # In production, use DTW for better alignment
            keep_ratio = min(1.0, n_edited / n_raw)
            labels = np.zeros(n_raw, dtype=np.float32)
            
            # Mark approximately keep_ratio beats as kept
            n_keep = int(n_raw * keep_ratio)
            keep_indices = np.linspace(0, n_raw - 1, n_keep, dtype=int)
            labels[keep_indices] = 1.0
        
        cache.save_labels(raw_path, edited_path, labels)
        
        return {
            "status": "computed",
            "raw": str(raw_path),
            "edited": str(edited_path),
            "n_beats": len(labels),
            "keep_ratio": float(labels.mean()) if len(labels) > 0 else 0.0,
        }
        
    except Exception as e:
        logger.error(f"Error caching labels for {raw_path}: {e}")
        return {"status": "error", "raw": str(raw_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache audio features for RL editor training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory (default: ./rl_editor/cache)",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        choices=["basic", "enhanced", "full"],
        default="enhanced",
        help="Feature extraction mode",
    )
    parser.add_argument(
        "--stems",
        action="store_true",
        help="Also cache Demucs stem separation (slow!)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (stems require GPU, use 1)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Skip caching edit labels",
    )
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    cache = FeatureCache(cache_dir=args.cache_dir, enabled=True)
    config = Config()
    
    # Setup feature extractor
    feature_extractor = None
    stem_processor = None
    
    if HAS_ENHANCED_FEATURES:
        if args.feature_mode == "enhanced":
            feat_config = get_enhanced_feature_config()
        elif args.feature_mode == "full":
            feat_config = get_enhanced_feature_config()
            feat_config.use_stem_features = True
        else:
            feat_config = get_basic_feature_config()
        
        feature_extractor = BeatFeatureExtractor(
            sr=config.audio.sample_rate,
            hop_length=config.audio.hop_length,
            n_fft=config.audio.n_fft,
            n_mels=config.audio.n_mels,
            config=feat_config,
        )
        logger.info(f"Feature mode: {args.feature_mode} ({feature_extractor.get_feature_dim()} dims)")
    
    # Setup stem processor if requested
    if args.stems and HAS_DEMUCS:
        stem_processor = StemProcessor(
            cache_dir=args.cache_dir,
            sr=config.audio.sample_rate,
        )
        logger.info("Stem separation enabled (Demucs)")
    elif args.stems:
        logger.warning("Stem separation requested but Demucs not available")
    
    # Find all audio files
    subdirs = [config.data.raw_subdir, config.data.edited_subdir, config.data.reference_subdir]
    all_files = find_audio_files(data_dir, subdirs)
    logger.info(f"Found {len(all_files)} audio files to process")
    
    # Find pairs for label caching
    pairs = find_pairs(data_dir, config.data.raw_subdir, config.data.edited_subdir)
    logger.info(f"Found {len(pairs)} raw/edited pairs")
    
    # === Phase 1: Cache Beat Features ===
    logger.info("\n=== Phase 1: Caching Beat Features ===")
    start_time = time.time()
    
    computed = 0
    cached = 0
    errors = 0
    
    for i, audio_path in enumerate(all_files):
        result = cache_file_features(
            audio_path, cache, feature_extractor, config, 
            stem_processor=stem_processor if args.stems else None,
            force=args.force
        )
        
        if result["status"] == "computed":
            computed += 1
            logger.info(
                f"[{i+1}/{len(all_files)}] Computed: {audio_path.name} "
                f"({result.get('n_beats', 0)} beats, {result.get('duration', 0):.1f}s, {result.get('feature_dim', 0)} dims)"
            )
        elif result["status"] == "cached":
            cached += 1
            if (i + 1) % 10 == 0:
                logger.info(f"[{i+1}/{len(all_files)}] Cached: {audio_path.name}")
        else:
            errors += 1
            logger.error(f"[{i+1}/{len(all_files)}] Error: {audio_path.name}")
    
    elapsed = time.time() - start_time
    logger.info(f"Features: {computed} computed, {cached} cached, {errors} errors ({elapsed:.1f}s)")
    
    # === Phase 2: Cache Stems (if enabled) ===
    if args.stems and stem_processor:
        logger.info("\n=== Phase 2: Caching Stems (Demucs) ===")
        start_time = time.time()
        
        computed = 0
        cached = 0
        errors = 0
        
        for i, audio_path in enumerate(all_files):
            result = cache_stems(audio_path, cache, stem_processor, force=args.force)
            
            if result["status"] == "computed":
                computed += 1
                logger.info(f"[{i+1}/{len(all_files)}] Separated: {audio_path.name}")
            elif result["status"] == "cached":
                cached += 1
            else:
                errors += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Stems: {computed} computed, {cached} cached, {errors} errors ({elapsed:.1f}s)")
    
    # === Phase 3: Cache Edit Labels ===
    if not args.no_labels and pairs:
        logger.info("\n=== Phase 3: Caching Edit Labels ===")
        start_time = time.time()
        
        computed = 0
        cached = 0
        errors = 0
        
        for i, (raw_path, edited_path) in enumerate(pairs):
            result = cache_pair_labels(
                raw_path, edited_path, cache, config, force=args.force
            )
            
            if result["status"] == "computed":
                computed += 1
                logger.info(
                    f"[{i+1}/{len(pairs)}] Labels: {raw_path.name} "
                    f"(keep ratio: {result.get('keep_ratio', 0):.1%})"
                )
            elif result["status"] == "cached":
                cached += 1
            else:
                errors += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Labels: {computed} computed, {cached} cached, {errors} errors ({elapsed:.1f}s)")
    
    # === Summary ===
    stats = cache.get_stats()
    logger.info("\n=== Cache Summary ===")
    logger.info(f"Cache directory: {stats['cache_dir']}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Total size: {stats['total_size_mb']:.1f} MB")
    for subdir, info in stats['subdirs'].items():
        logger.info(f"  {subdir}: {info['files']} files ({info['size_mb']:.1f} MB)")
    
    logger.info("\nPre-caching complete!")


if __name__ == "__main__":
    main()

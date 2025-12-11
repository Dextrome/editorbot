"""Data loading and processing for RL editor."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

from rl_editor.config import Config
from rl_editor.utils import load_audio, compute_mel_spectrogram, clear_audio_cache, get_audio_cache_stats

# Import centralized cache system
try:
    from rl_editor.cache import FeatureCache, get_cache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False
    FeatureCache = None
    get_cache = None

# Import enhanced features and augmentation
try:
    from rl_editor.features import (
        BeatFeatureExtractor,
        FeatureConfig,
        StemProcessor,
        StemFeatureExtractor,
        get_basic_feature_config,
        get_enhanced_feature_config,
        normalize_features,
    )
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    logging.warning("Enhanced features module not available.")

try:
    from rl_editor.augmentation import (
        AudioAugmentor,
        AugmentationConfig,
        get_default_augmentation_config,
    )
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False
    logging.warning("Augmentation module not available.")

# Assuming shared is importable from root
try:
    from shared.demucs_wrapper import DemucsSeparator
    HAS_DEMUCS = True
except ImportError:
    logging.warning("Could not import DemucsSeparator. Stem separation will not be available.")
    DemucsSeparator = None
    HAS_DEMUCS = False

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset for loading audio files and their features."""

    def __init__(
        self,
        data_dir: str,
        config: Config,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[callable] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing audio files
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            cache_dir: Directory to cache computed features
            transform: Optional transform to apply to features
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform

        # Find all audio files
        self.file_list = self._find_files()
        logger.info(f"Found {len(self.file_list)} files in {data_dir} ({split})")

        # Initialize separators/extractors if needed
        self.demucs = DemucsSeparator() if DemucsSeparator else None

    def _find_files(self) -> List[Path]:
        """Find audio files in data directory."""
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        files = []
        # Simple recursive search
        for p in self.data_dir.rglob("*"):
            if p.suffix.lower() in extensions:
                files.append(p)
        return sorted(files)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get item from dataset.

        Returns:
            Dictionary containing:
            - audio: Raw audio tensor
            - stems: Stem tensors (if available)
            - mel: Mel spectrogram
            - beats: Beat frames
            - path: File path
        """
        file_path = self.file_list[idx]
        #import time
        #t0 = time.perf_counter()
        
        # Check cache first
        if self.cache_dir:
            cached_data = self._load_from_cache(file_path)
            if cached_data:
                return cached_data

        # Load and process
        try:
            data = self._process_file(file_path)
            
            # Save to cache if enabled
            if self.cache_dir:
                self._save_to_cache(file_path, data)

            #t1 = time.perf_counter()
            #print(f"__getitem__({idx}) took {t1-t0:.4f}s")                
            return data
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            # Return a dummy item or raise (depending on robustness needs)
            # For now, raise to be visible
            raise e

    def _process_file(self, file_path: Path) -> Dict[str, Union[torch.Tensor, str]]:
        """Process a single audio file."""
        # Load audio
        y, sr = load_audio(str(file_path), sr=self.config.audio.sample_rate)
        
        # Compute Mel Spectrogram
        mel = compute_mel_spectrogram(
            y, 
            sr=sr, 
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length
        )

        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Compute beat features (onset, centroid, zcr, rms)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # Synchronize features to beats
        # We need to aggregate features between beats
        beat_features = []
        frames = librosa.time_to_frames(beat_times, sr=sr)
        frames = np.concatenate(([0], frames, [len(onset_env)]))
        
        for i in range(len(beats)):
            start = frames[i]
            end = frames[i+1]
            if start >= len(onset_env): break
            if end > len(onset_env): end = len(onset_env)
            if start == end: end = start + 1
            
            b_onset = np.mean(onset_env[start:end])
            b_centroid = np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))])
            b_zcr = np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))])
            b_rms = np.mean(rms[min(start, len(rms)-1):min(end, len(rms))])
            beat_features.append([b_onset, b_centroid, b_zcr, b_rms])
            
        beat_features = np.array(beat_features) if beat_features else np.zeros((0, 4))
        # Ensure length matches beats
        if len(beat_features) < len(beats):
            # Pad with last value
            pad_width = len(beats) - len(beat_features)
            beat_features = np.pad(beat_features, ((0, pad_width), (0, 0)), mode='edge')
        elif len(beat_features) > len(beats):
            beat_features = beat_features[:len(beats)]

        # Stem separation (if enabled and available)
        stems = None
        if self.config.data.use_stems and self.demucs:
            # This is slow! Should be done offline/cached.
            # For on-the-fly, we might skip or warn.
            # Here we assume it's cached or we do it (very slow)
            # For now, let's return a placeholder or implement it if critical
            # Real implementation would call self.demucs.separate(...)
            pass

        # Convert to tensors
        data = {
            "audio": torch.from_numpy(y).float(),
            "mel": torch.from_numpy(mel).float(),
            "beats": torch.from_numpy(beats).long(),
            "beat_times": torch.from_numpy(beat_times).float(),
            "beat_features": torch.from_numpy(beat_features).float(),
            "tempo": torch.tensor(tempo).float(),
            "path": str(file_path),
            "duration": torch.tensor(len(y) / sr).float()
        }
        
        if stems is not None:
            data["stems"] = torch.from_numpy(stems).float()

        return data

    def _get_cache_path(self, file_path: Path) -> Path:
        """Get cache path for a file."""
        if not self.cache_dir:
            return None
        # Create a unique name based on path hash or relative path
        rel_path = file_path.relative_to(self.data_dir)
        cache_path = self.cache_dir / rel_path.with_suffix(".pt")
        return cache_path

    def _load_from_cache(self, file_path: Path) -> Optional[Dict]:
        """Load processed data from cache."""
        cache_path = self._get_cache_path(file_path)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_path}: {e}")
        return None

    def _save_to_cache(self, file_path: Path, data: Dict) -> None:
        """Save processed data to cache."""
        cache_path = self._get_cache_path(file_path)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache_path)


class PairedAudioDataset(Dataset):
    """Dataset for loading paired raw/edited audio files for imitation learning.
    
    Expected structure:
        data_dir/
            input/           # Raw audio files (*_raw.wav, *_raw.mp3, etc.)
            desired_output/  # Edited audio files (*_edit.wav, etc.)
            reference/       # Optional: additional finished tracks (no pairs)
    
    Files are matched by stripping '_raw' and '_edit' suffixes.
    Reference tracks are included as "self-paired" (the track is both input and output,
    with all beats labeled as KEEP - teaching the agent what good finished music sounds like).
    """

    def __init__(
        self,
        data_dir: str,
        config: Config,
        cache_dir: Optional[str] = None,
        transform: Optional[callable] = None,
        include_reference: bool = True,
        use_augmentation: bool = False,
    ):
        """Initialize paired dataset.

        Args:
            data_dir: Directory containing input/ and desired_output/ subdirs
            config: Configuration object
            cache_dir: Directory to cache computed features (default from config)
            transform: Optional transform to apply to features
            include_reference: Whether to include reference tracks (default True)
            use_augmentation: Whether to apply data augmentation (default False, enable for training)
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.transform = transform
        self.include_reference = include_reference
        self.use_augmentation = use_augmentation

        # Use centralized cache system
        self.cache_dir = Path(cache_dir) if cache_dir else Path(config.data.cache_dir)
        self.feature_cache = None
        if HAS_CACHE and config.data.cache_features:
            self.feature_cache = FeatureCache(cache_dir=self.cache_dir, enabled=True)
            logger.info(f"Feature cache enabled: {self.cache_dir}")

        # Get subdir names from config
        self.raw_subdir = config.data.raw_subdir
        self.edited_subdir = config.data.edited_subdir
        self.reference_subdir = config.data.reference_subdir

        # Find and match pairs
        self.pairs = self._find_pairs()
        self.reference_files = self._find_reference() if include_reference else []
        
        logger.info(f"Found {len(self.pairs)} raw/edited pairs in {data_dir}")
        if self.reference_files:
            logger.info(f"Found {len(self.reference_files)} reference tracks")

        # Initialize separators/extractors if needed
        self.demucs = DemucsSeparator() if DemucsSeparator else None
        
        # Initialize enhanced feature extractor
        self.feature_extractor = None
        if HAS_ENHANCED_FEATURES:
            feature_mode = getattr(config.features, 'feature_mode', 'basic') if hasattr(config, 'features') else 'basic'
            if feature_mode == "enhanced":
                feat_config = get_enhanced_feature_config()
            elif feature_mode == "full":
                feat_config = get_enhanced_feature_config()
                feat_config.use_stem_features = True
            else:
                feat_config = get_basic_feature_config()
            
            self.feature_extractor = BeatFeatureExtractor(
                sr=config.audio.sample_rate,
                hop_length=config.audio.hop_length,
                n_fft=config.audio.n_fft,
                n_mels=config.audio.n_mels,
                config=feat_config,
            )
            logger.info(f"Using {feature_mode} features: {self.feature_extractor.get_feature_dim()} dims")
        
        # Initialize augmentor
        self.augmentor = None
        if use_augmentation and HAS_AUGMENTATION:
            aug_cfg = config.augmentation if hasattr(config, 'augmentation') else None
            if aug_cfg and aug_cfg.enabled:
                self.augmentor = AudioAugmentor(
                    sr=config.audio.sample_rate,
                    config=AugmentationConfig(
                        enabled=aug_cfg.enabled,
                        pitch_shift_enabled=aug_cfg.pitch_shift_enabled,
                        pitch_shift_range=(aug_cfg.pitch_shift_min, aug_cfg.pitch_shift_max),
                        pitch_shift_prob=aug_cfg.pitch_shift_prob,
                        time_stretch_enabled=aug_cfg.time_stretch_enabled,
                        time_stretch_range=(aug_cfg.time_stretch_min, aug_cfg.time_stretch_max),
                        time_stretch_prob=aug_cfg.time_stretch_prob,
                        noise_enabled=aug_cfg.noise_enabled,
                        noise_snr_range=(aug_cfg.noise_snr_min, aug_cfg.noise_snr_max),
                        noise_prob=aug_cfg.noise_prob,
                        gain_enabled=aug_cfg.gain_enabled,
                        gain_range=(aug_cfg.gain_min, aug_cfg.gain_max),
                        gain_prob=aug_cfg.gain_prob,
                        eq_enabled=aug_cfg.eq_enabled,
                        eq_gain_range=(aug_cfg.eq_gain_min, aug_cfg.eq_gain_max),
                        eq_prob=aug_cfg.eq_prob,
                        augment_prob=aug_cfg.augment_prob,
                        max_augments=aug_cfg.max_augments,
                    ),
                )
                logger.info("Data augmentation enabled")
        
        # Initialize stem processor for Demucs separation
        self.stem_processor = None
        feature_mode = getattr(config.features, 'feature_mode', 'basic') if hasattr(config, 'features') else 'basic'
        use_stems = config.data.use_stems or feature_mode == "full"
        
        if use_stems and HAS_ENHANCED_FEATURES and HAS_DEMUCS:
            stem_cache_dir = config.data.cache_dir if config.data.cache_features else None
            self.stem_processor = StemProcessor(
                cache_dir=stem_cache_dir,
                sr=config.audio.sample_rate,
            )
            logger.info("Stem separation enabled (Demucs)")
        
        # Pre-load all audio into memory cache for faster training
        self._preload_audio_cache()

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edited audio pairs."""
        input_dir = self.data_dir / self.raw_subdir
        output_dir = self.data_dir / self.edited_subdir
        
        if not input_dir.exists():
            logger.warning(f"Input directory not found: {input_dir}")
            return []
        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            return []

        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        
        # Build mapping from base name to input file
        input_files = {}
        for p in input_dir.iterdir():
            if p.suffix.lower() in extensions:
                # Strip _raw suffix to get base name
                base_name = p.stem
                if base_name.endswith("_raw"):
                    base_name = base_name[:-4]
                input_files[base_name] = p

        # Find matching output files
        pairs = []
        for p in output_dir.iterdir():
            if p.suffix.lower() in extensions:
                # Strip _edit suffix to get base name
                base_name = p.stem
                if base_name.endswith("_edit"):
                    base_name = base_name[:-5]
                
                if base_name in input_files:
                    pairs.append((input_files[base_name], p))
                else:
                    logger.warning(f"No matching input for output: {p}")

        return sorted(pairs, key=lambda x: x[0].stem)

    def _find_reference(self) -> List[Path]:
        """Find reference audio files (finished tracks without raw pairs)."""
        ref_dir = self.data_dir / self.reference_subdir
        
        if not ref_dir.exists():
            return []
        
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        files = []
        for p in ref_dir.iterdir():
            if p.suffix.lower() in extensions:
                files.append(p)
        
        return sorted(files)
    
    def _preload_audio_cache(self) -> None:
        """Pre-load all audio files into memory cache for faster training.
        
        This is called at initialization to avoid repeated disk I/O during training.
        Audio is loaded into the global in-memory cache in utils.py.
        """
        all_files = []
        
        # Collect all files to preload
        for raw_path, edited_path in self.pairs:
            all_files.append(raw_path)
            all_files.append(edited_path)
        
        for ref_path in self.reference_files:
            all_files.append(ref_path)
        
        if not all_files:
            return
        
        logger.info(f"Pre-loading {len(all_files)} audio files into memory cache...")
        
        loaded = 0
        for file_path in all_files:
            try:
                # This will cache the audio in memory via load_audio's cache
                load_audio(str(file_path), sr=self.config.audio.sample_rate)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to preload {file_path}: {e}")
        
        # Log cache stats
        stats = get_audio_cache_stats()
        logger.info(f"Pre-loaded {loaded}/{len(all_files)} files ({stats['size_mb']:.1f} MB in cache)")

    def __len__(self) -> int:
        return len(self.pairs) + len(self.reference_files)
    
    def is_reference_track(self, idx: int) -> bool:
        """Check if index corresponds to a reference track (not a paired track)."""
        return idx >= len(self.pairs)
    
    def get_num_paired(self) -> int:
        """Get number of paired (input/output) tracks."""
        return len(self.pairs)
    
    def get_num_reference(self) -> int:
        """Get number of reference tracks."""
        return len(self.reference_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """Get paired item from dataset.

        Returns:
            Dictionary containing:
            - raw: Dict with raw audio features
            - edited: Dict with edited audio features  
            - edit_labels: Per-beat labels (KEEP=1, CUT=0) inferred from alignment
            - pair_id: Identifier for the pair
            - is_reference: True if this is a reference track (not a true pair)
        """
        #import time
        #t0 = time.perf_counter()

        # Handle reference tracks (indexed after pairs)
        if idx >= len(self.pairs):
            ref_idx = idx - len(self.pairs)
            return self._get_reference_item(ref_idx)
        
        raw_path, edited_path = self.pairs[idx]

        # Load and process both files (with augmentation if enabled)
        try:
            # Apply same augmentation to both raw and edited to preserve labels
            apply_augment = self.use_augmentation and self.augmentor is not None
            
            if apply_augment:
                # Load raw audio for pair augmentation
                y_raw, sr = load_audio(str(raw_path), sr=self.config.audio.sample_rate)
                y_edited, _ = load_audio(str(edited_path), sr=self.config.audio.sample_rate)
                
                # Apply same augmentation to both
                y_raw_aug, y_edited_aug, aug_info = self.augmentor.augment_pair(
                    y_raw, y_edited, return_info=True
                )
                
                # Now process the augmented audio
                raw_data = self._process_audio_array(y_raw_aug, sr)
                edited_data = self._process_audio_array(y_edited_aug, sr)
            else:
                # Check for cached edit labels first
                cached_labels = None
                if self.feature_cache:
                    cached_labels = self.feature_cache.load_labels(raw_path, edited_path)
                
                raw_data = self._process_file(raw_path)
                edited_data = self._process_file(edited_path)
            
            # Infer edit labels (use cached if available)
            if apply_augment or cached_labels is None:
                edit_labels = self._infer_edit_labels(raw_data, edited_data)
                # Cache the labels for next time
                if not apply_augment and self.feature_cache:
                    self.feature_cache.save_labels(raw_path, edited_path, edit_labels)
            else:
                edit_labels = cached_labels
            
            data = {
                "raw": raw_data,
                "edited": edited_data,
                "edit_labels": torch.from_numpy(edit_labels).float(),
                "pair_id": raw_path.stem.replace("_raw", ""),
                "raw_path": str(raw_path),
                "edited_path": str(edited_path),
                "is_reference": False,
            }

            #t1 = time.perf_counter()
            #print(f"__getitem__({idx}) took {t1-t0:.4f}s")    

            return data
        except Exception as e:
            logger.error(f"Error processing pair {raw_path} / {edited_path}: {e}")
            raise e
    
    def _process_audio_array(self, y: np.ndarray, sr: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Process an audio array directly (for augmented data).
        
        Args:
            y: Audio array
            sr: Sample rate
        """
        # Compute Mel Spectrogram
        mel = compute_mel_spectrogram(
            y, 
            sr=sr, 
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length
        )

        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Use enhanced feature extractor if available
        if self.feature_extractor is not None and len(beats) > 0:
            beat_features = self.feature_extractor.extract_features(
                y, beats, beat_times, tempo
            )
        else:
            # Fall back to basic features
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
                if start >= len(onset_env): break
                if end > len(onset_env): end = len(onset_env)
                if start == end: end = start + 1
                
                b_onset = np.mean(onset_env[start:end])
                b_centroid = np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))])
                b_zcr = np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))])
                b_rms = np.mean(rms[min(start, len(rms)-1):min(end, len(rms))])
                beat_features.append([b_onset, b_centroid, b_zcr, b_rms])
                
            beat_features = np.array(beat_features) if beat_features else np.zeros((0, 4))
        
        # Ensure length matches beats
        if len(beat_features) < len(beats):
            pad_width = len(beats) - len(beat_features)
            beat_features = np.pad(beat_features, ((0, pad_width), (0, 0)), mode='edge')
        elif len(beat_features) > len(beats):
            beat_features = beat_features[:len(beats)]
        
        # Note: For augmented audio arrays, we skip stem separation as it requires
        # writing to temp file and is very slow. Stems are best pre-cached.

        return {
            "audio": torch.from_numpy(y).float(),
            "mel": torch.from_numpy(mel).float(),
            "beats": torch.from_numpy(beats).long(),
            "beat_times": torch.from_numpy(beat_times).float(),
            "beat_features": torch.from_numpy(beat_features).float(),
            "tempo": torch.tensor(tempo).float(),
            "duration": torch.tensor(len(y) / sr).float(),
            "sample_rate": sr,
        }

    def _get_reference_item(self, ref_idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """Get a reference track item (self-paired, all beats KEEP).
        
        Reference tracks are finished songs - the agent should learn that
        keeping all beats of a finished song is correct (100% keep).
        """
        ref_path = self.reference_files[ref_idx]
        
        # Check cache
        if self.cache_dir:
            cached_data = self._load_from_cache(ref_path, prefix="ref_")
            if cached_data:
                return cached_data
        
        try:
            # Process the reference track
            ref_data = self._process_file(ref_path)
            
            # All beats should be KEPT (it's already a finished track)
            n_beats = len(ref_data["beats"])
            edit_labels = np.ones(n_beats, dtype=np.float32)
            
            data = {
                "raw": ref_data,  # Same as edited for reference
                "edited": ref_data,
                "edit_labels": torch.from_numpy(edit_labels).float(),
                "pair_id": f"ref_{ref_path.stem}",
                "raw_path": str(ref_path),
                "edited_path": str(ref_path),
                "is_reference": True,
            }
            
            if self.feature_cache:
                self.feature_cache.save_full(ref_path, {
                    k: v.numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                    if isinstance(v, (torch.Tensor, np.ndarray))
                })
            
            return data
        except Exception as e:
            logger.error(f"Error processing reference {ref_path}: {e}")
            raise e

    def _process_file(self, file_path: Path, apply_augment: bool = False) -> Dict[str, Union[torch.Tensor, str]]:
        """Process a single audio file.
        
        Uses centralized cache when available for faster loading.
        
        Args:
            file_path: Path to audio file
            apply_augment: Whether to apply augmentation (default False)
        """
        # Try to load from centralized cache first (only if not augmenting)
        if not apply_augment and self.feature_cache:
            cached = self.feature_cache.load_features(file_path)
            if cached is not None:
                # We have cached beat features, load audio minimally
                y, sr = load_audio(str(file_path), sr=self.config.audio.sample_rate)
                
                # Load cached mel or compute
                mel = self.feature_cache.load_mel(file_path)
                if mel is None:
                    mel = compute_mel_spectrogram(
                        y, sr=sr,
                        n_mels=self.config.audio.n_mels,
                        n_fft=self.config.audio.n_fft,
                        hop_length=self.config.audio.hop_length
                    )
                    self.feature_cache.save_mel(file_path, mel)
                
                beat_features = cached["beat_features"]
                beats = cached["beats"]
                beat_times = cached["beat_times"]
                tempo = float(cached["tempo"])
                
                # Check if cached features already include stems (121 dims vs 109)
                # Only recompute stem features if cached features are base-only
                expected_full_dim = 121  # 109 base + 12 stem features
                cached_dim = beat_features.shape[-1] if beat_features.ndim > 1 else 0
                
                if self.stem_processor is not None and len(beats) > 0 and cached_dim < expected_full_dim:
                    # Cached features don't include stems - need to add them
                    stems = self.feature_cache.load_stems(file_path)
                    if stems is None:
                        stems = self.stem_processor.separate(str(file_path))
                        if stems:
                            self.feature_cache.save_stems(file_path, stems)
                    
                    if stems:
                        stem_features = self.stem_processor.get_stem_features(
                            stems, beats, hop_length=self.config.audio.hop_length
                        )
                        beat_features = np.concatenate([beat_features, stem_features], axis=1)
                
                return {
                    "audio": torch.from_numpy(y).float(),
                    "mel": torch.from_numpy(mel).float(),
                    "beats": torch.from_numpy(beats).long(),
                    "beat_times": torch.from_numpy(beat_times).float(),
                    "beat_features": torch.from_numpy(beat_features).float(),
                    "tempo": torch.tensor(tempo).float(),
                    "duration": torch.tensor(len(y) / sr).float(),
                    "sample_rate": sr,
                }
        
        # Not cached or augmenting - compute from scratch
        # Load audio
        y, sr = load_audio(str(file_path), sr=self.config.audio.sample_rate)
        
        # Apply augmentation if enabled (only for training)
        if apply_augment and self.augmentor is not None:
            y = self.augmentor(y)
        
        # Compute Mel Spectrogram
        mel = compute_mel_spectrogram(
            y, 
            sr=sr, 
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length
        )

        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Use enhanced feature extractor if available
        if self.feature_extractor is not None and len(beats) > 0:
            beat_features = self.feature_extractor.extract_features(
                y, beats, beat_times, tempo
            )
        else:
            # Fall back to basic features (onset, centroid, zcr, rms)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            
            # Aggregate features per beat
            beat_features = []
            frames = librosa.time_to_frames(beat_times, sr=sr)
            frames = np.concatenate(([0], frames, [len(onset_env)]))
            
            for i in range(len(beats)):
                start = frames[i]
                end = frames[i+1]
                if start >= len(onset_env): break
                if end > len(onset_env): end = len(onset_env)
                if start == end: end = start + 1
                
                b_onset = np.mean(onset_env[start:end])
                b_centroid = np.mean(centroid[min(start, len(centroid)-1):min(end, len(centroid))])
                b_zcr = np.mean(zcr[min(start, len(zcr)-1):min(end, len(zcr))])
                b_rms = np.mean(rms[min(start, len(rms)-1):min(end, len(rms))])
                beat_features.append([b_onset, b_centroid, b_zcr, b_rms])
                
            beat_features = np.array(beat_features) if beat_features else np.zeros((0, 4))
        
        # Ensure length matches beats
        if len(beat_features) < len(beats):
            pad_width = len(beats) - len(beat_features)
            beat_features = np.pad(beat_features, ((0, pad_width), (0, 0)), mode='edge')
        elif len(beat_features) > len(beats):
            beat_features = beat_features[:len(beats)]
        
        # Save beat features to centralized cache (before stem features added)
        if not apply_augment and self.feature_cache:
            self.feature_cache.save_features(
                file_path,
                beat_features=beat_features,
                beat_times=beat_times,
                beats=beats,
                tempo=float(tempo) if hasattr(tempo, '__float__') else tempo,
            )
            self.feature_cache.save_mel(file_path, mel)
        
        # Extract stem features if stem processor is available
        stem_features = None
        if self.stem_processor is not None and len(beats) > 0:
            # Check cache first
            stems = None
            if self.feature_cache:
                stems = self.feature_cache.load_stems(file_path)
            
            if stems is None:
                stems = self.stem_processor.separate(str(file_path))
                if stems and self.feature_cache:
                    self.feature_cache.save_stems(file_path, stems)
            
            if stems:
                stem_features = self.stem_processor.get_stem_features(
                    stems, beats, hop_length=self.config.audio.hop_length
                )
                # Concatenate stem features to beat features
                beat_features = np.concatenate([beat_features, stem_features], axis=1)

        result = {
            "audio": torch.from_numpy(y).float(),
            "mel": torch.from_numpy(mel).float(),
            "beats": torch.from_numpy(beats).long(),
            "beat_times": torch.from_numpy(beat_times).float(),
            "beat_features": torch.from_numpy(beat_features).float(),
            "tempo": torch.tensor(tempo).float(),
            "duration": torch.tensor(len(y) / sr).float(),
            "sample_rate": sr,
        }
        
        # Optionally include raw stem data
        if stem_features is not None:
            result["has_stems"] = True
        
        return result

    def _infer_edit_labels(
        self, 
        raw_data: Dict, 
        edited_data: Dict
    ) -> np.ndarray:
        """Infer per-beat KEEP/CUT labels by comparing raw and edited audio.
        
        Uses a simple duration-based heuristic combined with audio similarity.
        The actual keep ratio is estimated from the duration ratio, then
        beats are labeled based on how well they match segments in the edited audio.
        
        Returns:
            Array of shape (n_raw_beats,) with 1=KEEP, 0=CUT
        """
        raw_audio = raw_data["audio"].numpy()
        edited_audio = edited_data["audio"].numpy()
        raw_beats = raw_data["beat_times"].numpy()
        raw_beat_features = raw_data["beat_features"].numpy()
        sr = raw_data["sample_rate"]
        
        n_beats = len(raw_beats)
        if n_beats == 0:
            return np.array([])
        
        # Estimate actual keep ratio from durations
        raw_duration = len(raw_audio) / sr
        edited_duration = len(edited_audio) / sr
        actual_keep_ratio = min(1.0, edited_duration / raw_duration)
        
        # Compute similarity scores for each beat using RMS energy matching
        # This is much faster than chroma matching
        raw_rms = librosa.feature.rms(y=raw_audio, hop_length=512)[0]
        edited_rms = librosa.feature.rms(y=edited_audio, hop_length=512)[0]
        
        # For each raw beat, compute a similarity score
        similarity_scores = np.zeros(n_beats)
        beat_intervals = np.diff(np.concatenate([raw_beats, [raw_duration]]))
        
        for i, (beat_time, interval) in enumerate(zip(raw_beats, beat_intervals)):
            start_frame = int(beat_time * sr / 512)
            end_frame = int((beat_time + interval) * sr / 512)
            
            if start_frame >= len(raw_rms):
                similarity_scores[i] = 0.0
                continue
            end_frame = min(end_frame, len(raw_rms))
            if start_frame >= end_frame:
                similarity_scores[i] = 0.0
                continue
            
            beat_rms = raw_rms[start_frame:end_frame]
            beat_energy = np.mean(beat_rms)
            
            # Higher energy beats are more likely to be kept
            # Use onset strength from beat features as additional signal
            onset_strength = raw_beat_features[i, 0] if i < len(raw_beat_features) else 0
            
            # Combined score: energy + onset strength
            similarity_scores[i] = beat_energy + 0.5 * onset_strength
        
        # Normalize scores
        if similarity_scores.max() > 0:
            similarity_scores = similarity_scores / similarity_scores.max()
        
        # Select top beats based on actual keep ratio
        n_keep = max(1, int(n_beats * actual_keep_ratio))
        threshold_idx = np.argsort(similarity_scores)[-n_keep]
        threshold = similarity_scores[threshold_idx]
        
        # Label beats: 1 if above threshold (KEEP), 0 if below (CUT)
        labels = (similarity_scores >= threshold).astype(np.float32)
        
        return labels

    def _get_cache_path(self, file_path: Path, prefix: str = "") -> Path:
        """Get cache path for a file."""
        if not self.cache_dir:
            return None
        stem = file_path.stem.replace('_raw', '').replace('_edit', '')
        cache_path = self.cache_dir / "paired" / f"{prefix}{stem}.pt"
        return cache_path

    def _load_from_cache(self, file_path: Path, prefix: str = "") -> Optional[Dict]:
        """Load processed data from cache."""
        cache_path = self._get_cache_path(file_path, prefix)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_path}: {e}")
        return None

    def _save_to_cache(self, file_path: Path, data: Dict, prefix: str = "") -> None:
        """Save processed data to cache."""
        cache_path = self._get_cache_path(file_path, prefix)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache_path)

    def get_keep_ratio_stats(self, exclude_reference: bool = True) -> Dict[str, float]:
        """Compute statistics on keep ratios across all pairs.
        
        Args:
            exclude_reference: If True, only compute stats on true pairs (not reference tracks)
        """
        keep_ratios = []
        n_items = len(self.pairs) if exclude_reference else len(self)
        for idx in range(n_items):
            item = self[idx]
            labels = item["edit_labels"].numpy()
            if len(labels) > 0:
                keep_ratios.append(np.mean(labels))
        
        if not keep_ratios:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": np.mean(keep_ratios),
            "std": np.std(keep_ratios),
            "min": np.min(keep_ratios),
            "max": np.max(keep_ratios),
        }


def create_dataloader(
    dataset: Union[AudioDataset, PairedAudioDataset],
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    
    def collate_fn(batch):
        # Custom collate to handle variable length audio/features
        # For now, just return list of dicts or pad
        # Simple version: return list
        return batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

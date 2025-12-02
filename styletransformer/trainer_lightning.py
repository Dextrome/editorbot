"""
Style Transfer Trainer with PyTorch Lightning - GPU-optimized training.

Training approach:
1. Contrastive learning for style encoder (same song = similar embedding)
2. Reinforcement learning for policy (reward = discriminator score)
3. Adversarial training for discriminator (real vs generated)

Leverages:
- PyTorch Lightning for multi-GPU training
- AudioCraft for optional music generation\
- Pedalboard for audio augmentation
- Mixed precision training (FP16/BF16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

# Try to import PyTorch Lightning
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

# Try to import pedalboard for augmentation
try:
    import pedalboard
    from pedalboard import (
        Pedalboard, Chorus, Reverb, Compressor, Gain, 
        LowpassFilter, HighpassFilter, PitchShift
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

# Try to import audiocraft
try:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    HAS_AUDIOCRAFT = True
except ImportError:
    HAS_AUDIOCRAFT = False

# Handle both package and standalone imports
try:
    from .style_encoder import StyleEncoder, StyleEncoderNet, FeatureExtractor, StyleFeatures
    from .remix_policy import RemixPolicy, RemixPolicyNet
    from .discriminator import StyleDiscriminator, MultiScaleDiscriminator
except ImportError:
    from style_encoder import StyleEncoder, StyleEncoderNet, FeatureExtractor, StyleFeatures
    from remix_policy import RemixPolicy, RemixPolicyNet
    from discriminator import StyleDiscriminator, MultiScaleDiscriminator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Paths
    data_dir: str
    output_dir: str
    
    # Training params
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Architecture
    style_dim: int = 256
    hidden_dim: int = 512
    
    # GPU settings
    accelerator: str = "auto"     # "cpu", "gpu", "auto"
    devices: int = 1              # Number of GPUs
    precision: str = "32"         # "32" for stability, "16-mixed" for speed
    
    # RL params
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Data augmentation
    augment_tempo: bool = True
    augment_pitch: bool = True
    augment_effects: bool = True
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 2
    resume_from: str = None  # Path to checkpoint to resume from
    
    # Logging
    use_wandb: bool = False
    project_name: str = "styletransformer"
    
    # DataLoader
    num_workers: int = 4  # Number of parallel data loading workers
    preload: bool = False  # Preload all audio into RAM (faster training, uses more memory)


class AudioAugmenter:
    """Audio augmentation using Pedalboard (if available).
    
    Note: Pedalboard objects can't be pickled, so we create them on-demand
    to support multiprocessing DataLoaders on Windows.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.has_pedalboard = HAS_PEDALBOARD
        self._augmentations = None  # Lazy init to avoid pickle issues
    
    def _get_augmentations(self):
        """Lazy initialization of augmentations (avoids pickle issues on Windows)."""
        if self._augmentations is None and self.has_pedalboard:
            self._augmentations = [
                Pedalboard([Chorus(rate_hz=1.5, depth=0.3, mix=0.3)]),
                Pedalboard([Reverb(room_size=0.3, wet_level=0.2)]),
                Pedalboard([Compressor(threshold_db=-20, ratio=3)]),
                Pedalboard([LowpassFilter(cutoff_frequency_hz=8000)]),
                Pedalboard([HighpassFilter(cutoff_frequency_hz=100)]),
                Pedalboard([Gain(gain_db=-3)]),
            ]
        return self._augmentations
    
    def augment(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply random augmentation to audio."""
        if not self.has_pedalboard:
            # Fallback: simple noise augmentation
            noise = np.random.randn(*audio.shape) * 0.01 * strength
            return audio + noise
        
        # Ensure audio is float32 and correct shape for pedalboard
        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)  # (channels, samples)
        
        # Apply random augmentation
        import random
        augmentations = self._get_augmentations()
        if augmentations and random.random() < strength:
            aug = random.choice(augmentations)
            audio = aug(audio, self.sample_rate)
        
        return audio.squeeze()
    
    def __getstate__(self):
        """For pickling - don't include the augmentations."""
        state = self.__dict__.copy()
        state['_augmentations'] = None  # Don't pickle Pedalboard objects
        return state


class SongDataset(Dataset):
    """Dataset of songs for style learning with GPU-friendly features."""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        max_duration: float = 300.0,
        segment_duration: float = 30.0,  # Train on 30-second segments
        cache_features: bool = True,
        augment: bool = True,
        preload: bool = False,  # Preload all audio into RAM for faster training
        cache_features_in_memory: bool = True  # Cache extracted features in RAM
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.segment_duration = segment_duration
        self.cache_features = cache_features
        self.augment = augment
        self.preload = preload
        self.cache_features_in_memory = cache_features_in_memory
        
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.augmenter = AudioAugmenter(sample_rate) if augment else None
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            self.audio_files.extend(self.data_dir.rglob(ext))
        
        logger.info(f"Found {len(self.audio_files)} audio files in {data_dir}")
        
        # Feature cache (disk)
        self.cache_dir = self.data_dir / '.feature_cache'
        if cache_features:
            self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory feature cache (much faster after first epoch)
        self.memory_feature_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Track failed file indices (to skip on subsequent iterations)
        self.failed_indices: set = set()
        
        # Preloaded audio storage
        self.preloaded_audio: Dict[int, np.ndarray] = {}
        if preload:
            self._preload_all_audio()
        
        # Preload features into memory (much smaller than raw audio)
        if cache_features_in_memory and not preload:
            self._preload_features()
    
    def _preload_features(self):
        """Preload all features into memory. Much smaller than raw audio (~100MB for 1000 songs)."""
        logger.info("Preloading features into memory...")
        loaded_from_disk = 0
        computed = 0
        failed = 0
        
        for idx, audio_path in enumerate(tqdm(self.audio_files, desc="Preloading features")):
            # Try disk cache first
            cache_path = self.cache_dir / f"{audio_path.stem}_{idx}.pt"
            if self.cache_features and cache_path.exists():
                try:
                    cached = torch.load(cache_path, weights_only=True)
                    self.memory_feature_cache[idx] = cached
                    loaded_from_disk += 1
                    continue
                except Exception:
                    pass
            
            # Need to compute features
            try:
                audio = self._get_audio_segment(idx, audio_path)
                if audio is None:
                    failed += 1
                    continue
                
                # No augmentation during preload - we want clean features
                features = self.feature_extractor.extract(audio, self.sample_rate)
                data = self._features_to_tensors(features)
                
                self.memory_feature_cache[idx] = data
                computed += 1
                
                # Save to disk cache too
                if self.cache_features:
                    try:
                        torch.save(data, cache_path)
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.warning(f"Failed to preload features for {audio_path}: {e}")
                failed += 1
        
        # Estimate memory usage
        if self.memory_feature_cache:
            sample = next(iter(self.memory_feature_cache.values()))
            bytes_per_item = sum(t.numel() * t.element_size() for t in sample.values())
            total_mb = (bytes_per_item * len(self.memory_feature_cache)) / 1e6
            logger.info(f"Preloaded {len(self.memory_feature_cache)} feature sets "
                       f"({loaded_from_disk} from disk, {computed} computed) - {total_mb:.1f} MB RAM")
        if failed > 0:
            logger.warning(f"Failed to preload {failed} files")
    
    def _preload_all_audio(self):
        """Preload all audio files into memory for faster training."""
        logger.info("Preloading all audio into memory...")
        total_bytes = 0
        failed = 0
        
        for idx, audio_path in enumerate(tqdm(self.audio_files, desc="Preloading audio")):
            try:
                # Load full audio file
                audio, sr = librosa.load(
                    str(audio_path),
                    sr=self.sample_rate,
                    mono=True,
                    duration=self.max_duration  # Cap at max_duration
                )
                
                min_samples = int(self.sample_rate * 1.0)  # At least 1 second
                if len(audio) >= min_samples:
                    self.preloaded_audio[idx] = audio.astype(np.float32)
                    total_bytes += audio.nbytes
                else:
                    logger.warning(f"Skipping {audio_path.name}: too short ({len(audio)} samples)")
                    failed += 1
                    
            except Exception as e:
                logger.warning(f"Failed to preload {audio_path}: {e}")
                failed += 1
        
        logger.info(f"Preloaded {len(self.preloaded_audio)} songs ({total_bytes / 1e9:.2f} GB RAM)")
        if failed > 0:
            logger.warning(f"Failed to preload {failed} files")
    
    def _get_audio_segment(self, idx: int, audio_path: Path) -> Optional[np.ndarray]:
        """Get audio segment, either from preloaded cache or disk."""
        if self.preload and idx in self.preloaded_audio:
            # Use preloaded audio - just pick a random segment
            full_audio = self.preloaded_audio[idx]
            segment_samples = int(self.segment_duration * self.sample_rate)
            
            if len(full_audio) <= segment_samples:
                return full_audio
            
            # Random offset
            max_offset = len(full_audio) - segment_samples
            offset = np.random.randint(0, max_offset)
            return full_audio[offset:offset + segment_samples]
        
        # Load from disk (fallback or non-preloaded mode)
        try:
            import soundfile as sf
            try:
                info = sf.info(str(audio_path))
                file_duration = info.duration
                if file_duration < 1.0:
                    return None
            except Exception:
                file_duration = self.segment_duration
            
            max_offset = max(0, file_duration - self.segment_duration)
            offset = np.random.uniform(0, max_offset) if max_offset > 0 else 0
            
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True,
                offset=offset,
                duration=self.segment_duration
            )
            
            min_samples = int(self.sample_rate * 1.0)
            if len(audio) < min_samples:
                return None
                
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]
        
        # 1. Check in-memory feature cache first (fastest)
        if self.cache_features_in_memory and idx in self.memory_feature_cache:
            cached = self.memory_feature_cache[idx]
            # Check if this is a failed marker
            if cached.get('_failed'):
                return self._get_dummy_data()
            cached = cached.copy()
            cached['path'] = str(audio_path)
            return cached
        
        # 2. Check disk cache (when not preloading audio)
        if not self.preload and self.cache_features:
            cache_path = self.cache_dir / f"{audio_path.stem}_{idx}.pt"
            if cache_path.exists():
                try:
                    cached = torch.load(cache_path, weights_only=True)
                    # Store in memory cache for next time
                    if self.cache_features_in_memory:
                        self.memory_feature_cache[idx] = {k: v for k, v in cached.items()}
                    cached['path'] = str(audio_path)
                    return cached
                except Exception:
                    pass
        
        # 3. Check if this index previously failed
        if idx in self.failed_indices:
            return self._get_dummy_data()
        
        # 4. Load and process audio
        audio = self._get_audio_segment(idx, audio_path)
        
        if audio is None:
            self._move_to_error_dir(audio_path, idx)
            return self._get_dummy_data()
        
        try:
            # Apply augmentation
            if self.augment and self.augmenter:
                audio = self.augmenter.augment(audio)
            
            # Extract features
            features = self.feature_extractor.extract(audio, self.sample_rate)
            
            # Convert to tensors
            data = self._features_to_tensors(features)
            
            # Cache in memory (without path)
            if self.cache_features_in_memory:
                self.memory_feature_cache[idx] = {k: v for k, v in data.items()}
            
            # Cache to disk (without path)
            if self.cache_features and not self.preload:
                cache_path = self.cache_dir / f"{audio_path.stem}_{idx}.pt"
                try:
                    torch.save({k: v for k, v in data.items()}, cache_path)
                except Exception:
                    pass
            
            data['path'] = str(audio_path)
            return data
            
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            return self._get_dummy_data()
    
    def _move_to_error_dir(self, audio_path: Path, idx: int = None):
        """Move a failed audio file to the error directory and mark index as failed."""
        # Mark index as failed in memory cache so we skip it on subsequent iterations
        # This persists across epochs since memory_feature_cache is on the main process
        if idx is not None:
            self.failed_indices.add(idx)
            # Also mark in memory cache with a special "failed" marker
            self.memory_feature_cache[idx] = {'_failed': True}
        
        try:
            error_dir = self.data_dir / 'error'
            error_dir.mkdir(exist_ok=True)
            
            # Preserve relative path structure
            rel_path = audio_path.relative_to(self.data_dir)
            dest_path = error_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            import shutil
            shutil.move(str(audio_path), str(dest_path))
            logger.info(f"Moved failed file to: {dest_path}")
        except Exception as move_err:
            logger.warning(f"Could not move {audio_path} to error dir: {move_err}")
    
    def _features_to_tensors(self, features: StyleFeatures) -> Dict[str, torch.Tensor]:
        """Convert StyleFeatures to PyTorch tensors."""
        target_length = 2048
        
        def resample_1d(arr, target_len):
            if len(arr) == target_len:
                return arr
            indices = np.linspace(0, len(arr) - 1, target_len)
            return np.interp(indices, np.arange(len(arr)), arr)
        
        def resample_2d(arr, target_len):
            """Resample 2D array (n_features, time) to fixed length."""
            result = np.zeros((arr.shape[0], target_len), dtype=np.float32)
            for i in range(arr.shape[0]):
                result[i] = resample_1d(arr[i], target_len)
            return result
        
        # Time features: (6, T) - energy, brightness, onset, ZCR, bandwidth, rolloff
        energy = resample_1d(features.energy_curve, target_length)
        brightness = resample_1d(features.brightness_curve, target_length)
        onset = resample_1d(features.onset_density_curve, target_length)
        zcr = resample_1d(features.zcr_curve, target_length)
        bandwidth = resample_1d(features.spectral_bandwidth, target_length)
        rolloff = resample_1d(features.spectral_rolloff, target_length)
        time_features = np.stack([energy, brightness, onset, zcr, bandwidth, rolloff], axis=0).astype(np.float32)
        
        # MFCC features: (13, T)
        mfcc_features = resample_2d(features.mfcc_sequence, target_length)
        
        # Spectral contrast features: (7, T)
        contrast_features = resample_2d(features.spectral_contrast, target_length)
        
        # Chroma features: (12, T)
        chroma_features = resample_2d(features.chroma_sequence, target_length)
        
        # Tonnetz features: (6, T)
        tonnetz_features = resample_2d(features.tonnetz_sequence, target_length)
        
        # Frequency balance features: (3, T)
        freqbal_features = resample_2d(features.frequency_balance_curve, target_length)
        
        # Global features
        global_features = self._extract_global_features(features)
        
        return {
            'time_features': torch.from_numpy(time_features),
            'mfcc_features': torch.from_numpy(mfcc_features),
            'contrast_features': torch.from_numpy(contrast_features),
            'chroma_features': torch.from_numpy(chroma_features),
            'tonnetz_features': torch.from_numpy(tonnetz_features),
            'freqbal_features': torch.from_numpy(freqbal_features),
            'global_features': torch.from_numpy(global_features),
        }
    
    def _extract_global_features(self, features: StyleFeatures) -> np.ndarray:
        """Extract global features from StyleFeatures."""
        global_features = np.zeros(20, dtype=np.float32)
        global_features[0] = features.tempo / 200.0
        global_features[1] = features.key / 12.0
        global_features[2] = features.avg_phrase_length / 16.0
        global_features[3] = features.energy_variance
        global_features[4] = len(features.phrase_lengths) / 50.0
        global_features[5] = features.duration / 600.0
        
        if features.phrase_lengths:
            global_features[6] = np.mean(features.phrase_lengths) / 8.0
            global_features[7] = np.std(features.phrase_lengths) / 4.0 if len(features.phrase_lengths) > 1 else 0
        
        if features.phrase_energies:
            global_features[8] = np.mean(features.phrase_energies)
            global_features[9] = np.std(features.phrase_energies) if len(features.phrase_energies) > 1 else 0
            global_features[10] = np.max(features.phrase_energies)
            global_features[11] = np.min(features.phrase_energies)
        
        if features.transition_types:
            total = len(features.transition_types)
            global_features[12] = features.transition_types.count("buildup") / total
            global_features[13] = features.transition_types.count("drop") / total
            global_features[14] = features.transition_types.count("sustain") / total
        
        # New global features
        global_features[15] = 1.0 - features.tempo_stability  # Tempo variation (inverse of stability)
        global_features[16] = features.avg_beat_strength  # Beat strength
        global_features[17] = features.stereo_width  # Stereo width
        global_features[18] = features.dynamics_range  # Dynamics range
        global_features[19] = features.tempo_stability  # Tempo stability
        
        return global_features
    
    def _get_dummy_data(self) -> Dict[str, torch.Tensor]:
        """Return dummy data for failed loads."""
        return {
            'time_features': torch.zeros((6, 2048), dtype=torch.float32),
            'mfcc_features': torch.zeros((13, 2048), dtype=torch.float32),
            'contrast_features': torch.zeros((7, 2048), dtype=torch.float32),
            'chroma_features': torch.zeros((12, 2048), dtype=torch.float32),
            'tonnetz_features': torch.zeros((6, 2048), dtype=torch.float32),
            'freqbal_features': torch.zeros((3, 2048), dtype=torch.float32),
            'global_features': torch.zeros(20, dtype=torch.float32),
            'path': '',  # Empty path marks this as dummy data
        }


def collate_skip_errors(batch):
    """Custom collate function that skips failed samples (empty paths)."""
    # Filter out failed samples (those with empty paths)
    valid_batch = [b for b in batch if b.get('path', '') != '']
    
    if not valid_batch:
        # All samples failed, return dummy batch
        return {
            'time_features': torch.zeros((1, 6, 2048), dtype=torch.float32),
            'mfcc_features': torch.zeros((1, 13, 2048), dtype=torch.float32),
            'contrast_features': torch.zeros((1, 7, 2048), dtype=torch.float32),
            'chroma_features': torch.zeros((1, 12, 2048), dtype=torch.float32),
            'tonnetz_features': torch.zeros((1, 6, 2048), dtype=torch.float32),
            'freqbal_features': torch.zeros((1, 3, 2048), dtype=torch.float32),
            'global_features': torch.zeros((1, 20), dtype=torch.float32),
        }
    
    # Stack tensors, exclude 'path' from stacking
    result = {}
    feature_keys = ['time_features', 'mfcc_features', 'contrast_features', 
                    'chroma_features', 'tonnetz_features', 'freqbal_features', 'global_features']
    for key in feature_keys:
        result[key] = torch.stack([b[key] for b in valid_batch])
    
    return result


class StyleTransferLightning(pl.LightningModule if HAS_LIGHTNING else nn.Module):
    """PyTorch Lightning module for style transfer training."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters() if HAS_LIGHTNING else None
        
        # Models
        self.style_encoder = StyleEncoderNet(
            embedding_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.discriminator = MultiScaleDiscriminator(
            style_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.policy = RemixPolicyNet(
            style_dim=config.style_dim,
            hidden_dim=config.hidden_dim
        )
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat):
        return self.style_encoder(time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat)
    
    def training_step(self, batch, batch_idx):
        time_feat = batch['time_features']
        mfcc_feat = batch['mfcc_features']
        contrast_feat = batch['contrast_features']
        chroma_feat = batch['chroma_features']
        tonnetz_feat = batch['tonnetz_features']
        freqbal_feat = batch['freqbal_features']
        global_feat = batch['global_features']
        
        B = time_feat.size(0)
        
        # Get embeddings
        embeddings = self.style_encoder(time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat)
        
        # Create augmented versions
        time_aug = time_feat + torch.randn_like(time_feat) * 0.1
        mfcc_aug = mfcc_feat + torch.randn_like(mfcc_feat) * 0.1
        contrast_aug = contrast_feat + torch.randn_like(contrast_feat) * 0.1
        chroma_aug = chroma_feat + torch.randn_like(chroma_feat) * 0.1
        tonnetz_aug = tonnetz_feat + torch.randn_like(tonnetz_feat) * 0.1
        freqbal_aug = freqbal_feat + torch.randn_like(freqbal_feat) * 0.1
        global_aug = global_feat + torch.randn_like(global_feat) * 0.05
        
        embeddings_aug = self.style_encoder(time_aug, mfcc_aug, contrast_aug, chroma_aug, tonnetz_aug, freqbal_aug, global_aug)
        
        # Contrastive loss (InfoNCE)
        embeddings = F.normalize(embeddings, dim=-1)
        embeddings_aug = F.normalize(embeddings_aug, dim=-1)
        
        similarity = torch.mm(embeddings, embeddings_aug.t()) / self.temperature
        labels = torch.arange(B, device=self.device)
        
        encoder_loss = F.cross_entropy(similarity, labels)
        
        # Discriminator loss
        with torch.no_grad():
            emb_detached = embeddings.detach()
            emb_aug_detached = embeddings_aug.detach()
        
        pos_score, _ = self.discriminator(emb_detached, emb_aug_detached)
        perm = torch.randperm(B, device=self.device)
        neg_score, _ = self.discriminator(emb_detached, emb_aug_detached[perm])
        
        # Use MSE loss instead of BCE for AMP compatibility
        # (discriminator already applies sigmoid, so scores are in [0,1])
        disc_loss = F.mse_loss(
            pos_score, torch.ones_like(pos_score)
        ) + F.mse_loss(
            neg_score, torch.zeros_like(neg_score)
        )
        
        # Total loss
        total_loss = encoder_loss + disc_loss * 0.5
        
        # Logging
        self.log('train/encoder_loss', encoder_loss, prog_bar=True)
        self.log('train/disc_loss', disc_loss, prog_bar=True)
        self.log('train/total_loss', total_loss)
        self.log('train/temperature', self.temperature)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        time_feat = batch['time_features']
        chroma_feat = batch['chroma_features']
        global_feat = batch['global_features']
        
        embeddings = self.style_encoder(time_feat, chroma_feat, global_feat)
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute pairwise similarity
        similarity = torch.mm(embeddings, embeddings.t())
        
        # Self-similarity should be high (diagonal)
        diag_sim = similarity.diag().mean()
        
        # Off-diagonal should be lower
        mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=self.device)
        off_diag_sim = similarity[mask].mean()
        
        self.log('val/self_similarity', diag_sim)
        self.log('val/cross_similarity', off_diag_sim)
        self.log('val/discriminability', diag_sim - off_diag_sim)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class StyleTransferTrainer:
    """
    High-level trainer interface.
    Uses PyTorch Lightning if available, falls back to manual training loop.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Dataset
        self.dataset = SongDataset(
            config.data_dir,
            augment=config.augment_effects,
            preload=config.preload
        )
        
        # Use Lightning if available
        self.use_lightning = HAS_LIGHTNING
        
        if self.use_lightning:
            self.model = StyleTransferLightning(config)
        else:
            # Manual setup
            self._setup_manual_training()
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if self.config.accelerator == "cpu":
            return torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            # Set matmul precision for better Tensor Core utilization
            torch.set_float32_matmul_precision('high')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple MPS")
            return torch.device("mps")
        else:
            logger.warning("No GPU available, using CPU")
            return torch.device("cpu")
    
    def _setup_manual_training(self):
        """Setup for manual training loop (no Lightning)."""
        self.style_encoder = StyleEncoderNet(
            embedding_dim=self.config.style_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.discriminator = MultiScaleDiscriminator(
            style_dim=self.config.style_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.policy = RemixPolicyNet(
            style_dim=self.config.style_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            list(self.style_encoder.parameters()) +
            list(self.discriminator.parameters()) +
            list(self.policy.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Load from checkpoint if specified
        self.start_epoch = 0
        if self.config.resume_from:
            self._load_manual_checkpoint(self.config.resume_from)
    
    def _load_manual_checkpoint(self, resume_path: str):
        """Load weights for manual training from checkpoint."""
        resume_path = Path(resume_path)
        
        if resume_path.is_dir():
            # Directory with .pt files
            encoder_path = resume_path / 'style_encoder.pt'
            disc_path = resume_path / 'discriminator.pt'
            policy_path = resume_path / 'policy.pt'
            
            if encoder_path.exists():
                self.style_encoder.load_state_dict(
                    torch.load(encoder_path, map_location=self.device, weights_only=True))
                logger.info(f"Loaded style encoder from {encoder_path}")
            if disc_path.exists():
                self.discriminator.load_state_dict(
                    torch.load(disc_path, map_location=self.device, weights_only=True))
                logger.info(f"Loaded discriminator from {disc_path}")
            if policy_path.exists():
                self.policy.load_state_dict(
                    torch.load(policy_path, map_location=self.device, weights_only=True))
                logger.info(f"Loaded policy from {policy_path}")
                
        elif resume_path.suffix == '.pt':
            # Single .pt checkpoint file
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=True)
            
            if 'style_encoder' in checkpoint:
                self.style_encoder.load_state_dict(checkpoint['style_encoder'])
                logger.info("Loaded style encoder from checkpoint")
            if 'discriminator' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator'])
                logger.info("Loaded discriminator from checkpoint")
            if 'policy' in checkpoint:
                self.policy.load_state_dict(checkpoint['policy'])
                logger.info("Loaded policy from checkpoint")
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']
                logger.info(f"Resuming from epoch {self.start_epoch}")
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("Loaded optimizer state")
        else:
            logger.warning(f"Unknown checkpoint format: {resume_path}")
    
    def train(self):
        """Run training."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Dataset size: {len(self.dataset)} songs")
        logger.info(f"Using Lightning: {self.use_lightning}")
        
        if self.use_lightning:
            self._train_lightning()
        else:
            self._train_manual()
    
    def _train_lightning(self):
        """Train using PyTorch Lightning."""
        num_workers = self.config.num_workers
        
        # Dataloader
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_skip_errors
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=self.config.output_dir,
                filename='best-{epoch:02d}-{val/discriminability:.3f}',
                monitor='val/discriminability',
                mode='max',
                save_top_k=3
            ),
            ModelCheckpoint(
                dirpath=self.config.output_dir,
                filename='checkpoint-{epoch:02d}',
                every_n_epochs=self.config.save_every
            ),
            LearningRateMonitor(logging_interval='step'),
        ]
        
        # Logger
        if self.config.use_wandb:
            pl_logger = WandbLogger(
                project=self.config.project_name,
                save_dir=self.config.output_dir
            )
        else:
            pl_logger = TensorBoardLogger(
                save_dir=self.config.output_dir,
                name='logs'
            )
        
        # Trainer
        # Validation loader (no shuffling)
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_skip_errors
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config.num_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            precision=self.config.precision,
            callbacks=callbacks,
            logger=pl_logger,
            gradient_clip_val=self.config.max_grad_norm,
            log_every_n_steps=10,
            check_val_every_n_epoch=self.config.eval_every,
        )
        
        # Determine checkpoint path for resuming
        ckpt_path = None
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if resume_path.suffix == '.ckpt':
                # Lightning checkpoint
                ckpt_path = str(resume_path)
                logger.info(f"Resuming from Lightning checkpoint: {ckpt_path}")
            elif resume_path.is_dir():
                # Directory with model files - load weights
                self._load_model_weights(resume_path)
                logger.info(f"Loaded model weights from: {resume_path}")
            elif resume_path.suffix == '.pt':
                # Single checkpoint file
                self._load_model_weights_from_pt(resume_path)
                logger.info(f"Loaded weights from checkpoint: {resume_path}")
        
        # Train
        trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
        
        # Save final model
        self._save_models_from_lightning()
    
    def _train_manual(self):
        """Manual training loop (fallback)."""
        # Use num_workers=0 when features are cached in memory (workers don't share memory)
        if self.dataset.cache_features_in_memory and self.dataset.memory_feature_cache:
            num_workers = 0
            logger.info("Using num_workers=0 for in-memory feature cache")
        else:
            num_workers = self.config.num_workers
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' and num_workers > 0 else False,
            collate_fn=collate_skip_errors
        )
        
        best_loss = float('inf')
        total_epochs = self.start_epoch + self.config.num_epochs
        
        for epoch in range(self.start_epoch, total_epochs):
            self.style_encoder.train()
            self.discriminator.train()
            
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            for batch in pbar:
                # Move to device
                time_feat = batch['time_features'].to(self.device)
                mfcc_feat = batch['mfcc_features'].to(self.device)
                contrast_feat = batch['contrast_features'].to(self.device)
                chroma_feat = batch['chroma_features'].to(self.device)
                tonnetz_feat = batch['tonnetz_features'].to(self.device)
                freqbal_feat = batch['freqbal_features'].to(self.device)
                global_feat = batch['global_features'].to(self.device)
                
                B = time_feat.size(0)
                if B < 2:
                    continue
                
                # Forward pass
                embeddings = self.style_encoder(time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat)
                
                # Augmented version
                time_aug = time_feat + torch.randn_like(time_feat) * 0.1
                mfcc_aug = mfcc_feat + torch.randn_like(mfcc_feat) * 0.1
                contrast_aug = contrast_feat + torch.randn_like(contrast_feat) * 0.1
                chroma_aug = chroma_feat + torch.randn_like(chroma_feat) * 0.1
                tonnetz_aug = tonnetz_feat + torch.randn_like(tonnetz_feat) * 0.1
                freqbal_aug = freqbal_feat + torch.randn_like(freqbal_feat) * 0.1
                global_aug = global_feat + torch.randn_like(global_feat) * 0.05
                
                embeddings_aug = self.style_encoder(time_aug, mfcc_aug, contrast_aug, chroma_aug, tonnetz_aug, freqbal_aug, global_aug)
                
                # Contrastive loss
                embeddings_norm = F.normalize(embeddings, dim=-1)
                embeddings_aug_norm = F.normalize(embeddings_aug, dim=-1)
                
                similarity = torch.mm(embeddings_norm, embeddings_aug_norm.t()) / 0.07
                labels = torch.arange(B, device=self.device)
                encoder_loss = F.cross_entropy(similarity, labels)
                
                # Discriminator loss
                pos_score, _ = self.discriminator(embeddings.detach(), embeddings_aug.detach())
                perm = torch.randperm(B, device=self.device)
                neg_score, _ = self.discriminator(embeddings.detach(), embeddings_aug[perm].detach())
                
                disc_loss = F.binary_cross_entropy(
                    pos_score, torch.ones_like(pos_score)
                ) + F.binary_cross_entropy(
                    neg_score, torch.zeros_like(neg_score)
                )
                
                # Combined loss
                loss = encoder_loss + disc_loss * 0.5
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.style_encoder.parameters()) +
                    list(self.discriminator.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(epoch + 1, is_best=True)
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        
        # Save learned style at end of training
        self._save_learned_style_manual()
    
    def _save_learned_style_manual(self):
        """Compute and save the average style embedding from training data (manual training)."""
        try:
            output_dir = Path(self.config.output_dir)
            
            logger.info("Computing learned style embedding from training data...")
            self.style_encoder.eval()
            
            embeddings = []
            with torch.no_grad():
                # Use ALL training data for the learned style
                for idx in tqdm(range(len(self.dataset)), desc="Computing learned style"):
                    try:
                        batch = self.dataset[idx]
                        if batch.get('path', '') == '':  # Skip dummy data
                            continue
                        
                        time_feat = batch['time_features'].unsqueeze(0).to(self.device)
                        mfcc_feat = batch['mfcc_features'].unsqueeze(0).to(self.device)
                        contrast_feat = batch['contrast_features'].unsqueeze(0).to(self.device)
                        chroma_feat = batch['chroma_features'].unsqueeze(0).to(self.device)
                        tonnetz_feat = batch['tonnetz_features'].unsqueeze(0).to(self.device)
                        freqbal_feat = batch['freqbal_features'].unsqueeze(0).to(self.device)
                        global_feat = batch['global_features'].unsqueeze(0).to(self.device)
                        
                        emb = self.style_encoder(time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat)
                        embeddings.append(emb.cpu().numpy())
                    except Exception:
                        continue
            
            if embeddings:
                # Average all embeddings to get "the learned style"
                learned_style = np.mean(np.vstack(embeddings), axis=0)
                np.save(output_dir / 'learned_style.npy', learned_style)
                logger.info(f"Saved learned style embedding ({len(embeddings)} samples averaged)")
            else:
                logger.warning("Could not compute learned style - no valid embeddings")
                
        except Exception as e:
            logger.warning(f"Failed to save learned style: {e}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': vars(self.config)
        }
        
        filename = 'best.pt' if is_best else f'checkpoint_{epoch}.pt'
        path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _save_models_from_lightning(self):
        """Save individual model files from Lightning checkpoint."""
        output_dir = Path(self.config.output_dir)
        
        torch.save(
            self.model.style_encoder.state_dict(),
            output_dir / 'style_encoder.pt'
        )
        torch.save(
            self.model.discriminator.state_dict(),
            output_dir / 'discriminator.pt'
        )
        torch.save(
            self.model.policy.state_dict(),
            output_dir / 'policy.pt'
        )
        
        # Compute and save learned style (average embedding from training data)
        self._save_learned_style(output_dir)
        
        logger.info(f"Saved individual model files to {output_dir}")
    
    def _save_learned_style(self, output_dir: Path):
        """Compute and save the average style embedding from training data."""
        try:
            import numpy as np
            
            logger.info("Computing learned style embedding from training data...")
            self.model.eval()
            
            embeddings = []
            with torch.no_grad():
                # Use ALL training data for the learned style
                for idx in tqdm(range(len(self.dataset)), desc="Computing learned style"):
                    try:
                        batch = self.dataset[idx]
                        if batch.get('path', '') == '':  # Skip dummy data
                            continue
                        
                        time_feat = batch['time_features'].unsqueeze(0).to(self.device)
                        mfcc_feat = batch['mfcc_features'].unsqueeze(0).to(self.device)
                        contrast_feat = batch['contrast_features'].unsqueeze(0).to(self.device)
                        chroma_feat = batch['chroma_features'].unsqueeze(0).to(self.device)
                        tonnetz_feat = batch['tonnetz_features'].unsqueeze(0).to(self.device)
                        freqbal_feat = batch['freqbal_features'].unsqueeze(0).to(self.device)
                        global_feat = batch['global_features'].unsqueeze(0).to(self.device)
                        
                        emb = self.model.style_encoder(time_feat, mfcc_feat, contrast_feat, chroma_feat, tonnetz_feat, freqbal_feat, global_feat)
                        embeddings.append(emb.cpu().numpy())
                    except Exception:
                        continue
            
            if embeddings:
                # Average all embeddings to get "the learned style"
                learned_style = np.mean(np.vstack(embeddings), axis=0)
                np.save(output_dir / 'learned_style.npy', learned_style)
                logger.info(f"Saved learned style embedding ({len(embeddings)} samples averaged)")
            else:
                logger.warning("Could not compute learned style - no valid embeddings")
                
        except Exception as e:
            logger.warning(f"Failed to save learned style: {e}")
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        if self.use_lightning:
            self.model.style_encoder.load_state_dict(checkpoint['style_encoder'])
            self.model.discriminator.load_state_dict(checkpoint['discriminator'])
            self.model.policy.load_state_dict(checkpoint['policy'])
        else:
            self.style_encoder.load_state_dict(checkpoint['style_encoder'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def _load_model_weights(self, model_dir: Path):
        """Load model weights from a directory containing .pt files."""
        encoder_path = model_dir / 'style_encoder.pt'
        disc_path = model_dir / 'discriminator.pt'
        policy_path = model_dir / 'policy.pt'
        
        if encoder_path.exists():
            state_dict = torch.load(encoder_path, map_location=self.device, weights_only=True)
            self.model.style_encoder.load_state_dict(state_dict)
            logger.info(f"Loaded style encoder from {encoder_path}")
        
        if disc_path.exists():
            state_dict = torch.load(disc_path, map_location=self.device, weights_only=True)
            self.model.discriminator.load_state_dict(state_dict)
            logger.info(f"Loaded discriminator from {disc_path}")
        
        if policy_path.exists():
            state_dict = torch.load(policy_path, map_location=self.device, weights_only=True)
            self.model.policy.load_state_dict(state_dict)
            logger.info(f"Loaded policy from {policy_path}")
    
    def _load_model_weights_from_pt(self, checkpoint_path: Path):
        """Load model weights from a single .pt checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        if 'style_encoder' in checkpoint:
            self.model.style_encoder.load_state_dict(checkpoint['style_encoder'])
            logger.info("Loaded style encoder from checkpoint")
        
        if 'discriminator' in checkpoint:
            self.model.discriminator.load_state_dict(checkpoint['discriminator'])
            logger.info("Loaded discriminator from checkpoint")
        
        if 'policy' in checkpoint:
            self.model.policy.load_state_dict(checkpoint['policy'])
            logger.info("Loaded policy from checkpoint")


def train_style_transfer(
    data_dir: str,
    output_dir: str,
    **kwargs
) -> StyleTransferTrainer:
    """
    Convenience function to train style transfer.
    
    Args:
        data_dir: Directory containing training songs
        output_dir: Directory for checkpoints
        **kwargs: Additional TrainingConfig options
        
    Returns:
        Trained StyleTransferTrainer
    """
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        **kwargs
    )
    
    trainer = StyleTransferTrainer(config)
    trainer.train()
    
    return trainer

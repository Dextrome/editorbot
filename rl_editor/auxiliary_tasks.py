"""Auxiliary tasks for multi-task learning.

Forces the encoder to learn richer representations by solving additional tasks
beyond just KEEP/CUT prediction. This prevents premature convergence.

Auxiliary Tasks:
1. Tempo Prediction - Predict local tempo at each beat
2. Energy Prediction - Predict energy level at each beat  
3. Phrase Boundary Detection - Binary classify phrase boundaries
4. Beat Feature Reconstruction - Predict next beat's features
5. Section Similarity - Predict if two sections are similar

The key insight: These tasks require understanding musical structure,
not just memorizing patterns. The encoder must learn generalizable features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryConfig:
    """Configuration for auxiliary tasks."""
    
    # Enable/disable individual tasks
    use_tempo_prediction: bool = True
    use_energy_prediction: bool = True
    use_phrase_boundary: bool = True
    use_beat_reconstruction: bool = True
    use_section_similarity: bool = False  # More complex, disabled by default
    
    # Loss weights - keep small relative to policy loss!
    # These are scaled down because cross-entropy can be ~2-4 naturally
    # and reconstruction MSE can explode if features aren't normalized
    tempo_loss_weight: float = 0.02      # Was 0.15
    energy_loss_weight: float = 0.01     # Was 0.15
    phrase_loss_weight: float = 0.03     # Was 0.2
    reconstruction_loss_weight: float = 0.002  # Was 0.1 - MSE can be huge!
    similarity_loss_weight: float = 0.02
    # Mel-spectrogram reconstruction (per-beat) from training pairs
    use_mel_reconstruction: bool = True
    mel_reconstruction_weight: float = 0.02
    mel_dim: int = 128
    # Chroma-based continuity loss (maps mel vectors to 12 chroma bins)
    use_chroma_continuity: bool = True
    chroma_loss_weight: float = 0.008
    mel_sample_rate: int = 22050
    # Disk cache directory for per-track mel/chroma npy files (relative to repo root or absolute)
    mel_chroma_cache: str = "feature_cache/chroma"
    
    # Binary good/bad edit classifier
    use_good_bad_classifier: bool = True
    good_bad_loss_weight: float = 1.0
    # Task-specific settings
    tempo_bins: int = 20  # Discretize tempo into bins (60-180 BPM)
    tempo_min: float = 60.0
    tempo_max: float = 180.0
    energy_bins: int = 10  # Discretize energy into bins
    phrase_length: int = 8  # Beats per phrase (for boundary detection)
    reconstruction_context: int = 3  # Beats of context for reconstruction
    
    # Curriculum learning - start with easier tasks
    warmup_epochs: int = 50  # Epochs before full auxiliary loss weight
    
    def get_total_weight(self) -> float:
        """Get total weight of all enabled auxiliary tasks."""
        total = 0.0
        if self.use_tempo_prediction:
            total += self.tempo_loss_weight
        if self.use_energy_prediction:
            total += self.energy_loss_weight
        if self.use_phrase_boundary:
            total += self.phrase_loss_weight
        if self.use_beat_reconstruction:
            total += self.reconstruction_loss_weight
        if self.use_section_similarity:
            total += self.similarity_loss_weight
        return total


class TempoPredictor(nn.Module):
    """Predict local tempo at each beat position.
    
    Task: Given beat features, predict what tempo range this beat is in.
    This forces the encoder to understand rhythmic structure.
    """
    
    def __init__(self, hidden_dim: int, n_bins: int = 20):
        super().__init__()
        self.n_bins = n_bins
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_bins),
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict tempo bin logits.
        
        Args:
            encoded: Encoded features from shared encoder (B, hidden_dim)
            
        Returns:
            Logits for tempo bins (B, n_bins)
        """
        return self.head(encoded)
    
    @staticmethod
    def compute_targets(
        beat_times: np.ndarray,
        tempo_min: float = 60.0,
        tempo_max: float = 180.0,
        n_bins: int = 20,
    ) -> np.ndarray:
        """Compute tempo bin targets from beat times.
        
        Args:
            beat_times: Array of beat times in seconds
            tempo_min: Minimum tempo for binning
            tempo_max: Maximum tempo for binning
            n_bins: Number of tempo bins
            
        Returns:
            Array of bin indices (n_beats,)
        """
        n_beats = len(beat_times)
        targets = np.zeros(n_beats, dtype=np.int64)
        
        for i in range(n_beats):
            # Compute local tempo from surrounding beats
            if i > 0 and i < n_beats - 1:
                # Use 3-beat window for local tempo
                dt = beat_times[min(i+1, n_beats-1)] - beat_times[max(i-1, 0)]
                local_tempo = 120.0 / (dt / 2.0) if dt > 0 else 120.0
            elif i == 0 and n_beats > 1:
                dt = beat_times[1] - beat_times[0]
                local_tempo = 60.0 / dt if dt > 0 else 120.0
            else:
                local_tempo = 120.0
            
            # Clamp and bin
            local_tempo = np.clip(local_tempo, tempo_min, tempo_max)
            bin_idx = int((local_tempo - tempo_min) / (tempo_max - tempo_min) * (n_bins - 1))
            targets[i] = np.clip(bin_idx, 0, n_bins - 1)
        
        return targets


class EnergyPredictor(nn.Module):
    """Predict energy level at each beat position.
    
    Task: Given beat features, predict the energy level bin.
    This forces the encoder to understand dynamics.
    """
    
    def __init__(self, hidden_dim: int, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_bins),
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict energy bin logits.
        
        Args:
            encoded: Encoded features from shared encoder (B, hidden_dim)
            
        Returns:
            Logits for energy bins (B, n_bins)
        """
        return self.head(encoded)
    
    @staticmethod
    def compute_targets(
        beat_features: np.ndarray,
        n_bins: int = 10,
        energy_feature_idx: int = 3,  # RMS energy is typically at index 3
    ) -> np.ndarray:
        """Compute energy bin targets from beat features.
        
        Args:
            beat_features: Beat feature array (n_beats, feature_dim)
            n_bins: Number of energy bins
            energy_feature_idx: Index of energy feature in beat_features
            
        Returns:
            Array of bin indices (n_beats,)
        """
        n_beats = len(beat_features)
        
        # Extract energy values
        if beat_features.ndim == 2 and beat_features.shape[1] > energy_feature_idx:
            energies = beat_features[:, energy_feature_idx]
        else:
            # Fallback: use mean of all features as proxy
            energies = np.mean(beat_features, axis=-1) if beat_features.ndim == 2 else beat_features
        
        # Normalize to [0, 1]
        e_min, e_max = energies.min(), energies.max()
        if e_max > e_min:
            energies_norm = (energies - e_min) / (e_max - e_min)
        else:
            energies_norm = np.ones(n_beats) * 0.5
        
        # Bin
        targets = (energies_norm * (n_bins - 1)).astype(np.int64)
        targets = np.clip(targets, 0, n_bins - 1)
        
        return targets


class GoodBadClassifier(nn.Module):
    """Binary classifier head for predicting whether an edited beat is 'good'.

    Takes encoder output (B, hidden_dim) and returns a single logit per sample.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded)


class PhraseBoundaryDetector(nn.Module):
    """Detect phrase boundaries (every 4 or 8 beats typically).
    
    Task: Binary classification - is this beat a phrase boundary?
    This forces the encoder to understand musical structure at phrase level.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict phrase boundary logit.
        
        Args:
            encoded: Encoded features from shared encoder (B, hidden_dim)
            
        Returns:
            Logit for phrase boundary (B, 1)
        """
        return self.head(encoded)
    
    @staticmethod
    def compute_targets(
        n_beats: int,
        phrase_length: int = 8,
    ) -> np.ndarray:
        """Compute phrase boundary targets.
        
        Args:
            n_beats: Number of beats
            phrase_length: Beats per phrase
            
        Returns:
            Binary array (n_beats,) where 1 = phrase boundary
        """
        targets = np.zeros(n_beats, dtype=np.float32)
        
        # Mark every phrase_length beats as boundary
        for i in range(0, n_beats, phrase_length):
            targets[i] = 1.0
        
        # Also mark half-phrase boundaries with lower weight (optional)
        # for i in range(phrase_length // 2, n_beats, phrase_length):
        #     targets[i] = 0.5
        
        return targets


class BeatReconstructor(nn.Module):
    """Reconstruct/predict next beat features.
    
    Task: Given current encoded state, predict features of next beat.
    This forces the encoder to understand temporal patterns and transitions.
    """
    
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict next beat features.
        
        Args:
            encoded: Encoded features from shared encoder (B, hidden_dim)
            
        Returns:
            Predicted features (B, feature_dim)
        """
        return self.head(encoded)
    
    @staticmethod
    def compute_targets(
        beat_features: np.ndarray,
        current_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute next beat feature targets.
        
        Args:
            beat_features: All beat features (n_beats, feature_dim)
            current_indices: Indices of current beats in batch
            
        Returns:
            Tuple of (targets, valid_mask) where valid_mask is 0 for last beat
        """
        n_beats = len(beat_features)
        batch_size = len(current_indices)
        feature_dim = beat_features.shape[1] if beat_features.ndim == 2 else 1
        
        targets = np.zeros((batch_size, feature_dim), dtype=np.float32)
        valid_mask = np.ones(batch_size, dtype=np.float32)
        
        for i, idx in enumerate(current_indices):
            if idx < n_beats - 1:
                targets[i] = beat_features[idx + 1]
            else:
                valid_mask[i] = 0.0
        
        return targets, valid_mask


class MelReconstructor(nn.Module):
    """Predict per-beat mel-spectrogram vector for the edited output.

    This uses the encoder's hidden representation to predict a mel vector
    representing the edited audio for the current beat (aligned to beats).
    """

    def __init__(self, hidden_dim: int, mel_dim: int = 128):
        super().__init__()
        self.mel_dim = mel_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mel_dim),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded)


class SectionSimilarityPredictor(nn.Module):
    """Predict if two sections are similar (e.g., verse-verse, chorus-chorus).
    
    Task: Given two encoded sections, predict similarity score.
    This forces the encoder to understand high-level song structure.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Siamese-style comparison
        self.comparison = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, encoded1: torch.Tensor, encoded2: torch.Tensor) -> torch.Tensor:
        """Predict similarity between two encoded sections.
        
        Args:
            encoded1: First section encoding (B, hidden_dim)
            encoded2: Second section encoding (B, hidden_dim)
            
        Returns:
            Similarity logit (B, 1)
        """
        combined = torch.cat([encoded1, encoded2], dim=-1)
        return self.comparison(combined)


class AuxiliaryTaskModule(nn.Module):
    """Combined module for all auxiliary tasks.
    
    Shares the encoder with the main policy network and adds
    task-specific prediction heads.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        beat_feature_dim: int,
        config: Optional[AuxiliaryConfig] = None,
    ):
        super().__init__()
        self.config = config or AuxiliaryConfig()
        self.hidden_dim = hidden_dim
        self.beat_feature_dim = beat_feature_dim
        
        # Initialize task heads based on config
        if self.config.use_tempo_prediction:
            self.tempo_predictor = TempoPredictor(hidden_dim, self.config.tempo_bins)
        else:
            self.tempo_predictor = None
            
        if self.config.use_energy_prediction:
            self.energy_predictor = EnergyPredictor(hidden_dim, self.config.energy_bins)
        else:
            self.energy_predictor = None
            
        if self.config.use_phrase_boundary:
            self.phrase_detector = PhraseBoundaryDetector(hidden_dim)
        else:
            self.phrase_detector = None
            
        if self.config.use_beat_reconstruction:
            self.reconstructor = BeatReconstructor(hidden_dim, beat_feature_dim)
        else:
            self.reconstructor = None
            
        if self.config.use_section_similarity:
            self.similarity_predictor = SectionSimilarityPredictor(hidden_dim)
        else:
            self.similarity_predictor = None
        
        logger.info(
            f"AuxiliaryTaskModule initialized: "
            f"tempo={self.config.use_tempo_prediction}, "
            f"energy={self.config.use_energy_prediction}, "
            f"phrase={self.config.use_phrase_boundary}, "
            f"reconstruction={self.config.use_beat_reconstruction}, "
            f"similarity={self.config.use_section_similarity}"
        )
    
    def forward(
        self,
        encoded: torch.Tensor,
        encoded2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all auxiliary task predictions.
        
        Args:
            encoded: Encoded features from shared encoder (B, hidden_dim)
            encoded2: Optional second encoding for similarity task
            
        Returns:
            Dict of task name -> predictions
        """
        predictions = {}
        
        if self.tempo_predictor is not None:
            predictions["tempo"] = self.tempo_predictor(encoded)
            
        if self.energy_predictor is not None:
            predictions["energy"] = self.energy_predictor(encoded)
            
        if self.phrase_detector is not None:
            predictions["phrase"] = self.phrase_detector(encoded)
            
        if self.reconstructor is not None:
            predictions["reconstruction"] = self.reconstructor(encoded)
            
        if self.similarity_predictor is not None and encoded2 is not None:
            predictions["similarity"] = self.similarity_predictor(encoded, encoded2)
        
        return predictions
    
    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all auxiliary losses.
        
        Args:
            predictions: Dict of task name -> predictions
            targets: Dict of task name -> targets
            epoch: Current epoch for curriculum learning
            
        Returns:
            Tuple of (total_loss, loss_breakdown_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Curriculum learning: gradually increase auxiliary loss weight
        warmup_factor = min(1.0, epoch / max(self.config.warmup_epochs, 1))
        
        if "tempo" in predictions and "tempo" in targets:
            tempo_loss = F.cross_entropy(predictions["tempo"], targets["tempo"])
            weight = self.config.tempo_loss_weight * warmup_factor
            total_loss = total_loss + weight * tempo_loss
            losses["tempo_loss"] = tempo_loss.item()
        
        if "energy" in predictions and "energy" in targets:
            energy_loss = F.cross_entropy(predictions["energy"], targets["energy"])
            weight = self.config.energy_loss_weight * warmup_factor
            total_loss = total_loss + weight * energy_loss
            losses["energy_loss"] = energy_loss.item()
        
        if "phrase" in predictions and "phrase" in targets:
            # Binary cross entropy for phrase boundaries
            phrase_pred = predictions["phrase"].squeeze(-1)
            phrase_target = targets["phrase"].float()
            phrase_loss = F.binary_cross_entropy_with_logits(phrase_pred, phrase_target)
            weight = self.config.phrase_loss_weight * warmup_factor
            total_loss = total_loss + weight * phrase_loss
            losses["phrase_loss"] = phrase_loss.item()
        
        if "reconstruction" in predictions and "reconstruction" in targets:
            recon_pred = predictions["reconstruction"]
            recon_target = targets["reconstruction"]
            recon_mask = targets.get("reconstruction_mask", torch.ones(len(recon_pred), device=recon_pred.device))
            
            # Normalize target features to prevent MSE explosion
            # Use batch-wise normalization
            target_mean = recon_target.mean()
            target_std = recon_target.std() + 1e-8
            recon_target_norm = (recon_target - target_mean) / target_std
            
            # Also normalize predictions to same scale
            pred_mean = recon_pred.mean()
            pred_std = recon_pred.std() + 1e-8
            recon_pred_norm = (recon_pred - pred_mean) / pred_std
            
            # MSE loss on normalized values - should be ~1.0 scale
            recon_loss = F.mse_loss(recon_pred_norm, recon_target_norm, reduction='none')
            recon_loss = (recon_loss.mean(dim=-1) * recon_mask).sum() / (recon_mask.sum() + 1e-8)
            
            # Clamp to prevent explosion
            recon_loss = torch.clamp(recon_loss, 0, 10.0)
            
            weight = self.config.reconstruction_loss_weight * warmup_factor
            total_loss = total_loss + weight * recon_loss
            losses["reconstruction_loss"] = recon_loss.item()
        
        if "similarity" in predictions and "similarity" in targets:
            sim_pred = predictions["similarity"].squeeze(-1)
            sim_target = targets["similarity"].float()
            sim_loss = F.binary_cross_entropy_with_logits(sim_pred, sim_target)
            weight = self.config.similarity_loss_weight * warmup_factor
            total_loss = total_loss + weight * sim_loss
            losses["similarity_loss"] = sim_loss.item()
        
        losses["total_auxiliary_loss"] = total_loss.item()
        losses["warmup_factor"] = warmup_factor
        
        return total_loss, losses


def compute_auxiliary_targets(
    beat_times: np.ndarray,
    beat_features: np.ndarray,
    beat_indices: np.ndarray,
    config: Optional[AuxiliaryConfig] = None,
) -> Dict[str, np.ndarray]:
    """Compute all auxiliary task targets for a batch.
    
    Args:
        beat_times: Array of beat times (n_beats,)
        beat_features: Beat feature array (n_beats, feature_dim)
        beat_indices: Current beat indices in batch (batch_size,)
        config: Auxiliary task configuration
        
    Returns:
        Dict of task name -> targets
    """
    config = config or AuxiliaryConfig()
    targets = {}
    n_beats = len(beat_times)
    batch_size = len(beat_indices)
    
    if config.use_tempo_prediction:
        # Compute tempo targets for all beats, then index into batch
        all_tempo_targets = TempoPredictor.compute_targets(
            beat_times, config.tempo_min, config.tempo_max, config.tempo_bins
        )
        targets["tempo"] = all_tempo_targets[np.clip(beat_indices, 0, n_beats-1)]
    
    if config.use_energy_prediction:
        all_energy_targets = EnergyPredictor.compute_targets(
            beat_features, config.energy_bins
        )
        targets["energy"] = all_energy_targets[np.clip(beat_indices, 0, n_beats-1)]
    
    if config.use_phrase_boundary:
        all_phrase_targets = PhraseBoundaryDetector.compute_targets(
            n_beats, config.phrase_length
        )
        targets["phrase"] = all_phrase_targets[np.clip(beat_indices, 0, n_beats-1)]
    
    if config.use_beat_reconstruction:
        recon_targets, recon_mask = BeatReconstructor.compute_targets(
            beat_features, beat_indices
        )
        targets["reconstruction"] = recon_targets
        targets["reconstruction_mask"] = recon_mask
    
    return targets


class AuxiliaryTaskModule(nn.Module):
    """Combined auxiliary task module used by the Agent.

    Provides multiple prediction heads and a unified loss computation.
    """

    def __init__(self, hidden_dim: int, beat_feature_dim: int, config: Optional[AuxiliaryConfig] = None):
        super().__init__()
        self.config = config or AuxiliaryConfig()
        self.hidden_dim = hidden_dim
        self.beat_feature_dim = beat_feature_dim

        # Heads
        self.tempo_predictor = TempoPredictor(hidden_dim, self.config.tempo_bins) if self.config.use_tempo_prediction else None
        self.energy_predictor = EnergyPredictor(hidden_dim, self.config.energy_bins) if self.config.use_energy_prediction else None
        self.phrase_detector = PhraseBoundaryDetector(hidden_dim) if self.config.use_phrase_boundary else None
        self.reconstructor = BeatReconstructor(hidden_dim, beat_feature_dim) if self.config.use_beat_reconstruction else None
        self.mel_reconstructor = MelReconstructor(hidden_dim, self.config.mel_dim) if self.config.use_mel_reconstruction else None
        self.good_bad_classifier = GoodBadClassifier(hidden_dim) if self.config.use_good_bad_classifier else None
        self.similarity_predictor = SectionSimilarityPredictor(hidden_dim) if self.config.use_section_similarity else None
        # Build mel->chroma mapping matrix (mel_dim x 12) and register as buffer
        self.register_buffer_name = None
        try:
            import librosa
            n_mels = self.config.mel_dim
            sr = getattr(self.config, "mel_sample_rate", 22050)
            # Compute mel center frequencies
            mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
            # Build mapping: for each mel bin, map to nearest MIDI pitch class (chroma)
            M = np.zeros((n_mels, 12), dtype=np.float32)
            for i, f in enumerate(mel_freqs):
                if f <= 0:
                    continue
                midi = 69 + 12.0 * np.log2(f / 440.0)
                chroma_idx = int(np.round(midi)) % 12
                M[i, chroma_idx] = 1.0
            mel_to_chroma = torch.from_numpy(M)
            self.register_buffer_name = 'mel_to_chroma'
            self.register_buffer('mel_to_chroma', mel_to_chroma)
        except Exception:
            # Fallback: simple evenly-distributed mapping if librosa unavailable
            try:
                n_mels = self.config.mel_dim
                M = np.zeros((n_mels, 12), dtype=np.float32)
                for i in range(n_mels):
                    M[i, i % 12] = 1.0
                mel_to_chroma = torch.from_numpy(M)
                self.register_buffer_name = 'mel_to_chroma'
                self.register_buffer('mel_to_chroma', mel_to_chroma)
            except Exception:
                # Last resort: don't register buffer
                self.mel_to_chroma = None

    def forward(self, encoded: torch.Tensor, encoded2: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        preds: Dict[str, torch.Tensor] = {}
        if self.tempo_predictor is not None:
            preds["tempo"] = self.tempo_predictor(encoded)
        if self.energy_predictor is not None:
            preds["energy"] = self.energy_predictor(encoded)
        if self.phrase_detector is not None:
            preds["phrase"] = self.phrase_detector(encoded)
        if self.reconstructor is not None:
            preds["reconstruction"] = self.reconstructor(encoded)
        if self.mel_reconstructor is not None:
            preds["mel_reconstruction"] = self.mel_reconstructor(encoded)
        if self.good_bad_classifier is not None:
            preds["good_bad"] = self.good_bad_classifier(encoded)
        if self.similarity_predictor is not None and encoded2 is not None:
            preds["similarity"] = self.similarity_predictor(encoded, encoded2)
        return preds

    def compute_losses(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, float] = {}
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        warmup_factor = min(1.0, epoch / max(self.config.warmup_epochs, 1))

        if "tempo" in predictions and "tempo" in targets:
            tempo_loss = F.cross_entropy(predictions["tempo"], targets["tempo"].to(device))
            w = self.config.tempo_loss_weight * warmup_factor
            total_loss = total_loss + w * tempo_loss
            losses["tempo_loss"] = float(tempo_loss.item())

        if "energy" in predictions and "energy" in targets:
            energy_loss = F.cross_entropy(predictions["energy"], targets["energy"].to(device))
            w = self.config.energy_loss_weight * warmup_factor
            total_loss = total_loss + w * energy_loss
            losses["energy_loss"] = float(energy_loss.item())

        if "phrase" in predictions and "phrase" in targets:
            phrase_pred = predictions["phrase"].squeeze(-1)
            phrase_t = targets["phrase"].to(device).float()
            phrase_loss = F.binary_cross_entropy_with_logits(phrase_pred, phrase_t)
            w = self.config.phrase_loss_weight * warmup_factor
            total_loss = total_loss + w * phrase_loss
            losses["phrase_loss"] = float(phrase_loss.item())

        if "reconstruction" in predictions and "reconstruction" in targets:
            pred = predictions["reconstruction"]
            tgt = targets["reconstruction"].to(device)
            mask = targets.get("reconstruction_mask", torch.ones(pred.shape[0], device=device))
            # Normalize
            tgt_mean = tgt.mean(); tgt_std = tgt.std() + 1e-8
            pred_mean = pred.mean(); pred_std = pred.std() + 1e-8
            pred_n = (pred - pred_mean) / pred_std
            tgt_n = (tgt - tgt_mean) / tgt_std
            recon_loss = F.mse_loss(pred_n, tgt_n, reduction='none')
            recon_loss = (recon_loss.mean(dim=-1) * mask).sum() / (mask.sum().clamp(min=1.0))
            recon_loss = torch.clamp(recon_loss, 0.0, 10.0)
            w = self.config.reconstruction_loss_weight * warmup_factor
            total_loss = total_loss + w * recon_loss
            losses["reconstruction_loss"] = float(recon_loss.item())

        if "mel_reconstruction" in predictions and "mel_reconstruction" in targets:
            pred = predictions["mel_reconstruction"]
            tgt = targets["mel_reconstruction"].to(device)
            mask = targets.get("mel_reconstruction_mask", torch.ones(pred.shape[0], device=device))
            pred_mean = pred.mean(); pred_std = pred.std() + 1e-8
            tgt_mean = tgt.mean(); tgt_std = tgt.std() + 1e-8
            pred_n = (pred - pred_mean) / pred_std
            tgt_n = (tgt - tgt_mean) / tgt_std
            mel_loss = F.mse_loss(pred_n, tgt_n, reduction='none')
            mel_loss = (mel_loss.mean(dim=-1) * mask).sum() / (mask.sum().clamp(min=1.0))
            mel_loss = torch.clamp(mel_loss, 0.0, 50.0)
            w = self.config.mel_reconstruction_weight * warmup_factor
            total_loss = total_loss + w * mel_loss
            losses["mel_reconstruction_loss"] = float(mel_loss.item())

        # Chroma continuity loss derived from mel vectors (differentiable linear mapping)
        if self.config.use_chroma_continuity and hasattr(self, 'mel_to_chroma') and self.mel_to_chroma is not None:
            if "mel_reconstruction" in predictions and "mel_reconstruction" in targets:
                try:
                    pred = predictions["mel_reconstruction"]
                    tgt = targets["mel_reconstruction"].to(device)
                    # Map mel -> chroma (B, 12)
                    M = self.mel_to_chroma.to(device)
                    pred_chroma = torch.matmul(pred, M)
                    tgt_chroma = torch.matmul(tgt, M)

                    # Normalize and compute MSE
                    p_mean = pred_chroma.mean(); p_std = pred_chroma.std() + 1e-8
                    t_mean = tgt_chroma.mean(); t_std = tgt_chroma.std() + 1e-8
                    pred_c_n = (pred_chroma - p_mean) / p_std
                    tgt_c_n = (tgt_chroma - t_mean) / t_std
                    chroma_loss = F.mse_loss(pred_c_n, tgt_c_n, reduction='none')
                    chroma_loss = chroma_loss.mean(dim=-1).mean()
                    chroma_loss = torch.clamp(chroma_loss, 0.0, 10.0)
                    w = self.config.chroma_loss_weight * warmup_factor
                    total_loss = total_loss + w * chroma_loss
                    losses["chroma_loss"] = float(chroma_loss.item())
                except Exception:
                    # If anything fails, skip chroma loss silently
                    pass

        if "good_bad" in predictions and "good_bad" in targets:
            gb_pred = predictions["good_bad"].squeeze(-1)
            gb_t = targets["good_bad"].to(device).float()
            gb_loss = F.binary_cross_entropy_with_logits(gb_pred, gb_t)
            w = self.config.good_bad_loss_weight * warmup_factor
            total_loss = total_loss + w * gb_loss
            losses["good_bad_loss"] = float(gb_loss.item())

        if "similarity" in predictions and "similarity" in targets:
            sim_pred = predictions["similarity"].squeeze(-1)
            sim_t = targets["similarity"].to(device).float()
            sim_loss = F.binary_cross_entropy_with_logits(sim_pred, sim_t)
            w = self.config.similarity_loss_weight * warmup_factor
            total_loss = total_loss + w * sim_loss
            losses["similarity_loss"] = float(sim_loss.item())

        losses["total_auxiliary_loss"] = float(total_loss.item())
        return total_loss, losses


class AuxiliaryTargetComputer:
    """Caches and computes auxiliary targets efficiently during training."""
    
    def __init__(self, config: Optional[AuxiliaryConfig] = None):
        self.config = config or AuxiliaryConfig()
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
    
    def get_targets(
        self,
        audio_id: str,
        beat_times: np.ndarray,
        beat_features: np.ndarray,
        beat_indices: np.ndarray,
        edited_mel: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Get auxiliary targets, using cache when possible.
        
        Args:
            audio_id: Unique identifier for this audio file
            beat_times: Array of beat times
            beat_features: Beat feature array
            beat_indices: Current beat indices in batch
            
        Returns:
            Dict of task name -> targets for batch
        """
        # Check if we have cached full-track targets
        if audio_id not in self._cache:
            self._cache[audio_id] = self._compute_full_targets(beat_times, beat_features)
        
        # Index into cached targets
        cached = self._cache[audio_id]
        n_beats = len(beat_times)
        targets = {}
        
        for key, full_targets in cached.items():
            if key == "reconstruction_full":
                continue  # Handle reconstruction separately
            safe_indices = np.clip(beat_indices, 0, n_beats - 1)
            targets[key] = full_targets[safe_indices]
        
        # Handle reconstruction (needs next beat)
        if self.config.use_beat_reconstruction:
            recon_targets, recon_mask = BeatReconstructor.compute_targets(
                beat_features, beat_indices
            )
            targets["reconstruction"] = recon_targets
            targets["reconstruction_mask"] = recon_mask

        # Handle mel reconstruction targets (if provided) or load from disk cache
        if self.config.use_mel_reconstruction:
            pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            cache_cfg = getattr(self.config, 'mel_chroma_cache', None)
            cache_dir = cache_cfg if (cache_cfg and os.path.isabs(cache_cfg)) else os.path.join(pkg_root, cache_cfg or 'training_data/feature_cache/chroma')

            # If edited_mel not provided, try loading per-beat mel from disk cache
            if edited_mel is None:
                try:
                    mel_path = os.path.join(cache_dir, f"{audio_id}_mel.npy")
                    chroma_path = os.path.join(cache_dir, f"{audio_id}_chroma.npy")
                    if os.path.exists(mel_path):
                        per_beat = np.load(mel_path)
                        n_beats = len(beat_times)
                        safe_indices = np.clip(beat_indices, 0, n_beats - 1)
                        targets["mel_reconstruction"] = per_beat[safe_indices]
                        targets["mel_reconstruction_mask"] = np.ones((len(safe_indices),), dtype=np.float32)
                        if os.path.exists(chroma_path):
                            try:
                                chroma_per_beat = np.load(chroma_path).astype(np.float32)
                                self._cache.setdefault(audio_id, {})["chroma_reconstruction_full"] = chroma_per_beat
                            except Exception:
                                pass
                        edited_mel = per_beat
                except Exception:
                    pass

            # If we have edited_mel data (either passed in or loaded), compute per-beat targets
            if edited_mel is not None:
                try:
                    n_beats = len(beat_times)
                    # If edited_mel already aligned per-beat
                    if isinstance(edited_mel, np.ndarray) and edited_mel.shape[0] == n_beats:
                        safe_indices = np.clip(beat_indices, 0, n_beats - 1)
                        targets["mel_reconstruction"] = edited_mel[safe_indices]
                        targets["mel_reconstruction_mask"] = np.ones((len(safe_indices),), dtype=np.float32)
                    else:
                        mel = np.array(edited_mel)
                        if mel.ndim == 2:
                            # Normalize shape to (n_mels, n_frames)
                            if mel.shape[0] == self.config.mel_dim:
                                pass
                            elif mel.shape[1] == self.config.mel_dim:
                                mel = mel.T
                            else:
                                mel = None

                            if mel is not None:
                                cached_per_beat = self._cache.get(audio_id, {}).get("mel_reconstruction_full")
                                if cached_per_beat is None:
                                    n_mels, n_frames = mel.shape
                                    frames_per_beat = int(np.ceil(n_frames / max(1, n_beats)))
                                    target_len = frames_per_beat * n_beats
                                    if target_len != n_frames:
                                        pad = target_len - n_frames
                                        mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)
                                        n_frames = mel.shape[1]

                                    mel_reshaped = mel.reshape(n_mels, n_beats, frames_per_beat)
                                    per_beat = mel_reshaped.mean(axis=2).T.astype(np.float32)
                                    try:
                                        self._cache.setdefault(audio_id, {})["mel_reconstruction_full"] = per_beat
                                    except Exception:
                                        pass
                                else:
                                    per_beat = cached_per_beat

                                safe_indices = np.clip(beat_indices, 0, n_beats - 1)
                                targets["mel_reconstruction"] = per_beat[safe_indices]
                                targets["mel_reconstruction_mask"] = np.ones((len(safe_indices),), dtype=np.float32)

                                # Precompute chroma per beat if enabled and not already cached
                                if self.config.use_chroma_continuity and self._cache.get(audio_id, {}).get("chroma_reconstruction_full") is None:
                                    try:
                                        import librosa
                                        mel_freqs = librosa.mel_frequencies(n_mels=self.config.mel_dim, fmin=0.0, fmax=getattr(self.config, 'mel_sample_rate', 22050) / 2.0)
                                        M = np.zeros((self.config.mel_dim, 12), dtype=np.float32)
                                        for i, f in enumerate(mel_freqs):
                                            if f <= 0:
                                                continue
                                            midi = 69 + 12.0 * np.log2(f / 440.0)
                                            chroma_idx = int(np.round(midi)) % 12
                                            M[i, chroma_idx] = 1.0
                                    except Exception:
                                        M = np.zeros((self.config.mel_dim, 12), dtype=np.float32)
                                        for i in range(self.config.mel_dim):
                                            M[i, i % 12] = 1.0

                                    chroma_per_beat = np.dot(per_beat, M).astype(np.float32)
                                    try:
                                        self._cache.setdefault(audio_id, {})["chroma_reconstruction_full"] = chroma_per_beat
                                    except Exception:
                                        pass
                except Exception:
                    pass

        return targets

    def _compute_full_targets(
        self,
        beat_times: np.ndarray,
        beat_features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute targets for all beats in a track (for caching)."""
        n_beats = len(beat_times)
        targets: Dict[str, np.ndarray] = {}

        if self.config.use_tempo_prediction:
            targets["tempo"] = TempoPredictor.compute_targets(
                beat_times, self.config.tempo_min, self.config.tempo_max, self.config.tempo_bins
            )

        if self.config.use_energy_prediction:
            targets["energy"] = EnergyPredictor.compute_targets(
                beat_features, self.config.energy_bins
            )

        if self.config.use_phrase_boundary:
            targets["phrase"] = PhraseBoundaryDetector.compute_targets(
                n_beats, self.config.phrase_length
            )

        return targets

    def clear_cache(self):
        """Clear the target cache."""
        self._cache.clear()

    def update_config(self, new_config: Optional[AuxiliaryConfig] = None):
        """Update internal config and clear caches if settings that affect targets changed.

        Clears cached per-track mel/chroma targets when `mel_dim` or
        `use_chroma_continuity` changes to avoid stale cached arrays.
        """
        if new_config is None:
            return

        # If mel_dim or chroma setting changed, clear cache to force recompute
        try:
            prev_mel_dim = getattr(self.config, 'mel_dim', None)
            prev_chroma = getattr(self.config, 'use_chroma_continuity', None)
            new_mel_dim = getattr(new_config, 'mel_dim', None)
            new_chroma = getattr(new_config, 'use_chroma_continuity', None)
            if prev_mel_dim != new_mel_dim or prev_chroma != new_chroma:
                self.clear_cache()
        except Exception:
            # On unexpected failures, be conservative and clear cache
            self.clear_cache()

        self.config = new_config

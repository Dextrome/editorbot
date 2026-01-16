"""Auxiliary task heads for Phase 2 training.

These auxiliary tasks help the encoder learn better representations
by providing additional supervision signals beyond the main policy objective.

Based on learnings from rl_editor:
- Multi-task learning improves feature representations
- Auxiliary tasks act as regularization
- Helps with sample efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..config import Phase2Config


class TempoPredictor(nn.Module):
    """Predict tempo bin from encoded state.

    Tempo prediction helps the encoder learn tempo-aware representations,
    which is important for edit decisions (e.g., cuts should align with beats).
    """

    def __init__(
        self,
        input_dim: int,
        n_tempo_bins: int = 10,
        hidden_dim: int = 128,
    ):
        """Initialize tempo predictor.

        Args:
            input_dim: Input dimension from encoder
            n_tempo_bins: Number of tempo bins (e.g., 60-180 BPM in 10 bins)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_tempo_bins = n_tempo_bins

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_tempo_bins),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict tempo bin logits.

        Args:
            encoded: Encoded state (B, T, D) or pooled (B, D)

        Returns:
            Tempo logits (B, n_tempo_bins)
        """
        # Global average pooling if sequence
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)  # (B, D)

        return self.head(encoded)  # (B, n_tempo_bins)


class EnergyPredictor(nn.Module):
    """Predict energy level bin from encoded state.

    Energy prediction helps the encoder understand dynamic levels,
    which is important for fade decisions and transitions.
    """

    def __init__(
        self,
        input_dim: int,
        n_energy_bins: int = 5,
        hidden_dim: int = 128,
    ):
        """Initialize energy predictor.

        Args:
            input_dim: Input dimension from encoder
            n_energy_bins: Number of energy bins (e.g., very low to very high)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_energy_bins = n_energy_bins

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_energy_bins),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict energy bin logits.

        Args:
            encoded: Encoded state (B, T, D) or pooled (B, D)

        Returns:
            Energy logits (B, n_energy_bins)
        """
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)

        return self.head(encoded)


class PhraseDetector(nn.Module):
    """Detect phrase boundaries from encoded state.

    Phrase detection helps the encoder understand musical structure,
    which is critical for making edit decisions that preserve musical coherence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        """Initialize phrase detector.

        Args:
            input_dim: Input dimension from encoder
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Frame-wise prediction
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Binary: is phrase boundary
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Predict phrase boundary probability.

        Args:
            encoded: Encoded state (B, T, D)

        Returns:
            Phrase boundary logits (B, T, 1) or (B, T)
        """
        return self.head(encoded).squeeze(-1)  # (B, T)


class MelReconstructor(nn.Module):
    """Reconstruct mel spectrogram from encoded state.

    This auxiliary task ensures the encoder preserves information
    needed for reconstruction, acting as a form of regularization.
    """

    def __init__(
        self,
        input_dim: int,
        n_mels: int = 128,
        hidden_dim: int = 256,
    ):
        """Initialize mel reconstructor.

        Args:
            input_dim: Input dimension from encoder
            n_mels: Number of mel bins to reconstruct
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_mels = n_mels

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct mel spectrogram.

        Args:
            encoded: Encoded state (B, T, D)

        Returns:
            Reconstructed mel (B, T, n_mels)
        """
        return self.head(encoded)


class AuxiliaryTaskModule(nn.Module):
    """Combined auxiliary task module.

    Manages all auxiliary task heads and computes combined loss.
    """

    def __init__(
        self,
        config: Phase2Config,
        encoder_dim: int,
    ):
        """Initialize auxiliary task module.

        Args:
            config: Phase2Config with auxiliary task settings
            encoder_dim: Dimension of encoder output
        """
        super().__init__()
        self.config = config
        self.encoder_dim = encoder_dim

        # Initialize heads if enabled
        self.tempo_predictor = None
        self.energy_predictor = None
        self.phrase_detector = None
        self.mel_reconstructor = None

        if config.aux_tempo_weight > 0:
            self.tempo_predictor = TempoPredictor(
                input_dim=encoder_dim,
                n_tempo_bins=10,
                hidden_dim=128,
            )

        if config.aux_energy_weight > 0:
            self.energy_predictor = EnergyPredictor(
                input_dim=encoder_dim,
                n_energy_bins=5,
                hidden_dim=128,
            )

        if config.aux_phrase_weight > 0:
            self.phrase_detector = PhraseDetector(
                input_dim=encoder_dim,
                hidden_dim=128,
            )

        if config.aux_mel_reconstruction_weight > 0:
            self.mel_reconstructor = MelReconstructor(
                input_dim=encoder_dim,
                n_mels=config.audio.n_mels,
                hidden_dim=256,
            )

    def forward(
        self,
        encoded: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run auxiliary task heads and optionally compute losses.

        Args:
            encoded: Encoded state (B, T, D)
            targets: Optional dict of targets:
                - tempo_bin: (B,) int - tempo class
                - energy_bin: (B,) int - energy class
                - phrase_boundaries: (B, T) float - phrase boundary labels
                - mel: (B, T, n_mels) float - target mel spectrogram

        Returns:
            Tuple of (predictions dict, losses dict)
        """
        predictions = {}
        losses = {}

        # Tempo prediction
        if self.tempo_predictor is not None:
            tempo_logits = self.tempo_predictor(encoded)
            predictions['tempo_logits'] = tempo_logits

            if targets is not None and 'tempo_bin' in targets:
                loss = F.cross_entropy(tempo_logits, targets['tempo_bin'])
                losses['tempo'] = loss * self.config.aux_tempo_weight

        # Energy prediction
        if self.energy_predictor is not None:
            energy_logits = self.energy_predictor(encoded)
            predictions['energy_logits'] = energy_logits

            if targets is not None and 'energy_bin' in targets:
                loss = F.cross_entropy(energy_logits, targets['energy_bin'])
                losses['energy'] = loss * self.config.aux_energy_weight

        # Phrase detection
        if self.phrase_detector is not None:
            phrase_logits = self.phrase_detector(encoded)
            predictions['phrase_logits'] = phrase_logits

            if targets is not None and 'phrase_boundaries' in targets:
                loss = F.binary_cross_entropy_with_logits(
                    phrase_logits, targets['phrase_boundaries']
                )
                losses['phrase'] = loss * self.config.aux_phrase_weight

        # Mel reconstruction
        if self.mel_reconstructor is not None:
            mel_pred = self.mel_reconstructor(encoded)
            predictions['mel_pred'] = mel_pred

            if targets is not None and 'mel' in targets:
                loss = F.l1_loss(mel_pred, targets['mel'])
                losses['mel_reconstruction'] = loss * self.config.aux_mel_reconstruction_weight

        return predictions, losses

    def get_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total auxiliary loss.

        Args:
            losses: Dict of individual losses

        Returns:
            Total loss (scalar)
        """
        if not losses:
            return torch.tensor(0.0)

        return sum(losses.values())


def compute_tempo_bin(tempo: float, n_bins: int = 10, min_bpm: float = 60, max_bpm: float = 180) -> int:
    """Convert tempo to bin index.

    Args:
        tempo: Tempo in BPM
        n_bins: Number of bins
        min_bpm: Minimum BPM for binning
        max_bpm: Maximum BPM for binning

    Returns:
        Bin index (0 to n_bins-1)
    """
    tempo = max(min_bpm, min(max_bpm, tempo))
    bin_idx = int((tempo - min_bpm) / (max_bpm - min_bpm) * n_bins)
    return min(bin_idx, n_bins - 1)


def compute_energy_bin(energy: float, n_bins: int = 5) -> int:
    """Convert RMS energy to bin index.

    Args:
        energy: Normalized RMS energy (0-1)
        n_bins: Number of bins

    Returns:
        Bin index (0 to n_bins-1)
    """
    energy = max(0, min(1, energy))
    bin_idx = int(energy * n_bins)
    return min(bin_idx, n_bins - 1)


def detect_phrase_boundaries(
    beat_times: torch.Tensor,
    tempo: float,
    beats_per_phrase: int = 16,
) -> torch.Tensor:
    """Create phrase boundary labels from beat times.

    Args:
        beat_times: Beat times in seconds
        tempo: Tempo in BPM
        beats_per_phrase: Number of beats per phrase (typically 8 or 16)

    Returns:
        Binary labels indicating phrase boundaries
    """
    n_beats = len(beat_times)
    boundaries = torch.zeros(n_beats)

    # Mark every beats_per_phrase beats as a boundary
    for i in range(0, n_beats, beats_per_phrase):
        boundaries[i] = 1.0

    return boundaries

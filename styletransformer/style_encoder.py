"""
Style Encoder - Extracts style embeddings from reference songs.

The style embedding captures:
- Structural patterns (verse-chorus-bridge progressions)
- Energy curves (how intensity builds and releases)
- Transition patterns (how phrases connect)
- Rhythmic texture patterns
- Harmonic progression tendencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StyleFeatures:
    """Raw features extracted from a song for style analysis."""
    # Time-series features (variable length)
    energy_curve: np.ndarray        # Energy over time (normalized)
    brightness_curve: np.ndarray    # Spectral centroid over time
    onset_density_curve: np.ndarray # Note density over time
    chroma_sequence: np.ndarray     # Pitch content over time (12 x T)
    
    # Structural features
    phrase_lengths: List[int]       # Lengths of detected phrases (in bars)
    phrase_energies: List[float]    # Energy of each phrase
    transition_types: List[str]     # e.g., "buildup", "drop", "sustain"
    
    # Global features
    tempo: float
    key: int                        # Dominant pitch class (0-11)
    avg_phrase_length: float
    energy_variance: float
    
    # Metadata
    duration: float
    sample_rate: int


class FeatureExtractor:
    """Extracts style features from audio."""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def extract(self, audio: np.ndarray, sr: int) -> StyleFeatures:
        """Extract all style features from audio."""
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        duration = len(audio) / sr
        
        # Tempo detection
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo)
        
        # Energy curve (RMS over time)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        energy_curve = rms / (rms.max() + 1e-8)  # Normalize
        
        # Brightness curve (spectral centroid)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
        brightness_curve = centroid / (centroid.max() + 1e-8)
        
        # Onset density curve
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
        # Smooth with a window to get density
        window_size = int(sr / self.hop_length)  # ~1 second window
        onset_density_curve = np.convolve(onset_env, np.ones(window_size)/window_size, mode='same')
        onset_density_curve = onset_density_curve / (onset_density_curve.max() + 1e-8)
        
        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=self.hop_length)
        
        # Dominant key (most common pitch class)
        avg_chroma = chroma.mean(axis=1)
        key = int(np.argmax(avg_chroma))
        
        # Phrase detection (simplified - based on energy valleys)
        phrase_lengths, phrase_energies, transition_types = self._detect_phrases(
            energy_curve, tempo, sr
        )
        
        # Global stats
        avg_phrase_length = np.mean(phrase_lengths) if phrase_lengths else 4.0
        energy_variance = float(np.var(energy_curve))
        
        return StyleFeatures(
            energy_curve=energy_curve,
            brightness_curve=brightness_curve,
            onset_density_curve=onset_density_curve,
            chroma_sequence=chroma,
            phrase_lengths=phrase_lengths,
            phrase_energies=phrase_energies,
            transition_types=transition_types,
            tempo=tempo,
            key=key,
            avg_phrase_length=avg_phrase_length,
            energy_variance=energy_variance,
            duration=duration,
            sample_rate=sr
        )
    
    def _detect_phrases(
        self, 
        energy_curve: np.ndarray, 
        tempo: float, 
        sr: int
    ) -> Tuple[List[int], List[float], List[str]]:
        """Detect phrases from energy curve."""
        # Frames per bar (assuming 4/4 time)
        frames_per_beat = (sr / self.hop_length) * (60 / tempo)
        frames_per_bar = frames_per_beat * 4
        
        # Find local minima in energy as phrase boundaries
        # Smooth first
        smoothed = np.convolve(energy_curve, np.ones(int(frames_per_bar))/frames_per_bar, mode='same')
        
        # Find valleys
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-smoothed, distance=int(frames_per_bar * 2))
        
        if len(valleys) < 2:
            # Not enough structure detected
            return [4], [float(np.mean(energy_curve))], ["sustain"]
        
        phrase_lengths = []
        phrase_energies = []
        transition_types = []
        
        boundaries = [0] + list(valleys) + [len(energy_curve)]
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            length_frames = end - start
            length_bars = int(round(length_frames / frames_per_bar))
            length_bars = max(1, min(16, length_bars))  # Clamp to 1-16 bars
            
            phrase_energy = float(np.mean(energy_curve[start:end]))
            phrase_lengths.append(length_bars)
            phrase_energies.append(phrase_energy)
            
            # Determine transition type
            if i > 0:
                prev_energy = phrase_energies[-2] if len(phrase_energies) > 1 else phrase_energy
                if phrase_energy > prev_energy * 1.2:
                    transition_types.append("buildup")
                elif phrase_energy < prev_energy * 0.8:
                    transition_types.append("drop")
                else:
                    transition_types.append("sustain")
            else:
                transition_types.append("start")
        
        return phrase_lengths, phrase_energies, transition_types


class StyleEncoderNet(nn.Module):
    """
    Neural network that encodes style features into a fixed-size embedding.
    
    Architecture:
    - 1D CNN for time-series features (energy, brightness, onset density)
    - Transformer for sequential patterns
    - MLP for global features
    - Combine all into style embedding
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Time-series encoder (for energy, brightness, onset curves)
        # Input: 3 channels (energy, brightness, onset)
        self.time_conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)  # Fixed output size
        )
        
        # Chroma encoder (for harmonic content)
        # Input: 12 channels (pitch classes)
        self.chroma_conv = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Transformer for sequence modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Global features MLP (tempo, key, variance, etc.)
        self.global_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Final projection to embedding
        # time_conv: 256*64 + chroma_conv: 128*32 + transformer: 256 + global: 128
        combined_dim = 256 * 64 + 128 * 32 + 256 + 128
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self,
        time_features: torch.Tensor,     # (B, 3, T)
        chroma_features: torch.Tensor,   # (B, 12, T)
        global_features: torch.Tensor    # (B, 16)
    ) -> torch.Tensor:
        """
        Encode style features into embedding.
        
        Returns:
            style_embedding: (B, embedding_dim)
        """
        # Time-series encoding
        time_enc = self.time_conv(time_features)  # (B, 256, 64)
        time_flat = time_enc.flatten(1)           # (B, 256*64)
        
        # Chroma encoding
        chroma_enc = self.chroma_conv(chroma_features)  # (B, 128, 32)
        chroma_flat = chroma_enc.flatten(1)             # (B, 128*32)
        
        # Transformer for sequential patterns
        time_seq = time_enc.permute(0, 2, 1)  # (B, 64, 256)
        trans_out = self.transformer(time_seq)  # (B, 64, 256)
        trans_pooled = trans_out.mean(dim=1)    # (B, 256)
        
        # Global features
        global_enc = self.global_mlp(global_features)  # (B, 128)
        
        # Combine all
        combined = torch.cat([time_flat, chroma_flat, trans_pooled, global_enc], dim=1)
        
        # Project to embedding
        embedding = self.projection(combined)
        
        return embedding


class StyleEncoder:
    """
    High-level interface for style encoding.
    Combines feature extraction and neural encoding.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_dim: int = 256,
        hidden_dim: int = 512
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.feature_extractor = FeatureExtractor()
        
        self.model = StyleEncoderNet(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.model.to(device)
        
        if model_path:
            self.load(model_path)
        
        self.model.eval()
    
    def load(self, path: str):
        """Load trained model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded style encoder from {path}")
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved style encoder to {path}")
    
    def _features_to_tensors(
        self, 
        features: StyleFeatures
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert StyleFeatures to model input tensors."""
        # Resample time series to fixed length
        target_length = 2048
        
        def resample_1d(arr, target_len):
            if len(arr) == target_len:
                return arr
            indices = np.linspace(0, len(arr) - 1, target_len)
            return np.interp(indices, np.arange(len(arr)), arr)
        
        # Time features: (3, T)
        energy = resample_1d(features.energy_curve, target_length)
        brightness = resample_1d(features.brightness_curve, target_length)
        onset = resample_1d(features.onset_density_curve, target_length)
        time_features = np.stack([energy, brightness, onset], axis=0)
        
        # Chroma features: (12, T)
        chroma_resampled = np.zeros((12, target_length))
        for i in range(12):
            chroma_resampled[i] = resample_1d(features.chroma_sequence[i], target_length)
        
        # Global features: encode various stats
        global_features = np.zeros(16)
        global_features[0] = features.tempo / 200.0  # Normalize tempo
        global_features[1] = features.key / 12.0     # Normalize key
        global_features[2] = features.avg_phrase_length / 16.0
        global_features[3] = features.energy_variance
        global_features[4] = len(features.phrase_lengths) / 50.0  # Num phrases normalized
        global_features[5] = features.duration / 600.0  # Duration normalized (10 min max)
        
        # Phrase length distribution
        if features.phrase_lengths:
            global_features[6] = np.mean(features.phrase_lengths) / 8.0
            global_features[7] = np.std(features.phrase_lengths) / 4.0
        
        # Energy distribution
        if features.phrase_energies:
            global_features[8] = np.mean(features.phrase_energies)
            global_features[9] = np.std(features.phrase_energies)
            global_features[10] = np.max(features.phrase_energies)
            global_features[11] = np.min(features.phrase_energies)
        
        # Transition type counts
        if features.transition_types:
            global_features[12] = features.transition_types.count("buildup") / len(features.transition_types)
            global_features[13] = features.transition_types.count("drop") / len(features.transition_types)
            global_features[14] = features.transition_types.count("sustain") / len(features.transition_types)
        
        # Convert to tensors
        time_tensor = torch.FloatTensor(time_features).unsqueeze(0).to(self.device)
        chroma_tensor = torch.FloatTensor(chroma_resampled).unsqueeze(0).to(self.device)
        global_tensor = torch.FloatTensor(global_features).unsqueeze(0).to(self.device)
        
        return time_tensor, chroma_tensor, global_tensor
    
    @torch.no_grad()
    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Encode audio into a style embedding.
        
        Args:
            audio: Audio samples (mono)
            sr: Sample rate
            
        Returns:
            style_embedding: (embedding_dim,) numpy array
        """
        # Extract features
        features = self.feature_extractor.extract(audio, sr)
        
        # Convert to tensors
        time_t, chroma_t, global_t = self._features_to_tensors(features)
        
        # Encode
        embedding = self.model(time_t, chroma_t, global_t)
        
        return embedding.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_file(self, audio_path: str) -> np.ndarray:
        """Encode audio file into style embedding."""
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.encode(audio, sr)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two style embeddings."""
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 > 0 and norm2 > 0:
            return float(dot / (norm1 * norm2))
        return 0.0

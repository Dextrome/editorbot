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
    
    # NEW: Additional time-series features
    mfcc_sequence: np.ndarray       # MFCCs for timbre/texture (13 x T)
    spectral_contrast: np.ndarray   # Spectral contrast (7 x T)
    spectral_bandwidth: np.ndarray  # Spectral bandwidth over time
    spectral_rolloff: np.ndarray    # Spectral rolloff over time
    zcr_curve: np.ndarray           # Zero crossing rate over time
    tonnetz_sequence: np.ndarray    # Tonnetz harmonic features (6 x T)
    tempo_curve: np.ndarray         # Local tempo variations over time
    beat_strength: np.ndarray       # Beat/pulse strength over time
    
    # Structural features
    phrase_lengths: List[int]       # Lengths of detected phrases (in bars)
    phrase_energies: List[float]    # Energy of each phrase
    transition_types: List[str]     # e.g., "buildup", "drop", "sustain"
    
    # Global features
    tempo: float
    key: int                        # Dominant pitch class (0-11)
    avg_phrase_length: float
    energy_variance: float
    
    # NEW: Additional global features
    dynamics_range: float           # Dynamic range (dB)
    frequency_balance_curve: np.ndarray   # Bass/mid/treble balance over time (3 x T)
    stereo_width: float             # Stereo width (0=mono, 1=full stereo)
    tempo_stability: float          # How stable the tempo is (0-1)
    avg_beat_strength: float        # Average beat strength
    
    # Metadata
    duration: float
    sample_rate: int


class FeatureExtractor:
    """Extracts style features from audio."""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def extract(self, audio: np.ndarray, sr: int, audio_stereo: np.ndarray = None) -> StyleFeatures:
        """Extract all style features from audio.
        
        Args:
            audio: Mono audio array
            sr: Sample rate
            audio_stereo: Optional stereo audio (2, N) for stereo width analysis
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        duration = len(audio) / sr
        
        # Tempo detection with beat tracking
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
        window_size = int(sr / self.hop_length)  # ~1 second window
        onset_density_curve = np.convolve(onset_env, np.ones(window_size)/window_size, mode='same')
        onset_density_curve = onset_density_curve / (onset_density_curve.max() + 1e-8)
        
        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=self.hop_length)
        
        # Dominant key (most common pitch class)
        avg_chroma = chroma.mean(axis=1)
        key = int(np.argmax(avg_chroma))
        
        # === NEW FEATURES ===
        
        # 1. MFCCs (13 coefficients) - captures timbre/texture
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # 2. Spectral contrast (7 bands) - peaks vs valleys in spectrum
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=self.hop_length)
        
        # 3. Spectral bandwidth - frequency spread
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
        spectral_bandwidth = bandwidth / (bandwidth.max() + 1e-8)
        
        # 4. Spectral rolloff - where energy is concentrated
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = rolloff / (sr / 2)  # Normalize to 0-1
        
        # 5. Zero crossing rate - noisiness/percussiveness
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)[0]
        zcr_curve = zcr / (zcr.max() + 1e-8)
        
        # 6. Tonnetz - harmonic relationships (requires chroma)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, hop_length=self.hop_length)
        
        # 7. Tempo variations / local tempo curve
        tempo_curve, tempo_stability = self._extract_tempo_curve(audio, sr, beats)
        
        # 8. Beat strength/groove
        beat_strength = self._extract_beat_strength(onset_env, beats, len(energy_curve))
        avg_beat_strength = float(np.mean(beat_strength))
        
        # 9. Dynamics range (in dB)
        dynamics_range = self._compute_dynamics_range(rms)
        
        # 10. Frequency balance (bass/mid/treble)
        frequency_balance = self._compute_frequency_balance(audio, sr)
        
        # 11. Stereo width (if stereo audio provided)
        stereo_width = self._compute_stereo_width(audio_stereo) if audio_stereo is not None else 0.0
        
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
            mfcc_sequence=mfcc,
            spectral_contrast=spectral_contrast,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            zcr_curve=zcr_curve,
            tonnetz_sequence=tonnetz,
            tempo_curve=tempo_curve,
            beat_strength=beat_strength,
            phrase_lengths=phrase_lengths,
            phrase_energies=phrase_energies,
            transition_types=transition_types,
            tempo=tempo,
            key=key,
            avg_phrase_length=avg_phrase_length,
            energy_variance=energy_variance,
            dynamics_range=dynamics_range,
            frequency_balance_curve=frequency_balance,
            stereo_width=stereo_width,
            tempo_stability=tempo_stability,
            avg_beat_strength=avg_beat_strength,
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
    
    def _extract_tempo_curve(
        self,
        audio: np.ndarray,
        sr: int,
        beats: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Extract local tempo variations and overall tempo stability."""
        if len(beats) < 3:
            # Not enough beats to compute tempo curve
            return np.array([1.0]), 1.0
        
        # Convert beat frames to time in seconds
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Compute inter-beat intervals
        ibis = np.diff(beat_times)
        
        # Local tempo in BPM
        local_tempos = 60.0 / (ibis + 1e-8)
        
        # Clip extreme values
        local_tempos = np.clip(local_tempos, 30, 300)
        
        # Normalize to 0-1 (relative to median tempo)
        median_tempo = np.median(local_tempos)
        if median_tempo > 0:
            tempo_curve = local_tempos / median_tempo
        else:
            tempo_curve = np.ones_like(local_tempos)
        
        # Tempo stability = inverse of coefficient of variation
        std_tempo = np.std(local_tempos)
        mean_tempo = np.mean(local_tempos)
        if mean_tempo > 0:
            cv = std_tempo / mean_tempo
            tempo_stability = 1.0 / (1.0 + cv)  # Higher = more stable
        else:
            tempo_stability = 1.0
        
        return tempo_curve, float(tempo_stability)
    
    def _extract_beat_strength(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """Extract beat strength curve (how strong each beat is)."""
        if len(beats) == 0:
            return np.zeros(target_length)
        
        # Get onset strength at each beat position
        beat_strengths = onset_env[beats]
        
        # Normalize
        if beat_strengths.max() > 0:
            beat_strengths = beat_strengths / beat_strengths.max()
        
        # Interpolate to target length
        if len(beat_strengths) == target_length:
            return beat_strengths
        
        indices = np.linspace(0, len(beat_strengths) - 1, target_length)
        return np.interp(indices, np.arange(len(beat_strengths)), beat_strengths)
    
    def _compute_dynamics_range(self, rms: np.ndarray) -> float:
        """Compute dynamics range in dB (normalized)."""
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        
        # Get 90th and 10th percentile (avoid outliers)
        high = np.percentile(rms_db, 90)
        low = np.percentile(rms_db, 10)
        
        dynamic_range = high - low  # In dB
        
        # Normalize (typical range 0-60 dB)
        return float(np.clip(dynamic_range / 60.0, 0, 1))
    
    def _compute_frequency_balance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute frequency balance (bass/mid/treble) over time."""
        # Compute STFT
        D = librosa.stft(audio, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Define frequency bands (Hz)
        bass_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        treble_mask = freqs >= 4000
        
        # Compute energy in each band over time
        bass = np.sum(S[bass_mask, :], axis=0)
        mid = np.sum(S[mid_mask, :], axis=0)
        treble = np.sum(S[treble_mask, :], axis=0)
        
        # Stack and normalize
        total = bass + mid + treble + 1e-8
        balance = np.stack([bass / total, mid / total, treble / total], axis=0)
        
        return balance
    
    def _compute_stereo_width(self, audio_stereo: Optional[np.ndarray]) -> float:
        """Compute stereo width (0 = mono, 1 = full stereo)."""
        if audio_stereo is None or audio_stereo.ndim != 2:
            return 0.0
        
        if audio_stereo.shape[0] != 2:
            return 0.0
        
        left = audio_stereo[0]
        right = audio_stereo[1]
        
        # Compute mid and side signals
        mid = left + right
        side = left - right
        
        # Stereo width = ratio of side energy to total energy
        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        total_energy = mid_energy + side_energy + 1e-8
        
        stereo_width = side_energy / total_energy
        
        return float(np.clip(stereo_width, 0, 1))


class StyleEncoderNet(nn.Module):
    """
    Neural network that encodes style features into a fixed-size embedding.
    
    Architecture:
    - 1D CNN for time-series features (energy, brightness, onset, ZCR, bandwidth, rolloff)
    - 1D CNN for MFCC features (13 coefficients)
    - 1D CNN for spectral contrast (7 bands)
    - 1D CNN for chroma features (12 pitch classes)
    - 1D CNN for tonnetz features (6 dimensions)
    - 1D CNN for frequency balance (3 bands)
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
        
        # Time-series encoder (for energy, brightness, onset, ZCR, bandwidth, rolloff)
        # Input: 6 channels
        self.time_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3),
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
        
        # MFCC encoder (for timbral content)
        # Input: 13 coefficients
        self.mfcc_conv = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Spectral contrast encoder (for texture)
        # Input: 7 bands
        self.contrast_conv = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
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
        
        # Tonnetz encoder (for harmonic relations)
        # Input: 6 dimensions
        self.tonnetz_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # Frequency balance encoder (low/mid/high bands)
        # Input: 3 bands
        self.freqbal_conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
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
        
        # Global features MLP (tempo, key, variance, dynamics, beat strength, tempo stability, stereo width, etc.)
        # Expanded from 16 to 20 to accommodate new global features
        self.global_mlp = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Final projection to embedding
        # time_conv: 256*64 + mfcc_conv: 128*32 + contrast_conv: 64*16 + 
        # chroma_conv: 128*32 + tonnetz_conv: 64*16 + freqbal_conv: 32*16 +
        # transformer: 256 + global: 128
        combined_dim = (256 * 64) + (128 * 32) + (64 * 16) + (128 * 32) + (64 * 16) + (32 * 16) + 256 + 128
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self,
        time_features: torch.Tensor,      # (B, 6, T) - energy, brightness, onset, ZCR, bandwidth, rolloff
        mfcc_features: torch.Tensor,      # (B, 13, T) - MFCC coefficients
        contrast_features: torch.Tensor,  # (B, 7, T) - spectral contrast bands
        chroma_features: torch.Tensor,    # (B, 12, T) - pitch classes
        tonnetz_features: torch.Tensor,   # (B, 6, T) - tonal centroid features
        freqbal_features: torch.Tensor,   # (B, 3, T) - frequency balance (low/mid/high)
        global_features: torch.Tensor     # (B, 20) - global statistics
    ) -> torch.Tensor:
        """
        Encode style features into embedding.
        
        Returns:
            style_embedding: (B, embedding_dim)
        """
        # Time-series encoding
        time_enc = self.time_conv(time_features)  # (B, 256, 64)
        time_flat = time_enc.flatten(1)           # (B, 256*64)
        
        # MFCC encoding
        mfcc_enc = self.mfcc_conv(mfcc_features)  # (B, 128, 32)
        mfcc_flat = mfcc_enc.flatten(1)           # (B, 128*32)
        
        # Spectral contrast encoding
        contrast_enc = self.contrast_conv(contrast_features)  # (B, 64, 16)
        contrast_flat = contrast_enc.flatten(1)                # (B, 64*16)
        
        # Chroma encoding
        chroma_enc = self.chroma_conv(chroma_features)  # (B, 128, 32)
        chroma_flat = chroma_enc.flatten(1)             # (B, 128*32)
        
        # Tonnetz encoding
        tonnetz_enc = self.tonnetz_conv(tonnetz_features)  # (B, 64, 16)
        tonnetz_flat = tonnetz_enc.flatten(1)               # (B, 64*16)
        
        # Frequency balance encoding
        freqbal_enc = self.freqbal_conv(freqbal_features)  # (B, 32, 16)
        freqbal_flat = freqbal_enc.flatten(1)               # (B, 32*16)
        
        # Transformer for sequential patterns (use time encoding)
        time_seq = time_enc.permute(0, 2, 1)  # (B, 64, 256)
        trans_out = self.transformer(time_seq)  # (B, 64, 256)
        trans_pooled = trans_out.mean(dim=1)    # (B, 256)
        
        # Global features
        global_enc = self.global_mlp(global_features)  # (B, 128)
        
        # Combine all
        combined = torch.cat([
            time_flat, mfcc_flat, contrast_flat, 
            chroma_flat, tonnetz_flat, freqbal_flat,
            trans_pooled, global_enc
        ], dim=1)
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert StyleFeatures to model input tensors."""
        # Resample time series to fixed length
        target_length = 2048
        
        def resample_1d(arr, target_len):
            if len(arr) == target_len:
                return arr
            indices = np.linspace(0, len(arr) - 1, target_len)
            return np.interp(indices, np.arange(len(arr)), arr)
        
        def resample_2d(arr, target_len):
            """Resample 2D array (n_features, time) to fixed length."""
            result = np.zeros((arr.shape[0], target_len))
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
        time_features = np.stack([energy, brightness, onset, zcr, bandwidth, rolloff], axis=0)
        
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
        
        # Global features: encode various stats (now 20 features)
        global_features = np.zeros(20)
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
        
        # New global features
        # Tempo variation derived from tempo_curve variance (1 - stability = variation)
        global_features[15] = 1.0 - features.tempo_stability  # Tempo variation (higher = more varied)
        global_features[16] = features.avg_beat_strength  # Beat strength
        global_features[17] = features.stereo_width  # Stereo width (0-1)
        global_features[18] = features.dynamics_range  # Dynamics range (normalized)
        global_features[19] = features.tempo_stability  # Tempo stability metric
        
        # Convert to tensors
        time_tensor = torch.FloatTensor(time_features).unsqueeze(0).to(self.device)
        mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(self.device)
        contrast_tensor = torch.FloatTensor(contrast_features).unsqueeze(0).to(self.device)
        chroma_tensor = torch.FloatTensor(chroma_features).unsqueeze(0).to(self.device)
        tonnetz_tensor = torch.FloatTensor(tonnetz_features).unsqueeze(0).to(self.device)
        freqbal_tensor = torch.FloatTensor(freqbal_features).unsqueeze(0).to(self.device)
        global_tensor = torch.FloatTensor(global_features).unsqueeze(0).to(self.device)
        
        return time_tensor, mfcc_tensor, contrast_tensor, chroma_tensor, tonnetz_tensor, freqbal_tensor, global_tensor
    
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
        time_t, mfcc_t, contrast_t, chroma_t, tonnetz_t, freqbal_t, global_t = self._features_to_tensors(features)
        
        # Encode
        embedding = self.model(time_t, mfcc_t, contrast_t, chroma_t, tonnetz_t, freqbal_t, global_t)
        
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

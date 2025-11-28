"""Neural network models for audio enhancement."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AudioEnhancementModel(nn.Module):
    """
    Neural network model for audio enhancement.
    Uses a U-Net style architecture for source separation and enhancement.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        hidden_channels: int = 64,
    ):
        """
        Initialize the enhancement model.

        Args:
            n_fft: FFT size for STFT.
            hop_length: Hop length for STFT.
            hidden_channels: Number of hidden channels in the network.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),  # Output mask between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input spectrogram tensor of shape (batch, 1, freq_bins, time_frames).

        Returns:
            Enhancement mask of the same shape.
        """
        # Encode
        encoded = self.encoder(x)

        # Bottleneck
        bottleneck = self.bottleneck(encoded)

        # Decode
        mask = self.decoder(bottleneck)

        return mask

    def enhance(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 44100,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Enhance audio using the trained model.

        Args:
            audio_data: Input audio as numpy array.
            sample_rate: Sample rate of the audio.
            device: Device to run inference on ('cuda' or 'cpu').

        Returns:
            Enhanced audio as numpy array.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        with torch.no_grad():
            # Convert to tensor and compute STFT
            audio_tensor = torch.from_numpy(audio_data).float().to(device)
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
            )

            # Get magnitude and phase
            magnitude = torch.abs(stft).unsqueeze(0).unsqueeze(0)
            phase = torch.angle(stft)

            # Normalize magnitude
            mag_max = magnitude.max()
            magnitude_norm = magnitude / (mag_max + 1e-8)

            # Get enhancement mask
            mask = self(magnitude_norm)

            # Apply mask
            enhanced_mag = magnitude * mask.squeeze()

            # Reconstruct complex spectrogram
            enhanced_stft = enhanced_mag.squeeze() * torch.exp(1j * phase)

            # Inverse STFT
            enhanced = torch.istft(
                enhanced_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=len(audio_data),
            )

            return enhanced.cpu().numpy()


class NoiseReductionModel(nn.Module):
    """
    Specialized model for noise reduction using a recurrent architecture.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize the noise reduction model.

        Args:
            n_fft: FFT size for STFT.
            hidden_size: Size of LSTM hidden state.
            num_layers: Number of LSTM layers.
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1

        self.lstm = nn.LSTM(
            input_size=self.n_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_bins),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input spectrogram tensor of shape (batch, time_frames, freq_bins).

        Returns:
            Noise mask of shape (batch, time_frames, freq_bins).
        """
        lstm_out, _ = self.lstm(x)
        mask = self.fc(lstm_out)
        return mask


def load_pretrained_model(
    model_type: str = "enhancement",
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Load a pretrained model.

    Args:
        model_type: Type of model ("enhancement" or "noise_reduction").
        weights_path: Path to pretrained weights (optional).

    Returns:
        Loaded model.
    """
    if model_type == "enhancement":
        model = AudioEnhancementModel()
    elif model_type == "noise_reduction":
        model = NoiseReductionModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    return model

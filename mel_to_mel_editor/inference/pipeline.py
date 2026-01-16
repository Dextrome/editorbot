"""Inference pipeline for mel-to-mel editor."""

import os
import sys
import json
import numpy as np
import torch
from typing import Optional
from pathlib import Path

from ..config import TrainConfig, ModelConfig
from ..models import MelUNet
from shared.audio_utils import (
    compute_mel_spectrogram_bigvgan_from_file,
    normalize_mel_for_model,
    denormalize_mel_for_vocoder,
)
from shared.audio_config import AudioConfig


class MelToMelPipeline:
    """End-to-end inference pipeline."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {model_path}")
        self.model = MelUNet.from_checkpoint(model_path)
        self.model = self.model.to(self.device).eval()

        # Audio config
        self.audio_config = AudioConfig()

        # Vocoder (lazy load)
        self._vocoder = None

    def extract_mel(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram from audio file."""
        mel_log, _ = compute_mel_spectrogram_bigvgan_from_file(
            audio_path,
            config=self.audio_config,
            normalize_volume=True,
            device='cpu',
        )
        mel_log = mel_log.T  # (n_mels, T) -> (T, n_mels)
        mel_norm = normalize_mel_for_model(mel_log).numpy()
        return mel_norm

    @torch.no_grad()
    def process(self, raw_mel: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
        """Process mel spectrogram through model.

        Args:
            raw_mel: Input mel (T, n_mels) normalized to [0, 1]
            chunk_size: Process in chunks to avoid OOM

        Returns:
            Edited mel (T, n_mels)
        """
        T = len(raw_mel)

        if T <= chunk_size:
            # Process all at once
            raw_tensor = torch.from_numpy(raw_mel).float().unsqueeze(0).to(self.device)
            pred_tensor = self.model(raw_tensor)
            return pred_tensor.squeeze(0).cpu().numpy()

        # Process in overlapping chunks
        overlap = 64
        chunks = []
        for start in range(0, T, chunk_size - overlap):
            end = min(start + chunk_size, T)
            chunk = raw_mel[start:end]

            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
            pred_chunk = self.model(chunk_tensor).squeeze(0).cpu().numpy()
            chunks.append((start, end, pred_chunk))

        # Stitch with crossfade
        output = np.zeros_like(raw_mel)
        weights = np.zeros(T)

        for start, end, chunk in chunks:
            chunk_len = end - start
            w = np.ones(chunk_len)

            # Fade edges
            fade_len = min(overlap // 2, chunk_len // 4)
            if start > 0:
                w[:fade_len] = np.linspace(0, 1, fade_len)
            if end < T:
                w[-fade_len:] = np.linspace(1, 0, fade_len)

            w = w.reshape(-1, 1)
            output[start:end] += chunk * w
            weights[start:end] += w.squeeze()

        weights = np.maximum(weights, 1e-8).reshape(-1, 1)
        return output / weights

    def _load_vocoder(self):
        """Load BigVGAN vocoder."""
        if self._vocoder is not None:
            return self._vocoder

        bigvgan_path = os.path.join(os.path.dirname(__file__), '..', '..', 'vocoder', 'BigVGAN')
        if bigvgan_path not in sys.path:
            sys.path.insert(0, bigvgan_path)

        from bigvgan import BigVGAN
        from env import AttrDict

        pretrained_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/"
            "snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
        )

        with open(os.path.join(pretrained_dir, 'config.json')) as f:
            config = AttrDict(json.load(f))

        vocoder = BigVGAN(config)
        state_dict = torch.load(
            os.path.join(pretrained_dir, 'bigvgan_generator.pt'),
            map_location='cpu'
        )
        vocoder.load_state_dict(state_dict.get('generator', state_dict))
        vocoder = vocoder.to(self.device).eval()
        vocoder.remove_weight_norm()

        print("BigVGAN vocoder loaded!")
        self._vocoder = vocoder
        return vocoder

    def mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio."""
        vocoder = self._load_vocoder()

        mel = np.clip(mel, 0, 1)
        mel_tensor = torch.from_numpy(mel).float()
        mel_log = denormalize_mel_for_vocoder(mel_tensor).numpy()

        mel_input = torch.FloatTensor(mel_log.T).unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio = vocoder(mel_input).squeeze().cpu().numpy()

        # Normalize volume
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9

        return audio

    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, audio, self.audio_config.sample_rate)

    def process_file(
        self,
        input_path: str,
        output_path: str,
    ) -> dict:
        """Process audio file end-to-end."""
        # Extract mel
        raw_mel = self.extract_mel(input_path)

        # Process through model
        edited_mel = self.process(raw_mel)

        # Convert to audio
        audio = self.mel_to_audio(edited_mel)

        # Save
        self.save_audio(audio, output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'raw_mel': raw_mel,
            'edited_mel': edited_mel,
            'audio': audio,
        }

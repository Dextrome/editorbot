"""Full inference pipeline for Super Editor.

End-to-end: audio file -> edited audio file
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path

from ..config import Phase1Config, Phase2Config, AudioConfig
from ..models import ReconstructionModel, EditPredictor
from ..data.preprocessing import MelExtractor


class SuperEditorPipeline:
    """End-to-end inference pipeline.

    Takes raw audio, predicts edit labels (Phase 2),
    then reconstructs edited audio (Phase 1).
    """

    def __init__(
        self,
        recon_model_path: str,
        predictor_model_path: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            recon_model_path: Path to trained Phase 1 model
            predictor_model_path: Path to trained Phase 2 model (optional)
            audio_config: Audio configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.audio_config = audio_config or AudioConfig()

        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load reconstruction model
        print(f"Loading reconstruction model from {recon_model_path}")
        self.recon_model = ReconstructionModel.from_checkpoint(recon_model_path)
        self.recon_model = self.recon_model.to(self.device)
        self.recon_model.eval()

        # Load predictor model (optional)
        self.predictor = None
        if predictor_model_path is not None:
            print(f"Loading edit predictor from {predictor_model_path}")
            self.predictor = self._load_predictor(predictor_model_path)

        # Mel extractor
        self.mel_extractor = MelExtractor(self.audio_config)

    def _load_predictor(self, path: str) -> EditPredictor:
        """Load edit predictor model."""
        checkpoint = torch.load(path, map_location=self.device)

        config = checkpoint.get('config', Phase2Config())
        predictor = EditPredictor(config).to(self.device)

        # Handle different checkpoint formats
        if 'actor_critic_state_dict' in checkpoint:
            # From Phase2Trainer - extract actor part
            ac_state = checkpoint['actor_critic_state_dict']
            actor_state = {k.replace('actor.', ''): v for k, v in ac_state.items() if k.startswith('actor.')}
            predictor.load_state_dict(actor_state)
        else:
            predictor.load_state_dict(checkpoint['model_state_dict'])

        predictor.eval()
        return predictor

    @torch.no_grad()
    def predict_labels(
        self,
        raw_mel: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict edit labels from raw mel spectrogram.

        Args:
            raw_mel: Raw mel spectrogram (T, n_mels)
            deterministic: Use argmax instead of sampling

        Returns:
            labels: Predicted edit labels (T,)
        """
        if self.predictor is None:
            raise ValueError("No predictor model loaded. Provide predictor_model_path.")

        if isinstance(raw_mel, np.ndarray):
            raw_mel = torch.from_numpy(raw_mel).float()

        # Add batch dimension
        raw_mel = raw_mel.unsqueeze(0).to(self.device)

        # Predict
        actions, _, _ = self.predictor.get_action(raw_mel, deterministic=deterministic)

        return actions.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def reconstruct(
        self,
        raw_mel: Union[np.ndarray, torch.Tensor],
        edit_labels: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Reconstruct edited mel from raw mel and labels.

        Args:
            raw_mel: Raw mel spectrogram (T, n_mels)
            edit_labels: Edit labels (T,)

        Returns:
            pred_mel: Predicted edited mel (T, n_mels)
        """
        if isinstance(raw_mel, np.ndarray):
            raw_mel = torch.from_numpy(raw_mel).float()
        if isinstance(edit_labels, np.ndarray):
            edit_labels = torch.from_numpy(edit_labels).long()

        raw_mel = raw_mel.unsqueeze(0).to(self.device)
        edit_labels = edit_labels.unsqueeze(0).to(self.device)

        pred_mel = self.recon_model(raw_mel, edit_labels)

        return pred_mel.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def process(
        self,
        raw_mel: Union[np.ndarray, torch.Tensor],
        edit_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        predict_labels: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Process raw mel through full pipeline.

        Args:
            raw_mel: Raw mel spectrogram (T, n_mels)
            edit_labels: Optional explicit labels (if not predicting)
            predict_labels: Whether to predict labels using Phase 2
            deterministic: Use argmax for label prediction

        Returns:
            Dictionary with:
                - 'edit_labels': Edit labels used
                - 'pred_mel': Reconstructed edited mel
        """
        if isinstance(raw_mel, np.ndarray):
            raw_mel_np = raw_mel
        else:
            raw_mel_np = raw_mel.numpy()

        # Get edit labels
        if edit_labels is None:
            if predict_labels and self.predictor is not None:
                edit_labels = self.predict_labels(raw_mel_np, deterministic)
            else:
                # Default: all KEEP
                edit_labels = np.ones(len(raw_mel_np), dtype=np.int64)

        # Reconstruct
        pred_mel = self.reconstruct(raw_mel_np, edit_labels)

        return {
            'edit_labels': edit_labels,
            'pred_mel': pred_mel,
        }

    def process_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        edit_labels: Optional[np.ndarray] = None,
        predict_labels: bool = True,
    ) -> Dict[str, Any]:
        """Process audio file through full pipeline.

        Args:
            audio_path: Path to input audio file
            output_path: Optional path to save output audio
            edit_labels: Optional explicit labels
            predict_labels: Whether to predict labels

        Returns:
            Dictionary with results including mel and optionally audio
        """
        # Extract mel
        raw_mel = self.mel_extractor.extract(audio_path)

        # Process
        result = self.process(raw_mel, edit_labels, predict_labels)
        result['raw_mel'] = raw_mel

        # Convert to audio if output path specified
        if output_path is not None:
            audio = self.mel_to_audio(result['pred_mel'])
            self.save_audio(audio, output_path)
            result['audio'] = audio
            result['output_path'] = output_path

        return result

    def mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio using Griffin-Lim.

        Args:
            mel: Mel spectrogram (T, n_mels) normalized to [0, 1]

        Returns:
            audio: Audio waveform
        """
        import librosa

        # Denormalize (assuming mel was normalized to [0, 1])
        # This is approximate - proper denormalization requires knowing original range
        mel_db = mel * 80 - 80  # Approximate dB range
        mel_power = librosa.db_to_power(mel_db)

        # Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power.T,  # librosa expects (n_mels, T)
            sr=self.audio_config.sample_rate,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length,
            win_length=self.audio_config.win_length,
            fmin=self.audio_config.fmin,
            fmax=self.audio_config.fmax,
        )

        return audio

    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, audio, self.audio_config.sample_rate)


def edit_audio(
    input_path: str,
    output_path: str,
    recon_model_path: str,
    predictor_model_path: Optional[str] = None,
    edit_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience function to edit an audio file.

    Args:
        input_path: Path to input audio
        output_path: Path to save edited audio
        recon_model_path: Path to Phase 1 model
        predictor_model_path: Optional path to Phase 2 model
        edit_labels: Optional explicit edit labels

    Returns:
        Results dictionary
    """
    pipeline = SuperEditorPipeline(
        recon_model_path=recon_model_path,
        predictor_model_path=predictor_model_path,
    )

    return pipeline.process_audio(
        input_path,
        output_path,
        edit_labels=edit_labels,
        predict_labels=(predictor_model_path is not None),
    )


def batch_edit_audio(
    input_paths: list,
    output_dir: str,
    recon_model_path: str,
    predictor_model_path: Optional[str] = None,
) -> list:
    """Edit multiple audio files.

    Args:
        input_paths: List of input audio paths
        output_dir: Directory to save edited audio
        recon_model_path: Path to Phase 1 model
        predictor_model_path: Optional path to Phase 2 model

    Returns:
        List of result dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SuperEditorPipeline(
        recon_model_path=recon_model_path,
        predictor_model_path=predictor_model_path,
    )

    results = []
    for input_path in input_paths:
        input_name = Path(input_path).stem
        output_path = output_dir / f"{input_name}_edited.wav"

        try:
            result = pipeline.process_audio(
                str(input_path),
                str(output_path),
                predict_labels=(predictor_model_path is not None),
            )
            results.append(result)
            print(f"Processed: {input_path} -> {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            results.append({'error': str(e), 'input_path': input_path})

    return results

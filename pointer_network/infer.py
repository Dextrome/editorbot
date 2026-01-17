"""Inference script for pointer network."""
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse

from .models import PointerNetwork, STOP_TOKEN
from .config import PointerNetworkConfig


def load_model(checkpoint_path: str, device: str = "cuda") -> PointerNetwork:
    """Load trained pointer network from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', None)
    if config and hasattr(config, 'model'):
        model_config = config.model
    else:
        model_config = PointerNetworkConfig()

    model = PointerNetwork(
        n_mels=model_config.n_mels,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_encoder_layers=model_config.n_encoder_layers,
        n_decoder_layers=model_config.n_decoder_layers,
        dropout=0.0,  # No dropout at inference
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', '?'):.4f}")

    return model, model_config


def compute_mel(audio_path: str, config: PointerNetworkConfig) -> np.ndarray:
    """Compute mel spectrogram from audio file."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sr, mono=True)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=2048,
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_norm, y, sr


def pointers_to_audio(pointers: np.ndarray, raw_audio: np.ndarray,
                      hop_length: int, sr: int) -> np.ndarray:
    """Reconstruct audio from pointer sequence.

    Args:
        pointers: Array of frame indices pointing into raw audio
        raw_audio: Original raw audio samples
        hop_length: Mel spectrogram hop length
        sr: Sample rate

    Returns:
        Reconstructed audio
    """
    # Each pointer points to a mel frame
    # Each mel frame corresponds to hop_length audio samples

    output_samples = []

    for i, ptr in enumerate(pointers):
        if ptr < 0 or ptr == STOP_TOKEN:
            break

        # Get audio samples for this frame
        start_sample = ptr * hop_length
        end_sample = start_sample + hop_length

        if end_sample <= len(raw_audio):
            output_samples.append(raw_audio[start_sample:end_sample])
        else:
            # Pad if we're at the end
            chunk = raw_audio[start_sample:]
            if len(chunk) > 0:
                output_samples.append(chunk)

    if not output_samples:
        return np.zeros(hop_length)

    # Concatenate with crossfade to reduce clicks
    output = np.concatenate(output_samples)

    return output


@torch.no_grad()
def run_inference(
    model: PointerNetwork,
    raw_mel: torch.Tensor,
    max_output_length: int = 10000,
    device: str = "cuda",
    temperature: float = 0.0,
    max_input_frames: int = 4096,  # ~47s at 22050Hz/256hop
) -> np.ndarray:
    """Run pointer network inference.

    Args:
        model: Trained pointer network
        raw_mel: Input mel spectrogram (n_mels, time)
        max_output_length: Maximum output sequence length
        device: Device to run on
        temperature: Sampling temperature (0 = greedy)
        max_input_frames: Maximum input frames (truncate if longer)

    Returns:
        Array of pointer indices
    """
    model.eval()

    # Truncate if too long (positional encoding limit)
    if raw_mel.shape[1] > max_input_frames:
        print(f"  Truncating input from {raw_mel.shape[1]} to {max_input_frames} frames")
        raw_mel = raw_mel[:, :max_input_frames]

    # Add batch dimension
    raw_mel = raw_mel.unsqueeze(0).to(device)  # (1, n_mels, time)

    # Run generation (let model predict length if max_output_length is None)
    outputs = model.generate(
        raw_mel=raw_mel,
        target_length=None,  # Let model predict length
        temperature=temperature,
        sample_style=False,  # Deterministic style
    )

    pointers = outputs['pointers']
    print(f"  Predicted length: {outputs['predicted_length']}")

    return pointers[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run pointer network inference")
    parser.add_argument("input", type=str, help="Input audio file (raw recording)")
    parser.add_argument("--checkpoint", type=str, default="models/pointer_network/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output audio file (default: input_edited.wav)")
    parser.add_argument("--max-length", type=int, default=10000,
                        help="Maximum output length in frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_edited.wav")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)

    # Load and process input audio
    print(f"Processing {args.input}...")
    mel, raw_audio, sr = compute_mel(args.input, config)
    print(f"  Audio: {len(raw_audio)/sr:.1f}s, {mel.shape[1]} mel frames")

    # Convert to tensor
    mel_tensor = torch.from_numpy(mel).float()

    # Run inference
    print("Running inference...")
    pointers = run_inference(model, mel_tensor, args.max_length, args.device)
    print(f"  Generated {len(pointers)} pointers")

    # Analyze pointers
    valid_pointers = pointers[pointers >= 0]
    if len(valid_pointers) > 0:
        print(f"  Pointer range: {valid_pointers.min()} - {valid_pointers.max()}")
        print(f"  Output/Input ratio: {len(valid_pointers)/mel.shape[1]:.2%}")

    # Reconstruct audio
    print("Reconstructing audio...")
    output_audio = pointers_to_audio(pointers, raw_audio, config.hop_length, sr)
    print(f"  Output: {len(output_audio)/sr:.1f}s")

    # Save output
    sf.write(args.output, output_audio, sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

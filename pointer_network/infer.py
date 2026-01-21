"""Inference script for pointer network."""
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse

from .models import PointerNetwork, STOP_TOKEN
from .config import PointerNetworkConfig


def load_stems(audio_path: str, cache_dir: str = "cache/stems_mel") -> np.ndarray | None:
    """Try to load cached stem mel spectrograms for an audio file.

    Returns (n_stems, n_mels, time) array or None if not found.
    """
    audio_name = Path(audio_path).stem
    cache_path = Path(cache_dir) / f"{audio_name}_stems.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        # Stems are stored per-stem as (n_mels, time)
        stems = []
        for key in ['drums', 'bass', 'vocals', 'other']:
            if key in data:
                stems.append(data[key])
        if stems:
            return np.stack(stems, axis=0)  # (n_stems, n_mels, time)
    return None


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
        use_pre_norm=getattr(model_config, 'use_pre_norm', False),
        use_edit_ops=getattr(model_config, 'use_edit_ops', False),
        use_stems=getattr(model_config, 'use_stems', False),
        n_stems=getattr(model_config, 'n_stems', 4),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', '?'):.4f}")

    return model, model_config


def compute_mel(audio_path: str, config: PointerNetworkConfig, cache_dir: str = "cache/features") -> np.ndarray:
    """Load mel from cache or compute from audio file.

    Args:
        audio_path: Path to audio file
        config: Pointer network config
        cache_dir: Directory with cached mel spectrograms

    Returns:
        mel: (n_mels, time) normalized mel spectrogram
        y: raw audio samples
        sr: sample rate
    """
    # Try to load from cache first (matches training exactly)
    audio_name = Path(audio_path).stem
    cache_path = Path(cache_dir) / f"{audio_name}.npz"

    # Load audio for reconstruction
    y, sr = librosa.load(audio_path, sr=config.sr, mono=True)

    if cache_path.exists():
        data = np.load(cache_path)
        mel = data['mel']
        # Ensure shape is (n_mels, time)
        if mel.shape[0] != config.n_mels:
            mel = mel.T
        print(f"  Loaded cached mel from {cache_path}")
        return mel, y, sr

    # Compute fresh if not cached
    print(f"  Computing mel spectrogram (not cached)")
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=2048,
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1] using fixed normalization (matches training cache)
    mel_norm = (mel_db + 80) / 80
    mel_norm = np.clip(mel_norm, 0, 1)

    return mel_norm, y, sr


def pointers_to_audio(pointers: np.ndarray, raw_audio: np.ndarray,
                      hop_length: int, sr: int,
                      crossfade_ms: float = 5.0) -> np.ndarray:
    """Reconstruct audio from pointer sequence using overlap-add.

    Each pointer specifies which raw audio frame to use for that output position.
    Uses overlap-add with Hann window for smooth transitions.

    Args:
        pointers: Array of frame indices pointing into raw audio
        raw_audio: Original raw audio samples
        hop_length: Mel spectrogram hop length
        sr: Sample rate
        crossfade_ms: Crossfade duration at discontinuities (ms)

    Returns:
        Reconstructed audio
    """
    # Filter valid pointers
    valid_pointers = []
    for ptr in pointers:
        if ptr < 0 or ptr == STOP_TOKEN:
            break
        valid_pointers.append(int(ptr))

    if not valid_pointers:
        return np.zeros(hop_length)

    n_frames = len(valid_pointers)

    # Frame size for overlap-add (2x hop for 50% overlap)
    frame_size = hop_length * 2

    # Output buffer
    output_len = n_frames * hop_length + hop_length
    output = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    # Hann window for smooth overlap-add
    window = np.hanning(frame_size)

    # Crossfade samples for discontinuities
    xfade_samples = int(sr * crossfade_ms / 1000)

    prev_ptr = None

    for i, ptr in enumerate(valid_pointers):
        # Output position
        out_start = i * hop_length
        out_end = out_start + frame_size

        if out_end > output_len:
            out_end = output_len
            frame_size_actual = out_end - out_start
        else:
            frame_size_actual = frame_size

        # Source position in raw audio
        src_center = ptr * hop_length + hop_length // 2
        src_start = src_center - frame_size_actual // 2
        src_end = src_start + frame_size_actual

        # Clamp to valid range
        if src_start < 0:
            src_start = 0
            src_end = frame_size_actual
        if src_end > len(raw_audio):
            src_end = len(raw_audio)
            src_start = max(0, src_end - frame_size_actual)

        # Get frame
        frame = raw_audio[src_start:src_end].copy()

        # Pad if needed
        if len(frame) < frame_size_actual:
            frame = np.pad(frame, (0, frame_size_actual - len(frame)))

        # Apply window
        win = window[:frame_size_actual] if frame_size_actual < frame_size else window

        # Check for discontinuity (jump > 1 frame)
        if prev_ptr is not None:
            jump = abs(ptr - prev_ptr)
            if jump > 1:
                # Apply extra fade at discontinuity
                fade_len = min(xfade_samples, len(frame) // 2)
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    frame[:fade_len] *= fade_in

        # Overlap-add
        output[out_start:out_start + len(frame)] += frame * win
        window_sum[out_start:out_start + len(win)] += win

        prev_ptr = ptr

    # Normalize by window sum (avoid divide by zero)
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum

    # Trim to actual length
    output = output[:n_frames * hop_length]

    return output


@torch.no_grad()
def run_inference_chunk(
    model: PointerNetwork,
    raw_mel: torch.Tensor,
    device: str = "cuda",
    temperature: float = 0.0,
    stems: torch.Tensor = None,
) -> tuple[np.ndarray, int]:
    """Run pointer network inference on a single chunk.

    Args:
        model: Trained pointer network
        raw_mel: Input mel spectrogram (n_mels, time) - must fit in memory
        device: Device to run on
        temperature: Sampling temperature (0 = greedy)
        stems: Optional stem mel spectrograms (n_stems, n_mels, time)

    Returns:
        Tuple of (pointer indices, predicted length)
    """
    model.eval()

    # Add batch dimension
    raw_mel = raw_mel.unsqueeze(0).to(device)  # (1, n_mels, time)

    # Prepare stems if provided
    stems_tensor = None
    if stems is not None:
        stems_tensor = stems.unsqueeze(0).to(device)  # (1, n_stems, n_mels, time)

    # Run generation
    outputs = model.generate(
        raw_mel=raw_mel,
        target_length=None,  # Let model predict length
        temperature=temperature,
        sample_style=False,  # Deterministic style
        stems=stems_tensor,
    )

    pointers = outputs['pointers'][0].cpu().numpy()
    predicted_length = outputs['predicted_length']

    return pointers, predicted_length


@torch.no_grad()
def run_inference(
    model: PointerNetwork,
    raw_mel: torch.Tensor,
    max_output_length: int = 10000,
    device: str = "cuda",
    temperature: float = 0.0,
    max_input_frames: int = 4096,  # ~47s at 22050Hz/256hop
    chunk_overlap: float = 0.25,  # 25% overlap between chunks
    stems: torch.Tensor = None,  # (n_stems, n_mels, time)
) -> np.ndarray:
    """Run pointer network inference with chunk-based processing for long audio.

    Args:
        model: Trained pointer network
        raw_mel: Input mel spectrogram (n_mels, time)
        max_output_length: Maximum output sequence length
        device: Device to run on
        temperature: Sampling temperature (0 = greedy)
        max_input_frames: Maximum input frames per chunk
        chunk_overlap: Fraction of overlap between chunks
        stems: Optional stem mel spectrograms (n_stems, n_mels, time)

    Returns:
        Array of pointer indices (global frame indices)
    """
    model.eval()
    total_frames = raw_mel.shape[1]

    # If input fits in one chunk, process directly
    if total_frames <= max_input_frames:
        pointers, pred_len = run_inference_chunk(model, raw_mel, device, temperature, stems)
        print(f"  Predicted length: {pred_len}")
        return pointers

    # Chunk-based processing for long audio
    print(f"  Processing {total_frames} frames in chunks of {max_input_frames}...")

    stride = int(max_input_frames * (1 - chunk_overlap))
    all_pointers = []
    chunk_idx = 0

    start = 0
    while start < total_frames:
        end = min(start + max_input_frames, total_frames)
        chunk_mel = raw_mel[:, start:end]

        # Get corresponding chunk of stems if available
        chunk_stems = None
        if stems is not None:
            chunk_stems = stems[:, :, start:end]

        # Pad if chunk is too small (last chunk)
        if chunk_mel.shape[1] < max_input_frames // 2:
            # Skip very small final chunks
            break

        pointers, pred_len = run_inference_chunk(model, chunk_mel, device, temperature, chunk_stems)

        # Filter valid pointers and offset to global frame indices
        valid_mask = (pointers >= 0) & (pointers != STOP_TOKEN)
        valid_pointers = pointers[valid_mask]

        # Offset pointers to global indices
        global_pointers = valid_pointers + start

        # Clip to valid range
        global_pointers = np.clip(global_pointers, 0, total_frames - 1)

        print(f"    Chunk {chunk_idx}: frames {start}-{end}, "
              f"output {len(global_pointers)} pointers (predicted {pred_len})")

        all_pointers.append(global_pointers)
        chunk_idx += 1
        start += stride

    # Concatenate all pointers
    if not all_pointers:
        return np.array([], dtype=np.int64)

    combined = np.concatenate(all_pointers)
    print(f"  Total: {len(combined)} pointers from {chunk_idx} chunks")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Run pointer network inference")
    parser.add_argument("input", type=str, help="Input audio file (raw recording)")
    parser.add_argument("--checkpoint", type=str, default="models/pointer_network/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output audio file (default: input_edited.wav)")
    parser.add_argument("--max-length", type=int, default=10000,
                        help="Maximum output length in frames")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Maximum frames per chunk (default: 4096 = ~47s)")
    parser.add_argument("--chunk-overlap", type=float, default=0.25,
                        help="Overlap between chunks as fraction (default: 0.25)")
    parser.add_argument("--stems-dir", type=str, default="cache/stems_mel",
                        help="Directory with cached stem mel spectrograms")
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

    # Try to load cached stems
    stems_tensor = None
    stems = load_stems(args.input, args.stems_dir)
    if stems is not None:
        print(f"  Loaded cached stems: {stems.shape}")
        # Ensure stems match mel length
        if stems.shape[2] != mel.shape[1]:
            print(f"  Adjusting stems length from {stems.shape[2]} to {mel.shape[1]}")
            stems = stems[:, :, :mel.shape[1]]
        stems_tensor = torch.from_numpy(stems).float()
    else:
        print("  No cached stems found (inference may be less accurate)")

    # Run inference
    print("Running inference...")
    pointers = run_inference(
        model, mel_tensor, args.max_length, args.device,
        max_input_frames=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        stems=stems_tensor,
    )
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

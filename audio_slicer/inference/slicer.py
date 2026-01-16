"""Audio Slicer - Select good segments from raw audio.

Uses FaceSwap-style dual autoencoder to score segments:
- Encode raw segment → decode with edited decoder
- Good segments transform cleanly (low reconstruction difference)
- Bad segments transform poorly (high reconstruction difference)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from pathlib import Path

from ..config import ModelConfig
from ..models import DualAutoencoder


class AudioSlicer:
    """Select good segments from raw audio using learned quality model."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {model_path}")
        self.model = DualAutoencoder.from_checkpoint(model_path)
        self.model = self.model.to(self.device).eval()

        # Vocoder (lazy load)
        self._vocoder = None

        # Audio config
        self.sample_rate = 44100
        self.hop_length = 512
        self.n_mels = 128

    def extract_mel(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        # Use shared audio utils
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from shared.audio_utils import compute_mel_spectrogram_bigvgan_from_file
        from shared.audio_config import AudioConfig

        config = AudioConfig()
        mel_log, _ = compute_mel_spectrogram_bigvgan_from_file(
            audio_path,
            config=config,
            normalize_volume=True,
            device='cpu',
        )

        # Normalize to [0, 1]
        mel_log = mel_log.T  # (n_mels, T) -> (T, n_mels)
        mel_min, mel_max = -11.5, 2.5  # BigVGAN range
        mel_norm = (mel_log - mel_min) / (mel_max - mel_min)
        mel_norm = np.clip(mel_norm, 0, 1)

        return mel_norm.numpy() if hasattr(mel_norm, 'numpy') else mel_norm

    @torch.no_grad()
    def score_segments(
        self,
        mel: np.ndarray,
        segment_frames: int = 128,
        hop_frames: int = 64,
    ) -> List[Tuple[int, int, float]]:
        """Score all segments in a mel spectrogram.

        Args:
            mel: (T, n_mels) full mel spectrogram
            segment_frames: Length of each segment
            hop_frames: Hop between segments

        Returns:
            List of (start, end, score) tuples
        """
        T = len(mel)
        segments = []

        for start in range(0, T - segment_frames + 1, hop_frames):
            end = start + segment_frames
            segment = mel[start:end]

            # Convert to tensor
            segment_tensor = torch.from_numpy(segment).float().unsqueeze(0).to(self.device)

            # Score using transformation quality
            score = self._compute_segment_score(segment_tensor)

            segments.append((start, end, score))

        return segments

    def _compute_segment_score(self, segment: torch.Tensor) -> float:
        """Compute quality score for a segment.

        Higher score = segment is compatible with "edited" quality.
        """
        # Method 1: Cross-reconstruction consistency
        # If raw segment matches edited pattern, transforming it should be clean

        # Encode raw
        z = self.model.encode(segment)
        T = segment.size(1)

        # Decode as edited
        as_edited = self.model.decode_edited(z, T)

        # Re-encode the "edited" version
        z_edited = self.model.encode(as_edited)

        # Decode back as raw
        back_as_raw = self.model.decode_raw(z_edited, T)

        # Cycle consistency: good segments survive the round-trip
        cycle_error = F.mse_loss(back_as_raw, segment).item()

        # Also check how different the edited version is
        # (good audio shouldn't change much, bad audio changes a lot)
        edit_diff = F.mse_loss(as_edited, segment).item()

        # Score: low error = high score
        # Weight cycle consistency more (it's more reliable)
        total_error = cycle_error * 2 + edit_diff
        score = np.exp(-total_error * 10)

        return float(score)

    def select_segments(
        self,
        mel: np.ndarray,
        threshold: float = 0.5,
        min_gap_frames: int = 32,
        segment_frames: int = 128,
        hop_frames: int = 64,
    ) -> List[Tuple[int, int]]:
        """Select good segments above threshold.

        Args:
            mel: Full mel spectrogram
            threshold: Minimum score to keep (0-1)
            min_gap_frames: Merge segments closer than this
            segment_frames: Segment length for scoring
            hop_frames: Hop for scoring

        Returns:
            List of (start, end) frame ranges to keep
        """
        # Score all segments
        scored = self.score_segments(mel, segment_frames, hop_frames)

        # Filter by threshold
        good_segments = [(s, e) for s, e, score in scored if score >= threshold]

        if not good_segments:
            print(f"Warning: No segments above threshold {threshold}")
            # Return everything if nothing passes
            return [(0, len(mel))]

        # Merge overlapping/close segments
        merged = self._merge_segments(good_segments, min_gap_frames)

        return merged

    def _merge_segments(
        self,
        segments: List[Tuple[int, int]],
        min_gap: int,
    ) -> List[Tuple[int, int]]:
        """Merge overlapping or close segments."""
        if not segments:
            return []

        # Sort by start
        segments = sorted(segments)

        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]

            # Merge if overlapping or close
            if start <= prev_end + min_gap:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged

    def slice_audio(
        self,
        input_path: str,
        output_path: str,
        threshold: float = 0.5,
        transform: bool = True,
    ) -> dict:
        """Slice audio file - keep only good segments.

        Args:
            input_path: Path to raw audio
            output_path: Path to save sliced audio
            threshold: Quality threshold (0-1)
            transform: If True, also apply raw→edited transformation

        Returns:
            Dict with stats
        """
        import soundfile as sf

        # Extract mel
        print(f"Extracting mel from {input_path}")
        mel = self.extract_mel(input_path)
        print(f"Mel shape: {mel.shape} ({mel.shape[0] / 86:.1f} seconds)")

        # Score and select
        print(f"Scoring segments (threshold={threshold})...")
        segments = self.select_segments(mel, threshold=threshold)

        print(f"Selected {len(segments)} segments:")
        total_kept = 0
        for i, (start, end) in enumerate(segments):
            duration = (end - start) / 86
            total_kept += end - start
            print(f"  {i+1}. frames {start}-{end} ({duration:.1f}s)")

        keep_ratio = total_kept / len(mel)
        print(f"Keeping {keep_ratio*100:.1f}% of audio")

        # Extract kept segments
        kept_mels = []
        for start, end in segments:
            segment_mel = mel[start:end]

            if transform:
                # Transform to "edited" style
                segment_tensor = torch.from_numpy(segment_mel).float().unsqueeze(0).to(self.device)
                z = self.model.encode(segment_tensor)
                transformed = self.model.decode_edited(z, segment_tensor.size(1))
                segment_mel = transformed.squeeze(0).detach().cpu().numpy()

            kept_mels.append(segment_mel)

        # Concatenate with crossfades
        output_mel = self._concatenate_with_crossfade(kept_mels)

        # Convert to audio
        print("Converting to audio...")
        audio = self._mel_to_audio(output_mel)

        # Save
        sf.write(output_path, audio, self.sample_rate)
        print(f"Saved: {output_path}")

        return {
            'input_path': input_path,
            'output_path': output_path,
            'input_frames': len(mel),
            'output_frames': len(output_mel),
            'keep_ratio': keep_ratio,
            'n_segments': len(segments),
            'segments': segments,
        }

    def _concatenate_with_crossfade(
        self,
        mels: List[np.ndarray],
        crossfade_frames: int = 16,
    ) -> np.ndarray:
        """Concatenate mel segments with crossfades."""
        if len(mels) == 1:
            return mels[0]

        result = mels[0]

        for mel in mels[1:]:
            # Crossfade
            if len(result) >= crossfade_frames and len(mel) >= crossfade_frames:
                # Blend end of result with start of mel
                fade_out = np.linspace(1, 0, crossfade_frames).reshape(-1, 1)
                fade_in = np.linspace(0, 1, crossfade_frames).reshape(-1, 1)

                result[-crossfade_frames:] = (
                    result[-crossfade_frames:] * fade_out +
                    mel[:crossfade_frames] * fade_in
                )
                result = np.concatenate([result, mel[crossfade_frames:]], axis=0)
            else:
                result = np.concatenate([result, mel], axis=0)

        return result

    def _load_vocoder(self):
        """Load BigVGAN vocoder."""
        if self._vocoder is not None:
            return self._vocoder

        bigvgan_path = str(Path(__file__).parent.parent.parent / 'vocoder' / 'BigVGAN')
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

    def _mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel to audio using BigVGAN."""
        vocoder = self._load_vocoder()

        # Denormalize
        mel = np.clip(mel, 0, 1)
        mel_min, mel_max = -11.5, 2.5
        mel_log = mel * (mel_max - mel_min) + mel_min

        # Process in chunks to avoid OOM
        chunk_size = 500
        all_audio = []

        for start in range(0, len(mel_log), chunk_size):
            end = min(start + chunk_size, len(mel_log))
            chunk = mel_log[start:end]

            mel_input = torch.FloatTensor(chunk.T).unsqueeze(0).to(self.device)
            with torch.no_grad():
                chunk_audio = vocoder(mel_input).squeeze().cpu().numpy()

            all_audio.append(chunk_audio)

            if len(all_audio) % 10 == 0:
                torch.cuda.empty_cache()

        # Concatenate
        audio = np.concatenate(all_audio)

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9

        return audio

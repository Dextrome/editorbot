"""Mel Decoder for Phase 1 reconstruction.

Takes latent representation and produces edited mel spectrogram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..config import Phase1Config


def find_contiguous_segments(labels: torch.Tensor, target_label: int) -> list:
    """Find start and end indices of contiguous segments with target_label.

    Args:
        labels: (T,) tensor of labels
        target_label: Label value to find segments of

    Returns:
        List of (start, end) tuples for each contiguous segment
    """
    segments = []
    mask = (labels == target_label)

    if not mask.any():
        return segments

    # Find transitions
    padded = F.pad(mask.float(), (1, 1), value=0)
    diff = padded[1:] - padded[:-1]

    starts = (diff == 1).nonzero(as_tuple=True)[0]
    ends = (diff == -1).nonzero(as_tuple=True)[0]

    for s, e in zip(starts.tolist(), ends.tolist()):
        segments.append((s, e))

    return segments


def apply_fade_in(mel: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Apply fade-in ramp (0→1) to a segment."""
    seg_len = end - start
    if seg_len <= 0:
        return mel

    # Create ramp from 0 to 1
    ramp = torch.linspace(0, 1, seg_len, device=mel.device, dtype=mel.dtype)
    ramp = ramp.view(-1, 1)  # (seg_len, 1) for broadcasting

    # Apply to segment
    result = mel.clone()
    result[start:end] = mel[start:end] * ramp
    return result


def apply_fade_out(mel: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Apply fade-out ramp (1→0) to a segment."""
    seg_len = end - start
    if seg_len <= 0:
        return mel

    # Create ramp from 1 to 0
    ramp = torch.linspace(1, 0, seg_len, device=mel.device, dtype=mel.dtype)
    ramp = ramp.view(-1, 1)  # (seg_len, 1) for broadcasting

    # Apply to segment
    result = mel.clone()
    result[start:end] = mel[start:end] * ramp
    return result


def apply_loop(mel: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Loop: repeat a segment of audio with crossfades to avoid artifacts.

    Takes audio from before the loop region and tiles it with smooth transitions.
    """
    seg_len = end - start
    if seg_len <= 0:
        return mel

    result = mel.clone()

    # Determine source length - use a reasonable chunk for looping
    # Not too short (causes buzzing) or too long (won't repeat)
    source_len = min(seg_len, start, 50)  # Max 50 frames per loop chunk

    if source_len <= 10:
        # Too short to loop properly - just keep original audio
        return mel

    # Get source segment
    source = mel[start - source_len:start].clone()  # (source_len, M)

    # Create crossfade ramps for smooth loop transitions
    fade_len = min(10, source_len // 4)
    fade_in = torch.linspace(0, 1, fade_len, device=mel.device, dtype=mel.dtype).view(-1, 1)
    fade_out = torch.linspace(1, 0, fade_len, device=mel.device, dtype=mel.dtype).view(-1, 1)

    # Apply fade out to end of source and fade in to start
    source[-fade_len:] = source[-fade_len:] * fade_out
    source[:fade_len] = source[:fade_len] * fade_in

    # Tile to fill the loop region
    n_repeats = (seg_len + source_len - 1) // source_len
    tiled = source.repeat(n_repeats, 1)[:seg_len]

    # Crossfade at the boundaries with surrounding audio
    if start > fade_len:
        # Fade in from previous audio at start of loop region
        for i in range(fade_len):
            alpha = i / fade_len
            tiled[i] = (1 - alpha) * mel[start + i] + alpha * tiled[i]

    if end < len(mel) - fade_len:
        # Fade out to next audio at end of loop region
        for i in range(fade_len):
            alpha = i / fade_len
            tiled[-(fade_len - i)] = alpha * mel[end - fade_len + i] + (1 - alpha) * tiled[-(fade_len - i)]

    result[start:end] = tiled
    return result


def apply_effect(mel: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Apply a noticeable effect: aggressive low-pass + slight boost to low frequencies.

    This creates a "muffled" or "underwater" sound effect.
    """
    seg_len = end - start
    if seg_len <= 0:
        return mel

    result = mel.clone()
    segment = mel[start:end]  # (seg_len, n_mels)
    n_mels = segment.shape[-1]

    # Create a more aggressive filter:
    # - Boost low frequencies (first 1/4 of mel bins)
    # - Keep mid frequencies
    # - Heavily attenuate high frequencies (last 1/2 of mel bins)
    filter_curve = torch.ones(n_mels, device=mel.device, dtype=mel.dtype)

    # Low freq boost (mel bins 0-31)
    low_end = n_mels // 4
    filter_curve[:low_end] = 1.3

    # Mid freq unchanged (mel bins 32-63)
    mid_end = n_mels // 2

    # High freq attenuation (mel bins 64-127) - aggressive rolloff
    filter_curve[mid_end:] = torch.linspace(0.8, 0.0, n_mels - mid_end, device=mel.device, dtype=mel.dtype)

    filter_curve = filter_curve.view(1, -1)  # (1, n_mels)
    result[start:end] = torch.clamp(segment * filter_curve, 0, 1)
    return result


def apply_transition(mel: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Apply transition: crossfade from current audio to silence and back.

    Creates a "dip" in volume that's clearly audible as a transition.
    """
    seg_len = end - start
    if seg_len <= 0:
        return mel

    result = mel.clone()
    segment = mel[start:end]  # (seg_len, n_mels)

    # Create a dip envelope: 1.0 -> 0.2 -> 1.0 (down and back up)
    half = seg_len // 2
    ramp_down = torch.linspace(1.0, 0.2, half, device=mel.device, dtype=mel.dtype)
    ramp_up = torch.linspace(0.2, 1.0, seg_len - half, device=mel.device, dtype=mel.dtype)
    envelope = torch.cat([ramp_down, ramp_up]).view(-1, 1)

    result[start:end] = segment * envelope
    return result


class MelDecoder(nn.Module):
    """Decodes latent to edited mel spectrogram.

    Architecture:
        latent -> TransformerEncoder -> Linear -> pred_mel
        pred_mel + α * raw_mel -> output (residual connection)

    The residual connection helps preserve unedited regions and
    provides a strong initialization for reconstruction.
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Transformer decoder (using encoder architecture for self-attention)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.decoder_dim),
        )

        # Output projection with refinement layers
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_dim, config.decoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_dim, config.decoder_dim // 2),
            nn.GELU(),
            nn.Linear(config.decoder_dim // 2, config.audio.n_mels),
        )

        # Learnable residual weight (for non-gated mode)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # Gating mechanism for residual connection
        # DISABLED: Gating can cause chaotic blending - use simple residual instead
        self.use_gating = False
        if self.use_gating:
            # Gate input: decoded features + raw mel
            gate_input_dim = config.decoder_dim + config.audio.n_mels
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, config.decoder_dim),
                nn.GELU(),
                nn.Linear(config.decoder_dim, config.audio.n_mels),
                nn.Sigmoid(),
            )

    def forward(
        self,
        latent: torch.Tensor,  # (B, T, decoder_dim)
        raw_mel: torch.Tensor,  # (B, T, n_mels) for residual
        edit_labels: Optional[torch.Tensor] = None,  # (B, T) int - for label-aware gating
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent: Encoded representation (B, T, decoder_dim)
            raw_mel: Original mel spectrogram (B, T, n_mels)
            mask: Valid frame mask (B, T), True = valid

        Returns:
            pred_mel: Predicted edited mel (B, T, n_mels)
        """
        # Attention mask
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        # Decode
        decoded = self.transformer(latent, src_key_padding_mask=attn_mask)  # (B, T, decoder_dim)

        # Project to mel dimension and clamp to valid range
        pred_mel = self.output_proj(decoded)  # (B, T, n_mels)
        pred_mel = torch.clamp(pred_mel, 0, 1)  # CRITICAL: Keep in [0, 1] for vocoder

        # Compute base output (gated or simple residual)
        if self.use_gating:
            gate_input = torch.cat([decoded, raw_mel], dim=-1)
            gate = self.gate(gate_input)  # (B, T, n_mels), values in [0, 1]
            # Blend prediction with raw mel using learned gate
            base_output = pred_mel * (1 - gate) + raw_mel * gate
        else:
            # Simple weighted residual
            base_output = pred_mel + self.residual_weight * raw_mel

        # ALWAYS apply hard overrides based on edit labels (this is critical!)
        # Each label type gets a deterministic DSP operation
        if edit_labels is not None:
            B, T, M = raw_mel.shape
            output = torch.zeros_like(raw_mel)

            for b in range(B):
                # Start with raw mel for this batch item
                batch_output = raw_mel[b].clone()  # (T, M)
                batch_labels = edit_labels[b]  # (T,)

                # CUT (0) -> silence
                cut_mask = (batch_labels == 0)
                batch_output[cut_mask] = 0.0

                # KEEP (1) -> raw_mel (already set)
                # No change needed

                # LOOP (2) -> repeat previous frame
                loop_segments = find_contiguous_segments(batch_labels, 2)
                for start, end in loop_segments:
                    batch_output = apply_loop(batch_output, start, end)

                # FADE_IN (3) -> ramp 0→1
                fade_in_segments = find_contiguous_segments(batch_labels, 3)
                for start, end in fade_in_segments:
                    batch_output = apply_fade_in(batch_output, start, end)

                # FADE_OUT (4) -> ramp 1→0
                fade_out_segments = find_contiguous_segments(batch_labels, 4)
                for start, end in fade_out_segments:
                    batch_output = apply_fade_out(batch_output, start, end)

                # EFFECT (5) -> low-pass filter (reduce high frequencies)
                effect_segments = find_contiguous_segments(batch_labels, 5)
                for start, end in effect_segments:
                    batch_output = apply_effect(batch_output, start, end)

                # TRANSITION (6) -> partial fade + blur
                transition_segments = find_contiguous_segments(batch_labels, 6)
                for start, end in transition_segments:
                    batch_output = apply_transition(batch_output, start, end)

                output[b] = batch_output
        else:
            output = base_output

        # Ensure output stays in valid range
        output = torch.clamp(output, 0, 1)
        return output


class MultiScaleMelDecoder(nn.Module):
    """Decoder that produces mel at multiple scales.

    Inspired by FaceSwap's multi-resolution outputs.
    Helps with coarse-to-fine reconstruction.
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.decoder_dim),
        )

        # Multi-scale output heads
        # Scale 1: Full resolution (n_mels)
        self.head_full = nn.Linear(config.decoder_dim, config.audio.n_mels)

        # Scale 2: Half resolution (n_mels // 2) - for coarse structure
        self.head_half = nn.Linear(config.decoder_dim, config.audio.n_mels // 2)

        # Scale 3: Quarter resolution (n_mels // 4) - for very coarse
        self.head_quarter = nn.Linear(config.decoder_dim, config.audio.n_mels // 4)

        # Upsampling for loss computation at full resolution
        self.upsample_half = nn.Linear(config.audio.n_mels // 2, config.audio.n_mels)
        self.upsample_quarter = nn.Linear(config.audio.n_mels // 4, config.audio.n_mels)

        # Residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        latent: torch.Tensor,
        raw_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_scales: bool = False,
    ) -> torch.Tensor:
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        decoded = self.transformer(latent, src_key_padding_mask=attn_mask)

        # Multi-scale outputs
        out_full = self.head_full(decoded)
        out_half = self.head_half(decoded)
        out_quarter = self.head_quarter(decoded)

        # Final output is full resolution with residual, clamped to valid range
        output = out_full + self.residual_weight * raw_mel
        output = torch.clamp(output, 0, 1)  # CRITICAL: Keep in [0, 1] for vocoder

        if return_all_scales:
            # Upsample coarser scales for loss computation
            out_half_up = self.upsample_half(out_half)
            out_quarter_up = self.upsample_quarter(out_quarter)
            return output, out_half_up, out_quarter_up

        return output

"""Pointer Network for Audio Editing.

A comprehensive model that learns to generate sequences of frame pointers
from raw audio, enabling cutting, looping, and reordering operations.

Features:
- Multi-scale encoding (frame/beat/bar levels)
- Music-aware positional encoding
- STOP token for variable length output
- Length prediction + duration conditioning
- Hierarchical attention (bar -> beat -> frame)
- Sparse attention for efficiency
- VAE for edit style diversity
- Structure prediction auxiliary task
- KV caching for fast inference
- Chunked processing for long sequences

V2 Full-Sequence Architecture:
- Linear Attention Encoder (O(n) complexity for full sequences)
- Position-Aware Windowed Cross-Attention (attend to expected position window)
- Delta Prediction (predict offset from expected position, not absolute)
- Global Summary Tokens (for long-range jumps/loops)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple, List, Any
import math


# =============================================================================
# CONSTANTS
# =============================================================================
STOP_TOKEN = -2  # Special token indicating sequence end
PAD_TOKEN = -1   # Padding token


# =============================================================================
# EDIT OPERATIONS
# =============================================================================
class EditOp:
    """Edit operation types for explicit edit labeling."""
    COPY = 0        # Copy frame at pointer position
    LOOP_START = 1  # Mark start of loop region
    LOOP_END = 2    # End loop, jump back to LOOP_START
    SKIP = 3        # Skip N frames (cut)
    FADE_IN = 4     # Apply fade in
    FADE_OUT = 5    # Apply fade out
    STOP = 6        # End of sequence

    NUM_OPS = 7

    @classmethod
    def names(cls):
        return ['COPY', 'LOOP_START', 'LOOP_END', 'SKIP', 'FADE_IN', 'FADE_OUT', 'STOP']


# =============================================================================
# V2: LINEAR ATTENTION (O(n) complexity for full sequences)
# =============================================================================
class LinearAttention(nn.Module):
    """Linear attention using ELU feature map for O(n) complexity.

    Instead of softmax(QK^T)V which is O(n²), we use:
    φ(Q)(φ(K)^T V) where φ is ELU+1 feature map.
    This allows computing (K^T V) first in O(n*d) then multiplying by Q.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 feature map for non-negative features."""
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: φ(Q)(φ(K)^T V)
        # First compute K^T V: (batch, heads, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)

        # Then multiply by Q: (batch, heads, seq_len, head_dim)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)

        # Normalize by sum of keys (for numerical stability)
        k_sum = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).squeeze(-1) + 1e-6
        out = out / normalizer.unsqueeze(-1)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.dropout(self.out_proj(out))

        return out


class LinearAttentionEncoderLayer(nn.Module):
    """Encoder layer with linear attention for O(n) complexity."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear_attn = LinearAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm linear attention
        x = x + self.linear_attn(self.norm1(x))
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionEncoder(nn.Module):
    """Full encoder using linear attention for O(n) complexity on full sequences."""

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_global_tokens: int = 64,  # Summary tokens for long-range
        global_token_stride: int = 1000,  # One global token per N frames
    ):
        super().__init__()
        self.d_model = d_model
        self.n_global_tokens = n_global_tokens
        self.global_token_stride = global_token_stride

        # Input projection
        self.mel_proj = nn.Linear(n_mels, d_model)

        # Positional encoding (sinusoidal for arbitrary length)
        self.pos_scale = nn.Parameter(torch.ones(1))

        # Encoder layers
        self.layers = nn.ModuleList([
            LinearAttentionEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Global token generator (pools local info into summary tokens)
        self.global_pool = nn.Conv1d(d_model, d_model, kernel_size=global_token_stride,
                                      stride=global_token_stride, padding=0)
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _sinusoidal_pos_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                            (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe * self.pos_scale

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: (batch, n_mels, time)
        Returns:
            dict with 'frame' and 'global' embeddings
        """
        batch_size, n_mels, time = mel.shape

        # Project mel to d_model
        x = mel.transpose(1, 2)  # (batch, time, n_mels)
        x = self.mel_proj(x)  # (batch, time, d_model)

        # Add positional encoding
        pos_enc = self._sinusoidal_pos_encoding(time, x.device)
        x = x + pos_enc.unsqueeze(0)

        # Run through linear attention layers
        for layer in self.layers:
            x = layer(x)

        frame_emb = self.final_norm(x)

        # Generate global summary tokens
        if time >= self.global_token_stride:
            # Pool to create global tokens
            global_tokens = self.global_pool(frame_emb.transpose(1, 2)).transpose(1, 2)
            global_tokens = self.global_proj(global_tokens)
        else:
            # For short sequences, just use mean
            global_tokens = frame_emb.mean(dim=1, keepdim=True)
            global_tokens = self.global_proj(global_tokens)

        return {
            'frame': frame_emb,
            'global': global_tokens,
        }


# =============================================================================
# V2: POSITION-AWARE WINDOWED CROSS-ATTENTION
# =============================================================================
class PositionAwareWindowedAttention(nn.Module):
    """Cross-attention that only attends to a window around expected position.

    For output position t with compression ratio r:
    - Expected raw position = t / r
    - Only attend to encoder[expected - window : expected + window]
    - Plus global tokens for long-range dependencies

    This makes cross-attention O(output_len * window_size) instead of O(output_len * input_len).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,  # Attend to ±256 frames around expected position
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Separate projections for global tokens
        self.global_k_proj = nn.Linear(d_model, d_model)
        self.global_v_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,  # (batch, tgt_len, d_model)
        encoder_out: torch.Tensor,  # (batch, src_len, d_model)
        global_tokens: torch.Tensor,  # (batch, n_global, d_model)
        compression_ratio: float,  # Expected output/input ratio
        cumulative_offset: Optional[torch.Tensor] = None,  # (batch, tgt_len) position adjustments
    ) -> torch.Tensor:
        """
        Args:
            query: Decoder queries (batch, tgt_len, d_model)
            encoder_out: Encoder frame embeddings (batch, src_len, d_model)
            global_tokens: Global summary tokens (batch, n_global, d_model)
            compression_ratio: Expected edit/raw ratio (e.g., 0.67)
            cumulative_offset: Optional position adjustments from previous predictions
        """
        batch_size, tgt_len, _ = query.shape
        src_len = encoder_out.shape[1]
        device = query.device

        # Compute expected positions for each output position
        positions = torch.arange(tgt_len, device=device).float()
        expected_positions = (positions / compression_ratio).long()

        if cumulative_offset is not None:
            expected_positions = expected_positions.unsqueeze(0) + cumulative_offset
            expected_positions = expected_positions.long()
        else:
            expected_positions = expected_positions.unsqueeze(0).expand(batch_size, -1)

        # Clamp to valid range
        expected_positions = expected_positions.clamp(0, src_len - 1)

        # Project queries
        q = self.q_proj(query)
        q = q.view(batch_size, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)

        # For efficiency, we compute attention in chunks
        # Each output position attends to its local window + global tokens

        # Project global tokens (same for all positions)
        global_k = self.global_k_proj(global_tokens)
        global_v = self.global_v_proj(global_tokens)
        n_global = global_tokens.shape[1]

        global_k = global_k.view(batch_size, n_global, self.n_heads, self.head_dim).transpose(1, 2)
        global_v = global_v.view(batch_size, n_global, self.n_heads, self.head_dim).transpose(1, 2)

        # Pre-project all encoder frames
        all_k = self.k_proj(encoder_out)
        all_v = self.v_proj(encoder_out)
        all_k = all_k.view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        all_v = all_v.view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Process in chunks for memory efficiency
        chunk_size = 256
        outputs = []

        for chunk_start in range(0, tgt_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, tgt_len)
            chunk_q = q[:, :, chunk_start:chunk_end, :]  # (batch, heads, chunk, head_dim)
            chunk_expected = expected_positions[:, chunk_start:chunk_end]  # (batch, chunk)

            # Gather local windows for this chunk
            # For simplicity, use a single window for the whole chunk based on center
            chunk_center = chunk_expected[:, chunk_expected.shape[1] // 2]  # (batch,)

            window_start = (chunk_center - self.window_size // 2).clamp(0, src_len - self.window_size)
            window_end = (window_start + self.window_size).clamp(0, src_len)

            # Gather window keys/values
            # For each batch, extract the window
            window_k_list = []
            window_v_list = []
            for b in range(batch_size):
                ws = window_start[b].item()
                we = window_end[b].item()
                window_k_list.append(all_k[b, :, ws:we, :])
                window_v_list.append(all_v[b, :, ws:we, :])

            # Pad to same size and stack
            max_window = max(wk.shape[1] for wk in window_k_list)
            window_k = torch.zeros(batch_size, self.n_heads, max_window, self.head_dim, device=device)
            window_v = torch.zeros(batch_size, self.n_heads, max_window, self.head_dim, device=device)
            window_mask = torch.ones(batch_size, max_window, device=device, dtype=torch.bool)

            for b, (wk, wv) in enumerate(zip(window_k_list, window_v_list)):
                wlen = wk.shape[1]
                window_k[b, :, :wlen, :] = wk
                window_v[b, :, :wlen, :] = wv
                window_mask[b, :wlen] = False

            # Concatenate global tokens
            full_k = torch.cat([global_k, window_k], dim=2)  # (batch, heads, n_global + window, head_dim)
            full_v = torch.cat([global_v, window_v], dim=2)
            full_mask = torch.cat([
                torch.zeros(batch_size, n_global, device=device, dtype=torch.bool),
                window_mask
            ], dim=1)

            # Compute attention
            attn = torch.matmul(chunk_q, full_k.transpose(-2, -1)) / self.scale
            attn = attn.masked_fill(full_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            chunk_out = torch.matmul(attn, full_v)
            outputs.append(chunk_out)

        # Concatenate chunks
        out = torch.cat(outputs, dim=2)  # (batch, heads, tgt_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)

        return self.out_proj(out)


# =============================================================================
# V2: DELTA PREDICTION HEAD
# =============================================================================
class DeltaPredictionHead(nn.Module):
    """Predicts pointer offset from expected position instead of absolute position.

    For output position t:
    - Expected raw position = t / compression_ratio
    - Predict delta ∈ {-max_delta, ..., +max_delta} OR large jump
    - Actual position = expected + delta

    This leverages the prior that edits are mostly sequential (delta=0 or 1).
    """

    def __init__(
        self,
        d_model: int,
        max_delta: int = 64,  # Max small delta (covers ~0.7 seconds at 22050/256)
        n_jump_buckets: int = 32,  # For large jumps, quantize into buckets
    ):
        super().__init__()
        self.d_model = d_model
        self.max_delta = max_delta
        self.n_jump_buckets = n_jump_buckets

        # Delta prediction: -max_delta to +max_delta (2*max_delta + 1 classes)
        self.n_delta_classes = 2 * max_delta + 1
        self.delta_head = nn.Linear(d_model, self.n_delta_classes)

        # Large jump prediction (for jumps > max_delta)
        # Predicts bucket index, which maps to relative jump size
        self.jump_head = nn.Linear(d_model, n_jump_buckets)

        # Whether to use large jump vs small delta
        self.use_jump_head = nn.Linear(d_model, 1)

        # Stop prediction
        self.stop_head = nn.Linear(d_model, 1)

    def forward(self, decoder_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_out: (batch, tgt_len, d_model)
        Returns:
            dict with 'delta_logits', 'jump_logits', 'use_jump_logits', 'stop_logits'
        """
        return {
            'delta_logits': self.delta_head(decoder_out),  # (batch, tgt_len, n_delta_classes)
            'jump_logits': self.jump_head(decoder_out),    # (batch, tgt_len, n_jump_buckets)
            'use_jump_logits': self.use_jump_head(decoder_out).squeeze(-1),  # (batch, tgt_len)
            'stop_logits': self.stop_head(decoder_out).squeeze(-1),  # (batch, tgt_len)
        }

    def delta_to_index(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert delta values to class indices."""
        # delta in [-max_delta, max_delta] -> index in [0, 2*max_delta]
        return (delta + self.max_delta).clamp(0, self.n_delta_classes - 1)

    def index_to_delta(self, index: torch.Tensor) -> torch.Tensor:
        """Convert class indices to delta values."""
        return index - self.max_delta

    def compute_targets(
        self,
        target_pointers: torch.Tensor,  # (batch, tgt_len)
        compression_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute delta targets from absolute pointer targets.

        Args:
            target_pointers: Absolute pointer positions (batch, tgt_len)
            compression_ratio: Expected edit/raw ratio
        Returns:
            dict with 'delta_targets', 'use_jump_targets', 'jump_targets'
        """
        batch_size, tgt_len = target_pointers.shape
        device = target_pointers.device

        # Compute expected positions
        positions = torch.arange(tgt_len, device=device).float()
        expected_positions = (positions / compression_ratio).unsqueeze(0).expand(batch_size, -1)

        # Compute deltas
        deltas = target_pointers.float() - expected_positions

        # Handle STOP tokens (mark as delta=0 for now, will be masked)
        stop_mask = target_pointers == STOP_TOKEN
        deltas = deltas.masked_fill(stop_mask, 0)

        # Determine if small delta or large jump
        use_jump = deltas.abs() > self.max_delta

        # Small delta targets (clamp to valid range)
        delta_targets = self.delta_to_index(deltas.round().long())

        # Large jump targets (quantize into buckets)
        # Map large jumps to bucket indices based on magnitude
        jump_magnitude = deltas.abs()
        jump_sign = deltas.sign()
        # Log-scale bucketing for jumps
        jump_log = torch.log1p(jump_magnitude - self.max_delta).clamp(0, 10)
        jump_bucket = (jump_log / 10 * (self.n_jump_buckets // 2 - 1)).long()
        # Encode sign in bucket index
        jump_targets = torch.where(
            jump_sign >= 0,
            jump_bucket + self.n_jump_buckets // 2,
            self.n_jump_buckets // 2 - 1 - jump_bucket
        ).clamp(0, self.n_jump_buckets - 1)

        return {
            'delta_targets': delta_targets,  # (batch, tgt_len)
            'use_jump_targets': use_jump,    # (batch, tgt_len) bool
            'jump_targets': jump_targets,    # (batch, tgt_len)
            'stop_targets': stop_mask,       # (batch, tgt_len) bool
            'expected_positions': expected_positions,  # (batch, tgt_len)
        }

    def decode_predictions(
        self,
        delta_logits: torch.Tensor,
        jump_logits: torch.Tensor,
        use_jump_logits: torch.Tensor,
        compression_ratio: float,
        src_len: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Decode predictions to absolute pointer positions.

        Args:
            delta_logits: (batch, tgt_len, n_delta_classes)
            jump_logits: (batch, tgt_len, n_jump_buckets)
            use_jump_logits: (batch, tgt_len)
            compression_ratio: Expected edit/raw ratio
            src_len: Source sequence length
            temperature: Sampling temperature (0 = greedy)
        Returns:
            pointers: (batch, tgt_len) absolute pointer positions
        """
        batch_size, tgt_len, _ = delta_logits.shape
        device = delta_logits.device

        # Expected positions
        positions = torch.arange(tgt_len, device=device).float()
        expected = (positions / compression_ratio).unsqueeze(0).expand(batch_size, -1)

        # Decide whether to use jump or delta
        use_jump = torch.sigmoid(use_jump_logits) > 0.5

        # Get delta predictions
        if temperature > 0:
            delta_probs = F.softmax(delta_logits / temperature, dim=-1)
            delta_idx = torch.multinomial(delta_probs.view(-1, self.n_delta_classes), 1)
            delta_idx = delta_idx.view(batch_size, tgt_len)
        else:
            delta_idx = delta_logits.argmax(dim=-1)

        deltas = self.index_to_delta(delta_idx).float()

        # Get jump predictions (for positions that need it)
        if temperature > 0:
            jump_probs = F.softmax(jump_logits / temperature, dim=-1)
            jump_idx = torch.multinomial(jump_probs.view(-1, self.n_jump_buckets), 1)
            jump_idx = jump_idx.view(batch_size, tgt_len)
        else:
            jump_idx = jump_logits.argmax(dim=-1)

        # Decode jump bucket to actual offset
        center = self.n_jump_buckets // 2
        jump_sign = torch.where(jump_idx >= center,
                                torch.ones_like(jump_idx),
                                -torch.ones_like(jump_idx)).float()
        jump_log_mag = torch.where(
            jump_idx >= center,
            (jump_idx - center).float() / (center - 1) * 10,
            (center - 1 - jump_idx).float() / (center - 1) * 10
        )
        jump_offset = jump_sign * (torch.expm1(jump_log_mag) + self.max_delta)

        # Combine delta and jump based on use_jump
        offset = torch.where(use_jump, jump_offset, deltas)

        # Compute absolute positions
        pointers = (expected + offset).round().long()
        pointers = pointers.clamp(0, src_len - 1)

        return pointers


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================
class MusicAwarePositionalEncoding(nn.Module):
    """Positional encoding incorporating musical structure (beat/bar/phrase)."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 65536,
        frames_per_beat: int = 43,
        beats_per_bar: int = 4,
        bars_per_phrase: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.bars_per_phrase = bars_per_phrase

        # Split d_model for different scales
        self.dim_per_scale = d_model // 4

        # Learnable embeddings for musical structure
        self.frame_embed = nn.Embedding(frames_per_beat, self.dim_per_scale)
        self.beat_embed = nn.Embedding(beats_per_bar, self.dim_per_scale)
        self.bar_embed = nn.Embedding(bars_per_phrase, self.dim_per_scale)

        # Sinusoidal for absolute position
        remaining_dim = d_model - 3 * self.dim_per_scale
        pe = torch.zeros(max_len, remaining_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, remaining_dim, 2).float() * (-math.log(10000.0) / remaining_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if remaining_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:remaining_dim // 2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

        # Cache for position indices (avoids recomputing modulo ops each forward)
        self._pos_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_position_indices(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cached position indices or compute them."""
        cache_key = (seq_len, str(device))
        if cache_key not in self._pos_cache:
            frames = torch.arange(seq_len, device=device)
            beat_pos = frames % self.frames_per_beat
            bar_pos = (frames // self.frames_per_beat) % self.beats_per_bar
            phrase_pos = (frames // (self.frames_per_beat * self.beats_per_bar)) % self.bars_per_phrase
            self._pos_cache[cache_key] = (beat_pos, bar_pos, phrase_pos)
        return self._pos_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get cached position indices
        beat_pos, bar_pos, phrase_pos = self._get_position_indices(seq_len, device)

        frame_pe = self.frame_embed(beat_pos).unsqueeze(0).expand(batch_size, -1, -1)
        beat_pe = self.beat_embed(bar_pos).unsqueeze(0).expand(batch_size, -1, -1)
        bar_pe = self.bar_embed(phrase_pos).unsqueeze(0).expand(batch_size, -1, -1)
        abs_pe = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

        pos_encoding = torch.cat([frame_pe, beat_pe, bar_pe, abs_pe], dim=-1)
        return self.dropout(x + pos_encoding)


# =============================================================================
# ATTENTION MODULES
# =============================================================================
class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV caching for fast autoregressive inference."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, tgt_len, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        new_cache = (k, v) if use_cache else None

        q = q.view(batch_size, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(out), new_cache


class HierarchicalAttention(nn.Module):
    """Hierarchical attention: coarse (bar) -> medium (beat) -> fine (frame)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        frames_per_beat: int = 43,
        beats_per_bar: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar

        self.bar_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.beat_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.frame_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

        self.gate = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Sigmoid())
        self.combine = nn.Linear(d_model * 3, d_model)

    def forward(
        self,
        query: torch.Tensor,
        frame_keys: torch.Tensor,
        beat_keys: torch.Tensor,
        bar_keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bar_out, _ = self.bar_attn(query, bar_keys, bar_keys, need_weights=False)
        beat_out, _ = self.beat_attn(query, beat_keys, beat_keys, need_weights=False)
        frame_out, _ = self.frame_attn(query, frame_keys, frame_keys, need_weights=False)

        combined = torch.cat([bar_out, beat_out, frame_out], dim=-1)
        gate = self.gate(combined)
        output = self.combine(combined) * gate + frame_out * (1 - gate)

        return output, None  # No longer returning weights for Flash Attention


class SparseAttention(nn.Module):
    """Sparse attention with local window + strided global patterns."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        local_window: int = 128,
        stride: int = 64,
        n_global_tokens: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local_window = local_window
        self.stride = stride
        self.n_global_tokens = n_global_tokens
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.global_tokens = nn.Parameter(torch.randn(n_global_tokens, d_model) * 0.02)

        # Cache for attention masks (avoids recreating on each forward)
        self._mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _get_or_create_mask(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        """Get cached mask or create new one."""
        cache_key = (tgt_len, src_len, str(device))
        if cache_key not in self._mask_cache:
            # Create sparse mask
            attn_mask = torch.zeros(tgt_len, src_len + self.n_global_tokens, device=device)
            attn_mask[:, :self.n_global_tokens] = 1  # Global tokens always attend

            # Vectorized local window computation (faster than loop)
            positions = torch.arange(tgt_len, device=device).unsqueeze(1)
            src_positions = torch.arange(src_len, device=device).unsqueeze(0)
            local_mask = (src_positions >= positions - self.local_window // 2) & \
                         (src_positions < positions + self.local_window // 2)
            attn_mask[:, self.n_global_tokens:] = local_mask.float()

            # Add strided positions
            strided = torch.arange(0, src_len, self.stride, device=device)
            attn_mask[:, self.n_global_tokens + strided] = 1

            self._mask_cache[cache_key] = attn_mask
        return self._mask_cache[cache_key]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len, d_model = query.shape
        src_len = key.shape[1]
        device = query.device

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Add global tokens
        global_k = self.global_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        k = torch.cat([global_k, k], dim=1)
        v = torch.cat([global_k, v], dim=1)

        # Get cached or create sparse mask
        attn_mask = self._get_or_create_mask(tgt_len, src_len, device)

        q = q.view(batch_size, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, d_model)
        return self.out_proj(out)


# =============================================================================
# PRE-LAYERNORM TRANSFORMER LAYERS (More stable training)
# =============================================================================
class PreNormTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm (more stable, no NaN)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm: normalize before attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm: normalize before FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x


class PreNormTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm (more stable)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        tgt_norm = self.norm1(tgt)
        self_attn_out, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout(self_attn_out)

        # Cross-attention with pre-norm
        tgt_norm = self.norm2(tgt)
        cross_attn_out, _ = self.cross_attn(
            tgt_norm, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout(cross_attn_out)

        # FFN with pre-norm
        tgt_norm = self.norm3(tgt)
        tgt = tgt + self.ffn(tgt_norm)
        return tgt


# =============================================================================
# STEM ENCODER (Multi-track support)
# =============================================================================
class StemEncoder(nn.Module):
    """Encodes multiple audio stems into a shared representation.

    Stems (drums, bass, vocals, other) are encoded separately then fused
    via attention. The same pointers will be applied to all stems for
    coherent editing.
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_stems: int = 4,  # drums, bass, vocals, other
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stems = n_stems
        self.d_model = d_model

        # Per-stem projection
        self.stem_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_mels, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_stems)
        ])

        # Stem type embeddings
        self.stem_embed = nn.Embedding(n_stems, d_model)

        # Fusion layer (concatenate + project)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_stems, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Attention-based fusion (alternative, more expressive)
        # NOTE: Disabled due to CUDA issues with large batch*seq_len
        self.use_attention_fusion = False
        if self.use_attention_fusion:
            self.fusion_attn = nn.MultiheadAttention(
                d_model, num_heads=4, dropout=dropout, batch_first=True
            )
            self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self,
        stems: Dict[str, torch.Tensor],
        stem_order: List[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            stems: Dict mapping stem name to mel spectrogram (B, n_mels, T) or (B, T, n_mels)
            stem_order: Order of stems for consistent processing

        Returns:
            Fused representation (B, T, d_model)
        """
        if stem_order is None:
            stem_order = ['drums', 'bass', 'vocals', 'other']

        # Get first stem to determine shape
        first_stem = next(iter(stems.values()))
        if first_stem.dim() == 3 and first_stem.shape[1] != first_stem.shape[2]:
            # Assume (B, n_mels, T) format, transpose to (B, T, n_mels)
            stems = {k: v.transpose(1, 2) if v.shape[1] < v.shape[2] else v for k, v in stems.items()}
            first_stem = next(iter(stems.values()))

        batch_size = first_stem.shape[0]
        seq_len = first_stem.shape[1]
        device = first_stem.device

        # Encode each stem
        stem_encodings = []
        for i, stem_name in enumerate(stem_order):
            if stem_name in stems:
                stem_mel = stems[stem_name]  # (B, T, n_mels)
            else:
                # Use zeros for missing stems
                n_mels = self.stem_proj[0][0].in_features
                stem_mel = torch.zeros(batch_size, seq_len, n_mels, device=device)

            # Project and add stem embedding
            encoded = self.stem_proj[i](stem_mel)  # (B, T, d_model)
            encoded = encoded + self.stem_embed.weight[i].unsqueeze(0).unsqueeze(0)
            stem_encodings.append(encoded)

        if self.use_attention_fusion:
            # Stack stems: (B, T, n_stems, d_model)
            stacked = torch.stack(stem_encodings, dim=2)
            # Reshape for attention: (B*T, n_stems, d_model)
            stacked = stacked.view(batch_size * seq_len, self.n_stems, self.d_model)
            # Query to fuse: (B*T, 1, d_model)
            query = self.fusion_query.expand(batch_size * seq_len, -1, -1)
            # Attention fusion
            fused, _ = self.fusion_attn(query, stacked, stacked, need_weights=False)
            # Reshape back: (B, T, d_model)
            fused = fused.view(batch_size, seq_len, self.d_model)
        else:
            # Concatenate and project
            concat = torch.cat(stem_encodings, dim=-1)  # (B, T, d_model * n_stems)
            fused = self.fusion(concat)  # (B, T, d_model)

        return fused


# =============================================================================
# ENCODER
# =============================================================================
class MultiScaleEncoder(nn.Module):
    """Encodes mel spectrograms at frame, beat, and bar levels."""

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        frames_per_beat: int = 43,
        beats_per_bar: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        use_pre_norm: bool = True,  # Pre-LayerNorm for stability
        dim_feedforward: int = None,  # Default: d_model * 4
    ):
        super().__init__()
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.frames_per_bar = frames_per_beat * beats_per_bar
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.use_pre_norm = use_pre_norm

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.mel_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = MusicAwarePositionalEncoding(d_model, dropout=dropout)

        # Frame encoder - use Pre-LayerNorm if specified
        if use_pre_norm:
            self.frame_encoder = nn.ModuleList([
                PreNormTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ])
            self.frame_encoder_norm = nn.LayerNorm(d_model)  # Final norm for pre-norm
        else:
            self.frame_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model, n_heads, dim_feedforward, dropout, 'gelu', batch_first=True
                ),
                num_layers=n_layers
            )
            self.frame_encoder_norm = None

        self.beat_pool = nn.Conv1d(d_model, d_model, frames_per_beat, stride=frames_per_beat)

        if use_pre_norm:
            self.beat_encoder = nn.ModuleList([
                PreNormTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(2)
            ])
            self.beat_encoder_norm = nn.LayerNorm(d_model)
        else:
            self.beat_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, 'gelu', batch_first=True),
                num_layers=2
            )
            self.beat_encoder_norm = None

        self.bar_pool = nn.Conv1d(d_model, d_model, beats_per_bar, stride=beats_per_bar)

        if use_pre_norm:
            self.bar_encoder = nn.ModuleList([
                PreNormTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(2)
            ])
            self.bar_encoder_norm = nn.LayerNorm(d_model)
        else:
            self.bar_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, 'gelu', batch_first=True),
                num_layers=2
            )
            self.bar_encoder_norm = None

    def _run_encoder(self, x: torch.Tensor, encoder, final_norm=None) -> torch.Tensor:
        """Run encoder (handles both ModuleList and TransformerEncoder)."""
        if self.use_pre_norm:
            # Pre-norm: run through ModuleList
            for layer in encoder:
                x = layer(x)
            if final_norm is not None:
                x = final_norm(x)
        else:
            # Post-norm: run through TransformerEncoder
            x = encoder(x)
        return x

    def _frame_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frame encoder forward (for checkpointing)."""
        return self._run_encoder(x, self.frame_encoder, self.frame_encoder_norm)

    def _beat_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Beat encoder forward (for checkpointing)."""
        return self._run_encoder(x, self.beat_encoder, self.beat_encoder_norm)

    def _bar_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Bar encoder forward (for checkpointing)."""
        return self._run_encoder(x, self.bar_encoder, self.bar_encoder_norm)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: (batch, n_mels, time)
        Returns:
            dict with 'frame', 'beat', 'bar' embeddings
        """
        batch_size, n_mels, time = mel.shape

        # Frame level
        x = mel.transpose(1, 2)
        x = self.mel_proj(x)
        x = self.pos_enc(x)

        if self.use_checkpoint and self.training:
            frame_emb = checkpoint(self._frame_encode, x, use_reentrant=False)
        else:
            frame_emb = self._run_encoder(x, self.frame_encoder, self.frame_encoder_norm)

        # Beat level
        if time >= self.frames_per_beat:
            beat_emb = self.beat_pool(frame_emb.transpose(1, 2)).transpose(1, 2)
            if self.use_checkpoint and self.training:
                beat_emb = checkpoint(self._beat_encode, beat_emb, use_reentrant=False)
            else:
                beat_emb = self._run_encoder(beat_emb, self.beat_encoder, self.beat_encoder_norm)
        else:
            beat_emb = frame_emb.mean(dim=1, keepdim=True)

        # Bar level
        n_beats = beat_emb.shape[1]
        if n_beats >= self.beats_per_bar:
            bar_emb = self.bar_pool(beat_emb.transpose(1, 2)).transpose(1, 2)
            if self.use_checkpoint and self.training:
                bar_emb = checkpoint(self._bar_encode, bar_emb, use_reentrant=False)
            else:
                bar_emb = self._run_encoder(bar_emb, self.bar_encoder, self.bar_encoder_norm)
        else:
            bar_emb = beat_emb.mean(dim=1, keepdim=True)

        return {'frame': frame_emb, 'beat': beat_emb, 'bar': bar_emb}


# =============================================================================
# AUXILIARY MODULES
# =============================================================================
class LengthPredictor(nn.Module):
    """Predicts output length from encoder output."""

    def __init__(self, d_model: int, max_length: int = 65536):
        super().__init__()
        self.max_length = max_length
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, encoder_out: torch.Tensor, src_len: int) -> torch.Tensor:
        pooled = encoder_out.mean(dim=1)
        ratio = torch.sigmoid(self.net(pooled).squeeze(-1))
        return (ratio * src_len * 1.5).clamp(1, self.max_length)


class DurationConditioning(nn.Module):
    """Conditions decoder on target duration."""

    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, embeddings: torch.Tensor, target_length: int) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        length_norm = torch.tensor([[target_length / 10000.0]], device=embeddings.device)
        length_norm = length_norm.expand(batch_size, 1)
        cond = self.embed(length_norm).unsqueeze(1)
        return embeddings + cond


class EditStyleVAE(nn.Module):
    """VAE for learning edit style latent space for diverse outputs."""

    def __init__(self, d_model: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(d_model // 2, latent_dim)
        self.logvar_head = nn.Linear(d_model // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, context: torch.Tensor, sample: bool = True):
        # bfloat16 is numerically stable, no need to disable AMP
        h = self.encoder(context.mean(dim=1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        conditioning = self.decoder(z)
        return conditioning, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp logvar to prevent extreme values
        logvar = logvar.clamp(-10, 10)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Clamp KL loss to prevent explosion (normal range is 0.0001-0.01)
        return kl.clamp(0, 1.0)


class StructurePredictionHead(nn.Module):
    """Auxiliary task: predict cut points and loop boundaries."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cut_head = nn.Linear(d_model, 1)
        self.loop_start_head = nn.Linear(d_model, 1)
        self.loop_end_head = nn.Linear(d_model, 1)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'cut_logits': self.cut_head(embeddings).squeeze(-1),
            'loop_start_logits': self.loop_start_head(embeddings).squeeze(-1),
            'loop_end_logits': self.loop_end_head(embeddings).squeeze(-1),
        }


# =============================================================================
# MAIN MODEL
# =============================================================================
class PointerNetwork(nn.Module):
    """Pointer Network for learning audio editing.

    Takes raw audio mel spectrogram and outputs a sequence of pointers
    into the raw audio, representing which frames to select and in what order.

    Architecture:
    - Linear attention encoder (O(n) complexity for full sequences)
    - Position-aware windowed cross-attention
    - Delta prediction (offset from expected position)
    - Global summary tokens for long-range jumps
    - Multi-stem encoding with shared pointers
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        latent_dim: int = 64,
        frames_per_beat: int = 43,
        beats_per_bar: int = 4,
        dropout: float = 0.1,
        max_length: int = 65536,
        use_checkpoint: bool = False,
        use_pre_norm: bool = True,      # Pre-LayerNorm for stability
        use_stems: bool = False,         # Enable multi-stem encoding
        n_stems: int = 4,                # Number of stems (drums, bass, vocals, other)
        dim_feedforward: int = None,     # FFN dimension (default: d_model * 4)
        label_smoothing: float = 0.1,    # Label smoothing for loss
        # Full-Sequence features
        compression_ratio: float = 0.67, # Expected output/input ratio
        attn_window_size: int = 512,     # Window size for position-aware attention
        max_delta: int = 64,             # Max delta for small adjustments
        n_global_tokens: int = 64,       # Number of global summary tokens
        global_token_stride: int = 1000, # Frames per global token
        # Deprecated V1 params (ignored, kept for checkpoint compat)
        use_edit_ops: bool = False,
        op_loss_weight: float = 0.1,
        use_full_sequence: bool = True,  # Always True now
    ):
        super().__init__()

        self.d_model = d_model
        self.n_mels = n_mels
        self.max_length = max_length
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.frames_per_bar = frames_per_beat * beats_per_bar
        self.use_checkpoint = use_checkpoint
        self.use_pre_norm = use_pre_norm
        self.use_stems = use_stems
        self.n_stems = n_stems
        self.label_smoothing = label_smoothing

        # Full-sequence settings
        self.compression_ratio = compression_ratio
        self.attn_window_size = attn_window_size
        self.max_delta = max_delta

        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.dim_feedforward = dim_feedforward

        # Precomputed constants
        self.scale = math.sqrt(d_model)  # Attention temperature scale

        # Cache for position indices
        self._position_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Multi-stem encoder (optional)
        if use_stems:
            self.stem_encoder = StemEncoder(n_mels, d_model, n_stems, dropout)

        # Linear attention encoder for O(n) complexity
        self.encoder = LinearAttentionEncoder(
            n_mels, d_model, n_heads, n_encoder_layers,
            dim_feedforward, dropout, n_global_tokens, global_token_stride
        )

        # VAE for style
        self.style_vae = EditStyleVAE(d_model, latent_dim)

        # Length prediction
        self.length_predictor = LengthPredictor(d_model, max_length)

        # Duration conditioning
        self.duration_cond = DurationConditioning(d_model)

        # Decoder queries
        self.query_embed = nn.Embedding(max_length + 1, d_model)  # +1 for STOP
        self.stop_embed_idx = max_length

        # Position-aware windowed cross-attention
        self.windowed_attn = PositionAwareWindowedAttention(
            d_model, n_heads, attn_window_size, dropout
        )

        # Delta prediction head
        self.delta_head = DeltaPredictionHead(d_model, max_delta)

        # Decoder transformer layers (Pre-LayerNorm for stability)
        self.decoder_layers = nn.ModuleList([
            PreNormTransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Auxiliary task (structure prediction)
        self.structure_head = StructurePredictionHead(d_model)

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and small gain for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_positions(self, length: int, device: torch.device) -> torch.Tensor:
        """Get cached position indices."""
        cache_key = (length, str(device))
        if cache_key not in self._position_cache:
            self._position_cache[cache_key] = torch.arange(length, device=device)
        return self._position_cache[cache_key]

    def forward(
        self,
        raw_mel: torch.Tensor,
        target_pointers: torch.Tensor,
        target_ops: Optional[torch.Tensor] = None,  # Edit operation targets
        target_length: Optional[int] = None,
        use_vae: bool = True,
        stems: Optional[Dict[str, torch.Tensor]] = None,  # Multi-stem input
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            raw_mel: (batch, n_mels, time)
            target_pointers: (batch, tgt_len) with STOP_TOKEN (-2) for end positions
            target_ops: (batch, tgt_len) edit operation labels (optional)
            stems: Dict of stem mels {name: (batch, n_mels, time)} (optional)
        """
        batch_size = raw_mel.shape[0]
        device = raw_mel.device

        # Multi-stem encoding (if enabled and stems provided)
        if self.use_stems and stems is not None:
            stem_emb = self.stem_encoder(stems)
            # Add stem embedding to the main encoder input
            # This is a simple approach - could also concatenate or use cross-attention
            raw_mel_for_encoder = raw_mel  # Still use raw_mel for positional structure
        else:
            raw_mel_for_encoder = raw_mel

        # Encode with linear attention encoder
        encoded = self.encoder(raw_mel_for_encoder)
        frame_emb = encoded['frame']
        global_tokens = encoded['global']
        src_len = frame_emb.shape[1]

        # If using stems, fuse the stem embedding with frame embedding
        if self.use_stems and stems is not None:
            if stem_emb.shape[1] != frame_emb.shape[1]:
                stem_emb = F.interpolate(
                    stem_emb.transpose(1, 2), size=frame_emb.shape[1], mode='linear', align_corners=False
                ).transpose(1, 2)
            frame_emb = frame_emb + stem_emb

        # Style VAE
        if use_vae:
            style_cond, mu, logvar = self.style_vae(frame_emb, sample=True)
            kl_loss = self.style_vae.kl_loss(mu, logvar)
        else:
            style_cond, mu, logvar = self.style_vae(frame_emb, sample=False)
            kl_loss = torch.tensor(0.0, device=device)

        # Target length
        tgt_len = target_pointers.shape[1]
        if target_length is None:
            target_length = tgt_len

        # Duration conditioning
        cond_frame_emb = self.duration_cond(frame_emb, target_length)
        cond_frame_emb = cond_frame_emb + style_cond.unsqueeze(1)

        # Decoder queries
        positions = self._get_positions(tgt_len, device).unsqueeze(0).expand(batch_size, -1)
        queries = self.query_embed(positions.clamp(0, self.stop_embed_idx - 1))

        # Position-aware windowed cross-attention
        decoder_out = self.windowed_attn(
            queries, cond_frame_emb, global_tokens,
            self.compression_ratio
        )

        # Decoder layers (Pre-LayerNorm)
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, cond_frame_emb)
        decoder_out = self.decoder_norm(decoder_out)

        # === DELTA PREDICTION ===
        stop_mask = target_pointers == STOP_TOKEN
        delta_outputs = self.delta_head(decoder_out)

        # Compute delta targets
        delta_targets = self.delta_head.compute_targets(
            target_pointers, self.compression_ratio
        )

        # Delta loss (small adjustments)
        delta_loss = F.cross_entropy(
            delta_outputs['delta_logits'].reshape(-1, self.delta_head.n_delta_classes),
            delta_targets['delta_targets'].reshape(-1),
            reduction='none',
            label_smoothing=self.label_smoothing,
        ).view(batch_size, tgt_len)
        delta_loss = (delta_loss * (~stop_mask).float()).sum() / (~stop_mask).sum().clamp(min=1)

        # Jump loss (large jumps)
        use_jump_mask = delta_targets['use_jump_targets'] & ~stop_mask
        if use_jump_mask.any():
            jump_loss = F.cross_entropy(
                delta_outputs['jump_logits'][use_jump_mask],
                delta_targets['jump_targets'][use_jump_mask],
                label_smoothing=self.label_smoothing,
            )
        else:
            jump_loss = torch.tensor(0.0, device=device)

        # Use-jump classification loss
        use_jump_loss = F.binary_cross_entropy_with_logits(
            delta_outputs['use_jump_logits'],
            delta_targets['use_jump_targets'].float(),
        )

        # Stop loss
        stop_logits = delta_outputs['stop_logits']
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_mask.float())

        # Combined pointer loss
        pointer_loss = delta_loss + 0.5 * jump_loss + 0.3 * use_jump_loss

        # Length loss
        predicted_length = self.length_predictor(frame_emb, src_len)
        length_scale = 1000.0
        length_loss = F.mse_loss(
            predicted_length / length_scale,
            torch.tensor([tgt_len / length_scale], device=device).float().expand(batch_size)
        )

        # Structure loss (predict cut points)
        structure_preds = self.structure_head(frame_emb)
        pointer_diff = torch.diff(target_pointers, dim=1)
        cut_labels = (pointer_diff.abs() > 50).float()
        cut_labels = F.pad(cut_labels, (0, 1), value=0)
        min_len = min(tgt_len, structure_preds['cut_logits'].shape[1], cut_labels.shape[1])
        if min_len > 0:
            structure_loss = F.binary_cross_entropy_with_logits(
                structure_preds['cut_logits'][:, :min_len],
                cut_labels[:, :min_len],
            )
        else:
            structure_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = (
            pointer_loss +
            0.5 * stop_loss +
            0.1 * kl_loss +
            0.01 * length_loss +
            0.1 * structure_loss
        )

        return {
            'delta_logits': delta_outputs['delta_logits'],
            'jump_logits': delta_outputs['jump_logits'],
            'use_jump_logits': delta_outputs['use_jump_logits'],
            'stop_logits': stop_logits,
            'delta_loss': delta_loss,
            'jump_loss': jump_loss,
            'use_jump_loss': use_jump_loss,
            'pointer_loss': pointer_loss,
            'stop_loss': stop_loss,
            'kl_loss': kl_loss,
            'length_loss': length_loss,
            'structure_loss': structure_loss,
            'loss': total_loss,
            'predicted_length': predicted_length,
            'delta_targets': delta_targets,
        }

    @torch.no_grad()
    def generate(
        self,
        raw_mel: torch.Tensor,
        target_length: Optional[int] = None,
        temperature: float = 1.0,
        sample_style: bool = True,
        n_samples: int = 1,
        stop_threshold: float = 0.5,
        stems: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate pointer sequence using V2 architecture with delta prediction.

        The generation process:
        1. Encode raw mel with LinearAttentionEncoder -> frame embeddings + global tokens
        2. Optionally fuse stem embeddings
        3. Apply VAE style conditioning and duration conditioning
        4. Run position-aware windowed cross-attention (attends to expected positions)
        5. Decoder layers refine the representation
        6. Delta head predicts offset from expected position for each output frame
        7. Decode predictions to absolute pointer positions

        Args:
            raw_mel: Input mel spectrogram (batch, n_mels, time)
            target_length: Target output length (None = use length predictor)
            temperature: Sampling temperature (0 = greedy)
            sample_style: Whether to sample from VAE latent space
            n_samples: Number of samples to generate
            stop_threshold: Threshold for stop prediction
            stems: Optional stem mel spectrograms

        Returns:
            Dict with 'pointers' (batch, tgt_len) and 'predicted_length'
        """
        batch_size = raw_mel.shape[0]
        device = raw_mel.device

        # Encode stems if provided and enabled
        stem_emb = None
        if self.use_stems and stems is not None:
            # stems can be a dict or a tensor (n_stems, n_mels, time) or (batch, n_stems, n_mels, time)
            if isinstance(stems, dict):
                stem_emb = self.stem_encoder(stems)
            else:
                # Convert tensor to dict format
                stem_names = ['drums', 'bass', 'vocals', 'other']
                if stems.dim() == 3:
                    # (n_stems, n_mels, time) -> add batch dim
                    stems = stems.unsqueeze(0)
                # (batch, n_stems, n_mels, time)
                stems_dict = {name: stems[:, i] for i, name in enumerate(stem_names)}
                stem_emb = self.stem_encoder(stems_dict)

        # Encode with linear attention encoder (O(n) complexity)
        encoded = self.encoder(raw_mel)
        frame_emb = encoded['frame']  # (batch, src_len, d_model)
        global_tokens = encoded['global']  # (batch, n_global, d_model)
        src_len = frame_emb.shape[1]

        # Fuse stem embedding with frame embedding
        if self.use_stems and stem_emb is not None:
            if stem_emb.shape[1] != frame_emb.shape[1]:
                stem_emb = F.interpolate(
                    stem_emb.transpose(1, 2), size=frame_emb.shape[1], mode='linear', align_corners=False
                ).transpose(1, 2)
            frame_emb = frame_emb + stem_emb

        # Predict target length if not provided
        if target_length is None:
            target_length = int(self.length_predictor(frame_emb, src_len).mean().item())
            target_length = max(1, min(target_length, self.max_length))

        all_pointers = []

        for _ in range(n_samples):
            # Style conditioning from VAE
            style_cond, _, _ = self.style_vae(frame_emb, sample=sample_style)

            # Duration conditioning
            cond_frame_emb = self.duration_cond(frame_emb, target_length)
            cond_frame_emb = cond_frame_emb + style_cond.unsqueeze(1)

            # Decoder queries (learned embeddings for each output position)
            positions = torch.arange(target_length, device=device).unsqueeze(0).expand(batch_size, -1)
            queries = self.query_embed(positions.clamp(0, self.stop_embed_idx - 1))

            # Position-aware windowed cross-attention
            # Attends to window around expected position + global tokens
            decoder_out = self.windowed_attn(
                queries, cond_frame_emb, global_tokens,
                self.compression_ratio
            )

            # Decoder transformer layers
            for layer in self.decoder_layers:
                decoder_out = layer(decoder_out, cond_frame_emb)
            decoder_out = self.decoder_norm(decoder_out)

            # Delta prediction
            delta_outputs = self.delta_head(decoder_out)

            # Decode predictions to absolute pointer positions
            pointers = self.delta_head.decode_predictions(
                delta_outputs['delta_logits'],
                delta_outputs['jump_logits'],
                delta_outputs['use_jump_logits'],
                self.compression_ratio,
                src_len,
                temperature=temperature,
            )

            # Apply stop tokens
            stop_logits = delta_outputs['stop_logits']
            stops = torch.sigmoid(stop_logits) > stop_threshold

            for b in range(batch_size):
                stop_pos = stops[b].nonzero(as_tuple=True)[0]
                if len(stop_pos) > 0:
                    pointers[b, stop_pos[0]:] = STOP_TOKEN

            all_pointers.append(pointers)

        if n_samples == 1:
            return {'pointers': all_pointers[0], 'predicted_length': target_length}
        return {'pointers': torch.stack(all_pointers, dim=1), 'predicted_length': target_length}

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda') -> 'PointerNetwork':
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, **kwargs):
        """Save model checkpoint with V2 architecture config."""
        checkpoint = {
            'model_config': {
                'n_mels': self.n_mels,
                'd_model': self.d_model,
                'max_length': self.max_length,
                'frames_per_beat': self.frames_per_beat,
                'beats_per_bar': self.beats_per_bar,
                'use_pre_norm': self.use_pre_norm,
                'use_stems': self.use_stems,
                'n_stems': self.n_stems,
                'dim_feedforward': self.dim_feedforward,
                'label_smoothing': self.label_smoothing,
                # V2 full-sequence parameters
                'compression_ratio': self.compression_ratio,
                'attn_window_size': self.attn_window_size,
                'max_delta': self.max_delta,
            },
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'architecture_version': 'v2',  # Mark as V2 for compatibility checking
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

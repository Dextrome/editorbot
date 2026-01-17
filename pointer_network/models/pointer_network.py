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
        bar_out, _ = self.bar_attn(query, bar_keys, bar_keys)
        beat_out, _ = self.beat_attn(query, beat_keys, beat_keys)
        frame_out, frame_weights = self.frame_attn(query, frame_keys, frame_keys)

        combined = torch.cat([bar_out, beat_out, frame_out], dim=-1)
        gate = self.gate(combined)
        output = self.combine(combined) * gate + frame_out * (1 - gate)

        return output, frame_weights


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
    ):
        super().__init__()
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.frames_per_bar = frames_per_beat * beats_per_bar
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        self.mel_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = MusicAwarePositionalEncoding(d_model, dropout=dropout)

        self.frame_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, dropout, 'gelu', batch_first=True
            ),
            num_layers=n_layers
        )

        self.beat_pool = nn.Conv1d(d_model, d_model, frames_per_beat, stride=frames_per_beat)
        self.beat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, 'gelu', batch_first=True),
            num_layers=2
        )

        self.bar_pool = nn.Conv1d(d_model, d_model, beats_per_bar, stride=beats_per_bar)
        self.bar_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, 'gelu', batch_first=True),
            num_layers=2
        )

    def _frame_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frame encoder forward (for checkpointing)."""
        return self.frame_encoder(x)

    def _beat_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Beat encoder forward (for checkpointing)."""
        return self.beat_encoder(x)

    def _bar_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Bar encoder forward (for checkpointing)."""
        return self.bar_encoder(x)

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
            frame_emb = self.frame_encoder(x)

        # Beat level
        if time >= self.frames_per_beat:
            beat_emb = self.beat_pool(frame_emb.transpose(1, 2)).transpose(1, 2)
            if self.use_checkpoint and self.training:
                beat_emb = checkpoint(self._beat_encode, beat_emb, use_reentrant=False)
            else:
                beat_emb = self.beat_encoder(beat_emb)
        else:
            beat_emb = frame_emb.mean(dim=1, keepdim=True)

        # Bar level
        n_beats = beat_emb.shape[1]
        if n_beats >= self.beats_per_bar:
            bar_emb = self.bar_pool(beat_emb.transpose(1, 2)).transpose(1, 2)
            if self.use_checkpoint and self.training:
                bar_emb = checkpoint(self._bar_encode, bar_emb, use_reentrant=False)
            else:
                bar_emb = self.bar_encoder(bar_emb)
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
        # Clamp logvar to prevent numerical instability
        logvar = logvar.clamp(-10, 10)
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


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
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.frames_per_bar = frames_per_beat * beats_per_bar
        self.use_checkpoint = use_checkpoint

        # Precomputed constants
        self.scale = math.sqrt(d_model)  # Attention temperature scale

        # Cache for position indices
        self._position_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Encoder
        self.encoder = MultiScaleEncoder(
            n_mels, d_model, n_heads, n_encoder_layers,
            frames_per_beat, beats_per_bar, dropout, use_checkpoint
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

        # Attention layers
        self.hierarchical_attn = HierarchicalAttention(
            d_model, n_heads, frames_per_beat, beats_per_bar, dropout
        )
        self.sparse_attn = SparseAttention(d_model, n_heads, dropout=dropout)

        # Decoder transformer layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, n_heads, d_model * 4, dropout, 'gelu', batch_first=True
            )
            for _ in range(n_decoder_layers)
        ])

        # Output heads - hierarchical pointers (bar -> beat -> frame)
        self.pointer_proj = nn.Linear(d_model, d_model)  # Frame-level pointer
        self.bar_pointer_proj = nn.Linear(d_model, d_model)  # Bar-level pointer
        self.beat_pointer_proj = nn.Linear(d_model, d_model)  # Beat-level pointer
        self.stop_head = nn.Linear(d_model, 1)

        # Auxiliary task
        self.structure_head = StructurePredictionHead(d_model)

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
        target_length: Optional[int] = None,
        use_vae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            raw_mel: (batch, n_mels, time)
            target_pointers: (batch, tgt_len) with STOP_TOKEN (-2) for end positions
        """
        batch_size = raw_mel.shape[0]
        device = raw_mel.device

        # Encode
        encoded = self.encoder(raw_mel)
        frame_emb = encoded['frame']
        beat_emb = encoded['beat']
        bar_emb = encoded['bar']
        src_len = frame_emb.shape[1]

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

        # Decoder queries (use cached positions)
        positions = self._get_positions(tgt_len, device).unsqueeze(0).expand(batch_size, -1)
        queries = self.query_embed(positions.clamp(0, self.stop_embed_idx - 1))

        # Hierarchical + sparse attention
        hier_out, _ = self.hierarchical_attn(queries, cond_frame_emb, beat_emb, bar_emb)
        sparse_out = self.sparse_attn(hier_out, cond_frame_emb, cond_frame_emb)
        decoder_out = hier_out + sparse_out

        # Decoder layers
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, cond_frame_emb)

        # === Hierarchical pointer logits (bar -> beat -> frame) ===
        # Frame-level pointers (use precomputed scale)
        pointer_queries = self.pointer_proj(decoder_out)
        pointer_logits = torch.matmul(pointer_queries, cond_frame_emb.transpose(1, 2)) / self.scale

        # Bar-level pointers
        bar_queries = self.bar_pointer_proj(decoder_out)
        bar_pointer_logits = torch.matmul(bar_queries, bar_emb.transpose(1, 2)) / self.scale

        # Beat-level pointers
        beat_queries = self.beat_pointer_proj(decoder_out)
        beat_pointer_logits = torch.matmul(beat_queries, beat_emb.transpose(1, 2)) / self.scale

        # Stop logits
        stop_logits = self.stop_head(decoder_out).squeeze(-1)

        # === Compute losses ===
        stop_mask = target_pointers == STOP_TOKEN
        target_clamped = target_pointers.clone()
        target_clamped[stop_mask] = 0
        target_clamped = target_clamped.clamp(0, src_len - 1)

        # Derive hierarchical targets from frame pointers
        n_bars = bar_emb.shape[1]
        n_beats = beat_emb.shape[1]
        target_bar = (target_clamped // self.frames_per_bar).clamp(0, n_bars - 1)
        target_beat = (target_clamped // self.frames_per_beat).clamp(0, n_beats - 1)

        # Frame pointer loss
        pointer_loss = F.cross_entropy(
            pointer_logits.reshape(-1, src_len),
            target_clamped.reshape(-1),
            reduction='none',
        ).view(batch_size, tgt_len)
        pointer_loss = (pointer_loss * (~stop_mask).float()).sum() / (~stop_mask).sum().clamp(min=1)

        # Bar pointer loss
        bar_pointer_loss = F.cross_entropy(
            bar_pointer_logits.reshape(-1, n_bars),
            target_bar.reshape(-1),
            reduction='none',
        ).view(batch_size, tgt_len)
        bar_pointer_loss = (bar_pointer_loss * (~stop_mask).float()).sum() / (~stop_mask).sum().clamp(min=1)

        # Beat pointer loss
        beat_pointer_loss = F.cross_entropy(
            beat_pointer_logits.reshape(-1, n_beats),
            target_beat.reshape(-1),
            reduction='none',
        ).view(batch_size, tgt_len)
        beat_pointer_loss = (beat_pointer_loss * (~stop_mask).float()).sum() / (~stop_mask).sum().clamp(min=1)

        # Stop loss
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_mask.float())

        # Length loss (normalized by scale factor to prevent huge values)
        predicted_length = self.length_predictor(encoded['frame'], src_len)
        length_scale = 1000.0  # Normalize to reasonable range
        length_loss = F.mse_loss(
            predicted_length / length_scale,
            torch.tensor([tgt_len / length_scale], device=device).float().expand(batch_size)
        )

        # Structure loss (handle shape mismatches carefully)
        structure_preds = self.structure_head(frame_emb)
        pointer_diff = torch.diff(target_pointers, dim=1)
        cut_labels = (pointer_diff.abs() > 50).float()
        cut_labels = F.pad(cut_labels, (0, 1), value=0)
        # Use minimum length to avoid indexing errors
        min_len = min(tgt_len, structure_preds['cut_logits'].shape[1], cut_labels.shape[1])
        if min_len > 0:
            structure_loss = F.binary_cross_entropy_with_logits(
                structure_preds['cut_logits'][:, :min_len],
                cut_labels[:, :min_len],
            )
        else:
            structure_loss = torch.tensor(0.0, device=device)

        # Combined hierarchical pointer loss (coarse-to-fine weighting)
        # Bar loss weighted lower since it's easier, frame loss weighted higher for precision
        hierarchical_pointer_loss = (
            0.2 * bar_pointer_loss +
            0.3 * beat_pointer_loss +
            1.0 * pointer_loss
        )

        total_loss = (
            hierarchical_pointer_loss +
            0.5 * stop_loss +
            0.1 * kl_loss +
            0.01 * length_loss +
            0.1 * structure_loss
        )

        return {
            'pointer_logits': pointer_logits,
            'bar_pointer_logits': bar_pointer_logits,
            'beat_pointer_logits': beat_pointer_logits,
            'stop_logits': stop_logits,
            'pointer_loss': pointer_loss,
            'bar_pointer_loss': bar_pointer_loss,
            'beat_pointer_loss': beat_pointer_loss,
            'stop_loss': stop_loss,
            'kl_loss': kl_loss,
            'length_loss': length_loss,
            'structure_loss': structure_loss,
            'loss': total_loss,
            'predicted_length': predicted_length,
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
    ) -> Dict[str, torch.Tensor]:
        """Generate pointer sequence."""
        batch_size = raw_mel.shape[0]
        device = raw_mel.device

        encoded = self.encoder(raw_mel)
        frame_emb = encoded['frame']
        beat_emb = encoded['beat']
        bar_emb = encoded['bar']
        src_len = frame_emb.shape[1]

        if target_length is None:
            target_length = int(self.length_predictor(frame_emb, src_len).mean().item())
            target_length = max(1, min(target_length, self.max_length))

        all_pointers = []

        for _ in range(n_samples):
            style_cond, _, _ = self.style_vae(frame_emb, sample=sample_style)
            cond_frame_emb = self.duration_cond(frame_emb, target_length)
            cond_frame_emb = cond_frame_emb + style_cond.unsqueeze(1)

            positions = torch.arange(target_length, device=device).unsqueeze(0).expand(batch_size, -1)
            queries = self.query_embed(positions.clamp(0, self.stop_embed_idx - 1))

            hier_out, _ = self.hierarchical_attn(queries, cond_frame_emb, beat_emb, bar_emb)
            sparse_out = self.sparse_attn(hier_out, cond_frame_emb, cond_frame_emb)
            decoder_out = hier_out + sparse_out

            for layer in self.decoder_layers:
                decoder_out = layer(decoder_out, cond_frame_emb)

            pointer_queries = self.pointer_proj(decoder_out)
            pointer_logits = torch.matmul(pointer_queries, cond_frame_emb.transpose(1, 2))

            if temperature > 0:
                probs = F.softmax(pointer_logits / temperature, dim=-1)
                pointers = torch.multinomial(probs.view(-1, src_len), 1).view(batch_size, target_length)
            else:
                pointers = pointer_logits.argmax(dim=-1)

            stop_logits = self.stop_head(decoder_out).squeeze(-1)
            stops = torch.sigmoid(stop_logits) > stop_threshold

            for b in range(batch_size):
                stop_pos = stops[b].nonzero(as_tuple=True)[0]
                if len(stop_pos) > 0:
                    pointers[b, stop_pos[0]:] = STOP_TOKEN

            all_pointers.append(pointers)

        if n_samples == 1:
            return {'pointers': all_pointers[0], 'predicted_length': target_length}
        return {'pointers': torch.stack(all_pointers, dim=1), 'predicted_length': target_length}

    @torch.no_grad()
    def generate_hierarchical(
        self,
        raw_mel: torch.Tensor,
        target_length: Optional[int] = None,
        temperature: float = 1.0,
        sample_style: bool = True,
        stop_threshold: float = 0.5,
        coarse_to_fine: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Generate pointer sequence using coarse-to-fine hierarchical sampling.

        Instead of directly predicting frame pointers, first sample bar,
        then sample beat within bar context, then sample frame within beat context.
        This provides more structured outputs that respect musical boundaries.
        """
        batch_size = raw_mel.shape[0]
        device = raw_mel.device

        encoded = self.encoder(raw_mel)
        frame_emb = encoded['frame']
        beat_emb = encoded['beat']
        bar_emb = encoded['bar']
        src_len = frame_emb.shape[1]
        n_beats = beat_emb.shape[1]
        n_bars = bar_emb.shape[1]

        if target_length is None:
            target_length = int(self.length_predictor(frame_emb, src_len).mean().item())
            target_length = max(1, min(target_length, self.max_length))

        style_cond, _, _ = self.style_vae(frame_emb, sample=sample_style)
        cond_frame_emb = self.duration_cond(frame_emb, target_length)
        cond_frame_emb = cond_frame_emb + style_cond.unsqueeze(1)

        positions = torch.arange(target_length, device=device).unsqueeze(0).expand(batch_size, -1)
        queries = self.query_embed(positions.clamp(0, self.stop_embed_idx - 1))

        hier_out, _ = self.hierarchical_attn(queries, cond_frame_emb, beat_emb, bar_emb)
        sparse_out = self.sparse_attn(hier_out, cond_frame_emb, cond_frame_emb)
        decoder_out = hier_out + sparse_out

        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, cond_frame_emb)

        if coarse_to_fine:
            # Step 1: Sample bar indices
            bar_queries = self.bar_pointer_proj(decoder_out)
            bar_logits = torch.matmul(bar_queries, bar_emb.transpose(1, 2))
            if temperature > 0:
                bar_probs = F.softmax(bar_logits / temperature, dim=-1)
                bar_indices = torch.multinomial(bar_probs.view(-1, n_bars), 1).view(batch_size, target_length)
            else:
                bar_indices = bar_logits.argmax(dim=-1)

            # Step 2: Sample beat indices (constrained to selected bar)
            beat_queries = self.beat_pointer_proj(decoder_out)
            beat_logits = torch.matmul(beat_queries, beat_emb.transpose(1, 2))

            # Mask beats outside selected bar
            beat_mask = torch.zeros(batch_size, target_length, n_beats, device=device)
            for b in range(batch_size):
                for t in range(target_length):
                    bar_idx = bar_indices[b, t].item()
                    beat_start = bar_idx * self.beats_per_bar
                    beat_end = min((bar_idx + 1) * self.beats_per_bar, n_beats)
                    beat_mask[b, t, beat_start:beat_end] = 1.0

            beat_logits = beat_logits.masked_fill(beat_mask == 0, float('-inf'))
            if temperature > 0:
                beat_probs = F.softmax(beat_logits / temperature, dim=-1)
                # Handle potential all-inf case
                beat_probs = beat_probs.nan_to_num(0)
                beat_probs = beat_probs / beat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                beat_indices = torch.multinomial(beat_probs.view(-1, n_beats), 1).view(batch_size, target_length)
            else:
                beat_indices = beat_logits.argmax(dim=-1)

            # Step 3: Sample frame indices (constrained to selected beat)
            pointer_queries = self.pointer_proj(decoder_out)
            frame_logits = torch.matmul(pointer_queries, cond_frame_emb.transpose(1, 2))

            # Mask frames outside selected beat
            frame_mask = torch.zeros(batch_size, target_length, src_len, device=device)
            for b in range(batch_size):
                for t in range(target_length):
                    beat_idx = beat_indices[b, t].item()
                    frame_start = beat_idx * self.frames_per_beat
                    frame_end = min((beat_idx + 1) * self.frames_per_beat, src_len)
                    frame_mask[b, t, frame_start:frame_end] = 1.0

            frame_logits = frame_logits.masked_fill(frame_mask == 0, float('-inf'))
            if temperature > 0:
                frame_probs = F.softmax(frame_logits / temperature, dim=-1)
                frame_probs = frame_probs.nan_to_num(0)
                frame_probs = frame_probs / frame_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                pointers = torch.multinomial(frame_probs.view(-1, src_len), 1).view(batch_size, target_length)
            else:
                pointers = frame_logits.argmax(dim=-1)
        else:
            # Standard frame-level generation (no hierarchy)
            pointer_queries = self.pointer_proj(decoder_out)
            pointer_logits = torch.matmul(pointer_queries, cond_frame_emb.transpose(1, 2))
            if temperature > 0:
                probs = F.softmax(pointer_logits / temperature, dim=-1)
                pointers = torch.multinomial(probs.view(-1, src_len), 1).view(batch_size, target_length)
            else:
                pointers = pointer_logits.argmax(dim=-1)
            bar_indices = pointers // self.frames_per_bar
            beat_indices = pointers // self.frames_per_beat

        # Apply stop tokens
        stop_logits = self.stop_head(decoder_out).squeeze(-1)
        stops = torch.sigmoid(stop_logits) > stop_threshold
        for b in range(batch_size):
            stop_pos = stops[b].nonzero(as_tuple=True)[0]
            if len(stop_pos) > 0:
                pointers[b, stop_pos[0]:] = STOP_TOKEN

        return {
            'pointers': pointers,
            'bar_indices': bar_indices,
            'beat_indices': beat_indices,
            'predicted_length': target_length,
        }

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda') -> 'PointerNetwork':
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, **kwargs):
        checkpoint = {
            'model_config': {'d_model': self.d_model, 'max_length': self.max_length},
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

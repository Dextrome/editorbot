"""Pointer decoder for generating frame index sequences."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PointerAttention(nn.Module):
    """Attention mechanism that produces pointer distributions.

    Given a query and a set of keys, produces a probability distribution
    over the keys (pointer distribution).
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, tgt_len, d_model) decoder states
            keys: (batch, src_len, d_model) encoder outputs (raw audio)
            key_padding_mask: (batch, src_len) True = masked

        Returns:
            pointer_logits: (batch, tgt_len, src_len) unnormalized pointer scores
        """
        batch_size, tgt_len, _ = query.shape
        src_len = keys.shape[1]

        # Project to multi-head
        q = self.q_proj(query).view(batch_size, tgt_len, self.n_heads, self.head_dim)
        k = self.k_proj(keys).view(batch_size, src_len, self.n_heads, self.head_dim)

        # (batch, n_heads, tgt_len, head_dim) @ (batch, n_heads, head_dim, src_len)
        q = q.transpose(1, 2)  # (batch, n_heads, tgt_len, head_dim)
        k = k.transpose(1, 2)  # (batch, n_heads, src_len, head_dim)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn: (batch, n_heads, tgt_len, src_len)

        # Average over heads for pointer distribution
        pointer_logits = attn.mean(dim=1)  # (batch, tgt_len, src_len)

        # Apply padding mask
        if key_padding_mask is not None:
            pointer_logits = pointer_logits.masked_fill(
                key_padding_mask.unsqueeze(1), float('-inf')
            )

        return pointer_logits


class PointerDecoder(nn.Module):
    """Autoregressive decoder that outputs pointer indices.

    At each step:
    1. Embed the previous pointer (index into raw audio)
    2. Self-attend over decoder history
    3. Cross-attend to raw audio embeddings
    4. Output pointer distribution over raw audio frames
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_raw_frames: int = 65536,
        max_output_len: int = 65536,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_raw_frames = max_raw_frames

        # Pointer embedding: embed the index of the previous pointer
        # We use a learned embedding + relative position info
        self.pointer_embed = nn.Embedding(max_raw_frames + 2, d_model)  # +2 for BOS, PAD
        self.bos_token = max_raw_frames
        self.pad_token = max_raw_frames + 1

        # Positional encoding for decoder positions
        self.pos_embed = nn.Embedding(max_output_len, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Pointer attention for output
        self.pointer_attn = PointerAttention(d_model, n_heads, dropout)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pointers: torch.Tensor,
        raw_embeddings: torch.Tensor,
        raw_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pointers: (batch, tgt_len) previous pointer indices (shifted right)
            raw_embeddings: (batch, src_len, d_model) encoded raw audio
            raw_padding_mask: (batch, src_len) True = masked position in raw
            tgt_padding_mask: (batch, tgt_len) True = masked position in target

        Returns:
            pointer_logits: (batch, tgt_len, src_len) logits over raw frames
        """
        batch_size, tgt_len = pointers.shape
        device = pointers.device

        # Clamp pointers to valid range (for embedding lookup)
        pointers_clamped = pointers.clamp(0, self.max_raw_frames + 1)

        # Embed pointers
        tgt = self.pointer_embed(pointers_clamped)  # (batch, tgt_len, d_model)

        # Add positional encoding
        positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt = tgt + self.pos_embed(positions)

        # Create causal mask for decoder self-attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

        # Transformer decoder
        decoder_out = self.transformer_decoder(
            tgt,
            raw_embeddings,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=raw_padding_mask,
        )

        decoder_out = self.norm(decoder_out)

        # Pointer attention
        pointer_logits = self.pointer_attn(decoder_out, raw_embeddings, raw_padding_mask)

        return pointer_logits

    def generate(
        self,
        raw_embeddings: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        raw_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively generate pointer sequence.

        Args:
            raw_embeddings: (batch, src_len, d_model) encoded raw audio
            max_length: maximum output length
            temperature: sampling temperature (1.0 = normal, <1 = sharper)
            raw_padding_mask: (batch, src_len) True = masked

        Returns:
            pointers: (batch, max_length) generated pointer indices
        """
        batch_size = raw_embeddings.shape[0]
        device = raw_embeddings.device

        # Start with BOS token
        pointers = torch.full(
            (batch_size, 1), self.bos_token, dtype=torch.long, device=device
        )

        for _ in range(max_length - 1):
            # Get logits for next position
            logits = self.forward(pointers, raw_embeddings, raw_padding_mask)
            next_logits = logits[:, -1, :]  # (batch, src_len)

            # Sample from distribution
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_pointer = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to sequence
            pointers = torch.cat([pointers, next_pointer], dim=1)

        return pointers[:, 1:]  # Remove BOS token

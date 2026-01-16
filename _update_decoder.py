
import torch
import torch.nn as nn
from typing import Optional

DECODER_CODE = """
import torch
import torch.nn as nn
from typing import Optional

from ..config import Phase1Config


class MelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_dim, nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.n_decoder_layers, norm=nn.LayerNorm(config.decoder_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_dim, config.decoder_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.decoder_dim, config.decoder_dim // 2), nn.GELU(),
            nn.Linear(config.decoder_dim // 2, config.audio.n_mels))
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.n_edit_labels = getattr(config, 'n_edit_labels', 8)
        self.label_embed_dim = 32
        self.label_embed = nn.Embedding(self.n_edit_labels, self.label_embed_dim)
        self.use_gating = True
        if self.use_gating:
            gate_input_dim = config.decoder_dim + config.audio.n_mels + self.label_embed_dim
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, config.decoder_dim), nn.GELU(),
                nn.Linear(config.decoder_dim, config.audio.n_mels), nn.Sigmoid())

    def forward(self, latent, raw_mel, edit_labels=None, mask=None):
        attn_mask = ~mask if mask is not None else None
        decoded = self.transformer(latent, src_key_padding_mask=attn_mask)
        pred_mel = self.output_proj(decoded)
        if self.use_gating and edit_labels is not None:
            label_feat = self.label_embed(edit_labels)
            gate_input = torch.cat([decoded, raw_mel, label_feat], dim=-1)
            gate = self.gate(gate_input)
            cut_mask = (edit_labels == 0).unsqueeze(-1).float()
            keep_mask = (edit_labels == 1).unsqueeze(-1).float()
            other_mask = 1.0 - cut_mask - keep_mask
            effective_gate = cut_mask * 0.0 + keep_mask * 1.0 + other_mask * gate
            output = pred_mel * (1 - effective_gate) + raw_mel * effective_gate
        elif self.use_gating:
            gate_input = torch.cat([decoded, raw_mel, torch.zeros(decoded.size(0), decoded.size(1), self.label_embed_dim, device=decoded.device)], dim=-1)
            gate = self.gate(gate_input)
            output = pred_mel * (1 - gate) + raw_mel * gate
        else:
            output = pred_mel + self.residual_weight * raw_mel
        return output
"""
print(DECODER_CODE)

import torch
import pytest
from src.ai.natten_wrapper import NATTEN1DAttention

def test_natten1d_attention_forward():
    batch, length, dim, heads = 2, 16, 8, 2
    x = torch.randn(batch, length, dim)
    attn = NATTEN1DAttention(dim=dim, num_heads=heads)
    out = attn(x)
    assert out.shape == (batch, length, dim)
    assert torch.isfinite(out).all()

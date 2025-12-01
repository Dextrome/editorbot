import torch
import pytest
from src.ai.natten_wrapper import NATTEN1DAttention
from natten.backends import get_compatible_backends

def test_natten1d_attention_forward():
    batch, length, dim, heads = 2, 16, 8, 2
    x = torch.randn(batch, length, dim)
    # If NATTEN cannot pick a compatible backend in this environment, skip the test.
    if not get_compatible_backends(torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8), False):
        pytest.skip("No NATTEN-compatible backend found for test environment; skipping natten-based test")
    attn = NATTEN1DAttention(dim=dim, num_heads=heads)
    out = attn(x)
    assert out.shape == (batch, length, dim)
    assert torch.isfinite(out).all()

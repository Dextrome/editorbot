
import torch
import numpy as np
import pytest
import os
from src.audio import AudioProcessor
from src.ai.natten_wrapper import NATTEN1DAttention

@pytest.mark.parametrize("audio_path", ["data/samples/wartsnall3xx.wav"])
def test_natten1d_attention_on_audio(audio_path):
    os.environ["NATTEN_LOG_LEVEL"] = "DEBUG"
    # Load audio (mono)
    processor = AudioProcessor(sample_rate=16000)
    audio, sr = processor.load(audio_path)
    audio = processor.normalize(audio)
    # Convert to torch tensor, add batch and feature dims
    x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1)  # (1, seqlen, 1)
    # Project to higher dim for NATTEN (e.g., 8)
    seqlen = x.shape[1]
    dim = 8
    proj = torch.nn.Linear(1, dim)
    x_proj = proj(x)
    # Pad sequence to multiple of 8 for kernel_size=7
    pad = (8 - seqlen % 8) % 8
    if pad > 0:
        x_proj = torch.nn.functional.pad(x_proj, (0, 0, 0, pad))
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_proj = x_proj.to(device)
    attn = NATTEN1DAttention(dim=dim, num_heads=1, kernel_size=7).to(device)
    print(f"x_proj shape: {x_proj.shape}, dtype: {x_proj.dtype}, device: {x_proj.device}")
    print(f"kernel_size: {attn.kernel_size}, num_heads: {attn.num_heads}, head_dim: {attn.head_dim}")
    print(f"is contiguous: {x_proj.is_contiguous()}")
    out = attn(x_proj)
    assert out.shape == x_proj.shape
    assert torch.isfinite(out).all()
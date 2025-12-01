import torch
import torch.nn as nn
from typing import Optional


class NattenFrameEncoder(nn.Module):
    """Module that projects framed audio to a low-dim embedding and runs NATTEN.

    Forward input: tensor of shape (1, n_frames, frame_size)
    Output: tensor of shape (1, n_frames, proj_dim)
    """

    def __init__(self, frame_size: int, proj_dim: int = 8, kernel_size: int = 7, num_heads: int = 1):
        super().__init__()
        self.frame_size = frame_size
        self.proj_dim = proj_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads

        self.proj = nn.Linear(frame_size, proj_dim)
        # Import natten wrapper lazily to avoid import cycles at package import time
        from .natten_wrapper import NATTEN1DAttention

        self.attn = NATTEN1DAttention(dim=proj_dim, num_heads=num_heads, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, n_frames, frame_size)
        y = self.proj(x)  # (1, n_frames, proj_dim)
        # NATTEN expects (batch, seqlen, num_heads, head_dim)
        # Our wrapper handles reshaping internally, so pass (batch, seqlen, proj_dim)
        y = self.attn(y)
        return y

    def state_dict_for_save(self):
        return self.state_dict()

    def load_state_dict_from(self, state_dict):
        self.load_state_dict(state_dict)

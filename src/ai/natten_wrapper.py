
import torch
import natten
import torch.nn as nn

class NATTEN1DAttention(nn.Module):
    """
    Wrapper for NATTEN 1D attention using natten.na1d function.
    Input: (batch, seqlen, dim)
    Output: (batch, seqlen, dim)
    """
    def __init__(self, dim, num_heads=1, kernel_size=7, dilation=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

    def forward(self, x):
        # x: (batch, seqlen, dim)
        B, L, D = x.shape
        H = self.num_heads
        x = x.view(B, L, H, self.head_dim)
        # Use x as query, key, value (self-attention)
        out = natten.na1d(x, x, x, kernel_size=self.kernel_size, dilation=self.dilation)
        out = out.view(B, L, D)
        return out

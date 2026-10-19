import torch
from torch import nn, einsum
import torch.nn.functional as F

# MPS has no memory-efficient attention kernel: the full (batch, heads, n, n) score matrix is
# materialised (2.4 GB for an 11 s Mel-Band Roformer chunk), so attention runs in batch slices whose
# score matrix stays under this size.
MPS_ATTENTION_SLICE_BYTES = 512 * 2**20


class Attend(nn.Module):
    """Scaled dot-product attention. Uses torch's fused kernel when `flash` is set."""

    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash and hasattr(F, 'scaled_dot_product_attention')

    def flash_attn(self, q, k, v):
        if self.scale is not None:
            q = q * (self.scale / q.shape[-1] ** -0.5)
        dropout_p = self.dropout if self.training else 0.

        batch, heads, q_len, k_len = q.shape[0], q.shape[1], q.shape[-2], k.shape[-2]
        slice_size = max(MPS_ATTENTION_SLICE_BYTES // (heads * q_len * k_len * q.element_size()), 1)
        if q.device.type != 'mps' or slice_size >= batch:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        return torch.cat([F.scaled_dot_product_attention(q[i:i + slice_size], k[i:i + slice_size], v[i:i + slice_size],
                                                         dropout_p=dropout_p)
                          for i in range(0, batch, slice_size)])

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum('b h i j, b h j d -> b h i d', attn, v)

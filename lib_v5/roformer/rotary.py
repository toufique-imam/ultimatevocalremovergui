"""Minimal rotary position embedding, numerically matching lucidrains' rotary-embedding-torch
(freqs_for='lang', no xpos/interpolation) which Roformer checkpoints were trained with.
The `freqs` parameter name is kept so checkpoints load with strict=True."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self._cache = None

    def _get_cos_sin(self, seq_len, device, dtype):
        cache = self._cache
        if cache is not None and cache[0].shape[0] == seq_len and cache[0].device == device and cache[0].dtype == dtype:
            return cache
        seq = torch.arange(seq_len, device=device, dtype=self.freqs.dtype)
        angles = torch.einsum('n, f -> n f', seq, self.freqs.to(device))
        self._cache = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self._cache

    def rotate_queries_or_keys(self, t):
        """Rotates `t` of shape (..., seq, dim) along the sequence axis (-2).
        Feature pairs (2i, 2i + 1) are rotated by the angle of frequency i, as in rotary-embedding-torch."""
        cos, sin = self._get_cos_sin(t.shape[-2], t.device, t.dtype)
        rot_dim = cos.shape[-1] * 2
        t_rot = t[..., :rot_dim].unflatten(-1, (-1, 2))
        x1, x2 = t_rot[..., 0], t_rot[..., 1]
        out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
        if rot_dim < t.shape[-1]:
            out = torch.cat((out, t[..., rot_dim:]), dim=-1)
        return out

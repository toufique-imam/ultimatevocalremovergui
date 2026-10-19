"""Building blocks shared by BS-Roformer and Mel-Band-Roformer.

Adapted from ZFTurbo/Music-Source-Separation-Training (MIT) and lucidrains/BS-RoFormer (MIT),
inference only. Parameter names match theirs so published checkpoints load unchanged."""

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from .attend import Attend


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rotary_embed=None, flash=True):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    """Linear attention from https://arxiv.org/abs/2106.09681 (El-Nouby et al.)."""

    def __init__(self, *, dim, dim_head=32, heads=8, scale=8, flash=False, dropout=0.):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.attend = Attend(scale=scale, dropout=dropout, flash=flash)

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x)
        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()
        out = self.attend(q, k, v)
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4,
                 norm_output=True, rotary_embed=None, flash_attn=True, linear_attn=False):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                 rotary_embed=rotary_embed, flash=flash_attn)

            self.layers.append(ModuleList([attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            self.to_features.append(nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)))

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)
        outs = [to_feature(split_input) for split_input, to_feature in zip(x, self.to_features)]
        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, num_hidden=1, activation=nn.Tanh):
    """`num_hidden` hidden layers. BS-Roformer uses depth - 1, Mel-Band-Roformer uses depth."""
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * num_hidden), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if not is_last:
            net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs, num_hidden, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            self.to_freqs.append(nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, num_hidden=num_hidden),
                nn.GLU(dim=-1)
            ))

    def forward(self, x):
        x = x.unbind(dim=-2)
        return torch.cat([mlp(band_features) for band_features, mlp in zip(x, self.to_freqs)], dim=-1)


def stft(x, window, **kwargs):
    """torch.stft returning a real (..., f, t, 2) tensor; falls back to CPU where the device lacks support (MPS)."""
    try:
        spec = torch.stft(x, window=window, return_complex=True, **kwargs)
        return torch.view_as_real(spec)
    except (RuntimeError, NotImplementedError):
        spec = torch.stft(x.cpu(), window=window.cpu(), return_complex=True, **kwargs)
        return torch.view_as_real(spec).to(x.device)


def istft(spec_real, window, length=None, **kwargs):
    """Inverse of `stft` taking a real (..., f, t, 2) tensor."""
    device = spec_real.device
    try:
        return torch.istft(torch.view_as_complex(spec_real.contiguous()), window=window, length=length, **kwargs)
    except (RuntimeError, NotImplementedError):
        spec = torch.view_as_complex(spec_real.cpu().contiguous())
        return torch.istft(spec, window=window.cpu(), length=length, **kwargs).to(device)


def complex_mul(a, b):
    """Complex product of real tensors whose last dim holds (real, imag). Avoids complex dtypes, which MPS lacks."""
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack((ar * br - ai * bi, ar * bi + ai * br), dim=-1)


def build_axial_layers(*, dim, depth, time_transformer_depth, freq_transformer_depth, linear_transformer_depth,
                       transformer_kwargs, dim_head, rotary_embed_cls):
    time_rotary_embed = rotary_embed_cls(dim=dim_head)
    freq_rotary_embed = rotary_embed_cls(dim=dim_head)

    layers = ModuleList([])
    for _ in range(depth):
        tran_modules = []
        if linear_transformer_depth > 0:
            tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
        tran_modules.append(Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs))
        tran_modules.append(Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs))
        layers.append(ModuleList(tran_modules))
    return layers


def run_axial_layers(layers, x, skip_connection=False):
    """Hierarchical time / frequency attention over x of shape (b, t, f, d)."""
    store = []

    for transformer_block in layers:
        if len(transformer_block) == 3:
            linear_transformer, time_transformer, freq_transformer = transformer_block
            x, ft_ps = pack([x], 'b * d')
            x = linear_transformer(x)
            x, = unpack(x, ft_ps, 'b * d')
        else:
            time_transformer, freq_transformer = transformer_block

        if skip_connection:
            for prev in store:
                x = x + prev

        x = rearrange(x, 'b t f d -> b f t d')
        x, ps = pack([x], '* t d')
        x = time_transformer(x)
        x, = unpack(x, ps, '* t d')

        x = rearrange(x, 'b f t d -> b t f d')
        x, ps = pack([x], '* f d')
        x = freq_transformer(x)
        x, = unpack(x, ps, '* f d')

        if skip_connection:
            store.append(x)

    return x

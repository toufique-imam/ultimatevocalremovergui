"""HTDemucs network without its STFT/iSTFT, for ONNX export.

forward(mix [B,2,TL], mag [B,4,F,T]) -> (spec_out [B,S,4,F,T], wave_out [B,S,2,TL])
where mag is HTDemucs._magnitude(HTDemucs._spec(mix)) (CaC layout: ch0 re, ch0 im, ch1 re, ch1 im).
The caller does _mask/_ispec and adds wave_out, exactly as HTDemucs.forward does.
"""
import math

import torch
from torch import nn
from einops import rearrange


class ChunkedAttention(nn.Module):
    """nn.MultiheadAttention (eval, no masks) computed over blocks of `chunk` queries.

    Softmax is per query row, so every row is computed exactly as before, but
    the [heads, Lq, Lk] score matrix never exists whole: HTDemucs's 2688-token
    self-attention otherwise needs 220 MB per score tensor, which pushed ONNX
    Runtime past 4 GB on phones.
    """

    def __init__(self, mha: nn.MultiheadAttention, chunk: int):
        super().__init__()
        assert mha._qkv_same_embed_dim and mha.in_proj_bias is not None
        self.e, self.h, self.batch_first, self.chunk = mha.embed_dim, mha.num_heads, mha.batch_first, chunk
        self.in_proj_weight, self.in_proj_bias, self.out_proj = mha.in_proj_weight, mha.in_proj_bias, mha.out_proj

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, **_):
        assert attn_mask is None and key_padding_mask is None
        if not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        e, h = self.e, self.h
        w, b = self.in_proj_weight, self.in_proj_bias
        q = torch.nn.functional.linear(query, w[:e], b[:e])
        k = torch.nn.functional.linear(key, w[e:2 * e], b[e:2 * e])
        v = torch.nn.functional.linear(value, w[2 * e:], b[2 * e:])
        bsz, lq, _ = q.shape
        d = e // h
        q = q.view(bsz, lq, h, d).transpose(1, 2)
        k = k.view(bsz, -1, h, d).transpose(1, 2)
        v = v.view(bsz, -1, h, d).transpose(1, 2)
        kt = k.transpose(-2, -1)
        scale = 1.0 / math.sqrt(d)
        outs = []
        for i in range(0, lq, self.chunk):
            s = torch.matmul(q[:, :, i:i + self.chunk], kt) * scale
            outs.append(torch.matmul(torch.softmax(s, dim=-1), v))
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(bsz, lq, e)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


def chunk_attention(model: nn.Module, chunk: int = 672) -> nn.Module:
    """Replace every nn.MultiheadAttention in `model` with ChunkedAttention (in place)."""
    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            setattr(model, name, ChunkedAttention(child, chunk))
        else:
            chunk_attention(child, chunk)
    return model


class HTDemucsCore(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def export_demucs(name, repo, out_dir):
    """Export Demucs v4 model `name` (a BagOfModels from `repo`) as <name>_<i>.onnx cores plus a
    <name>.demucs manifest for the Android app. Returns [(file, max |onnx − torch| / peak)]."""
    import json, os
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path
    from demucs.pretrained import get_model

    torch.backends.mha.set_fastpath_enabled(False)   # the fused MHA op has no ONNX export
    bag = get_model(name, repo=Path(repo))
    entries, report = [], []
    with torch.no_grad():
        for i, (m, w) in enumerate(zip(bag.models, bag.weights)):
            m.eval()
            tl = int(m.segment * m.samplerate)
            mix = torch.randn(1, 2, tl, generator=torch.Generator().manual_seed(i)) * 0.3
            z = m._spec(mix)
            mag = m._magnitude(z)
            ref = m(mix)
            f = f"{name}_{i}.onnx"
            chunk_attention(m)
            torch.onnx.export(HTDemucsCore(m).eval(), (mix, mag), os.path.join(out_dir, f),
                              input_names=["mix", "mag"], output_names=["spec", "wave"], opset_version=17)
            sx, sw = ort.InferenceSession(os.path.join(out_dir, f)).run(None, {"mix": mix.numpy(), "mag": mag.numpy()})
            y = m._ispec(m._mask(z, torch.from_numpy(sx)), tl) + torch.from_numpy(sw)
            report.append((f, ((y - ref).abs().max() / ref.abs().max()).item()))
            entries.append({"file": f, "weights": [float(x) for x in w]})
    m0 = bag.models[0]
    manifest = {"arch": "demucs_v4", "name": name, "sources": list(m0.sources), "samplerate": m0.samplerate,
                "segment_samples": int(m0.segment * m0.samplerate), "nfft": m0.nfft,
                "hop_length": m0.hop_length, "models": entries}
    with open(os.path.join(out_dir, f"{name}.demucs"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return report

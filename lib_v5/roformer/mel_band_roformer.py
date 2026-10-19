"""Mel-Band-Roformer (https://arxiv.org/abs/2310.01809), inference only.

Adapted from ZFTurbo/Music-Source-Separation-Training (MIT) and lucidrains/BS-RoFormer (MIT)."""

from functools import partial

import torch
from torch import nn
from torch.nn import Module
from einops import rearrange, pack, unpack, reduce, repeat
from librosa import filters

from .modules import (MaskEstimator, BandSplit, build_axial_layers, run_axial_layers,
                      stft, istft, complex_mul, default)
from .rotary import RotaryEmbedding


class MelBandRoformer(Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn=None,
            zero_dc=True,
            mask_estimator_depth=1,
            mlp_expansion_factor=4,
            match_input_audio_length=False,
            skip_connection=False,
            **training_only_kwargs,  # multi-resolution loss settings, use_torch_checkpoint, ...
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.match_input_audio_length = match_input_audio_length

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        self.layers = build_axial_layers(
            dim=dim, depth=depth, time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth, linear_transformer_depth=linear_transformer_depth,
            transformer_kwargs=transformer_kwargs, dim_head=dim_head, rotary_embed_cls=RotaryEmbedding)

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = stft_n_fft // 2 + 1

        # binary mel band membership, as in section 2 of the paper (masks are averaged where bands overlap)
        mel_filter_bank = torch.from_numpy(filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands))

        # librosa leaves out the first freq, and some environments give 0.0 instead of ~1.9e-18 for the last
        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=freqs_per_bands_with_complex, num_hidden=mask_estimator_depth,
                          mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(num_stems)
        ])

        self.zero_dc = zero_dc

    def forward(self, raw_audio):
        """raw_audio: (b, s, t) or (b, t). Returns (b, n, s, t)."""
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). ' \
            'also need to be False if mono (channel dimension of 1)'

        raw_audio, batch_audio_channel_packed_shape = pack([raw_audio], '* t')

        stft_window = self.stft_window_fn(device=device)
        stft_repr = stft(raw_audio, stft_window, **self.stft_kwargs)
        stft_repr, = unpack(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # index out all frequencies for all frequency ranges across bands ascending in one go
        x = stft_repr[:, self.freq_indices]
        x = rearrange(x, 'b f t c -> b t (f c)')
        x = self.band_split(x)

        x = run_axial_layers(self.layers, x, skip_connection=self.skip_connection)

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        # average the estimated mask over the frequencies shared by overlapping bands
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        scatter_indices = repeat(self.freq_indices, 'f -> b n f t c', b=batch, n=num_stems, t=stft_repr.shape[-2], c=2)
        masks_summed = torch.zeros((batch, num_stems, *stft_repr.shape[2:]), dtype=masks.dtype, device=device)
        masks_summed = masks_summed.scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1 1', r=channels)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = complex_mul(stft_repr, masks_averaged)

        stft_repr = rearrange(stft_repr, 'b n (f s) t c -> (b n s) f t c', s=self.audio_channels)

        if self.zero_dc:
            stft_repr[:, 0] = 0.

        recon_audio = istft(stft_repr, stft_window, length=istft_length, **self.stft_kwargs)

        return rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)

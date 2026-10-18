"""BS-Roformer / Mel-Band-Roformer support (checkpoints in ZFTurbo/Music-Source-Separation-Training format)."""

import torch

from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer

BS_ROFORMER = 'BS-Roformer'
MEL_BAND_ROFORMER = 'Mel-Band-Roformer'


def _model_config(config):
    model_cfg = config.model if hasattr(config, 'model') else config['model']
    return model_cfg.to_dict() if hasattr(model_cfg, 'to_dict') else dict(model_cfg)


def get_roformer_type(config):
    """Returns BS_ROFORMER / MEL_BAND_ROFORMER for a Roformer yaml config, else None (e.g. MDX23C)."""
    try:
        model_cfg = _model_config(config)
    except (AttributeError, KeyError, TypeError):
        return None
    if 'dim' not in model_cfg or 'depth' not in model_cfg:
        return None
    if 'num_bands' in model_cfg:
        return MEL_BAND_ROFORMER
    if 'freqs_per_bands' in model_cfg:
        return BS_ROFORMER
    return None


def _clean_state_dict(checkpoint):
    state_dict = checkpoint
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
    prefixes = ('module.', '_orig_mod.', 'model.')
    cleaned = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        cleaned[k] = v
    return cleaned


def load_roformer(config, model_path, device='cpu'):
    """Builds the Roformer described by `config` and loads the checkpoint at `model_path`."""
    roformer_type = get_roformer_type(config)
    if roformer_type is None:
        raise ValueError('Unknown model type in the configuration (expected a BS-Roformer or Mel-Band-Roformer).')

    model_cfg = _model_config(config)
    if model_cfg.get('use_pope'):
        raise ValueError('Roformer models using PoPE positional embeddings are not supported.')

    state_dict = _clean_state_dict(torch.load(model_path, map_location='cpu'))

    # "Value residual" checkpoints (e.g. BS_Inst_EXP_VRL) come from lucidrains' BS-RoFormer 0.5.0, whose value
    # residual mix was a no-op (it lerped v with itself), so its weights are unused. Measured on MUSDB18, applying
    # the mix as 0.5.1 does drops that model from 9.3 to 3.0 dB. That code also never zeroes the DC bin.
    value_residual_keys = [k for k in state_dict if '.to_value_residual_mix.' in k]
    if value_residual_keys:
        for k in value_residual_keys:
            del state_dict[k]
        model_cfg.setdefault('zero_dc', False)

    model_cls = MelBandRoformer if roformer_type == MEL_BAND_ROFORMER else BSRoformer
    model = model_cls(**model_cfg)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

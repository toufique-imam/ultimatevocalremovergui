"""Models in the benchmark: key -> (UVR process method attribute, name as desktop UVR lists it,
model file on the phone (None: desktop only), file the desktop loads — for the model size).

This is the 18 Oct 2026 benchmark's model list, kept so its results can be re-scored. Since then
VR 10–16, Kim Inst and Inst HQ 2 were deleted from the desktop models folder and VR 10, 11 and 14
from the app, so re-running those entries needs the files back (UVR's download center; VR via
convert.ipynb)."""

REPO_MODELS = '/Users/nuhash/Desktop/other-codes/ultimatevocalremovergui/models'

MODELS = {
    **{f'vr{n}': ('VR_ARCH_PM', name, f'{name}_vr.ptl', f'{REPO_MODELS}/VR_Models/{name}.pth') for n, name in [
        (10, '10_SP-UVR-2B-32000-1'), (11, '11_SP-UVR-2B-32000-2'), (12, '12_SP-UVR-3B-44100'),
        (13, '13_SP-UVR-4B-44100-1'), (14, '14_SP-UVR-4B-44100-2'), (15, '15_SP-UVR-MID-44100-1'),
        (16, '16_SP-UVR-MID-44100-2')]},
    'kim_vocal2': ('MDX_ARCH_TYPE', 'Kim Vocal 2', 'Kim_Vocal_2.onnx', f'{REPO_MODELS}/MDX_Net_Models/Kim_Vocal_2.onnx'),
    'kim_inst': ('MDX_ARCH_TYPE', 'Kim Inst', 'Kim_Inst.onnx', f'{REPO_MODELS}/MDX_Net_Models/Kim_Inst.onnx'),
    'inst_hq2': ('MDX_ARCH_TYPE', 'UVR-MDX-NET Inst HQ 2', 'UVR-MDX-NET-Inst_HQ_2.onnx',
                 f'{REPO_MODELS}/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_2.onnx'),
    # Older vocal/instrumental VR models, desktop only (not converted for the app), added 18 Oct 2026
    **{key: ('VR_ARCH_PM', name, None, f'{REPO_MODELS}/VR_Models/{name}.pth') for key, name in [
        ('hp1', '1_HP-UVR'), ('hp2', '2_HP-UVR'), ('hp_vocal3', '3_HP-Vocal-UVR'), ('hp_vocal4', '4_HP-Vocal-UVR'),
        ('hp2_7', '7_HP2-UVR'), ('hp2_8', '8_HP2-UVR'), ('hp2_9', '9_HP2-UVR'),
        ('mgm_main', 'MGM_MAIN_v4'), ('mgm_high', 'MGM_HIGHEND_v4'), ('mgm_low_a', 'MGM_LOWEND_A_v4'), ('mgm_low_b', 'MGM_LOWEND_B_v4')]},
    # Newer models already downloaded, desktop only for now, added 18 Oct 2026
    'voc_ft': ('MDX_ARCH_TYPE', 'UVR-MDX-NET-Voc_FT', None, f'{REPO_MODELS}/MDX_Net_Models/UVR-MDX-NET-Voc_FT.onnx'),
    **{key: ('MDX_ARCH_TYPE', f'UVR-MDX-NET Inst HQ {n}', None, f'{REPO_MODELS}/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_{n}.onnx')
       for key, n in [('inst_hq3', 3), ('inst_hq4', 4), ('inst_hq5', 5)]},
    'mdx23c': ('MDX_ARCH_TYPE', 'MDX23C-InstVoc HQ', None, f'{REPO_MODELS}/MDX_Net_Models/MDX23C-8KFFT-InstVoc_HQ.ckpt'),
    'htdemucs_ft': ('DEMUCS_ARCH_TYPE', 'htdemucs_ft', None, f'{REPO_MODELS}/Demucs_Models/v3_v4_repo/f7e0c4bc-ba3fe64a.th'),
    'htdemucs': ('DEMUCS_ARCH_TYPE', 'htdemucs', 'htdemucs.demucs', f'{REPO_MODELS}/Demucs_Models/v3_v4_repo/955717e8-8726e21a.th'),
    'htdemucs_6s': ('DEMUCS_ARCH_TYPE', 'htdemucs_6s', 'htdemucs_6s.demucs', f'{REPO_MODELS}/Demucs_Models/v3_v4_repo/5c90dfd2-34c22ccb.th'),
}

# Files the phone needs in the app's external models folder (VR models are bundled in the app)
PHONE_FILES = {
    'kim_vocal2': [f'{REPO_MODELS}/MDX_Net_Models/Kim_Vocal_2.onnx'],
    'kim_inst': [f'{REPO_MODELS}/MDX_Net_Models/Kim_Inst.onnx'],
    'inst_hq2': [f'{REPO_MODELS}/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_2.onnx'],
    'htdemucs': [f'{REPO_MODELS}/../onnx_models/htdemucs.demucs', f'{REPO_MODELS}/../onnx_models/htdemucs_0.onnx'],
    'htdemucs_6s': [f'{REPO_MODELS}/../onnx_models/htdemucs_6s.demucs', f'{REPO_MODELS}/../onnx_models/htdemucs_6s_0.onnx'],
}

# Demucs uses shifts = 0 on both sides: desktop picks shift offsets at random, so shifted runs can't match
PHONE_EXTRA = {'htdemucs': ['-e', 'shifts', '0'], 'htdemucs_6s': ['-e', 'shifts', '0']}

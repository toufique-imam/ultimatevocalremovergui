"""UVR MDX-Net .onnx -> .mnn for the Android MNN CPU engine.

The app runs <name>.mnn (beside UVR's original <name>.onnx) on MNN's CPU backend in FP32 when its
first non-silent chunk matches ONNX Runtime to FP32 rounding and it is faster (Kim_Vocal_2 on a
Galaxy M12: 29 s per chunk instead of 37). The conversion is MNN's own converter, keeping FP32 weights
and exact GELU (its default swaps erf-GELU for the tanh approximation).

Validation runs the .mnn with the pip MNN against ONNX Runtime. The app builds MNN without
Strassen matrix products and switches off Winograd convolutions, which reaches ~118 dB SNR on
Kim_Vocal_2; pip MNN keeps Strassen, so expect ~90-100 dB here. The check is for conversion bugs,
which land far lower.

Needs the MNN pip package (pin it to the app's MNN version, see android/app/src/main/cpp/CMakeLists.txt),
so run it in the uvr-convert env (see convert.ipynb):
    conda run -n uvr-convert python mnn_export.py <model.onnx> <out.mnn>
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort

MNN_VERSION = "3.6.1"
MIN_SNR_DB = 80.0   # pip MNN (Winograd off, Strassen on) vs ONNX Runtime; a broken conversion is far below


def realistic_input(shape):
    """An STFT-like chunk: random noise with a spectrum falling with frequency (axis 2)."""
    rng = np.random.default_rng(0)
    falloff = np.exp(-np.linspace(0, 6, shape[2]))[None, None, :, None]
    return (rng.standard_normal(shape) * falloff * 20).astype(np.float32)


def run_mnn(path, x):
    import MNN
    net = MNN.Interpreter(path)
    net.setSessionHint(3, 0)   # WINOGRAD_MEMORY_LEVEL 0: no Winograd convolutions, as in the app
    session = net.createSession({'backend': 'CPU', 'precision': 'high', 'numThread': 4})
    inp = net.getSessionInput(session)
    net.resizeTensor(inp, x.shape)
    net.resizeSession(session)
    inp.copyFrom(MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe))
    net.runSession(session)
    out = net.getSessionOutput(session)
    shape = out.getShape()
    host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32), MNN.Tensor_DimensionType_Caffe)
    out.copyToHostTensor(host)
    return np.array(host.getData(), np.float32).reshape(shape)


def export(src, dest):
    import MNN
    if MNN.version() != MNN_VERSION:
        raise RuntimeError(f"MNN {MNN.version()} installed; the app runs {MNN_VERSION} (pip install MNN=={MNN_VERSION})")
    work = tempfile.mkdtemp()
    try:
        out = os.path.join(work, os.path.basename(dest))
        subprocess.run(['MNNConvert', '-f', 'ONNX', '--modelFile', src, '--MNNModel', out, '--bizCode', 'uvr',
                        '--useGeluApproximation', '0'], check=True, capture_output=True)
        session = ort.InferenceSession(src)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
        x = realistic_input(shape)
        ref = session.run(None, {session.get_inputs()[0].name: x})[0].astype(np.float64)
        got = run_mnn(out, x)
        snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((got - ref) ** 2), 1e-30))
        if not snr >= MIN_SNR_DB:
            raise ValueError(f"MNN vs ONNX Runtime SNR {snr:.1f} dB < {MIN_SNR_DB} dB")
        shutil.move(out, dest)
        return snr
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    print(f"{export(sys.argv[1], sys.argv[2]):.1f}")

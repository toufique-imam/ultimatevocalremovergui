"""UVR MDX-Net .onnx -> .tflite for the Android GPU delegate.

The app runs <name>.tflite (beside UVR's original <name>.onnx) on the GPU in FP16 when that is
faster than ONNX Runtime on the CPU and its first non-silent chunk matches the CPU's within 30 dB
(Adreno 650: ~2.4x faster per chunk, SDR within 0.01 dB). Two rewrites make the graph convert and
run as one GPU partition; both are exact:
  - MatMul on the frequency (last) axis -> Transpose + 1x1 Conv + Transpose (onnx2tf can't place
    the original in NHWC).
  - ConvTranspose 2x2 / stride 2 -> 1x1 Conv to 4*Cout channels + DepthToSpace (DCR); the GPU
    delegate's TRANSPOSE_CONV otherwise ends the GPU partition halfway through the net.

Needs TensorFlow, so run it in the uvr-convert env (see convert.ipynb):
    conda run -n uvr-convert python mdx_tflite.py <model.onnx> <out.tflite>
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

MAX_REL_ERR = 1e-4   # FP32 TFLite (CPU) vs the original .onnx, as a fraction of the output peak


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    for io in list(model.graph.input) + list(model.graph.output):
        io.type.tensor_type.shape.dim[0].dim_value = 1   # the GPU delegate needs a fixed batch
    inits = {i.name: i for i in model.graph.initializer}
    nodes = []
    for n in model.graph.node:
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in n.attribute}
        if (n.op_type == 'ConvTranspose' and attrs.get('kernel_shape') == [2, 2] and attrs.get('strides') == [2, 2]
                and attrs.get('pads', [0, 0, 0, 0]) == [0, 0, 0, 0] and attrs.get('group', 1) == 1):
            # Output channel (a*2 + b)*Cout + co of the 1x1 conv holds W[:, co, a, b]
            w = numpy_helper.to_array(inits[n.input[1]])   # [Cin, Cout, 2, 2]
            cout = w.shape[1]
            conv_w = numpy_helper.from_array(
                np.ascontiguousarray(w.transpose(2, 3, 1, 0).reshape(4 * cout, w.shape[0]))[:, :, None, None], n.name + '_d2s_w')
            bias = numpy_helper.to_array(inits[n.input[2]]) if len(n.input) > 2 else np.zeros(cout, w.dtype)
            conv_b = numpy_helper.from_array(np.tile(bias, 4), n.name + '_d2s_b')
            model.graph.initializer.extend([conv_w, conv_b])
            c = n.name + '_d2s_conv'
            nodes += [helper.make_node('Conv', [n.input[0], conv_w.name, conv_b.name], [c], kernel_shape=[1, 1], name=c),
                      helper.make_node('DepthToSpace', [c], [n.output[0]], blocksize=2, mode='DCR', name=n.name + '_d2s')]
        elif n.op_type == 'MatMul' and n.input[1] in inits:
            w = numpy_helper.to_array(inits[n.input[1]])   # [F, F']
            conv_w = numpy_helper.from_array(np.ascontiguousarray(w.T)[:, :, None, None], n.name + '_convw')
            model.graph.initializer.append(conv_w)
            t, c = n.name + '_nchw', n.name + '_conv'
            nodes += [helper.make_node('Transpose', [n.input[0]], [t], perm=[0, 3, 1, 2], name=t + '_t'),       # [1,C,T,F] -> [1,F,C,T]
                      helper.make_node('Conv', [t, conv_w.name], [c], kernel_shape=[1, 1], name=c + '_c'),         # -> [1,F',C,T]
                      helper.make_node('Transpose', [c], [n.output[0]], perm=[0, 2, 3, 1], name=n.name + '_back')]  # -> [1,C,T,F']
        else:
            nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def export_mdx_tflite(src: str, dest: str) -> float:
    """Writes dest and returns max|tflite - onnx| / output peak on a spectrogram-like chunk."""
    from ai_edge_litert.interpreter import Interpreter

    work = tempfile.mkdtemp()
    try:
        model = rewrite(onnx.load(src))
        rewritten = os.path.join(work, 'm.onnx')
        onnx.save(model, rewritten)
        subprocess.run(['onnxsim', rewritten, rewritten], check=True, capture_output=True)
        subprocess.run(['onnx2tf', '-i', rewritten, '-o', os.path.join(work, 'tf'), '-n'], check=True, capture_output=True)
        converted = os.path.join(work, 'tf', 'm_float32.tflite')

        # A chunk with an STFT's dynamic range (falling with frequency); plain noise isn't representative
        _, ch, freq, frames = (d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((1, ch, freq, frames)) * np.exp(-np.linspace(0, 6, freq))[None, None, :, None] * 20).astype(np.float32)
        ref = ort.InferenceSession(src).run(None, {model.graph.input[0].name: x})[0]
        it = Interpreter(converted)
        it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]['index'], x.transpose(0, 2, 3, 1).copy())   # NCHW -> NHWC
        it.invoke()
        y = it.get_tensor(it.get_output_details()[0]['index']).transpose(0, 3, 1, 2)
        rel_err = float(np.abs(y - ref).max() / np.abs(ref).max())
        if rel_err > MAX_REL_ERR:
            raise ValueError(f'max|tflite-onnx| = {rel_err:.1e} of peak > {MAX_REL_ERR}')
        shutil.copy(converted, dest)
        return rel_err
    finally:
        shutil.rmtree(work)


if __name__ == '__main__':
    print(f'{export_mdx_tflite(sys.argv[1], sys.argv[2]):.2e}')

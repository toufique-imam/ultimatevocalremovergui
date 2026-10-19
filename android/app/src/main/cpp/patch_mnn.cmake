# Keeps MNN's CPU 1x1 convolutions (which also run every MatMul with constant weights, such as
# MDX-Net's frequency layers) exact: MNN splits large ones with Strassen's algorithm, which
# trades accuracy for speed. On Kim_Vocal_2 that alone holds the output to ~99 dB SNR against
# ONNX Runtime; plain matrix products give ~118 dB, the level of FP32 rounding.
# (Winograd 3x3 convolutions, the other inexact shortcut, are switched off at run time.)
# Usage: cmake -DMNN_DIR=<MNN source> -P patch_mnn.cmake   (idempotent)
set(file ${MNN_DIR}/source/backend/cpu/compute/Convolution1x1Strassen.cpp)
file(READ ${file} text)
set(original "    int maxDepth = 5;\n")
set(patched "    int maxDepth = 0; // UVR: no Strassen recursion (accuracy)\n")
string(FIND "${text}" "${patched}" done)
if(done EQUAL -1)
    string(FIND "${text}" "${original}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "patch_mnn.cmake: Strassen depth not found in ${file} — check the patch for this MNN version")
    endif()
    string(REPLACE "${original}" "${patched}" text "${text}")
    file(WRITE ${file} "${text}")
    message(STATUS "patch_mnn.cmake: Strassen disabled in ${file}")
endif()

package com.uvr.android.ml

/**
 * Common interface for ML inference backends.
 *
 * Two implementations:
 *  - [OnnxInferenceEngine]  — for .onnx models (via ONNX Runtime Android)
 *  - [PtlInferenceEngine]   — for .ptl models  (via PyTorch Lite Android)
 *
 * Separators (VRSeparator, MDXSeparator, DemucsSeparator) are written against
 * this interface and are therefore format-agnostic.
 */
interface InferenceEngine : AutoCloseable {

    /**
     * Run a single forward pass.
     * [data] is a flat float array; [shape] describes its dimensions.
     * Returns the model output as a flat float array.
     */
    fun runBatch(data: FloatArray, shape: LongArray): FloatArray

    /** Names of all model inputs (useful for multi-input models). */
    val inputNames: Set<String>

    /** Names of all model outputs. */
    val outputNames: Set<String>

    /**
     * Returns the shape declared by the model for [inputName], or null if
     * the backend does not expose shape metadata.
     * Dynamic dimensions are represented as -1.
     */
    fun expectedInputShape(inputName: String): LongArray?

    /**
     * Returns the rank (number of dimensions) the model declares for
     * [inputName], or -1 if unknown.
     */
    fun expectedInputRank(inputName: String): Int
}

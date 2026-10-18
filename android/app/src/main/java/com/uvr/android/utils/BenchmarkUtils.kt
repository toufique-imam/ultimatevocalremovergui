package com.uvr.android.utils

import android.util.Log
import kotlin.system.measureTimeMillis

/**
 * Utility for measuring and logging performance of ML inference and audio processing.
 * Mirrors the "Benchmark" mode in the original UVR desktop app.
 */
object BenchmarkUtils {
    const val TAG = "UVR-Benchmark"

    /**
     * Measures the execution time of [block] and logs it with [label].
     * @return The result of the [block] and the elapsed time in ms.
     */
    inline fun <T> measure(label: String, block: () -> T): Pair<T, Long> {
        var result: T
        val time = measureTimeMillis {
            result = block()
        }
        Log.i(TAG, "$label took ${time}ms")
        return result to time
    }

    /**
     * Data class to hold metrics for a single separation run.
     */
    data class PerformanceMetrics(
        val modelName: String,
        val totalTimeMs: Long,
        val inferenceTimeMs: Long,
        val processingTimeMs: Long,
        val inputDurationSec: Double,
        val speedFactor: Double // (inputDurationSec / (totalTimeMs / 1000.0))
    ) {
        fun log() {
            Log.i(TAG, """
                --- Benchmark Results: $modelName ---
                Total Time:      ${totalTimeMs}ms
                Inference Only:  ${inferenceTimeMs}ms
                Pre/Post Proc:   ${processingTimeMs}ms
                File Duration:   ${"%.2f".format(inputDurationSec)}s
                Speed Factor:    ${"%.2f".format(speedFactor)}x real-time
                -------------------------------------
            """.trimIndent())
        }
    }
}

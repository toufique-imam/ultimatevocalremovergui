package com.uvr.android.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VerifiedEngineTest {

    /** Returns [transform] of the input after [delayMs]; counts calls and closes. */
    private class FakeEngine(val delayMs: Long, val transform: (FloatArray) -> FloatArray) : InferenceEngine {
        var calls = 0
        var closed = false
        override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
            calls++
            if (delayMs > 0) Thread.sleep(delayMs)
            return transform(data)
        }
        override val inputNames = setOf("input")
        override val outputNames = setOf("output")
        override fun expectedInputShape(inputName: String): LongArray? = null
        override fun expectedInputRank(inputName: String) = -1
        override fun close() { closed = true }
    }

    private val shape = longArrayOf(1, 1000)
    private val loud = FloatArray(1000) { if (it % 2 == 0) 1f else -1f }
    private val silent = FloatArray(1000)

    @Test
    fun fastEngineWithinSnrIsKept() {
        val ref = FakeEngine(30) { it.copyOf() }
        val fast = FakeEngine(0) { x -> FloatArray(x.size) { x[it] * 1.001f } }   // error 60 dB down
        val engine = VerifiedEngine(fast, { ref }, minSnrDb = 30.0, minLevel = 0.05)
        engine.runBatch(loud, shape)
        engine.runBatch(loud, shape)
        assertEquals("gpu", engine.backend)
        assertEquals(2, fast.calls)
        assertTrue(ref.closed)
    }

    @Test
    fun fastEngineBelowSnrFallsBack() {
        val ref = FakeEngine(30) { it.copyOf() }
        val fast = FakeEngine(0) { x -> FloatArray(x.size) { x[it] * 1.1f } }     // error 20 dB down
        val engine = VerifiedEngine(fast, { ref }, minSnrDb = 30.0, minLevel = 0.05)
        engine.runBatch(loud, shape)
        engine.runBatch(loud, shape)
        assertEquals("cpu", engine.backend)
        assertEquals(1, fast.calls)
        assertTrue(fast.closed)
    }

    @Test
    fun quietWindowsDeferTheCheck() {
        val ref = FakeEngine(30) { it.copyOf() }
        val fast = FakeEngine(0) { x -> FloatArray(x.size) { x[it] * 1.001f + 1e-4f } }  // poor SNR on silence
        val engine = VerifiedEngine(fast, { ref }, minSnrDb = 30.0, minLevel = 0.05)
        engine.runBatch(silent, shape)
        engine.runBatch(silent, shape)
        assertEquals("pending", engine.backend)
        assertEquals(0, fast.calls)
        engine.runBatch(loud, shape)
        assertEquals("gpu", engine.backend)
        assertEquals(3, ref.calls)
    }

    @Test
    fun withoutSnrTheFp32ToleranceApplies() {
        val ref = FakeEngine(30) { it.copyOf() }
        val fast = FakeEngine(0) { x -> FloatArray(x.size) { x[it] * 1.001f } }   // |Δ| = 1e-3 > tolerance
        val engine = VerifiedEngine(fast, { ref }, tolerance = 1e-4f)
        engine.runBatch(loud, shape)
        assertEquals("cpu", engine.backend)
        // Silence isn't deferred in this mode: it matches exactly, so the fast engine is kept
        val again = VerifiedEngine(FakeEngine(0) { x -> FloatArray(x.size) { x[it] * 1.001f } }, { FakeEngine(30) { it.copyOf() } }, tolerance = 1e-4f)
        again.runBatch(silent, shape)
        assertEquals("gpu", again.backend)
    }

    @Test
    fun lazyFastEngineIsCreatedAfterTheReferenceRan() {
        val ref = FakeEngine(30) { it.copyOf() }
        var refCallsAtCreation = -1
        val fast = FakeEngine(0) { it.copyOf() }
        val engine = VerifiedEngine(LazyEngine { refCallsAtCreation = ref.calls; fast }, { ref },
            minSnrDb = 100.0, fastName = "mnn", referenceName = "ort")
        engine.runBatch(loud, shape)
        assertEquals(1, refCallsAtCreation)
        assertEquals("mnn", engine.backend)
        assertTrue(ref.closed)
    }

    @Test
    fun fastEngineMustBeatTheReferenceByTheMargin() {
        val slightlyFaster = VerifiedEngine(FakeEngine(90) { it.copyOf() }, { FakeEngine(100) { it.copyOf() } },
            minSnrDb = 100.0, minSpeedup = 1.5)
        slightlyFaster.runBatch(loud, shape)
        assertEquals("cpu", slightlyFaster.backend)
        val muchFaster = VerifiedEngine(FakeEngine(10) { it.copyOf() }, { FakeEngine(100) { it.copyOf() } },
            minSnrDb = 100.0, minSpeedup = 1.5)
        muchFaster.runBatch(loud, shape)
        assertEquals("gpu", muchFaster.backend)
    }

    @Test
    fun failingLazyFastEngineFallsBack() {
        val ref = FakeEngine(30) { it.copyOf() }
        val engine = VerifiedEngine(LazyEngine { throw IllegalStateException("no MNN") }, { ref },
            minSnrDb = 100.0, fastName = "mnn", referenceName = "ort")
        engine.runBatch(loud, shape)
        engine.runBatch(loud, shape)
        assertEquals("ort", engine.backend)
        assertEquals(2, ref.calls)
    }
}

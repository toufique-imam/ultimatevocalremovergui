package com.uvr.android.audio

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.webkit.MimeTypeMap
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFmpegSession
import com.arthenica.ffmpegkit.FFprobeKit
import com.arthenica.ffmpegkit.ReturnCode
import com.uvr.android.utils.FileUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Handles all audio I/O for UVR Android.
 *
 * All decoded audio stays on disk as a temp raw PCM file (interleaved f32le,
 * no header) — it is NEVER loaded into a FloatArray in the Java heap.
 * Separators stream chunks from the file.
 */
class AudioProcessor(private val context: Context) {

    companion object {
        private const val TAG = "AudioProcessor"
        const val TARGET_SAMPLE_RATE = 44100
        const val TARGET_CHANNELS = 2

        /** Folder under the shared Music collection where stems are published. */
        const val OUTPUT_RELATIVE_DIR = "Music/UVR"

        /** Folder under the shared Movies collection where videos with a stem as their audio go. */
        const val VIDEO_OUTPUT_RELATIVE_DIR = "Movies/UVR"

        /** Why FFmpeg failed: its exception, or else the last lines of its log. */
        private fun failureReason(session: FFmpegSession): String =
            session.failStackTrace
                ?: session.output?.trim()?.lines()?.takeLast(2)?.joinToString(" ")
                ?: "unknown error"
    }

    /**
     * Decode any audio format (or the sound of a video) to a temp raw PCM file (f32le, 44100 Hz, stereo).
     * Raw rather than WAV: FFmpeg's float WAV header is not a fixed 44 bytes
     * (WAVE_FORMAT_EXTENSIBLE / LIST chunks), so skipping 44 bytes misaligned every sample.
     * With [keepVideo], a copy of the file is kept as [DecodedAudio.videoFile] if it has a video
     * stream, so stems can be put back into the video later.
     * Returns a [DecodedAudio] whose files must be deleted ([DecodedAudio.delete]) when done.
     */
    suspend fun decode(uri: Uri, keepVideo: Boolean = false): DecodedAudio = withContext(Dispatchers.IO) {
        Log.i(TAG, "Decoding: $uri")

        val tempInput = createTempFile("uvr_input", suffix(uri))
        val tempPcm = createTempFile("uvr_pcm", ".f32")
        var videoFile: File? = null
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                tempInput.outputStream().use { output -> input.copyTo(output) }
            } ?: throw AudioDecodeException("Cannot open the selected file")

            val cmd = "-y -i \"${tempInput.absolutePath}\" -vn " +
                    "-ar $TARGET_SAMPLE_RATE -ac $TARGET_CHANNELS " +
                    "-f f32le -acodec pcm_f32le \"${tempPcm.absolutePath}\""

            Log.d(TAG, "FFmpeg: $cmd")
            val session = FFmpegKit.execute(cmd)
            if (!ReturnCode.isSuccess(session.returnCode) || tempPcm.length() == 0L) {
                throw AudioDecodeException("Not a readable audio file (${failureReason(session)})")
            }
            if (keepVideo && hasVideoStream(tempInput)) videoFile = tempInput
        } catch (e: Throwable) {
            tempPcm.delete()
            throw e
        } finally {
            if (videoFile == null) tempInput.delete()
        }

        val totalSamples = tempPcm.length() / 4
        val durationSec  = totalSamples / TARGET_CHANNELS / TARGET_SAMPLE_RATE
        Log.i(TAG, "Decoded ${durationSec}s → ${tempPcm.absolutePath} (${tempPcm.length() / 1_000_000}MB)" +
                if (videoFile != null) ", keeping the video" else "")
        DecodedAudio(tempPcm, TARGET_SAMPLE_RATE, TARGET_CHANNELS, totalSamples, videoFile)
    }

    /** True if [file] has a real video stream — not just the cover art some audio files carry. */
    private fun hasVideoStream(file: File): Boolean =
        FFprobeKit.getMediaInformation(file.absolutePath).mediaInformation?.streams.orEmpty().any { stream ->
            stream.type == "video" &&
                stream.getProperty("disposition")?.optInt("attached_pic") != 1
        }

    /**
     * Save [video] with [stemPcm] (raw interleaved f32le) as its audio. The video stream is
     * copied as is, so this is quick and lossless. Writes [output] (.mp4), or an .mkv next to
     * it when MP4 can't hold the video codec, and returns the file written.
     */
    suspend fun muxVideo(
        video: File,
        stemPcm: File,
        output: File,
        sampleRate: Int = TARGET_SAMPLE_RATE,
        channels: Int = TARGET_CHANNELS,
        onProgress: (Float) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {
        // Encoding the AAC is the slow part (FFmpeg's "fast" coder: ~2x realtime on an Exynos 850)
        val inputs = "-y -i \"${video.absolutePath}\" " +
                "-f f32le -ar $sampleRate -ac $channels -i \"${stemPcm.absolutePath}\" " +
                "-map 0:v:0 -map 1:a:0 -map_metadata 0 -c:v copy " +
                "-ar $TARGET_SAMPLE_RATE -c:a aac -aac_coder fast -b:a 256k"
        val durationMs = stemPcm.length() / 4.0 / channels / sampleRate * 1000

        val mp4 = executeWithProgress("$inputs -movflags +faststart \"${output.absolutePath}\"", durationMs, onProgress)
        if (ReturnCode.isSuccess(mp4.returnCode)) return@withContext output
        output.delete()
        Log.w(TAG, "MP4 mux failed, trying MKV: ${failureReason(mp4)}")

        val mkvFile = File(output.parentFile, "${output.nameWithoutExtension}.mkv")
        val mkv = executeWithProgress("$inputs \"${mkvFile.absolutePath}\"", durationMs, onProgress)
        if (ReturnCode.isSuccess(mkv.returnCode)) return@withContext mkvFile
        mkvFile.delete()
        throw AudioEncodeException("Could not save the video (${failureReason(mkv)})")
    }

    /** Run FFmpeg [cmd], reporting how much of [durationMs] it has written. Cancelling the coroutine stops FFmpeg. */
    private suspend fun executeWithProgress(cmd: String, durationMs: Double, onProgress: (Float) -> Unit): FFmpegSession =
        suspendCancellableCoroutine { cont ->
            val session = FFmpegKit.executeAsync(cmd,
                { session -> if (cont.isActive) cont.resume(session) },
                null,
                { stats -> if (durationMs > 0) onProgress((stats.time / durationMs).toFloat().coerceIn(0f, 1f)) })
            cont.invokeOnCancellation { FFmpegKit.cancel(session.sessionId) }
        }

    /**
     * Encode a raw f32le PCM file ([sampleRate] Hz) to the chosen output format at
     * [outputSampleRate] via FFmpegKit.
     * The [rawPcmFile] is read directly — no FloatArray ever enters the Java heap.
     */
    suspend fun encode(
        rawPcmFile: File,
        outputFile: File,
        format: OutputFormat = OutputFormat.WAV,
        sampleRate: Int = TARGET_SAMPLE_RATE,
        channels: Int = TARGET_CHANNELS,
        outputSampleRate: Int = TARGET_SAMPLE_RATE
    ): File = withContext(Dispatchers.IO) {
        Log.i(TAG, "Encoding ${rawPcmFile.name} → ${outputFile.name} (format=$format)")

        val codecArgs = when (format) {
            OutputFormat.WAV  -> "-acodec pcm_s16le"
            OutputFormat.MP3  -> "-acodec libmp3lame -q:a 2"
            OutputFormat.FLAC -> "-acodec flac"
            OutputFormat.OGG  -> "-acodec libvorbis -q:a 6"
            OutputFormat.M4A  -> "-acodec aac -b:a 256k"
        }

        val cmd = "-y -f f32le -ar $sampleRate -ac $channels " +
                "-i \"${rawPcmFile.absolutePath}\" -ar $outputSampleRate $codecArgs \"${outputFile.absolutePath}\""

        val session = FFmpegKit.execute(cmd)

        if (!ReturnCode.isSuccess(session.returnCode)) {
            outputFile.delete()
            throw AudioEncodeException("Could not encode $format (${failureReason(session)})")
        }

        Log.i(TAG, "Encoded → ${outputFile.absolutePath}")
        outputFile
    }

    /**
     * Move an encoded stem into the shared Music/UVR collection (videos: Movies/UVR) so it
     * shows up in the Files app and media players. [file] is deleted afterwards.
     *
     * Android 10+ uses MediaStore (no permission needed). Older versions keep the
     * file in [fallbackDir] (app-specific storage) and share it via FileProvider.
     */
    suspend fun publish(file: File, displayName: String, fallbackDir: File): OutputStem =
        withContext(Dispatchers.IO) {
            val mime = MimeTypeMap.getSingleton()
                .getMimeTypeFromExtension(file.extension.lowercase()) ?: "audio/*"
            val size = file.length()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val isVideo = mime.startsWith("video/")
                val relativeDir = if (isVideo) VIDEO_OUTPUT_RELATIVE_DIR else OUTPUT_RELATIVE_DIR
                val resolver = context.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                    put(MediaStore.MediaColumns.MIME_TYPE, mime)
                    put(MediaStore.MediaColumns.RELATIVE_PATH, relativeDir)
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                val collection = if (isVideo) MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                                 else MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                val uri = resolver.insert(collection, values)
                    ?: throw AudioEncodeException("Could not create $displayName in $relativeDir")
                try {
                    resolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
                        ?: throw AudioEncodeException("Could not write $displayName")
                    resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
                } catch (e: Throwable) {
                    resolver.delete(uri, null, null)
                    throw e
                } finally {
                    file.delete()
                }
                // MediaStore renames on collisions ("name (1).wav"); report the real name
                val finalName = resolver.query(uri, arrayOf(MediaStore.MediaColumns.DISPLAY_NAME), null, null, null)
                    ?.use { c -> if (c.moveToFirst()) c.getString(0) else null } ?: displayName
                Log.i(TAG, "Published → $relativeDir/$finalName ($uri)")
                OutputStem(uri, finalName, mime, size, "$relativeDir/$finalName")
            } else {
                fallbackDir.mkdirs()
                // Like MediaStore, don't overwrite an earlier output ("name (1).wav"): a folder
                // can hold files of the same name in different subfolders
                val base = displayName.substringBeforeLast('.')
                val ext = displayName.substringAfterLast('.', "").let { if (it.isEmpty()) "" else ".$it" }
                val dest = generateSequence(0) { it + 1 }
                    .map { n -> File(fallbackDir, if (n == 0) displayName else "$base ($n)$ext") }
                    .first { !it.exists() }
                file.copyTo(dest)
                file.delete()
                OutputStem(FileUtils.shareableUri(context, dest), dest.name, mime, size, dest.absolutePath)
            }
        }

    /**
     * Generate a waveform preview by seeking to 1000 evenly-spaced positions.
     * Uses O(1) memory — never reads the full PCM into a FloatArray.
     */
    fun generateWaveformPreview(audio: DecodedAudio, targetPoints: Int = 1000): FloatArray {
        val monoSamples = audio.totalSamples / audio.channels
        if (monoSamples <= 0L) return FloatArray(0)

        val step    = maxOf(1L, monoSamples / targetPoints)
        val count   = minOf(targetPoints, (monoSamples / step).toInt())
        val result  = FloatArray(count)
        val frameBuf = ByteArray(audio.channels * 4)
        val bb       = ByteBuffer.wrap(frameBuf).order(ByteOrder.LITTLE_ENDIAN)

        try {
            RandomAccessFile(audio.pcmFile, "r").use { raf ->
                for (i in 0 until count) {
                    val filePos = i.toLong() * step * audio.channels * 4
                    if (filePos + frameBuf.size > raf.length()) break
                    raf.seek(filePos)
                    raf.readFully(frameBuf)
                    bb.rewind()
                    var peak = 0f
                    repeat(audio.channels) {
                        val v = kotlin.math.abs(bb.float)
                        if (v > peak) peak = v
                    }
                    result[i] = peak
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Waveform preview error: ${e.message}")
        }
        return result
    }

    fun createTempFile(prefix: String, suffix: String): File =
        File.createTempFile(prefix, suffix, context.cacheDir)

    private fun suffix(uri: Uri): String {
        val path = uri.path ?: return ".tmp"
        return ".${path.substringAfterLast('.', "tmp")}"
    }
}

/**
 * Decoded audio backed by a temp raw PCM file on disk (interleaved f32le, no header).
 * Call [delete] when done to free disk space.
 */
data class DecodedAudio(
    val pcmFile: File,
    val sampleRate: Int,
    val channels: Int,
    val totalSamples: Long,   // total float32 count = file.length / 4
    /** A copy of the input when it is a video, for saving videos with a stem as their audio. */
    val videoFile: File? = null
) {
    fun delete() {
        pcmFile.delete()
        videoFile?.delete()
    }
}

enum class OutputFormat(val extension: String, val mimeType: String) {
    WAV("wav",  "audio/wav"),
    MP3("mp3",  "audio/mpeg"),
    FLAC("flac","audio/flac"),
    OGG("ogg",  "audio/ogg"),
    M4A("m4a",  "audio/mp4")
}

/** A separated stem saved for the user: a content URI plus what the UI shows about it. */
data class OutputStem(
    val uri: Uri,
    val displayName: String,
    val mimeType: String,
    val sizeBytes: Long,
    val location: String
)

class AudioDecodeException(message: String) : Exception(message)
class AudioEncodeException(message: String) : Exception(message)

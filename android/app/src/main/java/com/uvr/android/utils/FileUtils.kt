package com.uvr.android.utils

import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import androidx.core.content.FileProvider
import java.io.File

/**
 * File utilities for sharing and opening output audio files.
 */
object FileUtils {

    private const val FILE_PROVIDER_AUTHORITY = "com.uvr.android.fileprovider"

    /**
     * Build a shareable URI for an output file using FileProvider.
     * Required on Android 7+ to share files between apps.
     */
    fun shareableUri(context: Context, file: File): Uri {
        return FileProvider.getUriForFile(context, FILE_PROVIDER_AUTHORITY, file)
    }

    /**
     * Build a share intent for a separated stem ([uri] from MediaStore or FileProvider).
     */
    fun buildShareIntent(uri: Uri, mimeType: String = "audio/*"): Intent {
        return Intent(Intent.ACTION_SEND).apply {
            type = mimeType
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    /**
     * Build an open-with intent for an audio file.
     */
    fun buildOpenIntent(context: Context, file: File): Intent {
        val uri = shareableUri(context, file)
        return Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "audio/*")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    /**
     * User-visible file name of [uri]. For picker/share URIs this is the provider's
     * DISPLAY_NAME — `lastPathSegment` is a document ID such as
     * `primary:Music/Song.mp3` or `audio:1234`.
     */
    fun displayName(context: Context, uri: Uri): String {
        if (uri.scheme == ContentResolver.SCHEME_CONTENT) {
            runCatching {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                    ?.use { c -> if (c.moveToFirst()) c.getString(0)?.takeIf { it.isNotBlank() } else null }
            }.getOrNull()?.let { return it }
        }
        return uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: "audio"
    }

    private val VIDEO_EXTENSIONS = setOf("mp4", "m4v", "mov", "mkv", "webm", "avi", "flv", "3gp", "ts", "mpg", "mpeg", "wmv")

    /** Desktop UVR's input extensions, for files a provider types as application/octet-stream. */
    private val AUDIO_EXTENSIONS = setOf("aac", "aiff", "alac", "flac", "m4a", "mp2", "mp3", "mpc", "mpc8",
        "ogg", "oga", "opus", "tta", "wav", "wave", "wma", "eac3", "ac3", "amr")

    /**
     * Whether [uri] may be a video, by its MIME type or [fileName]'s extension. Only a hint:
     * decoding checks for a real video stream (an .mp4 can be audio only).
     */
    fun looksLikeVideo(context: Context, uri: Uri, fileName: String): Boolean {
        val type = runCatching { context.contentResolver.getType(uri) }.getOrNull()
        return looksLikeVideo(type, fileName)
    }

    private fun looksLikeVideo(mimeType: String?, fileName: String): Boolean =
        mimeType?.startsWith("video/") == true ||
            fileName.substringAfterLast('.', "").lowercase() in VIDEO_EXTENSIONS

    /** Name of the folder [treeUri] (from the system folder picker) points at. */
    fun folderName(context: Context, treeUri: Uri): String =
        displayName(context, DocumentsContract.buildDocumentUriUsingTree(treeUri, DocumentsContract.getTreeDocumentId(treeUri)))

    /**
     * The audio and video files in the folder [treeUri] (from the system folder picker) and its
     * subfolders — like desktop UVR when a folder is dropped on it — sorted by path.
     * Hidden files and folders are skipped.
     */
    fun listMediaFiles(context: Context, treeUri: Uri): List<MediaFile> {
        val resolver = context.contentResolver
        val found = mutableListOf<Pair<String, MediaFile>>()
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )
        fun walk(documentId: String, path: String) {
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, documentId)
            resolver.query(children, projection, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val id = c.getString(0) ?: continue
                    val name = c.getString(1)?.takeIf { it.isNotBlank() && !it.startsWith(".") } ?: continue
                    val mime = c.getString(2)
                    when {
                        mime == DocumentsContract.Document.MIME_TYPE_DIR -> walk(id, "$path$name/")
                        isMediaFile(mime, name) -> found += "$path$name" to MediaFile(
                            DocumentsContract.buildDocumentUriUsingTree(treeUri, id), name, looksLikeVideo(mime, name))
                    }
                }
            }
        }
        walk(DocumentsContract.getTreeDocumentId(treeUri), "")
        return found.sortedWith(compareBy(String.CASE_INSENSITIVE_ORDER) { it.first }).map { it.second }
    }

    private fun isMediaFile(mimeType: String?, fileName: String): Boolean =
        mimeType?.startsWith("audio/") == true || looksLikeVideo(mimeType, fileName) ||
            fileName.substringAfterLast('.', "").lowercase() in AUDIO_EXTENSIONS

    /**
     * Delete leftover `uvr_*` temp files (decoded PCM, raw stems) in [dir] that are
     * older than [before]. Runs are interrupted by process death, so nothing else
     * would ever remove them — a single 8-minute song leaves ~170 MB behind.
     */
    fun deleteStaleTempFiles(dir: File, before: Long): Long {
        var freed = 0L
        dir.listFiles { f -> f.isFile && f.name.startsWith("uvr_") && f.lastModified() < before }
            ?.forEach { f ->
                val size = f.length()
                if (f.delete()) freed += size
            }
        return freed
    }

    /** Filename base for output files: extension removed, path/reserved characters replaced. */
    fun sanitizeBaseName(name: String): String {
        return name.substringBeforeLast('.')
            .replace(Regex("[\\\\/:*?\"<>|\\p{Cntrl}]"), "_")
            .trim()
            .ifEmpty { "output" }
    }
}

/** An audio or video file found in a folder the user picked. [isVideo] is a hint, as in [FileUtils.looksLikeVideo]. */
data class MediaFile(val uri: Uri, val name: String, val isVideo: Boolean)

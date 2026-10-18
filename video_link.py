"""Video support: download a video with yt-dlp, hand its audio to UVR, and
put a separated stem back into the video with FFmpeg."""

import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from gui_data.constants import ALL_STEMS, KEEP_VIDEO_LENGTH, CUT_VIDEO_SILENCE

VIDEO_DOWNLOADS_FOLDER = 'Video Downloads'
VIDEO_EXTENSIONS = ('.mp4', '.m4v', '.mov', '.mkv', '.webm', '.avi', '.flv', '.f4v', '.mpeg', '.mpg', '.wmv', '.ts')

# UVR names its outputs "<n>_<input name>_(<stem>).<ext>"
STEM_FILE_PATTERN = re.compile(r'_\(([^()]+)\)\.(wav|flac|mp3)$', re.IGNORECASE)
JS_RUNTIMES = ('deno', 'node', 'bun')
URL_PATTERN = re.compile(r'^https?://\S+$', re.IGNORECASE)

class VideoLinkError(Exception):
    pass

class SilentLogger:
    """Keeps yt-dlp off the console; its errors are raised as exceptions instead."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

def is_url(text):
    return bool(URL_PATTERN.match(text.strip()))

def is_yt_dlp_installed():
    return importlib.util.find_spec('yt_dlp') is not None

def find_ffmpeg():
    """FFmpeg bundled next to UVR (where separate.py looks for it on macOS), else the one on PATH."""

    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')

    return bundled if os.path.isfile(bundled) else shutil.which('ffmpeg')

def run_ffmpeg(ffmpeg_path, args):
    startupinfo = None
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.run([ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error', *args],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)

    if process.returncode != 0:
        raise VideoLinkError(process.stderr.decode(errors='replace').strip() or f'FFmpeg exited with code {process.returncode}')

    return process.stdout.decode(errors='replace')

@functools.lru_cache(maxsize=None)
def video_encoder_args(ffmpeg_path):
    """H.264 for videos that have to be re-encoded, when FFmpeg has the encoder; else its MP4 default"""

    if ' libx264 ' in run_ffmpeg(ffmpeg_path, ['-encoders']):
        return ('-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p')

    return ('-pix_fmt', 'yuv420p')

def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

def download_video(url, download_dir, ffmpeg_path=None, set_progress=None):
    """Downloads the best video and audio of the link (merged into one file) and returns its path.
    set_progress receives the overall download progress from 0 to 1."""

    import yt_dlp

    format_ids = []

    def progress_hook(d):
        if not set_progress or d['status'] != 'downloading':
            return

        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        if not total:
            return

        # Video and audio are separate downloads; spread them over the whole bar
        format_id = d['info_dict'].get('format_id')
        part, parts = (format_ids.index(format_id), len(format_ids)) if format_id in format_ids else (0, 1)
        set_progress((part + min(d.get('downloaded_bytes', 0) / total, 1)) / parts)

    options = {
        'format': 'bv*+ba/b',
        # The video stream is copied into the output MP4 as is. QuickTime and many players can't play
        # VP9 (YouTube's usual best), so take H.264 when the site has it, even at a lower resolution
        'format_sort': ['vcodec:h264', 'res'],
        'outtmpl': os.path.join(download_dir, '%(title).80B [%(id)s].%(ext)s'),
        'windowsfilenames': True,
        'noplaylist': True,
        'extract_flat': 'in_playlist',
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'logger': SilentLogger(),
        'progress_hooks': [progress_hook],
        # YouTube needs a JavaScript runtime; yt-dlp only looks for deno unless told otherwise
        'js_runtimes': {runtime: {} for runtime in JS_RUNTIMES if shutil.which(runtime)} or {'deno': {}},
    }

    if ffmpeg_path:
        options['ffmpeg_location'] = ffmpeg_path

    os.makedirs(download_dir, exist_ok=True)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)

            if info.get('_type') == 'playlist':
                raise VideoLinkError('The link is a playlist. Please use a link to a single video.')

            format_ids.extend(f['format_id'] for f in info.get('requested_formats') or [info])
            info = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise VideoLinkError(re.sub(r'^ERROR:\s*', '', str(e))) from e

    downloads = info.get('requested_downloads') or []
    video_path = downloads[0].get('filepath') if downloads else None

    if not video_path or not os.path.isfile(video_path):
        raise VideoLinkError('yt-dlp did not produce a video file.')

    return video_path

def extract_audio(ffmpeg_path, video_path, audio_path):
    """Writes the first audio stream of the video to a 32-bit float WAV for UVR."""

    try:
        run_ffmpeg(ffmpeg_path, ['-i', video_path, '-map', '0:a:0', '-vn', '-c:a', 'pcm_f32le', audio_path])
    except VideoLinkError as e:
        if 'matches no streams' in str(e):
            raise VideoLinkError('The downloaded video has no audio.') from e
        raise

    return audio_path

def find_stem_files(folders, output_prefix, since):
    """Returns {stem: path} for the UVR outputs starting with output_prefix ("<file number>_<input name>",
    after the timestamp test mode may add) written at or after since (with slack for filesystems that
    store modification times in 2 second steps)."""

    stems = {}
    name_pattern = re.compile(rf'^(\d+_)?{re.escape(output_prefix)}_')

    for folder in dict.fromkeys(folders):
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            match = STEM_FILE_PATTERN.search(file)
            if match and name_pattern.match(file) and os.path.getmtime(path) >= since - 2:
                stems[match.group(1)] = path

    return stems

def select_stems(stems, chosen_stem):
    """Picks the stem files to put back into the video: all of them, or the one named chosen_stem."""

    if chosen_stem == ALL_STEMS:
        return stems

    return {stem: path for stem, path in stems.items() if stem.lower() == chosen_stem.lower()}

def replace_audio(ffmpeg_path, video_path, audio_path, output_base):
    """Copies the video stream and encodes audio_path as its soundtrack. Writes output_base.mp4,
    or output_base.mkv if the video codec doesn't fit in an MP4, and returns the path written."""

    inputs = ['-i', video_path, '-i', audio_path, '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '0', '-c:v', 'copy']
    mp4_path = f'{output_base}.mp4'

    try:
        run_ffmpeg(ffmpeg_path, inputs + ['-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', mp4_path])
        return mp4_path
    except VideoLinkError:
        if os.path.isfile(mp4_path):
            os.remove(mp4_path)

    mkv_path = f'{output_base}.mkv'
    run_ffmpeg(ffmpeg_path, inputs + ['-c:a', 'flac', mkv_path])

    return mkv_path

def make_video(ffmpeg_path, video_path, audio_path, output_base, video_fit=KEEP_VIDEO_LENGTH, trim=None):
    """Saves the video with audio_path as its audio and returns the path written. trim is Trim Silence's
    (kept (start, end) seconds, original length) when it shortened audio_path; video_fit then decides
    whether the video is left as is, cut the same way, or sped up to the new length."""

    try:
        if not trim or video_fit == KEEP_VIDEO_LENGTH:
            return replace_audio(ffmpeg_path, video_path, audio_path, output_base)

        keep_ranges, original_length = trim

        if video_fit == CUT_VIDEO_SILENCE:
            video_filter = cut_video_filter(keep_ranges)
        else:
            kept_length = sum(end - start for start, end in keep_ranges)
            video_filter = f'setpts=(PTS-STARTPTS)*{kept_length / original_length:.6f}'

        # Changing the frames means re-encoding the video instead of copying it
        output_path = f'{output_base}.mp4'
        run_ffmpeg(ffmpeg_path, ['-i', video_path, '-i', audio_path, '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '0',
                                 '-vf', video_filter, *video_encoder_args(ffmpeg_path),
                                 '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', output_path])
        return output_path
    except VideoLinkError as e:
        if 'matches no streams' in str(e):
            raise VideoLinkError('The input has no video stream.') from e
        raise

def cut_video_filter(keep_ranges):
    """Keeps the frames inside keep_ranges and closes the gaps between them,
    so each kept frame lands where its audio does in the trimmed stem"""

    keep_frames, shifts, removed, previous_end = [], [], 0, 0

    for start, end in keep_ranges:
        removed += start - previous_end
        previous_end = end
        keep_frames.append(f'gte(t,{start:.6f})*lt(t,{end:.6f})')
        shifts.append(f'gte(T,{start:.6f})*lt(T,{end:.6f})*{removed:.6f}')

    return f"setpts=PTS-STARTPTS,select='{'+'.join(keep_frames)}',setpts='(T-({'+'.join(shifts)}))/TB'"

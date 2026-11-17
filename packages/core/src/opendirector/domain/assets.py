"""Shared media input size limits for model adapters and result caching."""

MAX_ASSET_BYTES = 50 * 1024 * 1024
# A super-resolved H3 clip carries the same frame count in four times the pixel
# area, so a 15 s 2688x1536 delivery reaches ~68 MB. Video results therefore get
# their own ceiling instead of the image-oriented one.
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
VIDEO_SUFFIXES = frozenset({'mp4', 'mov', 'webm', 'mkv'})


def asset_limit(suffix):
    return MAX_VIDEO_BYTES if str(suffix).lower() in VIDEO_SUFFIXES else MAX_ASSET_BYTES


def size_label(limit):
    return f'{limit // (1024 * 1024)} MB'

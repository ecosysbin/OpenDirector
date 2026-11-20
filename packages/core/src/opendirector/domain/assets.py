"""模型适配器与结果缓存共用的媒体输入体积上限。"""

MAX_ASSET_BYTES = 50 * 1024 * 1024
# 超分后的 H3 片段帧数不变、像素面积翻四倍，15 秒 2688x1536 的交付可达约 68 MB。
# 因此视频结果单独给一档上限，而不是沿用面向图片的那一档。
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
VIDEO_SUFFIXES = frozenset({'mp4', 'mov', 'webm', 'mkv'})


def asset_limit(suffix):
    """按结果后缀选择上限：视频按 512 MB，其余仍按 50 MB。"""
    return MAX_VIDEO_BYTES if str(suffix).lower() in VIDEO_SUFFIXES else MAX_ASSET_BYTES


def size_label(limit):
    """把字节上限转成报错文案里的可读单位（仅按整 MB 展示）。"""
    return f'{limit // (1024 * 1024)} MB'

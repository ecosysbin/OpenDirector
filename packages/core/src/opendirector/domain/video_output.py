"""Resolve explicit output resolution without changing reference files or old tasks.

在不改动参考文件或旧任务的前提下，解析显式的输出分辨率。
"""
import math

from opendirector.domain.input_rules import input_rules
from opendirector.domain.projects import ProjectError
from opendirector.domain.rule_facts import VIDEO_ASPECT_RATIOS as RATIOS
from opendirector.domain.video_capabilities import validate_video_duration


def _ratio(value):
    width, height = map(int, value.split(':'))
    return width / height


def h3_output_size(resolution, ratio):
    """Use H3's documented landscape buckets and a 32-pixel canvas grid.

    采用 H3 文档中的横屏档位，并按 32 像素画布网格取整。
    """
    edge = {'480P': 480, '768P': 768}[resolution]
    if ratio in {'16:9', '9:16'}:
        long_edge = 832 if edge == 480 else 1344
        width, height = long_edge, edge
        if ratio == '9:16': width, height = height, width
    else:
        aspect = _ratio(ratio)
        width, height = (edge * aspect, edge) if aspect >= 1 else (edge, edge / aspect)
        width, height = (32 * math.floor(value / 32 + .5) for value in (width, height))
    return f'{width}x{height}'


def apply_video_resolution(parameters, node, records, config, output):
    rules = input_rules(config)
    # Super resolution is per-request, so it survives the automatic-sizing path
    # that returns before a frame is resolved.
    # 超分是按请求生效的，因此在还没确定画幅就返回的自动尺寸路径里也要保留。
    if output.get('upscale'):
        parameters['upscale'] = True
    capabilities = rules.get('videoCapabilities')
    chosen = output.get('resolution')
    resolution = chosen if chosen is not None else (None if node.get('videoMode') == 'continuation'
                                                    else (capabilities or {}).get('defaultResolution'))
    # Only a user selection is checked against the declared duration grid: an
    # adapter may infer fractional seconds from the prompt or the source clip.
    # 只有用户显式选择的时长才按声明的时长网格校验：
    # 适配器可能从提示词或源片段推断出小数秒。
    seconds = output.get('durationSeconds')
    if resolution is None or str(resolution).strip().upper() in {'', 'AUTO', '自动'}:
        if seconds is not None: validate_video_duration(config, seconds, None)
        # Preserve the adapter's existing automatic sizing and recovery.
        return  # 保留适配器既有的自动尺寸与恢复行为。
    resolution = str(resolution).strip().upper()
    if node.get('videoMode') == 'continuation':
        # The source clip decides the canvas, so a chosen tier is ignored rather
        # than rejected; the request frame is overwritten with the source pixels.
        # 画布由源片段决定，所选档位被忽略而非拒绝；
        # 请求画幅会被源像素覆盖。
        if seconds is not None: validate_video_duration(config, seconds, None)
        return
    if resolution not in rules.get('outputResolutions', []):
        raise ProjectError('当前模型不支持所选分辨率，请重新选择；未提交生成。')
    if seconds is not None: validate_video_duration(config, seconds, resolution)
    sources = [records[key] for key in node['referenceIds'] if key in records]
    if (node.get('videoMode') == 'reference' and any(item['kind'] == 'image' for item in sources)
            and resolution not in rules.get('referenceImageResolutions', rules['outputResolutions'])):
        raise ProjectError('当前模型的图片参考模式不支持所选分辨率，请重新选择；未提交生成。')
    preset = next((item for item in (capabilities or {}).get('outputs', []) if item['resolution'] == resolution), {})
    if rules['profile'] in {'fasth3-omni', 'vllm-omni-h3', 'standard-video'}:
        # Explicit framing wins; infer it from references only for automatic fit.
        # 显式画幅优先；只有自动适配时才从参考素材推断。
        visuals = [item for kind in ('video', 'image') for item in sources
                   if item['kind'] == kind and item.get('width') and item.get('height')]
        ratio = output.get('aspectRatio')
        if ratio in {None, 'adaptive'} and visuals:
            aspect = visuals[0]['width'] / visuals[0]['height']
            ratio = min(RATIOS, key=lambda candidate: abs(math.log(_ratio(candidate) / aspect)))
        elif ratio in {None, 'adaptive'}:
            ratio = '16:9'
        if ratio not in RATIOS:
            raise ProjectError('当前模型不支持所选画幅比例；未提交生成。')
        if preset.get('width') and preset.get('height'):
            width, height = preset['width'], preset['height']
            if ratio == '9:16':
                width, height = height, width
            elif ratio != '16:9':
                aspect, edge = _ratio(ratio), height
                width, height = (edge * aspect, edge) if aspect >= 1 else (edge, edge / aspect)
                step = capabilities['dimensionStep']
                width, height = (step * math.floor(value / step + .5) for value in (width, height))
            parameters['size'] = f'{width}x{height}'
        elif rules['profile'] in {'fasth3-omni', 'vllm-omni-h3'} and resolution in {'480P', '768P'}:
            parameters['size'] = h3_output_size(resolution, ratio)
        else:
            raise ProjectError('该 /videos 输出规格尚未配置像素宽高，请在模型接入中补齐；未提交生成。')
    parameters['resolution'] = resolution.lower()

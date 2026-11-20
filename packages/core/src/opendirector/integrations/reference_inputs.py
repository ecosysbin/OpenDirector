"""Resolve local references once per request and encode only defined inputs."""
import base64
import hashlib
import json

from opendirector.domain.input_rules import MAX_REFERENCES, input_rules
from opendirector.domain.rule_facts import (
    FASTH3_MEDIA_REFERENCE_LIMIT, REFERENCE_MEDIA_SECONDS, TEXT_REFERENCE_BYTES,
)
from opendirector.domain.projects import ProjectError
from opendirector.domain.minimax_image import validate_minimax_image_reference
from opendirector.domain.references import project_content


def reference_text(data):
    if len(data) > TEXT_REFERENCE_BYTES:
        raise ProjectError('文本参考文件应不超过 64 KB，请缩短文件后重新添加。')
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ProjectError('文本参考文件需使用 UTF-8 编码。') from None


def image_type(data):
    return ('image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else
            'image/jpeg' if data.startswith(b'\xff\xd8\xff') else
            'image/gif' if data.startswith((b'GIF87a', b'GIF89a')) else
            'image/webp' if data[:4] == b'RIFF' and data[8:12] == b'WEBP' else None)


def references(store, owner, ids, overrides=None):
    ids = list(dict.fromkeys(map(str, ids)))
    if len(ids) > MAX_REFERENCES:
        raise ProjectError(f'参考数量超过系统上限 {MAX_REFERENCES}。')
    for asset_id in ids:
        data, meta = (overrides or {}).get(asset_id) or store.asset(owner, asset_id)
        fingerprint = meta.get('fingerprint', '')
        if fingerprint.startswith('sha256:') and fingerprint != 'sha256:' + hashlib.sha256(data).hexdigest():
            raise ProjectError(f'“{meta["name"]}”内容已变化，请重新核对参考。', 409)
        mime = meta['type']
        kind = meta.get('kind') or ('text' if mime.startswith('text/') or meta['name'].lower().endswith(('.txt', '.md')) else mime.split('/')[0])
        yield data, meta, kind


def data_uri(data, mime):
    return f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')


def request_size(payload, maximum):
    if len(json.dumps(payload, ensure_ascii=False).encode()) > maximum:
        raise ProjectError('正文和参考编码后的请求过大，请减少参考或缩小文件；当前输入已保留。', 413)


def reference_content(store, owner, message, *, manifest=None, overrides=None):
    agent_rules = input_rules({'kind': 'agent', 'url': '', 'modelId': ''})
    structured = message.get('content') is not None
    text = project_content(message['content'])[0] if structured else message['text']
    blocks = [{'type': 'text', 'text': text}] if text else []
    for index, (data, meta, kind) in enumerate(references(store, owner, [asset['id'] for asset in message['assets']], overrides), 1):
        entry = {'assetId': meta['id'], 'fingerprint': meta['fingerprint'], 'kind': kind,
                 'transmission': 'metadata' if kind in {'video', 'audio'} else 'content'}
        if manifest is not None:
            manifest.append(entry)
        if structured:
            blocks.append({'type': 'text', 'text': f'参考{index}：{meta["name"]}（资产 ID：{meta["id"]}）'})
        if kind == 'text':
            blocks.append({'type': 'text', 'text': f'参考文件：{meta["name"]}\n{reference_text(data)}'})
        elif kind in {'video', 'audio'}:
            info = {key: meta[key] for key in ('id', 'name', 'kind', 'durationMs', 'width', 'height', 'sampleRate', 'channels') if meta.get(key) is not None}
            info.setdefault('kind', kind)
            blocks.append({'type': 'text', 'text': '素材信息（未提供音画内容，不得推测或声称已观看/听取；是否可作生成参考由所选生成模型决定）：' + json.dumps(info, ensure_ascii=False)})
        else:
            mime = image_type(data)
            if kind != 'image' or mime is None:
                raise ProjectError('当前对话支持图片及文本内容，视频和音频仅传素材信息；此文件格式不支持引用。')
            if len(data) > agent_rules['imageBytes']:
                raise ProjectError('单张对话参考图片应不超过 32 MB。', 413)
            blocks.append({'type': 'image_url', 'image_url': {'url': data_uri(data, mime)}})
    content = blocks if message['assets'] else text
    request_size(content, agent_rules['requestBytes'])
    return content


def continuation_source(store, project_id, meta, config):
    """Remote job ID of an earlier clip this same H3 service generated; its latents stay server-side."""
    message = f'“{meta["name"]}”不能接续：需引用由当前 vLLM-Omni H3 服务生成并已完成的视频，上传或其他模型生成的视频不能接续。'
    if not meta.get('generationTaskId'):
        raise ProjectError(message, 409)
    try:
        source = store.generation_task(project_id, meta['generationTaskId'])
    except (OSError, ValueError, KeyError, ProjectError):
        raise ProjectError(message, 409) from None
    if (source['kind'] != 'video' or source['status'] != 'succeeded' or not source.get('externalTaskId')
            or source['baseURL'].rstrip('/') != config['url'].rstrip('/') or source['model'] != config['modelId']):
        raise ProjectError(message, 409)
    return source['externalTaskId']


def generation_canvas(width, height, config, upscaled=None):
    """The canvas H3 generated on, before any delivered upscaling.

    Continuing a clip reuses the service's cached latents, which belong to the
    generated canvas. A clip that asked for upscaling comes back multiplied by
    `scale`, so the tail's request frame has to undo that factor or H3 rejects
    the continuation with a canvas mismatch.

    H3 实际生成时的画布尺寸（未叠加交付侧超分）。

    接续复用服务缓存的 latent，而 latent 属于生成画布。若上游片段请求了超分，
    返回的宽高是被 `scale` 放大过的交付尺寸，接续请求必须把这个倍数还原回去，
    否则 H3 会以画布不匹配为由拒绝接续。
    """
    if not width or not height:
        return (width, height)
    settings = (((config.get('videoCapabilities') or {}).get('inference') or {}).get('upscale')) or {}
    scale = float(settings.get('scale') or 2.0)
    if scale <= 1:
        return (width, height)
    # 768P is the widest generated edge; anything beyond it is a delivered size.
    # 1344 是 768P 画布的最长边；超过它的尺寸只可能是超分后的交付尺寸。
    ceiling = 1344
    if upscaled is False:
        return (width, height)
    if upscaled or max(width, height) > ceiling:
        return (round(width / scale), round(height / scale))
    return (width, height)


def continuation_node_source(store, project_id, node_id, config):
    """Job ID of the clip a declared source node produced last.

    A chain is laid out before any clip exists, so the source is a node ID: the
    job ID is read from that node's newest finished clip when this clip runs.

    解析「续接自」声明的源节点，返回其最新一条已完成片段的（任务 ID、接续画布）。

    整条链可以在任何片段存在之前先排好，所以声明里只有节点 ID；
    真正的任务 ID 要等这一段开始运行时，才从源节点最新的已完成片段上读出来。
    """
    found = store.latest_clip_for_node(project_id, node_id, config)
    if not found or not found[0]:
        raise ProjectError('「续接自」的源节点还没有已完成的视频：请先生成上游片段再生成这一段，或改用引用素材接续。',
                           409, code='continuation_source_pending')
    job_id, width, height, upscaled = found
    return job_id, generation_canvas(width, height, config, upscaled)


def generation_references(store, task, config, *, complete=True, overrides=None):
    kind = task['kind']
    rules = input_rules({**config, 'kind': 'agent' if kind == 'text' else kind})
    mode = task.get('videoMode', 'first_frame')
    prompt = project_content(task['promptContent'], speech=kind == 'audio')[0] if task.get('promptContent') is not None else task['prompt']
    result = {'prompt': prompt, 'images': [], 'imageRoles': [], 'audios': [], 'videos': [], 'manifest': [], 'rules': rules,
              'continueFrom': None, 'continueSize': None, 'refineSource': None}
    if kind == 'text':
        result['content'] = reference_content(store, task['projectId'], {'text': task['prompt'], 'content': task.get('promptContent'), 'assets': [{'id': asset_id} for asset_id in task['referenceIds']]}, manifest=result['manifest'], overrides=overrides)
        request_size(result['content'], rules['requestBytes'])
        return result
    if kind == 'audio':
        if rules.get('referencesAllowed') is False and task['referenceIds']:
            raise ProjectError('MOSS 语音生成不支持参考素材，请直接填写朗读文本和音色 ID。', 422, code='moss_speech_reference')
        # Historical context remains recorded; it is never appended to speech.
        result['manifest'] = [{'assetId': meta['id'], 'fingerprint': meta['fingerprint'], 'kind': source_kind, 'transmission': 'context_only'}
                              for _, meta, source_kind in references(store, task['projectId'], task['referenceIds'], overrides)]
        return result
    if kind == 'video' and mode not in rules['videoModes']:
        raise ProjectError('当前模型不支持接续生成，请选择 vLLM-Omni MiniMax-H3。' if mode == 'continuation'
                           else '当前模型接口尚未接通素材参考模式，请选择该模型支持的生成方式。')
    declared = (task.get('parameters') or {}).get('continuationOf') or task.get('continuationOf')
    # The wired edge is the source of truth once it has a clip. The remembered
    # node id only stands in while that node is still empty, so an attached
    # continuation video must not be reported as a second source.
    # 连线一旦有片段就以连线为准。记住的节点 ID 只在该节点还空着时顶替，
    # 因此已附带的接续视频不能被当成第二个来源。
    attached_continuation = bool(
        kind == 'video' and mode == 'continuation'
        and any(source_kind == 'video'
                for _, _, source_kind in references(store, task['projectId'], task['referenceIds'], overrides)))
    if kind == 'video' and mode == 'continuation' and declared and not attached_continuation:
        # Declared source wins over attached references: the chain is stated by
        # node ID and resolved against that node's newest finished clip.
        # 声明的来源优先于附带的引用素材：链是按节点 ID 表达的，
        # 并按该节点最新的已完成片段解析。
        result['continueFrom'], result['continueSize'] = continuation_node_source(
            store, task['projectId'], declared, config)
    total_audio_ms, total_video_ms, media_bytes = 0, 0, 0
    source_metadata, text_context = [], []
    role_by_asset = {str(value['assetId']): value['role'] for value in task.get('referenceRoles') or []}
    slot_roles = set()
    for index, (data, meta, source_kind) in enumerate(references(store, task['projectId'], task['referenceIds'], overrides), 1):
        entry = {'assetId': meta['id'], 'fingerprint': meta['fingerprint'], 'kind': source_kind, 'transmission': 'content'}
        if source_kind == 'video' and kind == 'video' and mode == 'continuation':
            if result['continueFrom']:
                raise ProjectError('接续生成只能有一个来源：已通过「续接自」指定上游节点，请移除引用的视频素材。'
                                   if declared else '接续生成只能引用一段上一段视频。')
            result['continueFrom'] = continuation_source(store, task['projectId'], meta, config)
            result['continueSize'] = generation_canvas(meta.get('width'), meta.get('height'), config)
            # The service continues from its cached latents; the file is not uploaded.
            entry.update(role='continuation_source', transmission='latent')
            result['manifest'].append(entry)
            continue
        media_bytes += len(data)
        source_metadata.append((meta, source_kind))
        if source_kind == 'video' and not (kind == 'video' and mode == 'reference' and rules.get('maxVideos')):
            raise ProjectError(f'“{meta["name"]}”：当前模型不支持视频参考，请尝试使用其他模型的素材参考模式。')
        if source_kind == 'text':
            if rules['profile'] == 'moss-image':
                try:
                    layout = json.loads(prompt)
                except ValueError:
                    layout = None
                if isinstance(layout, dict):
                    raise ProjectError('MOSS 布局 JSON 不支持附加文本参考，请将要求写入 JSON 或移除文本参考。', 422, code='moss_layout_reference')
            result['prompt'] += f'\n\n参考{index}：{meta["name"]}\n{reference_text(data)}'
            text_context.append(f'\n\n参考{index}：{meta["name"]}\n{reference_text(data)}')
            entry['role'] = 'text'
        elif source_kind == 'image':
            mime = image_type(data)
            if mime is None:
                raise ProjectError(f'“{meta["name"]}”不是已支持的参考图片格式。')
            if len(data) > rules['imageBytes']:
                raise ProjectError(f'“{meta["name"]}”超过当前接口参考图片大小限制。', 413)
            width, height = meta.get('width'), meta.get('height')
            if rules['profile'] == 'seedance-2.0' and (width is None or height is None or not 300 <= width <= 6000 or not 300 <= height <= 6000 or not .4 <= width / height <= 2.5):
                raise ProjectError(f'“{meta["name"]}”不符合视频参考图片尺寸要求：边长 300—6000 像素，宽高比 0.4—2.5。')
            if rules['profile'] == 'seedream-5.0-pro' and (width is None or height is None or min(width, height) <= 14 or not 196 <= width * height <= 36000000 or not 1 / 16 <= width / height <= 16):
                raise ProjectError(f'“{meta["name"]}”不符合当前图片模型的参考尺寸要求。')
            if rules['profile'] == 'minimax-h3-image':
                validate_minimax_image_reference(mime, width, height)
            role = 'reference_image'
            if kind == 'video' and mode == 'first_frame':
                # Frame slots are per-slot: one image each, and the slot the
                # caller asked for is kept so Ark receives last_frame as such.
                # A plain 'reference' role is a first frame, matching what a
                # single image meant before last_frame existed; it never turns
                # into a last frame implicitly.
                requested = role_by_asset.get(str(meta['id']))
                role = requested if requested in {'first_frame', 'last_frame'} else 'first_frame'
                if role in slot_roles:
                    raise ProjectError('首尾帧每个槽位只能使用一张图片，请在画布上明确指定首帧和尾帧。')
                slot_roles.add(role)
            result['images'].append(data_uri(data, mime))
            result['imageRoles'].append(role)
            number = len(result['images'])
            result['prompt'] += f'\n参考{index}：{meta["name"]}，对应第{number}张参考图片（图片{number}，<Picture {number}>）。'
            entry.update(role=role, number=number)
        elif source_kind == 'video':
            duration = meta.get('durationMs')
            if meta['type'] not in {'video/mp4', 'video/quicktime'} or len(data) > rules['videoBytes']:
                raise ProjectError(f'参考视频需为不超过 {rules["videoBytes"] // (1024 * 1024)} MB 的 MP4 或 MOV。')
            if duration is None or not 0 < duration <= REFERENCE_MEDIA_SECONDS * 1000:
                raise ProjectError(f'参考视频需为不超过 {REFERENCE_MEDIA_SECONDS} 秒的有效视频。')
            total_video_ms += duration
            result['videos'].append(data_uri(data, meta['type']))
            if rules['profile'] == 'vllm-omni-h3-refine':
                # The draft is the whole input, not a <Video n> reference the
                # prompt describes: the service refines its latents directly, so
                # nothing is appended to the text condition.
                # 草稿片段就是全部输入，而不是提示词里要描述的 <Video n> 参考：
                # 服务直接重绘它的 latent，所以文本条件下不追写任何说明。
                result['refineSource'] = {'durationMs': duration, 'width': meta.get('width'), 'height': meta.get('height')}
                entry.update(role='refine_source', number=len(result['videos']))
            else:
                number = len(result['videos'])
                result['prompt'] += f'\n参考{index}：{meta["name"]}，对应 <Video {number}>。'
                entry.update(role='reference_video', number=number)
        elif source_kind == 'audio' and kind == 'video' and mode in {'reference', 'continuation'} and rules['maxAudios']:
            if meta['type'] not in {'audio/wav', 'audio/x-wav', 'audio/mpeg'} or len(data) > 15 * 1024 * 1024:
                raise ProjectError('参考音频需为不超过 15 MB 的 WAV 或 MP3。')
            duration = meta.get('durationMs')
            if duration is None or not 2000 <= duration <= REFERENCE_MEDIA_SECONDS * 1000:
                raise ProjectError(f'参考音频需为 2—{REFERENCE_MEDIA_SECONDS} 秒，请先检查文件时长。')
            total_audio_ms += duration
            result['audios'].append(data_uri(data, 'audio/wav' if meta['type'] == 'audio/x-wav' else meta['type']))
            number = len(result['audios'])
            result['prompt'] += f'\n参考{index}：{meta["name"]}，对应音频{number}（<Audio {number}>）。'
            entry.update(role='reference_audio', number=number)
        else:
            raise ProjectError(f'“{meta["name"]}”：当前模型和生成模式暂不支持此类参考，当前输入已保留。')
        result['manifest'].append(entry)
    # A first/last keyframe pair is two images; models that only take a
    # first frame stay at one.
    frame_limit = 2 if 'last_frame' in (rules.get('frameRoles') or ()) else 1
    maximum = frame_limit if kind == 'video' and mode == 'first_frame' else rules['maxImages']
    if result['images'] and not maximum:
        raise ProjectError('当前模型只按文字生成，不支持参考图片；请移除图片参考，或把外观要求写进提示词。当前输入已保留。')
    if len(result['images']) > maximum:
        raise ProjectError(f'当前生成模式最多使用 {maximum} 张参考图片；多图视频请使用素材参考模式。')
    if len(result['audios']) > rules['maxAudios'] or total_audio_ms > REFERENCE_MEDIA_SECONDS * 1000:
        raise ProjectError(f'参考音频最多 3 段，总时长不超过 {REFERENCE_MEDIA_SECONDS} 秒。')
    if complete and mode == 'continuation' and not result['continueFrom']:
        raise ProjectError('接续生成需要引用上一段由 vLLM-Omni H3 生成的视频。')
    if complete and rules['profile'] == 'vllm-omni-h3-refine' and not result['videos']:
        raise ProjectError('精修服务需要引用一段已完成的视频作为草稿：请先用 768P 生成一段片段，再把该片段引用到精修节点。',
                           422, code='refine_source_missing')
    if len(result['videos']) > rules.get('maxVideos', 0) or total_video_ms > REFERENCE_MEDIA_SECONDS * 1000:
        raise ProjectError('精修服务只能引用一段视频草稿。' if rules['profile'] == 'vllm-omni-h3-refine'
                           else f'视频参考最多 3 段，总时长不超过 {REFERENCE_MEDIA_SECONDS} 秒。')
    if rules['profile'] == 'fasth3-omni' and len(result['images']) + len(result['videos']) + len(result['audios']) > FASTH3_MEDIA_REFERENCE_LIMIT:
        raise ProjectError(f'FastH3-Omni 的图片、视频和音频参考合计最多 {FASTH3_MEDIA_REFERENCE_LIMIT} 项。')
    if media_bytes > 100 * 1024 * 1024:
        raise ProjectError('参考素材合计超过 100 MB，请缩小文件。', 413)
    if complete and result['audios'] and not (result['images'] or result['videos']) and mode != 'continuation':
        raise ProjectError('音频参考须搭配至少一张参考图片或一段参考视频。')
    if rules['profile'] == 'fasth3-omni':
        from opendirector.integrations.fasth3 import native_prompt
        result['prompt'] = native_prompt(task, source_metadata) + ''.join(text_context)
    if rules['profile'] == 'moss-image':
        from opendirector.domain.moss_image import validate_moss_image_prompt
        validate_moss_image_prompt(result['prompt'])
    if len(result['prompt']) > 20000:
        raise ProjectError('正文及文本参考超过当前生成长度限制，请缩短后重试。')
    request_size({key: result[key] for key in ('prompt', 'images', 'audios', 'videos')}, rules['requestBytes'] - 4096)
    return result

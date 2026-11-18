import asyncio
import base64
from email.parser import BytesParser
from email.policy import default
import io
import json
import shutil
import subprocess
from uuid import uuid4

import httpx
from PIL import Image
import pytest

from opendirector.application.creation_generation import CreationGeneration
from opendirector.application.creation_tool_assets import CreationToolAssets
from generation_helpers import new_project, prepared as prepare_command, tasks
from opendirector.domain.input_rules import input_rules
from opendirector.domain.projects import ProjectError
from opendirector.infrastructure.local_projects import LocalProjects
from opendirector.infrastructure.model_configs import ModelConfigs
from opendirector.integrations.providers import GenerationProviders
from opendirector.integrations.vllm_omni import VllmOmniGeneration, h3_frames
from opendirector.integrations.volcengine import VolcengineGeneration

H3 = {'protocol': 'vllm_omni', 'kind': 'video', 'url': 'http://gpu-box:8000/v1', 'modelId': 'MiniMaxAI/MiniMax-H3', 'apiKey': 'token'}


def png():
    out = io.BytesIO()
    Image.new('RGB', (16, 12)).save(out, 'PNG')
    return out.getvalue()


def data_uri(data, mime):
    return f'data:{mime};base64,' + base64.b64encode(data).decode()


def multipart(request):
    request.read()
    message = BytesParser(policy=default).parsebytes(
        b'Content-Type: ' + request.headers['content-type'].encode() + b'\r\n\r\n' + request.content)
    fields, files = {}, []
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if part.get_filename():
            files.append((name, part.get_filename(), part.get_content_type()))
        else:
            fields[name] = part.get_content()
    return fields, files


def test_h3_profile_follows_the_served_model_on_any_host():
    rules = input_rules(H3)
    assert rules['profile'] == 'vllm-omni-h3' and rules['videoModes'] == ['first_frame', 'reference', 'continuation']
    # The continuation source is cached server-side, never uploaded: no video references.
    assert rules['maxVideos'] == 0 and 'video' in rules['allowedKinds']
    assert input_rules({**H3, 'url': 'https://example.test/omni/v1', 'modelId': 'MiniMax-H3'})['profile'] == 'vllm-omni-h3'
    assert input_rules({**H3, 'modelId': 'MiniMaxAI/MiniMax-H3-Turbo'})['profile'] == 'standard-video'
    assert input_rules({**H3, 'protocol': 'volcengine'})['profile'] == 'compatible'
    assert input_rules({**H3, 'kind': 'image'})['profile'] == 'compatible'
    providers = GenerationProviders(None)
    assert providers._profile_adapter(H3) is providers.vllm_omni
    assert providers._profile_adapter({**H3, 'protocol': 'vllm_omni'}) is providers.vllm_omni
    # Other vLLM-Omni video models keep the FastH3 profile and the protocol adapters.
    fasth3 = {**H3, 'modelId': 'FastH3-Omni', 'protocol': 'vllm_omni'}
    assert input_rules(fasth3)['profile'] == 'fasth3-omni' and providers._profile_adapter(fasth3) is None
    assert providers._profile_adapter({**H3, 'url': 'https://ark.cn-beijing.volces.com/api/plan/v3', 'modelId': 'doubao-seedance-2.0'}) is None


@pytest.mark.parametrize('seconds,frames', [(5, 124), (1, 107), (4, 107), (10, 243), (60, 345)])
def test_lengths_snap_to_the_h3_frame_grid(seconds, frames):
    assert h3_frames(seconds) == frames


def prepared(images=(), audios=(), continue_from=None, size=None):
    return {'prompt': '画面', 'images': list(images), 'audios': list(audios), 'continueFrom': continue_from, 'continueSize': size}


@pytest.mark.parametrize('mode,refs,extra,files', [
    ('first_frame', prepared(), {'task': 't2va'}, []),
    ('first_frame', prepared([data_uri(png(), 'image/png')]), {'task': 'fl2va'}, [('input_reference', 'first_frame.png', 'image/png')]),
    ('reference', prepared([data_uri(png(), 'image/png')], [data_uri(b'ID3', 'audio/mpeg')]), {'task': 'ref2va'},
     [('input_references', 'image_1.png', 'image/png'), ('input_references', 'audio_1.mp3', 'audio/mpeg')]),
    ('continuation', prepared([data_uri(png(), 'image/png')], continue_from='video_gen_a', size=(1344, 768)),
     {'task': 'ref2va', 'continue_from': 'video_gen_a'}, [('input_references', 'image_1.png', 'image/png')]),
])
def test_request_form_per_mode(mode, refs, extra, files):
    task = {'videoMode': mode, 'parameters': {'duration': 5, 'ratio': '9:16'}}
    fields, uploads = VllmOmniGeneration(None).form(H3, task, refs)
    assert json.loads(fields['extra_params']) == extra
    assert fields['model'] == 'MiniMaxAI/MiniMax-H3' and fields['fps'] == '24'
    assert [(name, upload[0], upload[2]) for name, upload in uploads] == files
    assert fields['num_frames'] == '124' and 'quality' not in fields
    assert fields.get('size') == ('1344x768' if mode == 'continuation' else None)
    # Top-level, and derived from a fixed frame so the two never disagree.
    assert fields['aspect_ratio'] == ('16:9' if mode == 'continuation' else '9:16')


def test_distilled_lora_settings_reach_the_request(tmp_path):
    store, _, _ = new_project(tmp_path, '蒸馏参数')
    config = ModelConfigs(store.local.repository).save(
        'video', H3['url'], 'minimax-h3', 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer',
        video_capabilities={'outputs': [{'resolution': '768P', 'durations': list(range(4, 16)), 'width': 1344, 'height': 768}],
                            'defaultResolution': '768P', 'defaultDuration': 5, 'dimensionStep': 32,
                            'inference': {'flowShift': 6, 'audioFlowShift': 3,
                                          'lora': {'name': '8step_v1.0_768p', 'path': '/models/Turbo-LoRA/8step-v1.0', 'scale': 1.0}}})
    task = {'videoMode': 'first_frame', 'parameters': {'duration': 5, 'size': '1344x768', 'generateAudio': True}}
    fields, _ = VllmOmniGeneration(None).form(config, task, prepared())
    # Steps come from the LoRA's own step count; the adapter and flow shift travel with them.
    assert (fields['num_inference_steps'], fields['flow_shift']) == ('8', '6')
    assert json.loads(fields['lora']) == {'name': '8step_v1.0_768p', 'path': '/models/Turbo-LoRA/8step-v1.0', 'scale': 1.0}
    assert fields['size'] == '1344x768' and fields['aspect_ratio'] == '16:9' and fields['generate_sound'] == 'true'
    assert json.loads(fields['extra_params']) == {'task': 't2va', 'audio_flow_shift': 3.0}


def test_upscale_is_a_per_request_choice(tmp_path):
    store, _, _ = new_project(tmp_path, '超分参数')
    config = ModelConfigs(store.local.repository).save(
        'video', H3['url'], 'minimax-h3', 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer',
        video_capabilities={'outputs': [{'resolution': '768P', 'durations': list(range(4, 16)), 'width': 1344, 'height': 768}],
                            'defaultResolution': '768P', 'defaultDuration': 5, 'dimensionStep': 32,
                            'inference': {'flowShift': 6, 'audioFlowShift': 3,
                                          'lora': {'name': '8step_v1.0_768p', 'path': '/models/Turbo-LoRA/8step-v1.0'}}})
    task = {'videoMode': 'first_frame', 'parameters': {'duration': 5, 'size': '1344x768'}}
    fields, _ = VllmOmniGeneration(None).form(config, task, prepared())
    # A draft keeps the generated frame; nothing changes unless the node asks.
    assert 'enable_upscale' not in fields
    task = {'videoMode': 'first_frame', 'parameters': {'duration': 5, 'size': '1344x768', 'upscale': True}}
    fields, _ = VllmOmniGeneration(None).form(config, task, prepared())
    assert fields['enable_upscale'] == 'true'
    assert fields['upscale_model_path'] == '/models/Upscaler/realesrgan-x4plus/RealESRGAN_x4plus.pth'
    assert (fields['upscale_scale'], fields['upscale_native_scale'], fields['upscale_tile']) == ('2', '4', '512')
    # Sampling settings travel with it: the service applies both in one request.
    assert (fields['num_inference_steps'], fields['flow_shift']) == ('8', '6')


def test_upscale_weights_follow_the_model_config(tmp_path):
    store, _, _ = new_project(tmp_path, '超分权重')
    config = ModelConfigs(store.local.repository).save(
        'video', H3['url'], 'minimax-h3', 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer',
        video_capabilities={'outputs': [{'resolution': '768P', 'durations': [5], 'width': 1344, 'height': 768}],
                            'dimensionStep': 32,
                            'inference': {'upscale': {'modelPath': '/models/Up/RealESRGAN_x2plus.pth', 'scale': 2,
                                                      'nativeScale': 2, 'tile': 128}}})
    task = {'videoMode': 'first_frame', 'parameters': {'duration': 5, 'size': '1344x768', 'upscale': True}}
    fields, _ = VllmOmniGeneration(None).form(config, task, prepared())
    assert fields['upscale_model_path'] == '/models/Up/RealESRGAN_x2plus.pth'
    assert (fields['upscale_scale'], fields['upscale_native_scale'], fields['upscale_tile']) == ('2', '2', '128')


def test_step_count_must_match_the_configured_lora(tmp_path):
    store, _, _ = new_project(tmp_path, '步数校验')
    with pytest.raises(ProjectError, match='视频输出规格无效'):
        ModelConfigs(store.local.repository).save(
            'video', H3['url'], 'minimax-h3', 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer',
            video_capabilities={'outputs': [{'resolution': '768P', 'durations': [5], 'width': 1344, 'height': 768}],
                                'dimensionStep': 32,
                                'inference': {'steps': 20, 'lora': {'name': '8step_v1.0_768p', 'path': '/models/x'}}})


class FakeOmni:
    """The async /v1/videos job API with a per-server latent cache of finished jobs."""

    def __init__(self, video):
        self.video, self.jobs, self.posts, self.cached = video, {}, [], set()

    def __call__(self, request):
        assert request.headers['authorization'] == 'Bearer token'
        path = request.url.path
        if request.method == 'POST' and path == '/v1/videos':
            fields, files = multipart(request)
            extra = json.loads(fields['extra_params'])
            self.posts.append((fields, files))
            if 'continue_from' in extra and extra['continue_from'] not in self.cached:
                return httpx.Response(400, json={'detail': f"continue_from video {extra['continue_from']!r} is unknown or has not completed."})
            job = f'video_gen_{len(self.posts)}'
            self.jobs[job] = 0
            return httpx.Response(200, json={'id': job, 'status': 'queued'})
        job = path.split('/')[3]
        if path.endswith('/content'):
            return httpx.Response(200, content=self.video)
        if request.method == 'DELETE':
            return httpx.Response(200, json={'id': job, 'deleted': True})
        self.jobs[job] += 1
        if self.jobs[job] < 2:
            return httpx.Response(200, json={'id': job, 'status': 'in_progress'})
        self.cached.add(job)
        return httpx.Response(200, json={'id': job, 'status': 'completed'})


@pytest.fixture
def project(tmp_path):
    store, pid, session = new_project(tmp_path, '接续测试')
    return store, {'id': str(pid)}, session, CreationToolAssets(store)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='media fixtures require ffmpeg')
def test_canvas_continues_a_previous_h3_clip_by_its_job_id(project, tmp_path):
    store, value, session, canvas = project
    clip = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x48:d=0.5', '-f', 'lavfi', '-i', 'sine=d=0.5',
                    '-shortest', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(clip)], check=True)
    server = FakeOmni(clip.read_bytes())
    configs = ModelConfigs(store.local.repository)
    config = configs.save('video', H3['url'], H3['modelId'], 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer')

    async def generate(inputs):
        parameters = dict(videoMode=inputs['videoMode'])
        task, pinned, private = prepare_command(manager, (store, value['id'], session), config,
            prompt=inputs['prompt'], parameters=parameters, references=inputs['referenceIds'])
        manager._launch(task, pinned, private)
        await asyncio.gather(*(job.future for job in manager.jobs.values()))
        return tasks(manager, value['id'])[-1]

    async def run():
        nonlocal manager
        manager = CreationGeneration(store, configs)
        manager.provider = GenerationProviders(manager.store, httpx.MockTransport(server), poll_interval=0)
        first = await generate({'prompt': '灯塔在黄昏亮起', 'referenceIds': [], 'videoMode': 'first_frame'})
        assert first['status'] == 'succeeded' and first['externalTaskId'] == 'video_gen_1', (first['error'], first['errorDetails'])
        assert json.loads(server.posts[0][0]['extra_params']) == {'task': 't2va'}
        assert server.posts[0][0]['aspect_ratio'] == '16:9'

        second = await generate({'prompt': '光束扫过海面', 'referenceIds': [first['assetId']], 'videoMode': 'continuation'})
        assert second['status'] == 'succeeded' and second['videoMode'] == 'continuation', (second['error'], second['errorDetails'])
        fields, files = server.posts[1]
        assert json.loads(fields['extra_params']) == {'task': 't2va', 'continue_from': 'video_gen_1'}
        assert (fields['size'], fields['num_frames']) == ('64x48', '124') and files == []
        reference = store.snapshot(value['id']).jobs[-1].inputSnapshot.references[0]
        assert reference.role == 'continuation' and reference.transmission == 'latent'
        assert str(reference.asset.assetId) == first['assetId']

        # The server lost its cache (e.g. restarted): the failure is explicit, not a silent new clip.
        server.cached.clear()
        third = await generate({'prompt': '海浪拍岸', 'referenceIds': [second['assetId']], 'videoMode': 'continuation'})
        assert third['status'] == 'failed' and '接续缓存' in third['errorDetails']['message']

        uploaded = canvas.import_file(value['id'], str(uuid4()), 'upload.mp4', clip.read_bytes(), session)
        with pytest.raises(ProjectError, match='不能接续'):
            await generate({'prompt': '接着上传的视频', 'referenceIds': [uploaded['id']], 'videoMode': 'continuation'})
        with pytest.raises(ProjectError, match='视频续写'):
            await generate({'prompt': '没有上一段', 'referenceIds': [], 'videoMode': 'continuation'})
        await manager.close()

    manager = None
    asyncio.run(run())


def test_other_models_keep_rejecting_video_references(project):
    store, value, _, _ = project
    seedance = {'kind': 'video', 'url': 'https://ark.cn-beijing.volces.com/api/plan/v3', 'modelId': 'doubao-seedance-2.0', 'apiKey': 'k'}
    with pytest.raises(ProjectError, match='不支持接续生成'):
        VolcengineGeneration(store).prepare_inputs(value['id'], {'kind': 'video', 'prompt': 'p', 'promptContent': None,
                                                                'referenceIds': [], 'videoMode': 'continuation'}, seedance)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='media fixtures require ffmpeg')
def test_requested_length_and_canvas_reach_the_h3_request(project, tmp_path):
    store, value, session, _ = project
    clip = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=green:s=64x48:d=0.5', '-pix_fmt', 'yuv420p', str(clip)], check=True)
    server = FakeOmni(clip.read_bytes())
    configs = ModelConfigs(store.local.repository)
    config = configs.save('video', H3['url'], H3['modelId'], 'token', connection_type='self_hosted', protocol='vllm_omni', auth_mode='bearer')
    async def run():
        manager = CreationGeneration(store, configs)
        manager.provider = GenerationProviders(manager.store, httpx.MockTransport(server), poll_interval=0)
        task, pinned, private = prepare_command(manager, (store, value['id'], session), config, prompt='雨夜街头',
            parameters=dict(videoMode='first_frame', durationSeconds=10, aspectRatio='9:16'))
        manager._launch(task, pinned, private)
        await asyncio.gather(*(job.future for job in manager.jobs.values()))
        done = tasks(manager, value['id'])[-1]
        assert done['status'] == 'succeeded', done['errorDetails']
        assert (done['parameters']['duration'], done['parameters']['ratio']) == (10, '9:16')
        intent = store.snapshot(value['id']).generationSpecs[-1]
        assert intent.parameters.durationSeconds == 10 and intent.prompt == '雨夜街头'
        fields, _ = server.posts[0]
        assert fields['num_frames'] == str(h3_frames(10))
        assert fields['size'] == '480x832' and json.loads(fields['extra_params']) == {'task': 't2va'}
        await manager.close()
    asyncio.run(run())


REFINE = {**H3, 'modelId': 'minimax-h3-regenerate-2k', 'videoAdapter': 'minimax_h3',
          'videoCapabilities': {'outputs': [{'resolution': '2K', 'durations': [5]}], 'defaultResolution': '2K',
                                'defaultDuration': 5, 'dimensionStep': 32, 'inference': {'regenerate': True}}}


def test_refine_profile_takes_one_draft_and_nothing_else():
    rules = input_rules(REFINE)
    assert rules['profile'] == 'vllm-omni-h3-refine'
    # The draft is the only input, and the service is never asked for a new clip.
    assert (rules['maxVideos'], rules['maxImages'], rules['maxAudios']) == (1, 0, 0)
    assert rules['videoModes'] == ['reference'] and rules['allowedKinds'] == ['text', 'video']
    assert input_rules({**REFINE, 'videoCapabilities': {'outputs': [{'resolution': '2K', 'durations': [5]}],
                                                        'defaultResolution': '2K', 'dimensionStep': 32}})['profile'] == 'vllm-omni-h3'
    assert input_rules({**REFINE, 'modelId': 'MiniMaxAI/MiniMax-H3', 'videoAdapter': 'auto',
                        'videoCapabilities': None})['profile'] == 'vllm-omni-h3'


def test_refine_form_carries_the_draft_and_nothing_the_service_owns():
    prepared = {'prompt': '蘑菇屋前的兔子', 'images': [], 'audios': [], 'videos': [data_uri(b'ID3', 'video/mp4')],
                'refineSource': {'durationMs': 5042, 'width': 1344, 'height': 768},
                'continueFrom': None, 'continueSize': None, 'manifest': [], 'rules': {}}
    task = {'videoMode': 'reference', 'parameters': {'duration': 5, 'ratio': '16:9'}}
    fields, uploads = VllmOmniGeneration(None).form(REFINE, task, prepared)
    # seconds must travel: without it the service falls back to its 8-second tier
    # and every draft fails the audio-latent check. num_frames stays out so the
    # draft decides the grid, and the delivered frame is the draft's, doubled.
    assert (fields['seconds'], fields['aspect_ratio']) == ('5', '16:9')
    assert 'num_frames' not in fields and 'size' not in fields
    assert json.loads(fields['extra_params']) == {'task': 't2va', 'regenerate': True}
    assert [(name, upload[0]) for name, upload in uploads] == [('source_video', 'draft.mp4')]
    for owned in ('num_inference_steps', 'flow_shift', 'lora', 'enable_upscale'):
        assert owned not in fields


def test_refine_form_requires_a_draft():
    task = {'videoMode': 'reference', 'parameters': {'duration': 5}}
    with pytest.raises(ProjectError, match='精修服务需要一段已完成的视频草稿'):
        VllmOmniGeneration(None).form(REFINE, task, {'prompt': '', 'images': [], 'audios': [], 'videos': [],
                                                     'refineSource': None, 'continueFrom': None, 'continueSize': None,
                                                     'manifest': [], 'rules': {}})


@pytest.mark.parametrize('inference,match', [
    ({'regenerate': True, 'lora': {'name': '8step_v1.0_768p', 'path': '/models/Turbo-LoRA/8step-v1.0'}}, '视频输出规格无效'),
    ({'regenerate': True, 'steps': 3}, '视频输出规格无效'),
    ({'regenerate': True, 'upscale': {'modelPath': '/models/Up/x2plus.pth', 'scale': 2, 'nativeScale': 2}}, '视频输出规格无效'),
])
def test_refine_config_rejects_settings_the_service_owns(tmp_path, inference, match):
    store, _, _ = new_project(tmp_path, '精修配置')
    with pytest.raises(ProjectError, match=match):
        ModelConfigs(store.local.repository).save(
            'video', H3['url'], 'minimax-h3-regenerate-2k', 'token', connection_type='self_hosted',
            protocol='vllm_omni', auth_mode='bearer', video_adapter='minimax_h3',
            video_capabilities={'outputs': [{'resolution': '2K', 'durations': [5]}], 'defaultResolution': '2K',
                                'defaultDuration': 5, 'dimensionStep': 32, 'inference': inference})

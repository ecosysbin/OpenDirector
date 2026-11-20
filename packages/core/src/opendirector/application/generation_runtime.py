"""服务商执行、恢复决策与并发控制，不涉及画布存储。"""
import asyncio
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import anyio
from opendirector.domain.projects import ProjectError
from opendirector.domain.generation import GenerationError, GenerationFailure
from opendirector.ports.gateway import gateway
from opendirector.integrations.http_settings import MODEL_JOB_CONTEXT
from opendirector.integrations.fasth3 import finalize_video_edit
from opendirector.integrations.generation_errors import transport_failure

def _self_hosted_video(task):
    profile = (task.get('referenceInputs') or {}).get('profile', '')
    return task['kind'] == 'video' and (task.get('protocol') == 'vllm_omni' or profile.startswith('vllm-omni-h3'))


def _execution_timeout(task):
    return 4 * 3600 if _self_hosted_video(task) else 1800


@dataclass
class GenerationJob:
    future: asyncio.Task
    project_id: str
    directory: str


class GenerationRuntime(ABC):
    def __init__(self):
        self.jobs: dict[str, GenerationJob] = {}
        self.accepting: dict[tuple[str, str], asyncio.Task] = {}
        self.cancelled = set()
        self.committing = set()
        self.limit = asyncio.Semaphore(2)
        self.remote_limits: dict[str, asyncio.Lock] = {}

    @abstractmethod
    def checkpoint(self, project_id, task, pinned):
        """通过项目的提交边界持久化执行事实。"""

    @abstractmethod
    def _result_asset(self, task, directory):
        """返回已提交且校验通过的结果（若存在）。"""

    @abstractmethod
    def _commit_output(self, task, pinned, suffix, data):
        """提交输出，并套用项目的采用策略。"""

    def _launch(self, task, pinned, config, mode='submit'):
        old = self.jobs.get(task['id'])
        if old and not old.future.done():
            return
        future = asyncio.create_task(self._run(dict(task), pinned, config, mode))
        self.jobs[task["id"]] = GenerationJob(future, task["projectId"], pinned)
        def done(completed):
            if self.jobs.get(task['id']) and self.jobs[task['id']].future is completed:
                self.jobs.pop(task['id'], None)
            if not completed.cancelled():
                completed.exception()
        future.add_done_callback(done)


    @staticmethod
    def _technical_check(asset):
        keys = ('kind', 'type', 'size', 'fingerprint', 'width', 'height', 'durationMs', 'frameRate', 'hasAudio', 'sampleRate', 'channels')
        return {'status': 'passed', 'facts': {key: asset[key] for key in keys if asset.get(key) is not None},
                'message': '文件已通过格式和可读取性检查；创意效果、人物一致性及内容质量仍需人工确认。', 'humanReviewRequired': True}


    def recovery(self, task, directory):
        """启动恢复与同一任务画布恢复共用的一张决策表。"""
        if task['status'] == 'succeeded':
            return 'done'
        if self._result_asset(task, directory):
            return 'local'
        cache = gateway().results(directory, task['id'])
        metadata = cache.metadata()
        if metadata and gateway().child(cache.directory, 'output').is_file():
            return 'local'
        details = task.get('errorDetails') or {}
        if details.get('providerStatus') in {'failed', 'cancelled', 'expired'}:
            return 'retry'  # 原远端任务已确认进入终态。
        # 重新查询视频任务，刷新可能已过期的结果定位。
        if task['kind'] == 'video' and task.get('externalTaskId'):
            return 'remote'
        if metadata and metadata.get('url'):
            return 'download'
        if task.get('phase') == 'queued':
            return 'submit'  # 调用任何服务商之前已确认落盘的检查点。
        if (details.get('stage') == 'submit' and details.get('httpStatus') in {400, 401, 402, 403, 404, 413, 422, 429}
                and details.get('category') not in {'submission_unknown', 'local', 'protocol'}):
            return 'retry'  # 显式接续可以创建一次新的尝试。
        return 'review'


    async def _run(self, task, pinned, config, mode='submit'):
        stage = 'save'
        cache = gateway().results(pinned, task['id'])
        trace_token = MODEL_JOB_CONTEXT.set((str(task['projectId']), str(task['id'])))
        async def save():
            # 取消 to_thread 并不会停止底层写入。保存终态之前先把它读完，
            # 避免旧的 running 回执覆盖之后的取消结果。
            pending = asyncio.create_task(asyncio.to_thread(self.checkpoint, task["projectId"], dict(task), pinned))
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                await pending
                raise

        async def submitted(external_id):
            nonlocal stage
            task["externalTaskId"] = external_id
            task['phase'] = 'poll'
            stage = 'save'
            await save()
            stage = 'submit'

        async def result_ready(suffix, url):
            nonlocal stage
            stage = 'save'
            task['phase'] = 'download'
            await asyncio.to_thread(cache.locator, suffix, url)
            await save()
            stage = 'download'

        try:
            remote_limit = (self.remote_limits.setdefault(task['baseURL'].rstrip('/'), asyncio.Lock())
                            if _self_hosted_video(task) else nullcontext())
            # 在全局并发限制与执行超时之外等待；排队中的任务仍是
            # 远端提交之前已确认的检查点。
            async with remote_limit, self.limit, asyncio.timeout(_execution_timeout(task)):
                task.update(status="running", resumeAllowed=False, error=None, errorDetails=None)
                await save()
                asset = await asyncio.to_thread(self._result_asset, task, pinned)
                cached = await asyncio.to_thread(cache.data) if asset is None else None
                if asset is None and cached is None:
                    if mode == 'local':
                        raise ProjectError('本地结果缺失，请保留任务并核对；不会重新生成。', 409)
                    if mode == 'download':
                        stage = 'download'
                        metadata = cache.metadata()
                        suffix, data = await self.provider.resume_result(config, metadata['suffix'], metadata['url'])
                    else:
                        stage = 'poll' if task.get('externalTaskId') else 'submit'
                        task['phase'] = stage
                        await save()  # 进入任何服务商调用之前先标记已提交。
                        if hasattr(self.provider, 'resume_result'):
                            suffix, data = await self.provider.generate(config, task, submitted, result_ready=result_ready)
                        else:
                            suffix, data = await self.provider.generate(config, task, submitted)
                    suffix, data = await asyncio.to_thread(finalize_video_edit, self.store, task, suffix, data)
                elif cached:
                    suffix, data = cached
                # 在写目录/链接之前先保住完整字节；取消会等待本地提交，
                # 恢复则校正未完成的提交。
                with anyio.CancelScope(shield=True):
                    if task["id"] in self.cancelled:
                        raise asyncio.CancelledError()
                    self.committing.add(task["id"])
                    stage = 'save'
                    if asset is None and cached is None:
                        task['phase'] = 'save'
                        await asyncio.to_thread(cache.save, suffix, data)
                        await save()
                    try:
                        asset = await asyncio.to_thread(self._commit_output, task, pinned, suffix if asset is None else None, data if asset is None else None)
                    except BaseException:
                        if task.get('phase') == 'adopt':
                            stage = 'adopt'
                        raise
                    task.update(status="succeeded", assetId=asset["id"], phase='complete', recovery='done')
        except asyncio.CancelledError:
            task["status"] = "cancelled" if task["id"] in self.cancelled else "interrupted"
            task['resumeAllowed'] = False
            task["error"] = "已停止本地生成；已提交任务和费用需在服务控制台核对。" if task["status"] == "cancelled" else "服务中断；有供应商任务 ID 的视频将在重启后查询，其他任务不会自动重复提交。"
            if task["status"] == "cancelled" and task["externalTaskId"]:
                try:
                    cancel_config = config or self.configs.resolve(task['modelConfigId'], task['kind'])
                    if cancel_config['url'] != task['baseURL'] or cancel_config['modelId'] != task['model'] or task.get('protocol') and cancel_config.get('protocol') != task['protocol']:
                        raise ProjectError('原模型配置已变化，供应商取消未确认。', 409)
                    await self.provider.cancel(cancel_config, task["externalTaskId"])
                    task["error"] = "已请求取消供应商任务，实际取消与费用以服务控制台为准。"
                except (ProjectError, OSError) as failure:
                    task["error"] = "本地已停止，供应商取消未确认，请在控制台核对任务。"
                    # 取消确认与本地停止是两回事；
                    # 绝不推断已退款或远端取消成功。
                    task['errorDetails'] = (failure.details if isinstance(failure, GenerationError) else GenerationFailure(
                        category='temporary', stage='cancel', message='供应商取消未确认。',
                        suggestion='请在控制台核对原任务、取消状态及费用。')).model_dump(mode='json')
        except (ProjectError, OSError, TimeoutError) as failure:
            task["status"] = "interrupted" if isinstance(failure, TimeoutError) or getattr(failure, "status", None) in {503, 504} else "failed"
            task["error"] = (str(failure) if isinstance(failure, ProjectError) else
                "等待超过允许时长，本地已停止查询；已提交的云端任务未自动取消，请恢复原任务继续查询，或取消云端任务。"
                if isinstance(failure, TimeoutError) and task.get('externalTaskId') and stage not in {'save', 'adopt'} else
                "生成或本地保存失败，请检查服务和磁盘；不会自动重新提交。")
            if isinstance(failure, GenerationError):
                details = failure.details
                if details.stage == 'check':
                    task['technicalCheck'] = {'status': 'failed', 'facts': {}, 'message': details.message, 'humanReviewRequired': True}
            elif isinstance(failure, TimeoutError) and stage not in {'save', 'adopt'}:
                details = transport_failure('poll' if task.get('externalTaskId') else stage, timeout=True).details
                if task.get('externalTaskId'):
                    details = details.model_copy(update={'message': task['error']})
            else:
                details = GenerationFailure(category='local' if stage in {'save', 'adopt'} else 'provider', stage=stage,
                    message=task['error'], suggestion='请检查本地项目、磁盘和采用状态，保留已有结果。' if stage in {'save', 'adopt'} else '请核对原任务和模型服务后再处理。')
            task['errorDetails'] = details.model_dump(mode='json')
        except Exception:
            # 不得泄露服务商响应、带签名的下载 URL 或凭据。
            task.update(status="failed", error="生成结果处理失败，请检查服务协议；不会自动重新提交。")
            task['errorDetails'] = GenerationFailure(category='local' if stage in {'save', 'adopt'} else 'protocol', stage=stage,
                message=task['error'], suggestion='请检查本地项目和保存结果。' if stage in {'save', 'adopt'} else '请核对原任务和接口协议。').model_dump(mode='json')
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    if task['status'] != 'succeeded':
                        try:
                            task['recovery'] = self.recovery(task, pinned)
                        except (ProjectError, OSError):
                            task['recovery'] = 'review'
                        task['resumeAllowed'] = task['recovery'] in {'submit', 'remote', 'download', 'local'}
                    await save()
                    if task['status'] == 'succeeded':
                        await asyncio.to_thread(cache.clear)
                except (OSError, ProjectError):
                    pass  # 原任务回执仍保留，供启动时核对。
            MODEL_JOB_CONTEXT.reset(trace_token)
            self.cancelled.discard(task["id"])
            self.committing.discard(task["id"])

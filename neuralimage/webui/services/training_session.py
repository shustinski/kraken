from __future__ import annotations

import base64
import io
import logging
import threading
import time
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from application.dto import MainWindowState, SettingsState, clone_main_window_state
from application.services import ActiveTaskMutationError, ProcessingSession, build_processing_start_error_message
from bootstrap.composition_root import create_web_presenter
from lib.logging_policy import MAX_LOG_MESSAGES, should_forward_log_event
from lib.message_bus import MessageBus
from presenter.web_presenter import WebPresenter

_LOG = logging.getLogger(__name__)


def _coerce_preview_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, 'detach') and hasattr(value, 'cpu') and hasattr(value, 'numpy'):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception:
            return None
    if array.size == 0:
        return None
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in {2, 3}:
        return None
    if array.ndim == 3 and array.shape[2] > 3:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(array)) <= 1.0 and float(np.min(array)) >= 0.0:
            array = array * 255.0
    else:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8, copy=False))


def _to_png_data_url(value: Any) -> str | None:
    array = _coerce_preview_array(value)
    if array is None:
        return None
    try:
        if array.ndim == 2:
            image = Image.fromarray(array, mode='L')
        elif array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array, mode='RGB')
        else:
            return None
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f'data:image/png;base64,{encoded}'
    except Exception:
        _LOG.exception('Failed to encode preview image for WebUI')
        return None


def _empty_preview_state() -> dict[str, Any]:
    return {
        'mode': 'train',
        'sample_name': '',
        'image_url': None,
        'label_url': None,
        'output_url': None,
    }


def _serialize_progress(progress: dict[str, int]) -> dict[str, int | str]:
    current = int(progress.get('current', 0))
    total = int(progress.get('total', 0))
    if total <= 0:
        return {
            'current': 0,
            'total': 0,
            'percent': 0,
            'text': '0%',
        }
    percent = max(0, min(100, int((current / total) * 100)))
    return {
        'current': current,
        'total': total,
        'percent': percent,
        'text': f'{percent}% ({current}/{total})',
    }


class TrainingSessionService:
    def __init__(self, presenter: WebPresenter) -> None:
        self._lock = threading.RLock()
        self._status = 'idle'
        self._events: list[dict[str, Any]] = []
        self._next_event_id = 1

        self._train_epoch: list[dict[str, float]] = []
        self._val_epoch: list[dict[str, float]] = []
        self._batch_by_epoch: dict[int, list[dict[str, float]]] = {}
        self._system_memory: dict[str, float] = {}
        self._validation_quality: dict[str, float] = {}
        self._train_perf: dict[str, float] = {}
        self._train_epoch_progress: dict[str, int] = {}
        self._train_batch_progress: dict[str, int] = {}
        self._recognition_progress: dict[str, int] = {}
        self._preview: dict[str, Any] = _empty_preview_state()
        self._train_speed_batches_per_sec: float | None = None
        self._recognition_speed_images_per_sec: float | None = None
        self._recognition_started_at: float | None = None
        self._recognition_last_current = 0
        self._recognition_last_total = 0

        self._processing_session = ProcessingSession()
        self._runner_thread: threading.Thread | None = None
        self._handler = None
        self._bus = MessageBus()
        self._bus.subscribe('logging', self._on_logging)
        self._bus.subscribe('training', self._on_training)
        self._bus.subscribe('metrics', self._on_metrics)

        self._presenter = presenter

    def _append_event(self, topic: str, message: str) -> None:
        with self._lock:
            event = {
                'id': self._next_event_id,
                'topic': topic,
                'message': str(message),
                'timestamp': time.time(),
            }
            self._next_event_id += 1
            self._events.append(event)
            if len(self._events) > MAX_LOG_MESSAGES:
                self._events[:] = self._events[-MAX_LOG_MESSAGES:]

    def _on_logging(self, payload: Any) -> None:
        if not should_forward_log_event('logging', payload):
            return
        self._append_event('logging', str(payload))

    def _on_training(self, payload: Any) -> None:
        if not should_forward_log_event('training', payload):
            return
        self._append_event('training', str(payload))

    def _update_recognition_speed(self, current: int, total: int) -> None:
        if total <= 0:
            self._recognition_speed_images_per_sec = None
            self._recognition_started_at = None
            self._recognition_last_current = 0
            self._recognition_last_total = 0
            return

        now = time.perf_counter()
        run_restarted = (
            self._recognition_started_at is None
            or total != self._recognition_last_total
            or current < self._recognition_last_current
            or current == 0
        )
        if run_restarted:
            self._recognition_started_at = now
            self._recognition_speed_images_per_sec = None
        elif self._recognition_started_at is not None and current > 0:
            elapsed_seconds = max(1e-6, now - self._recognition_started_at)
            self._recognition_speed_images_per_sec = current / elapsed_seconds

        self._recognition_last_current = current
        self._recognition_last_total = total

    def _on_metrics(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        metric_type = payload.get('type')
        epoch = int(payload.get('epoch', 0))
        loss = float(payload.get('loss', 0.0))

        with self._lock:
            if metric_type == 'train_epoch':
                self._train_epoch.append({'epoch': epoch, 'loss': loss})
                return
            if metric_type == 'val_epoch':
                val_point: dict[str, float] = {'epoch': epoch, 'loss': loss}
                if 'iou' in payload:
                    val_point['iou'] = float(payload.get('iou', 0.0))
                if 'dice' in payload:
                    val_point['dice'] = float(payload.get('dice', 0.0))
                if 'f1' in payload:
                    val_point['f1'] = float(payload.get('f1', 0.0))
                self._val_epoch.append(val_point)
                quality: dict[str, float] = {'epoch': epoch}
                if 'iou' in val_point:
                    quality['iou'] = float(val_point['iou'])
                if 'dice' in val_point:
                    quality['dice'] = float(val_point['dice'])
                if 'f1' in val_point:
                    quality['f1'] = float(val_point['f1'])
                self._validation_quality = quality
                return
            if metric_type == 'train_batch':
                batch_index = float(payload.get('batch_index', 0.0))
                points = self._batch_by_epoch.setdefault(epoch, [])
                points.append({'batch_index': batch_index, 'loss': loss})
                return
            if metric_type in {'train_epoch_progress', 'train_batch_progress', 'recognition_progress'}:
                progress = {
                    'current': int(payload.get('current', 0)),
                    'total': int(payload.get('total', 0)),
                }
                if metric_type == 'train_epoch_progress':
                    self._train_epoch_progress = progress
                    return
                if metric_type == 'train_batch_progress':
                    self._train_batch_progress = progress
                    return
                self._recognition_progress = progress
                self._update_recognition_speed(progress['current'], progress['total'])
                return
            if metric_type == 'train_perf':
                self._train_perf = {
                    'epoch': float(epoch),
                    'batch_index': float(payload.get('batch_index', 0.0)),
                    'data_wait_ms': float(payload.get('data_wait_ms', 0.0)),
                    'augmentation_ms': float(payload.get('augmentation_ms', 0.0)),
                    'forward_ms': float(payload.get('forward_ms', 0.0)),
                    'backward_ms': float(payload.get('backward_ms', 0.0)),
                    'optimizer_ms': float(payload.get('optimizer_ms', 0.0)),
                    'total_ms': float(payload.get('total_ms', 0.0)),
                }
                total_ms = float(self._train_perf.get('total_ms', 0.0))
                self._train_speed_batches_per_sec = 1000.0 / total_ms if total_ms > 0.0 else None
                return
            if metric_type == 'train_perf_epoch':
                self._train_perf = {
                    'epoch': float(epoch),
                    'data_wait_ms': float(payload.get('data_wait_ms', 0.0)),
                    'augmentation_ms': float(payload.get('augmentation_ms', 0.0)),
                    'forward_ms': float(payload.get('forward_ms', 0.0)),
                    'backward_ms': float(payload.get('backward_ms', 0.0)),
                    'optimizer_ms': float(payload.get('optimizer_ms', 0.0)),
                    'total_ms': float(payload.get('total_ms', 0.0)),
                }
                total_ms = float(self._train_perf.get('total_ms', 0.0))
                self._train_speed_batches_per_sec = 1000.0 / total_ms if total_ms > 0.0 else None
                return
            if metric_type == 'system_memory':
                data: dict[str, float] = {}
                if 'ram_mb' in payload:
                    data['ram_mb'] = float(payload.get('ram_mb', 0.0))
                if 'vram_allocated_mb' in payload:
                    data['vram_allocated_mb'] = float(payload.get('vram_allocated_mb', 0.0))
                if 'vram_reserved_mb' in payload:
                    data['vram_reserved_mb'] = float(payload.get('vram_reserved_mb', 0.0))
                self._system_memory = data
                return
            if metric_type == 'train_batch_preview':
                self._preview = {
                    'mode': 'train',
                    'sample_name': str(payload.get('sample_name', payload.get('frame_name', ''))).strip(),
                    'image_url': _to_png_data_url(payload.get('image')),
                    'label_url': _to_png_data_url(payload.get('label')),
                    'output_url': _to_png_data_url(payload.get('outputs', payload.get('output'))),
                }
                return
            if metric_type == 'recognition_preview':
                self._preview = {
                    'mode': 'recognition',
                    'sample_name': str(payload.get('sample_name', payload.get('frame_name', ''))).strip(),
                    'image_url': _to_png_data_url(payload.get('image')),
                    'label_url': None,
                    'output_url': _to_png_data_url(payload.get('outputs', payload.get('output', payload.get('result')))),
                }

    def _clear_runtime_metrics(self) -> None:
        with self._lock:
            self._train_epoch.clear()
            self._val_epoch.clear()
            self._batch_by_epoch.clear()
            self._system_memory.clear()
            self._validation_quality.clear()
            self._train_perf.clear()
            self._train_epoch_progress.clear()
            self._train_batch_progress.clear()
            self._recognition_progress.clear()
            self._preview = _empty_preview_state()
            self._train_speed_batches_per_sec = None
            self._recognition_speed_images_per_sec = None
            self._recognition_started_at = None
            self._recognition_last_current = 0
            self._recognition_last_total = 0

    def _run_handler(self, task_id: int) -> None:
        try:
            handler = self._handler
            if handler is not None:
                handler.start()
        except Exception as error:
            _LOG.exception('TrainingSessionService handler execution failed')
            self._append_event('error', f'Ошибка выполнения задачи #{task_id}: {error}')
        finally:
            with self._lock:
                result = self._processing_session.complete_active_task()
                self._status = 'idle'
                self._handler = None
                self._runner_thread = None
            if result.task is not None:
                if result.stop_requested:
                    self._append_event('logging', f'Задача #{result.task.task_id} остановлена.')
                else:
                    self._append_event('logging', f'Задача #{result.task.task_id} завершена.')
            self._start_next_task_if_possible()

    def _start_next_task_if_possible(self) -> bool:
        with self._lock:
            if self._handler is not None or self._status in {'running', 'stopping'}:
                return False
            decision = self._processing_session.next_task_to_start(worker_running=False)
            task = decision.task
            if task is None:
                self._status = 'idle'
                return False
            self._status = 'running'

        build_result = self._presenter.build_handler(
            main_state=task.main_window_state,
            settings_state=task.settings_state,
            message_bus=self._bus,
            question_module=self._question_yes,
            callback=self._on_finished,
        )
        if build_result.error is not None:
            with self._lock:
                self._processing_session.drop_task(task.task_id)
                self._status = 'idle'
            self._append_event('error', f'Не удалось запустить задачу #{task.task_id}: {build_result.error}')
            return self._start_next_task_if_possible()

        self._clear_runtime_metrics()
        owner_suffix = f' ({task.owner_display_name})' if getattr(task, 'owner_display_name', '') else ''
        self._append_event('logging', f'Запуск задачи #{task.task_id}{owner_suffix}...')
        with self._lock:
            self._handler = build_result.handler
            self._runner_thread = threading.Thread(target=self._run_handler, args=(task.task_id,), daemon=True)
            self._runner_thread.start()
        return True

    def _queue_index_by_task_id(self, task_id: int) -> int | None:
        for index, item in enumerate(self._processing_session.queue_snapshot()):
            if item.task_id == int(task_id):
                return index
        return None

    def get_task(self, task_id: int):
        with self._lock:
            row = self._queue_index_by_task_id(task_id)
            if row is None:
                return None
            return self._processing_session.get_task_by_index(row)

    def _question_yes(
        self,
        text: str,
        header: str,
        default_answer: bool = True,
        timeout_seconds: int | None = None,
    ) -> bool:
        default_label = 'Да' if default_answer else 'Нет'
        timeout_suffix = f', автоответ через {int(timeout_seconds)} сек.' if timeout_seconds else ''
        self._append_event('question', f'{header}: {text}. Ответ по умолчанию: {default_label}{timeout_suffix}')
        return bool(default_answer)

    def _on_finished(self) -> None:
        self._append_event('logging', 'Обработка завершена.')

    def start(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        *,
        owner_username: str,
        owner_display_name: str,
    ) -> tuple[bool, str | None, str | None]:
        validation_error = build_processing_start_error_message(main_state, settings_state)
        if validation_error:
            return False, validation_error, None

        with self._lock:
            task = self._processing_session.enqueue_task(
                main_state=clone_main_window_state(main_state),
                settings_state=replace(settings_state),
                owner_username=owner_username,
                owner_display_name=owner_display_name,
            )
        self._append_event('logging', f'Задача #{task.task_id} добавлена в очередь пользователем {owner_display_name}.')
        started = self._start_next_task_if_possible()
        if started:
            return True, None, f'Задача #{task.task_id} запущена.'
        return True, None, f'Задача #{task.task_id} добавлена в очередь.'

    def stop(self, *, owner_username: str) -> tuple[bool, str | None]:
        with self._lock:
            handler = self._handler
            active_task = self._processing_session.active_task
            if self._status != 'running' or handler is None or active_task is None:
                return False, 'Нет активной обработки.'
            if str(active_task.owner_username or '') != str(owner_username or ''):
                return False, 'Можно останавливать только свою активную задачу.'
            self._processing_session.request_stop()
            self._status = 'stopping'

        try:
            handler.stop_execution()
            self._append_event('logging', f'Останов активной задачи #{active_task.task_id} запрошен пользователем.')
            return True, None
        except Exception as error:
            _LOG.exception('TrainingSessionService stop failed')
            with self._lock:
                self._status = 'idle'
            return False, f'Ошибка остановки: {error}'

    def remove_task(self, task_id: int, *, owner_username: str) -> tuple[bool, str | None]:
        try:
            with self._lock:
                row = self._queue_index_by_task_id(task_id)
                if row is None:
                    return False, 'Задача не найдена.'
                current_task = self._processing_session.get_task_by_index(row)
                if current_task is None:
                    return False, 'Задача не найдена.'
                if str(current_task.owner_username or '') != str(owner_username or ''):
                    return False, 'Можно изменять только свои задачи.'
                task = self._processing_session.remove_task_by_index(row)
        except ActiveTaskMutationError as error:
            return False, f'Нельзя удалить активную задачу #{error.task_id}.'

        if task is None:
            return False, 'Задача не найдена.'
        self._append_event('logging', f'Задача #{task.task_id} удалена из очереди.')
        return True, None

    def toggle_pause_task(self, task_id: int, *, owner_username: str) -> tuple[bool, str | None]:
        try:
            with self._lock:
                row = self._queue_index_by_task_id(task_id)
                if row is None:
                    return False, 'Задача не найдена.'
                current_task = self._processing_session.get_task_by_index(row)
                if current_task is None:
                    return False, 'Задача не найдена.'
                if str(current_task.owner_username or '') != str(owner_username or ''):
                    return False, 'Можно изменять только свои задачи.'
                task = self._processing_session.toggle_pause_by_index(row)
        except ActiveTaskMutationError as error:
            return False, f'Нельзя поставить на паузу активную задачу #{error.task_id}.'

        if task is None:
            return False, 'Задача не найдена.'
        state = 'поставлена на паузу' if task.paused else 'снята с паузы'
        self._append_event('logging', f'Задача #{task.task_id} {state}.')
        if not task.paused:
            self._start_next_task_if_possible()
        return True, None

    def snapshot(self, after_event_id: int = 0, *, current_username: str = '') -> dict[str, Any]:
        with self._lock:
            new_events = [e for e in self._events if e['id'] > after_event_id]
            latest_epoch = self._train_epoch[-1]['epoch'] if self._train_epoch else 0
            batch_points = self._batch_by_epoch.get(latest_epoch, [])
            active_task = self._processing_session.active_task
            active_owner_username = str(getattr(active_task, 'owner_username', '') or '')
            queue_items = [
                {
                    'task_id': item.task_id,
                    'work_mode': item.work_mode,
                    'status': item.status,
                    'owner_username': item.owner_username,
                    'owner_display_name': item.owner_display_name,
                    'is_owner': bool(item.owner_username and item.owner_username == current_username),
                }
                for item in self._processing_session.queue_snapshot()
            ]
            return {
                'status': self._status,
                'events': new_events,
                'last_event_id': self._events[-1]['id'] if self._events else 0,
                'permissions': {
                    'can_stop_active_task': bool(
                        active_owner_username
                        and current_username
                        and active_owner_username == current_username
                        and self._status in {'running', 'stopping'}
                    ),
                },
                'queue': queue_items,
                'metrics': {
                    'train_epoch': list(self._train_epoch),
                    'val_epoch': list(self._val_epoch),
                    'batch_epoch': latest_epoch,
                    'train_batch': self._sparsify_batch_points(batch_points),
                    'system_memory': dict(self._system_memory),
                    'validation_quality': dict(self._validation_quality),
                    'train_perf': dict(self._train_perf),
                    'train_speed_batches_per_sec': self._train_speed_batches_per_sec,
                    'recognition_speed_images_per_sec': self._recognition_speed_images_per_sec,
                    'progress': {
                        'epoch': _serialize_progress(self._train_epoch_progress),
                        'batch': _serialize_progress(self._train_batch_progress),
                        'recognition': _serialize_progress(self._recognition_progress),
                    },
                    'preview': dict(self._preview),
                },
            }

    @staticmethod
    def _sparsify_batch_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
        if len(points) > 1000:
            return list(points[::2])
        return list(points)

    def load_initial_states(self) -> tuple[MainWindowState, SettingsState]:
        return self._presenter.load_initial_states()


_session_singleton: TrainingSessionService | None = None


def get_session_service() -> TrainingSessionService:
    global _session_singleton
    if _session_singleton is None:
        _session_singleton = TrainingSessionService(presenter=create_web_presenter())
    return _session_singleton

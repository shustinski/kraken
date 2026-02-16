from __future__ import annotations

import logging
import threading
import time
from typing import Any

from lib.message_bus import MessageBus
from presenter.web_presenter import WebPresenter
from view.window_dataclasses import MainWindowState, SettingsState

_LOG = logging.getLogger(__name__)


class TrainingSessionService:
    def __init__(self) -> None:
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

        self._runner_thread: threading.Thread | None = None
        self._handler = None
        self._bus = MessageBus()
        self._bus.subscribe('logging', self._on_logging)
        self._bus.subscribe('training', self._on_training)
        self._bus.subscribe('metrics', self._on_metrics)

        self._presenter = WebPresenter()

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
            if len(self._events) > 5000:
                self._events[:] = self._events[-5000:]

    def _on_logging(self, payload: Any) -> None:
        self._append_event('logging', str(payload))

    def _on_training(self, payload: Any) -> None:
        self._append_event('training', str(payload))

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
                if len(points) > 200:
                    points[:] = points[-200:]
                return
            if metric_type == 'train_perf':
                self._train_perf = {
                    'epoch': float(epoch),
                    'batch_index': float(payload.get('batch_index', 0.0)),
                    'data_wait_ms': float(payload.get('data_wait_ms', 0.0)),
                    'forward_ms': float(payload.get('forward_ms', 0.0)),
                    'backward_ms': float(payload.get('backward_ms', 0.0)),
                    'optimizer_ms': float(payload.get('optimizer_ms', 0.0)),
                    'total_ms': float(payload.get('total_ms', 0.0)),
                }
                return
            if metric_type == 'train_perf_epoch':
                self._train_perf = {
                    'epoch': float(epoch),
                    'data_wait_ms': float(payload.get('data_wait_ms', 0.0)),
                    'forward_ms': float(payload.get('forward_ms', 0.0)),
                    'backward_ms': float(payload.get('backward_ms', 0.0)),
                    'optimizer_ms': float(payload.get('optimizer_ms', 0.0)),
                    'total_ms': float(payload.get('total_ms', 0.0)),
                }
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

    def _clear_runtime_buffers(self) -> None:
        with self._lock:
            self._events.clear()
            self._next_event_id = 1
            self._train_epoch.clear()
            self._val_epoch.clear()
            self._batch_by_epoch.clear()
            self._system_memory.clear()
            self._validation_quality.clear()
            self._train_perf.clear()

    def _run_handler(self) -> None:
        try:
            if self._handler is not None:
                self._handler.start()
        except Exception as error:
            _LOG.exception('TrainingSessionService handler execution failed')
            self._append_event('error', f'Ошибка выполнения: {error}')
        finally:
            with self._lock:
                self._status = 'idle'
                self._handler = None

    def _question_yes(self, text: str, header: str) -> bool:
        self._append_event('question', f'{header}: {text}. Ответ по умолчанию: Да')
        return True

    def _on_finished(self) -> None:
        self._append_event('logging', 'Обработка завершена.')

    def start(self, main_state: MainWindowState, settings_state: SettingsState) -> tuple[bool, str | None]:
        with self._lock:
            if self._status == 'running':
                return False, 'Обработка уже запущена.'

        build_result = self._presenter.build_handler(
            main_state=main_state,
            settings_state=settings_state,
            message_bus=self._bus,
            question_module=self._question_yes,
            callback=self._on_finished,
        )
        if build_result.error is not None:
            return False, build_result.error

        self._clear_runtime_buffers()
        self._append_event('logging', 'Запуск обработки...')
        with self._lock:
            self._handler = build_result.handler
            self._status = 'running'
            self._runner_thread = threading.Thread(target=self._run_handler, daemon=True)
            self._runner_thread.start()

        return True, None

    def stop(self) -> tuple[bool, str | None]:
        with self._lock:
            if self._status != 'running' or self._handler is None:
                return False, 'Нет активной обработки.'
            self._status = 'stopping'
            handler = self._handler

        try:
            handler.stop_execution()
            self._append_event('logging', 'Останов запрошен пользователем.')
            return True, None
        except Exception as error:
            _LOG.exception('TrainingSessionService stop failed')
            with self._lock:
                self._status = 'idle'
            return False, f'Ошибка остановки: {error}'

    def snapshot(self, after_event_id: int = 0) -> dict[str, Any]:
        with self._lock:
            new_events = [e for e in self._events if e['id'] > after_event_id]
            latest_epoch = self._train_epoch[-1]['epoch'] if self._train_epoch else 0
            batch_points = self._batch_by_epoch.get(latest_epoch, [])
            return {
                'status': self._status,
                'events': new_events,
                'last_event_id': self._events[-1]['id'] if self._events else 0,
                'metrics': {
                    'train_epoch': list(self._train_epoch),
                    'val_epoch': list(self._val_epoch),
                    'batch_epoch': latest_epoch,
                    'train_batch': list(batch_points),
                    'system_memory': dict(self._system_memory),
                    'validation_quality': dict(self._validation_quality),
                    'train_perf': dict(self._train_perf),
                },
            }

    def load_initial_states(self) -> tuple[MainWindowState, SettingsState]:
        return self._presenter.load_initial_states()


_session_singleton: TrainingSessionService | None = None


def get_session_service() -> TrainingSessionService:
    global _session_singleton
    if _session_singleton is None:
        _session_singleton = TrainingSessionService()
    return _session_singleton


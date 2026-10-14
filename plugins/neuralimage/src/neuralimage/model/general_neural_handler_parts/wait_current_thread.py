"""general_neural_handler: wait current thread."""
from __future__ import annotations
import shutil
import os
import time
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from neuralimage.model.NeuralNetwork.model_train_and_recognition import ModelRecognizer


class WaitCurrentThreadMixin:
    def _write_training_manifest(self, model_save_path: Path) -> None:
        def normalize(value: Any) -> Any:
            if is_dataclass(value):
                return normalize(asdict(value))
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, 'value'):
                return normalize(value.value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        payload = {
            'schema': 'neuralimage_training_run',
            'version': 1,
            'model_path': str(model_save_path),
            'training': normalize(self.tranining_parameters),
        }
        manifest_path = model_save_path.parent / 'training_manifest.json'
        temporary_path = manifest_path.with_suffix('.json.tmp')
        temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary_path, manifest_path)

    def _stop_training_callback(self):
        if self._need_stop:
            return
        self.message_bus.publish('logging', 'Обучение завершено')

    def _start_recognition(self):
        self.message_bus.publish('metrics', {'type': 'workflow_phase', 'phase': 'recognition'})
        self.current_thread = ModelRecognizer(
            self.recognition_parameters,
            message_bus=self.message_bus,
            callback=None,
            multithreading=bool(getattr(self.recognition_parameters, 'recognition_multiprocessing_enabled', True)),
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('recognition')
        if not getattr(self.current_thread, 'succeeded', False):
            if (not self._need_stop) and getattr(self.current_thread, 'error_message', None) is None:
                self.message_bus.publish('error', 'Распознавание завершилось с ошибкой.')
        self._stop_recognition_callback()

    def _stop_recognition_callback(self):
        self._drop_runtime_references()
        if self.callback is not None:
            self.callback()

    def stop_execution(self):
        self._need_stop = True
        if self.current_thread is None:
            self._drop_runtime_references()
            return
        if hasattr(self.current_thread, 'stop'):
            self.current_thread.stop()

    def pause_execution(self):
        self._need_pause = True
        self._need_stop = True
        if self.current_thread is None:
            self._drop_runtime_references()
            return
        pause = getattr(self.current_thread, 'pause', None)
        if callable(pause):
            pause()
        elif hasattr(self.current_thread, 'stop'):
            self.current_thread.stop()

    def _wait_for_current_thread(self, operation_name: str) -> None:
        thread = self.current_thread
        if thread is None:
            return
        join_fn = getattr(thread, 'join', None)
        is_alive_fn = getattr(thread, 'is_alive', None)
        if not callable(join_fn):
            return
        if not callable(is_alive_fn):
            join_fn()
            return
        stop_wait_started_at: float | None = None
        while is_alive_fn():
            join_fn(timeout=self.STOP_JOIN_POLL_SECONDS)
            if not self._need_stop:
                continue
            if stop_wait_started_at is None:
                stop_wait_started_at = time.monotonic()
                continue
            waited_after_stop = time.monotonic() - stop_wait_started_at
            if waited_after_stop < self.STOP_JOIN_GRACE_SECONDS:
                continue
            self.message_bus.publish(
                'error',
                (
                    f'Не удалось корректно завершить {operation_name} в течение '
                    f'{int(self.STOP_JOIN_GRACE_SECONDS)} сек. Операция переведена в аварийное завершение.'
                ),
            )
            break

    def _check_folder_existance(self, folder: Path):
        answer = False
        if folder.exists():
            answer = self.question(
                f'Папка {folder.name} существует, использовать данные из неё?',
                'Папка существует',
                default_answer=self.EXISTING_FOLDER_DEFAULT_ANSWER,
                timeout_seconds=self.EXISTING_FOLDER_TIMEOUT_SECONDS,
            )
            if not answer:
                shutil.rmtree(folder, ignore_errors=False)
        return answer

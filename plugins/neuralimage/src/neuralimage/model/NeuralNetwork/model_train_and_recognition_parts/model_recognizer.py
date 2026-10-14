"""model_train_and_recognition: model recognizer."""
from __future__ import annotations
import os
import time
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any
import multiprocessing as mp
import threading
from neuralimage.lib.data_interfaces import (
    RecognitionParameters,
)
from neuralimage.lib.file_func import filter_images
from neuralimage.lib.message_bus import AbstractMessageBus

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import (
    _drain_process_queue,
    _join_process_with_escalation,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import RecognizerProcess

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _release_torch_memory

class ModelRecognizer(threading.Thread):
    def __init__(
        self,
        recognition_parameters: RecognitionParameters,
        message_bus: AbstractMessageBus,
        callback: Callable[..., None] | None = None,
        multithreading: bool | None = None,
    ):
        super().__init__()
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        if multithreading is None:
            multithreading = bool(getattr(recognition_parameters, 'recognition_multiprocessing_enabled', True))
        self._multithreading = bool(multithreading)
        self._stop_event = threading.Event()
        self._process_stop_event = mp.Event()
        self.message_queue = mp.Queue()
        self._inline_recognizer: NeuralRecognizer | None = None
        self.succeeded = False
        self.error_message: str | None = None
        self._manifest_path = Path(
            getattr(recognition_parameters, 'resume_manifest_path', None)
            or (Path(recognition_parameters.result_folder) / '.neuralimage-recognition-progress.json')
        )
        self._completed_outputs: dict[str, str] = {}

    def _load_or_reset_manifest(self) -> None:
        resume = bool(getattr(self._parameters, 'resume_from_manifest', False))
        if not resume:
            self._completed_outputs = {}
            try:
                self._manifest_path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        try:
            payload = json.loads(self._manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            payload = {}
        completed = payload.get('completed', []) if isinstance(payload, dict) else []
        self._completed_outputs = {
            str(item.get('source_path')): str(item.get('output_path', ''))
            for item in completed
            if isinstance(item, dict) and str(item.get('source_path', '')).strip()
        }

        if not self._parameters.source_files:
            source_folder = getattr(self._parameters, 'source_folder', None)
            if source_folder is not None and str(source_folder).strip():
                self._parameters.source_files = filter_images(
                    Path(source_folder),
                    recursive=bool(getattr(self._parameters, 'recursive_file_search', False)),
                )
        self._parameters.source_files = [
            path for path in self._parameters.source_files
            if str(path) not in self._completed_outputs
        ]

    def _record_completed_file(self, source_path: str, output_path: str) -> None:
        source_path = str(source_path or '').strip()
        if not source_path:
            return
        self._completed_outputs[source_path] = str(output_path or '')
        payload = {
            'version': 1,
            'last_completed_file': source_path,
            'completed': [
                {'source_path': source, 'output_path': output}
                for source, output in self._completed_outputs.items()
            ],
        }
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._manifest_path.with_suffix(f'{self._manifest_path.suffix}.tmp')
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp_path, self._manifest_path)

    @staticmethod
    def _elapsed_suffix(since: float) -> str:
        elapsed_seconds = int(max(0.0, time.perf_counter() - since))
        elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
        elapsed_minutes, elapsed_secs = divmod(remainder, 60)
        return f' Прошло: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}'

    def _publish_recognition_message(
        self,
        message: Any,
        *,
        append_elapsed_suffix: bool,
        started_at: float,
    ) -> None:
        topic, payload = message[0], message[1]
        if topic == 'metrics' and isinstance(payload, dict) and payload.get('type') == 'recognition_completed':
            self._record_completed_file(
                str(payload.get('source_path', '') or ''),
                str(payload.get('output_path', '') or ''),
            )
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(started_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)

    def _drain_recognition_queue(self, *, append_elapsed_suffix: bool, started_at: float) -> None:
        def _publish(queued_message: Any) -> None:
            self._publish_recognition_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                started_at=started_at,
            )
        _drain_process_queue(self.message_queue, _publish)

    def _finalize_recognition_result(self, recognition_process: Any) -> bool:
        if self._stop_event.is_set():
            self.succeeded = False
            return False
        exit_code = recognition_process.exitcode if recognition_process is not None else 1
        if exit_code not in (0, None):
            if self.error_message is None:
                self.error_message = f'Recognition error: process exited with code {exit_code}.'
                self._bus.publish('error', self.error_message)
            self.succeeded = False
            return False
        self.succeeded = True
        return True

    def run(self):
        recognition_process: Any = None
        try:
            self._load_or_reset_manifest()
            recognition_process = RecognizerProcess(
                self._parameters,
                self.message_queue,
                self._process_stop_event,
                multithreading=self._multithreading,
            )
            recognition_process.start()
            started_at = time.perf_counter()
            while recognition_process.is_alive():
                if self._stop_event.is_set():
                    self._process_stop_event.set()
                    _join_process_with_escalation(recognition_process)
                    break
                self._drain_recognition_queue(
                    append_elapsed_suffix=True,
                    started_at=started_at,
                )
                time.sleep(1)

            _join_process_with_escalation(recognition_process)
            self._drain_recognition_queue(append_elapsed_suffix=False, started_at=started_at)
            self._finalize_recognition_result(recognition_process)
        except Exception as error:
            self.succeeded = False
            self.error_message = f'Recognition error: {error}'
            self._bus.publish('error', self.error_message)
        finally:
            self._inline_recognizer = None
            if recognition_process is not None and recognition_process.is_alive():
                _join_process_with_escalation(recognition_process)
            try:
                self.message_queue.close()
            except Exception:
                pass
            _release_torch_memory()
            if self.callback is not None:
                self.callback()

    def stop(self):
        self._stop_event.set()
        self._process_stop_event.set()
        inline_recognizer = self._inline_recognizer
        if inline_recognizer is not None:
            inline_recognizer.stop()

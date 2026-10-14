"""model_train_and_recognition: resolve output threshold."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
from typing import Any
import multiprocessing as mp
import threading
import torch
import torch.nn as nn
from neuralimage.lib import System
from neuralimage.lib.data_interfaces import (
    RecognitionParameters,
)
from neuralimage.lib.file_func import filter_images
from kraken_core.asset_source import AssetSource
from neuralimage.lib.message_bus import AbstractMessageBus

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import RECOGNITION_AUX_WORKERS_PER_GPU

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import RecognitionRuntimePlan

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _release_torch_memory


class ResolveOutputThresholdMixin:
    def __init__(self, recognition_parameters:RecognitionParameters,
                 message_bus:AbstractMessageBus, callback:Callable[..., None]|None = None,
                 *, asset_source: AssetSource | None = None):
        super().__init__()
        self.asset_source = asset_source
        self.number_of_threads:int = 1
        self.devices_list: list[torch.device] = []
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        result_folder = self._parameters.result_folder
        result_folder.mkdir(parents=True, exist_ok=True)
        self.colors = 1
        self.model:nn.Module|None = None
        self._thread_stop_event = threading.Event()
        self.stop_event = mp.Event()
        self._resolved_output_threshold: float | None = 0.5
        self._use_context_branch = False
        self._use_cross_attention = True
        self._context_crop_size: tuple[int, int] | None = None
        self._context_input_size: tuple[int, int] | None = None

    def _ensure_source_files_indexed(self) -> None:
        if self._parameters.source_files:
            self._bus.publish('logging', f'Images queued for recognition: {len(self._parameters.source_files)}')
            return

        source_folder = getattr(self._parameters, 'source_folder', None)
        if source_folder is None or not str(source_folder).strip():
            self._parameters.source_files = []
            self._bus.publish('logging', 'Images queued for recognition: 0')
            return

        self._bus.publish('logging', 'Индексация файлов для распознавания выполняется в отдельном потоке...')
        recursive = bool(getattr(self._parameters, 'recursive_file_search', False))
        self._parameters.source_files = filter_images(Path(source_folder), recursive=recursive)
        self._bus.publish('logging', f'Images queued for recognition: {len(self._parameters.source_files)}')

    def run(self, multithreading: bool | None = None):
        try:
            self._ensure_source_files_indexed()
            self.prepare_model()
            if multithreading is None:
                multithreading = bool(getattr(self._parameters, 'recognition_multiprocessing_enabled', True))
            runtime_plan = self._build_runtime_plan(multithreading=multithreading)
            self.number_of_threads = runtime_plan.threads
            self.devices_list = runtime_plan.devices

            if runtime_plan.use_multiprocessing:
                self.run_multiprocessing(runtime_plan)
            else:
                if not multithreading:
                    self._bus.publish(
                        'logging',
                        'Recognition multiprocessing disabled by settings; using single-thread mode.',
                    )
                elif multithreading and not isinstance(self._parameters.model, (str, Path)):
                    self._bus.publish(
                        'logging',
                        'Многопроцессное распознавание доступно только при загрузке модели из файла. Переход в однопоточный режим.',
                    )
                elif len(self._parameters.source_files) < 2:
                    self._bus.publish(
                        'logging',
                        'Recognition multiprocessing requires at least 2 source images; falling back to single-thread mode.',
                    )
                self.run_one_thread()
        finally:
            self.model = None
            _release_torch_memory()
            if self.callback is not None:
                self.callback()

    def _build_runtime_plan(self, multithreading: bool) -> RecognitionRuntimePlan:
        source_count = len(self._parameters.source_files)
        cpu_threads = mp.cpu_count()
        devices, gpu_count = self._resolve_devices()

        can_use_multiprocessing_model = isinstance(self._parameters.model, (str, Path))
        enough_inputs_for_pipeline = source_count >= 2
        use_multiprocessing = bool(multithreading and can_use_multiprocessing_model and enough_inputs_for_pipeline)
        if gpu_count > 0:
            predict_workers = int(gpu_count)
            cut_workers = int(RECOGNITION_AUX_WORKERS_PER_GPU * gpu_count)
            sew_workers = int(RECOGNITION_AUX_WORKERS_PER_GPU * gpu_count)
            threads = int(predict_workers + cut_workers + sew_workers)
        else:
            threads = min(source_count, cpu_threads)
            predict_workers = max(1, len(devices))
            remaining_workers = max(2, threads - predict_workers)
            cut_workers = max(1, remaining_workers // 2)
            sew_workers = max(1, remaining_workers - cut_workers)

        return RecognitionRuntimePlan(
            threads=threads,
            devices=devices,
            gpu_count=gpu_count,
            use_multiprocessing=use_multiprocessing,
            cut_workers=cut_workers,
            predict_workers=predict_workers,
            sew_workers=sew_workers,
        )

    @staticmethod
    def _resolve_devices() -> tuple[list[torch.device], int]:
        if not torch.cuda.is_available():
            return [torch.device('cpu')], 0
        gpu_count = System.check_gpu_availability()
        if gpu_count <= 0:
            return [torch.device('cpu')], 0
        return [torch.device(f'cuda:{gpu}') for gpu in range(gpu_count)], gpu_count

    @staticmethod
    def _normalize_output_threshold(value: Any, *, fallback: float = 0.5) -> float:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            threshold = float(fallback)
        return float(min(max(threshold, 0.0), 1.0))

    @staticmethod
    def _extract_recommended_threshold(model: nn.Module) -> float | None:
        metadata = getattr(model, '_neuralimage_artifact_metadata', None)
        if not isinstance(metadata, dict):
            return None
        inference = metadata.get('inference')
        if not isinstance(inference, dict):
            return None
        threshold = inference.get('recommended_threshold')
        if threshold is None:
            return None
        return NeuralRecognizer._normalize_output_threshold(threshold, fallback=0.5)

    def _resolve_output_threshold(self, model: nn.Module) -> None:
        manual_threshold = self._normalize_output_threshold(
            getattr(self._parameters, 'threshold', 0.5),
            fallback=0.5,
        )
        if not bool(getattr(self._parameters, 'binarize_output', True)):
            self._resolved_output_threshold = None
            self._bus.publish(
                'logging',
                'Recognition output binarization disabled: saving probability maps.',
            )
            return

        if bool(getattr(self._parameters, 'use_auto_threshold', True)):
            recommended_threshold = self._extract_recommended_threshold(model)
            if recommended_threshold is not None:
                self._resolved_output_threshold = recommended_threshold
                self._bus.publish(
                    'logging',
                    f'Recognition threshold: using recommended model threshold {recommended_threshold:.2f}.',
                )
                return
            self._bus.publish(
                'logging',
                (
                    'Recognition threshold: model metadata does not contain a recommended '
                    f'threshold, using fallback {manual_threshold:.2f}.'
                ),
            )
        else:
            self._bus.publish(
                'logging',
                f'Recognition threshold: using manual value {manual_threshold:.2f}.',
            )

        self._resolved_output_threshold = manual_threshold

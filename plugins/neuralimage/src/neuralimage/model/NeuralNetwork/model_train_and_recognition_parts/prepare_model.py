"""model_train_and_recognition: prepare model."""
from __future__ import annotations
from typing import TYPE_CHECKING
import os
import sys
import hashlib
from pathlib import Path
from typing import cast
import torch
import torch.nn as nn
from neuralimage.lib.func import get_input_channels
from neuralimage.preprocessing.config import build_preprocessing_config
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    RecognitionWorkload,
    WorkerCounts,
    run_multiprocessing_recognition,
    run_single_thread_recognition,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import STOP_TOKEN

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.collect_memory_metrics import _collect_memory_metrics

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import (
    _get_torch_compile_unavailable_reason,
    _resolve_torch_compile_mode,
)


class PrepareModelMixin:
    def prepare_model(self):
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import load_model_artifact

        loaded_model: str | Path | nn.Module = self._parameters.model
        if isinstance(loaded_model, (str, Path)):
            loaded_model = load_model_artifact(loaded_model, map_location='cpu')
        if not isinstance(loaded_model, nn.Module):
            raise TypeError('Recognition model must be a torch.nn.Module or a model path.')
        self._apply_artifact_preprocessing(loaded_model)
        self._resolve_output_threshold(loaded_model)
        if not bool(getattr(sys, 'frozen', False)):
            compile_enabled = str(os.getenv('NEURALIMAGE_TORCH_COMPILE', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
            compile_fn = getattr(torch, 'compile', None)
            target_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            compile_unavailable_reason = _get_torch_compile_unavailable_reason(target_device_type)
            can_compile = (
                compile_enabled
                and compile_fn is not None
                and compile_unavailable_reason is None
            )
            if can_compile:
                try:
                    compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type)
                    compiled_model = compile_fn(loaded_model, mode=compile_mode, dynamic=False)
                    for attr_name in (
                        '_neuralimage_model_name',
                        '_neuralimage_input_channels',
                        '_neuralimage_model_kwargs',
                        '_neuralimage_artifact_metadata',
                    ):
                        if hasattr(loaded_model, attr_name):
                            setattr(compiled_model, attr_name, getattr(loaded_model, attr_name))
                    loaded_model = compiled_model
                    self._bus.publish(
                        'logging',
                        f'Распознавание: torch.compile enabled (mode={compile_mode}, reason={mode_reason}).',
                    )
                except Exception as error:
                    self._bus.publish('logging', f'Распознавание: torch.compile отключен (fallback): {error}')
            elif not compile_enabled:
                self._bus.publish('logging', 'Распознавание: torch.compile отключен переменной NEURALIMAGE_TORCH_COMPILE=0.')
            elif compile_fn is None:
                self._bus.publish('logging', 'Распознавание: torch.compile недоступен в этой версии PyTorch.')
            else:
                self._bus.publish(
                    'logging',
                    f'Распознавание: torch.compile отключен, {compile_unavailable_reason}',
                )
        self.model = loaded_model
        self.colors = get_input_channels(loaded_model)
        self._resolve_context_branch_settings(loaded_model)

    def _apply_artifact_preprocessing(self, model: nn.Module) -> None:
        metadata = getattr(model, '_neuralimage_artifact_metadata', None)
        preprocessing_payload = metadata.get('preprocessing') if isinstance(metadata, dict) else None
        if not isinstance(preprocessing_payload, dict) or not isinstance(preprocessing_payload.get('config'), dict):
            return
        artifact_config = build_preprocessing_config(preprocessing_payload['config'])
        artifact_hash = str(preprocessing_payload.get('hash') or artifact_config.stable_hash())
        requested = getattr(self._parameters, 'preprocessing', None)
        if requested is not None and requested.any_enabled() and requested.stable_hash() != artifact_hash:
            raise ValueError(
                'Recognition preprocessing override is incompatible with the model artifact. '
                f'artifact={artifact_hash}, requested={requested.stable_hash()}.'
            )
        self._parameters.preprocessing = artifact_config

    def run_multiprocessing(self, runtime_plan: RecognitionRuntimePlan | None = None):
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import run_multiprocessing_recognition

        if self.asset_source is not None:
            raise ValueError('Server streams require single-thread recognition')
        if runtime_plan is None:
            runtime_plan = self._build_runtime_plan(multithreading=True)
        workload = RecognitionWorkload(
            source_files=list(self._parameters.source_files),
            result_folder=self._parameters.result_folder,
            part_size=self._parameters.part_size,
            overlap=self._parameters.overlap,
            batch_size=self._parameters.batch_size,
            colors=self.colors,
            jpeg_quality=int(getattr(self._parameters, 'jpeg_quality', 95)),
            binarize_output=bool(getattr(self._parameters, 'binarize_output', True)),
            threshold=self._resolved_output_threshold,
            postprocess_enabled=bool(getattr(self._parameters, 'postprocess_enabled', False)),
            postprocess_kernel_size=max(1, int(getattr(self._parameters, 'postprocess_kernel_size', 3))),
            recognition_tta_enabled=bool(getattr(self._parameters, 'recognition_tta_enabled', False)),
            confidence_tta_enabled=bool(getattr(self._parameters, 'confidence_tta_enabled', False)),
            confidence_save_mode=str(getattr(self._parameters, 'confidence_save_mode', 'off') or 'off'),
            devices=list(self.devices_list),
            model_source=cast(str | Path, self._parameters.model),
            use_context_branch=bool(self._use_context_branch),
            use_cross_attention=bool(self._use_cross_attention),
            context_crop_size=self._context_crop_size,
            context_input_size=self._context_input_size,
            compression_factor=max(1, int(getattr(self._parameters, 'compression_factor', 1))),
            source_root=(
                Path(self._parameters.source_folder)
                if bool(getattr(self._parameters, 'recursive_file_search', False))
                and self._parameters.source_folder is not None
                else None
            ),
            preprocessing=getattr(self._parameters, 'preprocessing', None),
            active_learning=getattr(self._parameters, 'active_learning', None),
            active_learning_metadata=self._build_active_learning_metadata(),
            uncertainty=getattr(self._parameters, 'uncertainty', None),
        )
        run_multiprocessing_recognition(
            workload=workload,
            worker_counts=WorkerCounts(
                cut=runtime_plan.cut_workers,
                predict=runtime_plan.predict_workers,
                sew=runtime_plan.sew_workers,
            ),
            stop_event=self.stop_event,
            publish=self._bus.publish,
            stop_token=STOP_TOKEN,
        )

    def run_one_thread(self):
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import run_single_thread_recognition

        model = self.model
        if model is None:
            raise RuntimeError('Model is not prepared before recognition start.')
        if not self.devices_list:
            self.devices_list = [torch.device('cpu')]
        device = self.devices_list[0]
        run_single_thread_recognition(
            asset_source=self.asset_source,
            source_files=list(self._parameters.source_files),
            result_folder=self._parameters.result_folder,
            part_size=self._parameters.part_size,
            overlap=self._parameters.overlap,
            batch_size=self._parameters.batch_size,
            colors=self.colors,
            model=model,
            device=device,
            stop_event=self._thread_stop_event,
            publish=self._bus.publish,
            collect_memory_metrics=_collect_memory_metrics,
            jpeg_quality=int(getattr(self._parameters, 'jpeg_quality', 95)),
            binarize_output=bool(getattr(self._parameters, 'binarize_output', True)),
            threshold=self._resolved_output_threshold,
            postprocess_enabled=bool(getattr(self._parameters, 'postprocess_enabled', False)),
            postprocess_kernel_size=max(1, int(getattr(self._parameters, 'postprocess_kernel_size', 3))),
            recognition_tta_enabled=bool(getattr(self._parameters, 'recognition_tta_enabled', False)),
            confidence_tta_enabled=bool(getattr(self._parameters, 'confidence_tta_enabled', False)),
            confidence_save_mode=str(getattr(self._parameters, 'confidence_save_mode', 'off') or 'off'),
            use_context_branch=bool(self._use_context_branch),
            use_cross_attention=bool(self._use_cross_attention),
            context_crop_size=self._context_crop_size,
            context_input_size=self._context_input_size,
            compression_factor=max(1, int(getattr(self._parameters, 'compression_factor', 1))),
            lossless_binary_png=bool(getattr(self._parameters, 'lossless_binary_png', False)),
            source_root=(
                Path(self._parameters.source_folder)
                if bool(getattr(self._parameters, 'recursive_file_search', False))
                and self._parameters.source_folder is not None
                else None
            ),
            preprocessing=getattr(self._parameters, 'preprocessing', None),
            active_learning=getattr(self._parameters, 'active_learning', None),
            active_learning_metadata=self._build_active_learning_metadata(),
            uncertainty=getattr(self._parameters, 'uncertainty', None),
        )

    def _build_active_learning_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        model_source = getattr(self._parameters, 'model', None)
        if isinstance(model_source, (str, Path)) and Path(model_source).is_file():
            digest = hashlib.sha256()
            with Path(model_source).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            metadata['model_hash'] = digest.hexdigest()
        preprocessing = getattr(self._parameters, 'preprocessing', None)
        if preprocessing is not None and hasattr(preprocessing, 'stable_hash'):
            metadata['preprocessing_hash'] = preprocessing.stable_hash()
        sem_config_hash = getattr(self._parameters, 'sem_config_hash', None)
        if sem_config_hash:
            metadata['config_hash'] = str(sem_config_hash)
        return metadata

    def stop(self):
        self._thread_stop_event.set()
        self.stop_event.set()

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import RecognitionRuntimePlan

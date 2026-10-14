"""model_train_and_recognition: resolve artifact runtime metadata."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from neuralimage.lib.func import get_input_channels
from neuralimage.model.NeuralNetwork.model_io import save_model_artifact

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import _ARTIFACT_NAME_SANITIZE_PATTERN


class ResolveArtifactRuntimeMetadataMixin:
    @property
    def _base_model(self) -> nn.Module:
        """Return the original model instance even if wrapped by DDP/DataParallel."""
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return self._model.module
        return self._model

    def _resolve_model_artifact_metadata(self) -> tuple[str, int, dict[str, Any]]:
        base_model = self._base_model
        model_name = str(getattr(base_model, '_neuralimage_model_name', base_model.__class__.__name__))
        input_channels = getattr(base_model, '_neuralimage_input_channels', None)
        if input_channels is None:
            input_channels = get_input_channels(base_model)
        model_kwargs = getattr(base_model, '_neuralimage_model_kwargs', {})
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}
        return model_name, int(input_channels), dict(model_kwargs)

    def _resolve_artifact_runtime_metadata(self) -> dict[str, Any]:
        threshold = float(min(max(getattr(self, '_recommended_inference_threshold', 0.5), 0.0), 1.0))
        metadata: dict[str, Any] = {
            'inference': {
                'recommended_threshold': threshold,
            },
        }
        preprocessing = self._resolve_dataset_attribute(self._train_dataloader, '_preprocessing')
        if preprocessing is not None:
            metadata['preprocessing'] = {
                'config': asdict(preprocessing),
                'hash': preprocessing.stable_hash(),
            }
        supervision = self._resolved_supervision_targets()
        if supervision is not None:
            metadata['supervision_targets'] = asdict(supervision)
        return metadata

    def _save_model_artifact(self) -> None:
        model_name, input_channels, model_kwargs = self._resolve_model_artifact_metadata()
        save_model_artifact(
            self._base_model,
            self._save_path,
            model_name=model_name,
            input_channels=input_channels,
            model_kwargs=model_kwargs,
            metadata=self._resolve_artifact_runtime_metadata(),
        )

    @staticmethod
    def _sanitize_artifact_name(value: str, *, fallback: str = 'sample') -> str:
        cleaned = _ARTIFACT_NAME_SANITIZE_PATTERN.sub('_', str(value or '').strip()).strip('._-')
        return cleaned or fallback

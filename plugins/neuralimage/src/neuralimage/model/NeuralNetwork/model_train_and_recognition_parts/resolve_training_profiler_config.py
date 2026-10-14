"""model_train_and_recognition: resolve training profiler config."""
from __future__ import annotations
import os
from contextlib import nullcontext
from typing import Any, ContextManager
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    RANDOM_PATCH_PROFILE_BATCHES,
    RANDOM_PATCH_PROFILING_ENABLED,
    TRAINING_PROFILER_ENABLED,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _TrainingProfilerConfig


class ResolveTrainingProfilerConfigMixin:
    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = str(os.getenv(name, '1' if default else '0')).strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _env_int(name: str, default: int, minimum: int = 0) -> int:
        raw = os.getenv(name)
        if raw is None:
            return max(minimum, int(default))
        try:
            return max(minimum, int(raw))
        except Exception:
            return max(minimum, int(default))

    def _resolve_training_profiler_config(self) -> _TrainingProfilerConfig:
        random_patch_profile = bool(RANDOM_PATCH_PROFILING_ENABLED and self._uses_variable_patch_sizes())
        enabled = bool(TRAINING_PROFILER_ENABLED or random_patch_profile)
        max_batches = (
            max(1, int(RANDOM_PATCH_PROFILE_BATCHES))
            if random_patch_profile
            else self._env_int('NEURALIMAGE_TRAIN_PROFILE_STEPS', 40, minimum=1)
        )
        row_limit = self._env_int('NEURALIMAGE_TRAIN_PROFILE_ROW_LIMIT', 15, minimum=5)
        output_dir_name = str(os.getenv('NEURALIMAGE_TRAIN_PROFILE_DIR', 'profiles')).strip() or 'profiles'
        return _TrainingProfilerConfig(
            enabled=enabled,
            max_batches=max_batches,
            record_shapes=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_RECORD_SHAPES', True),
            profile_memory=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_MEMORY', True),
            with_stack=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_WITH_STACK', False),
            row_limit=row_limit,
            output_dir_name=output_dir_name,
        )

    def _random_patch_profiling_active(self) -> bool:
        return bool(RANDOM_PATCH_PROFILING_ENABLED and self._uses_variable_patch_sizes())

    def _profile_region(self, name: str) -> ContextManager[Any]:
        if not self._random_patch_profiling_active() or not hasattr(torch, 'profiler'):
            return nullcontext()
        return torch.profiler.record_function(f'random_patch/{name}')

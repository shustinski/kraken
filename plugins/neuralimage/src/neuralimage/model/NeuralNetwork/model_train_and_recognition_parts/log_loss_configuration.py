"""model_train_and_recognition: log loss configuration."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, ContextManager
import torch
from neuralimage.lib.data_interfaces import (
    MixedPrecisionMode,
)
from neuralimage.lib.loss_config import format_loss_formula

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.collect_memory_metrics import _collect_memory_metrics


class LogLossConfigurationMixin:
    def _step_training_profiler(self, active_profiler: _ActiveTrainProfiler | None) -> None:
        if active_profiler is None or active_profiler.is_closed:
            return
        if active_profiler.steps_left <= 0:
            self._stop_training_profiler(active_profiler)
            return
        try:
            active_profiler.profiler.step()
        finally:
            active_profiler.steps_left -= 1
            if active_profiler.steps_left <= 0:
                self._stop_training_profiler(active_profiler)

    def _stop_training_profiler(self, active_profiler: _ActiveTrainProfiler | None) -> None:
        if active_profiler is None or active_profiler.is_closed:
            return
        try:
            summary = active_profiler.profiler.key_averages().table(
                sort_by=active_profiler.summary_sort_key,
                row_limit=active_profiler.row_limit,
            )
            self._bus.put(['logging', f'Сводка профайлера (top {active_profiler.row_limit}):\n{summary}'])
            self._bus.put(['logging', f'Профиль сохранен: {active_profiler.trace_path}'])
        except Exception as profile_error:
            self._bus.put(['logging', f'Не удалось получить сводку профайлера: {profile_error}'])
        finally:
            try:
                active_profiler.profiler.__exit__(None, None, None)
            finally:
                active_profiler.is_closed = True

    def _prepare_training_device(self, device: torch.device, *, is_main_process: bool, distributed: bool) -> None:
        if device.type == 'cuda':
            variable_patch_sizes = self._uses_variable_patch_sizes()
            torch.backends.cudnn.benchmark = not variable_patch_sizes
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if variable_patch_sizes and is_main_process:
                self._bus.put([
                    'logging',
                    'cuDNN benchmark disabled: random patch sizes change input geometry between batches.',
                ])
        if not distributed:
            self._try_compile_model(is_main_process=is_main_process, device=device)

    def _resolve_train_loader_context(self) -> tuple[int, Any, bool, _TrainLoopStrides]:
        train_size = self._safe_loader_len(self._train_dataloader)
        train_sampler = getattr(self._train_dataloader, 'sampler', None) if self._train_dataloader is not None else None
        wrapped_sampler = getattr(getattr(self._train_dataloader, 'batch_sampler', None), 'sampler', None)
        if hasattr(wrapped_sampler, 'update_batch_losses'):
            train_sampler = wrapped_sampler
        supports_loss_aware_sampling = hasattr(train_sampler, 'update_batch_losses')
        strides = self._build_train_loop_strides(train_size, self._log_update_frequency)
        return train_size, train_sampler, supports_loss_aware_sampling, strides

    def _publish_training_start_metrics(self, train_size: int) -> None:
        self._bus.put(['metrics', {'type': 'train_epoch_progress', 'current': 0, 'total': int(self._epochs)}])
        self._bus.put(['metrics', {'type': 'train_batch_progress', 'current': 0, 'total': int(train_size)}])
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload}])

    def _log_training_start(self, *, is_main_process: bool, device: torch.device) -> None:
        if is_main_process:
            self._bus.put(['logging', f'Training device detected: {device}'])
        self._bus.put(['logging', 'Preparing data and starting the training loop'])

    def _log_mixed_precision_mode(
        self,
        *,
        resolved_mode: MixedPrecisionMode,
        device: torch.device,
    ) -> None:
        if resolved_mode != self._mixed_precision:
            self._bus.put([
                'logging',
                (
                    f'Mixed precision mode "{self._mixed_precision.value}" is unavailable on {device.type}. '
                    f'Using mode "{resolved_mode.value}".'
                ),
            ])
            return
        self._bus.put(['logging', f'Mixed precision mode: {resolved_mode.value}.'])

    def _log_loss_configuration(self) -> None:
        resolved_loss_mode = self._resolved_loss_function()
        resolved_loss_weights = self._resolved_loss_term_weights()
        if (not getattr(self, '_loss_term_weights', {})) and resolved_loss_mode != self._loss_function:
            self._bus.put(['logging', f'Unknown loss "{self._loss_function}". Using "bce".'])
        self._bus.put(['logging', f'Loss formula: {format_loss_formula(resolved_loss_weights)}'])
        for loss_mode, coefficient in resolved_loss_weights.items():
            self._bus.put(['logging', f'Loss term: {coefficient:.2f} * {loss_mode}.'])
        if self._hard_pixel_mining_enabled():
            supported_terms = [
                loss_mode for loss_mode in resolved_loss_weights
                if self._loss_supports_hard_pixel_mining(loss_mode)
            ]
            if supported_terms:
                self._bus.put([
                    'logging',
                    (
                        f'Hard pixel mining is enabled: keep_ratio={self._resolved_hard_pixel_keep_ratio():.2f}. '
                        f'Applied only to terms with pointwise loss: {", ".join(supported_terms)}.'
                    ),
                ])
            else:
                self._bus.put([
                    'logging',
                    (
                        'Hard pixel mining is enabled but ignored because all active loss terms '
                        'lack a pointwise pixel term.'
                    ),
                ])
        self._bus.put(['logging', 'Loss normalization: per-sample mean (batch-size invariant).'])
        if self._topograph_is_active():
            self._bus.put([
                'logging',
                (
                    f'Topograph loss enabled: weight={self._topograph_loss_weight:.2f}, '
                    f'connectivity=4, use_c={self._topograph_use_c}, '
                    f'num_processes={self._topograph_num_processes}.'
                ),
            ])
            if self._topograph_debug_viz:
                self._bus.put(['logging', 'Topograph debug visualization is enabled for batch previews.'])

    def _log_sampling_configuration(self, *, train_sampler: Any, supports_loss_aware_sampling: bool) -> None:
        if self._skip_uniform_labels:
            self._bus.put(['logging', 'Skipping uniform 0/1 labels is enabled for training batches.'])
        if not supports_loss_aware_sampling:
            return
        self._bus.put([
            'logging',
            'Hard-frame sampling is enabled; selected frames are repeated once in the next epoch.',
        ])

    @staticmethod
    def _build_autocast_context(
        *,
        device_type: str,
        autocast_dtype: torch.dtype | None,
    ) -> Callable[[], ContextManager[Any]]:
        if autocast_dtype is None:
            return lambda: nullcontext()
        return lambda: torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=True)

    def _log_warmup_configuration(self, scheduler: Any) -> None:
        if scheduler is None:
            return
        self._bus.put([
            'logging',
            (
                f'Warmup enabled: epochs={int(self._warmup_params.epochs)}, '
                f'start_factor={float(self._warmup_params.start_factor):.4f}.'
            ),
        ])

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _ActiveTrainProfiler, _TrainLoopStrides

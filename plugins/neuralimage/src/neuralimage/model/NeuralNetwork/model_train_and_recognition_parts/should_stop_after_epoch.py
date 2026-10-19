"""model_train_and_recognition: should stop after epoch."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.collect_memory_metrics import _collect_memory_metrics


class ShouldStopAfterEpochMixin:
    def _should_stop_after_epoch(
        self,
        *,
        epoch: int,
        device: torch.device,
        distributed: bool,
        runtime_state: _TrainingRuntimeState,
    ) -> bool:
        if runtime_state.automatic_stop_requested:
            if runtime_state.is_main_process:
                policy = self._automatic_early_stopping_policy
                best_batch = None if policy is None else policy.best_actual_batch
                self._bus.put([
                    'logging',
                    f'Automatic early stopping completed at epoch {epoch + 1}; best checkpoint batch={best_batch}.',
                ])
            return True
        should_stop = bool(
            runtime_state.early_stopping_config.enabled
            and runtime_state.early_stopping_state.bad_epochs > 0
            and runtime_state.early_stopping_state.bad_epochs >= runtime_state.early_stopping_config.patience
        )
        should_stop = self._sync_early_stopping_signal(
            should_stop,
            is_main_process=runtime_state.is_main_process,
            distributed=distributed,
            device=device,
        )
        if not should_stop:
            return False

        if runtime_state.is_main_process:
            self._bus.put([
                'logging',
                (
                    f'Early stopping triggered at epoch {epoch + 1}. '
                    f'Best val_loss reached at epoch {runtime_state.early_stopping_state.best_epoch}.'
                ),
            ])
        return True

    def _publish_epoch_end_memory(self, *, epoch: int, is_main_process: bool) -> None:
        if not is_main_process:
            return
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])

    def _finalize_training_success(self, *, is_main_process: bool) -> None:
        if is_main_process:
            self._save_random_patch_profile_csv()
            if self._automatic_early_stopping_config is not None:
                policy = self._automatic_early_stopping_policy
                if policy is not None and self._global_batch >= (
                    self._automatic_early_stopping_config.max_epochs
                    * self._automatic_early_stopping_config.batches_per_epoch
                ):
                    self._bus.put([
                        'logging',
                        f'Early stopping guard reached max_epochs={self._automatic_early_stopping_config.max_epochs}.',
                    ])
            best_payload = self._checkpoint_manager.restore_best(self._base_model)
            if best_payload is not None:
                self._recommended_inference_threshold = float(
                    best_payload.get('recommended_inference_threshold', self._recommended_inference_threshold)
                )
            self._save_model_artifact()
            try:
                self._save_metric_charts()
            except Exception as error:
                self._bus.put(['logging', f'Failed to save training metric charts: {error}'])
            try:
                log_path = self._save_path.parent / 'training_log.txt'
                self._bus.put(['logging', f'Training log saved: {log_path}'])
                self._save_training_log()
            except Exception as error:
                self._bus.put(['logging', f'Failed to save training log: {error}'])
        if self.callback is not None and is_main_process:
            self.callback()
        if is_main_process:
            print('Model saved successfully!')

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _TrainingRuntimeState

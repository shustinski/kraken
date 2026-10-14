"""model_train_and_recognition: run single training epoch."""
from __future__ import annotations
import time
from typing import cast
import torch


class RunSingleTrainingEpochMixin:
    def _run_single_training_epoch(
        self,
        *,
        epoch: int,
        device: torch.device,
        distributed: bool,
        runtime_state: _TrainingRuntimeState,
    ) -> None:
        run_context = runtime_state.run_context
        epoch_started_at = time.perf_counter()
        self._publish_epoch_start(epoch, run_context, distributed)

        train_stage_start = time.perf_counter()
        epoch_stats = self._run_train_epoch(
            epoch,
            device,
            run_context,
            is_main_process=runtime_state.is_main_process,
            distributed=distributed,
            active_profiler=runtime_state.active_profiler,
        )
        runtime_state.automatic_stop_requested = bool(getattr(epoch_stats, 'early_stop_requested', False))
        train_stage_ms = (time.perf_counter() - train_stage_start) * 1000.0

        global_train_loss, global_train_samples = self._reduce_epoch_train_stats(
            epoch_stats,
            distributed=distributed,
            device=device,
        )
        average_train_loss = (
            float(global_train_loss) / float(global_train_samples)
            if int(global_train_samples) > 0
            else None
        )
        if runtime_state.is_main_process:
            self._publish_epoch_train_metrics(
                epoch=epoch,
                run_context=run_context,
                epoch_stats=epoch_stats,
                global_train_loss=global_train_loss,
                global_train_samples=global_train_samples,
            )

        validation_start = time.perf_counter()
        validation_already_checked = bool(
            getattr(self, '_val_dataloader', None) is not None
            and getattr(self, '_latest_validation_result', None) is not None
            and getattr(self, '_latest_validation_global_batch', None) == self._global_batch
        )
        if validation_already_checked:
            validation_result = dict(cast(dict[str, float], self._latest_validation_result))
        else:
            validation_result = self._handle_validation(
                epoch=epoch,
                device=device,
                run_context=run_context,
                early_stopping_state=runtime_state.early_stopping_state,
                early_stopping_config=runtime_state.early_stopping_config,
                is_main_process=runtime_state.is_main_process,
            )
        if (
            runtime_state.is_main_process
            and getattr(self, '_automatic_early_stopping_policy', None) is None
            and getattr(self, '_checkpoint_manager', None) is not None
        ):
            checkpoint_loss = (
                float(validation_result['loss'])
                if validation_result is not None
                else average_train_loss
            )
            if checkpoint_loss is not None:
                self._checkpoint_manager.save_best(
                    model=self._base_model,
                    loss=float(checkpoint_loss),
                    global_batch=self._global_batch,
                    extra={
                        'metric_source': 'validation' if validation_result is not None else 'train',
                        'recommended_inference_threshold': float(self._recommended_inference_threshold),
                    },
                )
        self._step_epoch_scheduler(
            run_context=run_context,
            validation_result=validation_result,
            train_loss=average_train_loss,
            distributed=distributed,
            device=device,
            is_main_process=runtime_state.is_main_process,
        )
        validation_ms = (time.perf_counter() - validation_start) * 1000.0

        checkpoint_start = time.perf_counter()
        self._save_epoch_artifacts(
            epoch=epoch,
            run_context=run_context,
            early_stopping_state=runtime_state.early_stopping_state,
            is_main_process=runtime_state.is_main_process,
        )
        checkpoint_ms = (time.perf_counter() - checkpoint_start) * 1000.0

        if runtime_state.is_main_process:
            try:
                self._save_metric_charts()
            except Exception as error:
                self._bus.put(['logging', f'Failed to save training metric charts after epoch {epoch + 1}: {error}'])
            self._publish_epoch_load_breakdown(
                epoch=epoch,
                epoch_stats=epoch_stats,
                train_stage_ms=train_stage_ms,
                validation_ms=validation_ms,
                checkpoint_ms=checkpoint_ms,
            )
            self._bus.put([
                'logging',
                (
                    f'Epoch [{epoch + 1}/{self._epochs}] completed in '
                    f'{self._format_elapsed_duration(time.perf_counter() - epoch_started_at)}.'
                ),
            ])
            self._bus.put([
                'metrics',
                {'type': 'train_epoch_progress', 'current': int(epoch + 1), 'total': int(self._epochs)},
            ])

        return None

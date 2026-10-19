"""model_train_and_recognition: run train epoch."""
from __future__ import annotations
from typing import TYPE_CHECKING
import time
from typing import cast
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import _TrainingPauseRequested

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _EpochStats

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.stop_process_descendants import (
    _SupportsLossAwareSampling,
    _SupportsSetEpoch,
)


class RunTrainEpochMixin:
    def _run_train_epoch(
        self,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        *,
        is_main_process: bool = True,
        distributed: bool = False,
        active_profiler: _ActiveTrainProfiler | None = None,
    ) -> _EpochStats:
        epoch_stats = _EpochStats()
        if not hasattr(self, '_global_batch'):
            self._global_batch = 0
        if not hasattr(self, '_resume_batch'):
            self._resume_batch = 0
        if not hasattr(self, '_resume_epoch_torch_rng_state'):
            self._resume_epoch_torch_rng_state = None
        if not hasattr(self, '_resume_runtime_torch_rng_state'):
            self._resume_runtime_torch_rng_state = None
        last_automatic_check_batch: int | None = None
        validation_check_interval = self._validation_check_interval()

        train_dataset = self._train_dataloader.dataset
        if hasattr(train_dataset, 'set_epoch'):
            cast(_SupportsSetEpoch, train_dataset).set_epoch()
        train_sampler = getattr(self._train_dataloader, 'sampler', None)
        batch_sampler = getattr(self._train_dataloader, 'batch_sampler', None)
        wrapped_sampler = getattr(batch_sampler, 'sampler', None)
        if hasattr(wrapped_sampler, 'update_batch_losses'):
            train_sampler = wrapped_sampler
        if hasattr(train_sampler, 'resize'):
            base_dataset = getattr(train_dataset, '_base_dataset', train_dataset)
            reset_sampling_state = bool(
                getattr(base_dataset, 'shuffle_frames', False)
                or getattr(base_dataset, '_dynamic_frame_lengths', False)
            ) and not hasattr(train_sampler, 'finalize_epoch')
            cast(_SupportsLossAwareSampling, train_sampler).resize(
                len(train_dataset),
                reset=reset_sampling_state,
            )
        start_epoch_fn = getattr(train_sampler, 'start_epoch', None)
        if callable(start_epoch_fn) and self._resume_batch <= 0:
            start_epoch_fn()

        if self._resume_batch > 0 and self._resume_epoch_torch_rng_state is not None:
            torch.set_rng_state(self._resume_epoch_torch_rng_state)
        self._epoch_start_torch_rng_state = torch.get_rng_state().clone()
        train_iterator = iter(self._train_dataloader)
        batch_index = 0
        while True:
            data_wait_started_at = time.perf_counter()
            try:
                with self._profile_region('data_wait_and_cut'):
                    batch = next(train_iterator)
            except StopIteration:
                break
            if batch_index < self._resume_batch:
                batch_index += 1
                continue
            if self._periodic_validation_due(
                self._global_batch,
                last_automatic_check_batch,
                validation_check_interval,
            ):
                epoch_stats.early_stop_requested = self._run_automatic_early_stopping_check(
                    epoch=epoch,
                    device=device,
                    run_context=run_context,
                    is_main_process=is_main_process,
                    distributed=distributed,
                )
                last_automatic_check_batch = self._global_batch
                if epoch_stats.early_stop_requested:
                    break
            if self._resume_runtime_torch_rng_state is not None:
                torch.set_rng_state(self._resume_runtime_torch_rng_state)
                self._resume_runtime_torch_rng_state = None
            enable_cuda_event_timing = bool(getattr(run_context, 'enable_cuda_event_timing', True))
            with self._profile_region('transfer_and_augment'):
                prepared_batch, skipped_here = self._prepare_train_batch(
                    batch=batch,
                    device=device,
                    data_wait_started_at=data_wait_started_at,
                    enable_cuda_event_timing=enable_cuda_event_timing,
                )
            if skipped_here > 0:
                epoch_stats.skipped_uniform_count += skipped_here

            if prepared_batch is None:
                self._step_training_profiler(active_profiler)
                self._global_batch += 1
                batch_index += 1
                continue

            with self._profile_region('model_train_step'):
                step_result = self._run_train_step(run_context=run_context, batch=prepared_batch)
            if step_result is None:
                epoch_stats.skipped_non_finite_count += 1
                self._step_training_profiler(active_profiler)
                self._global_batch += 1
                batch_index += 1
                continue
            batch_total_ms = (time.perf_counter() - prepared_batch.batch_start) * 1000.0

            epoch_stats.add_batch(
                batch_samples=step_result.batch_samples,
                batch_loss=step_result.metric_loss,
                data_wait_ms=prepared_batch.data_wait_ms,
                augmentation_ms=prepared_batch.augmentation_ms,
                forward_ms=step_result.forward_ms,
                backward_ms=step_result.backward_ms,
                optimizer_ms=step_result.optimizer_ms,
                total_ms=batch_total_ms,
            )
            self._update_loss_aware_sampling(
                run_context=run_context,
                sample_indices=prepared_batch.sample_indices,
                per_sample_loss=step_result.per_sample_loss,
                mixup_pair_indices=prepared_batch.mixup_pair_indices,
                mixup_lambdas=prepared_batch.mixup_lambdas,
            )
            self._publish_train_batch_runtime(
                epoch=epoch,
                batch_index=batch_index,
                run_context=run_context,
                batch=prepared_batch,
                step_result=step_result,
                batch_total_ms=batch_total_ms,
            )
            self._record_random_patch_profile_batch(
                epoch=epoch,
                batch_index=batch_index,
                batch=prepared_batch,
                step_result=step_result,
                batch_total_ms=batch_total_ms,
            )
            self._step_training_profiler(active_profiler)
            self._global_batch += 1
            batch_index += 1
            pause_event = getattr(self, '_pause_event', None)
            if pause_event is not None and pause_event.is_set():
                raise _TrainingPauseRequested(epoch, batch_index)

        if (
            not epoch_stats.early_stop_requested
            and self._epoch_end_validation_due(
                self._global_batch,
                last_automatic_check_batch,
                validation_check_interval,
            )
        ):
            epoch_stats.early_stop_requested = self._run_automatic_early_stopping_check(
                epoch=epoch,
                device=device,
                run_context=run_context,
                is_main_process=is_main_process,
                distributed=distributed,
            )

        self._resume_batch = 0
        self._resume_epoch_torch_rng_state = None
        self._resume_runtime_torch_rng_state = None
        finalize_sampler_epoch = getattr(train_sampler, 'finalize_epoch', None)
        if callable(finalize_sampler_epoch):
            hard_frames = finalize_sampler_epoch()
            self._bus.put([
                'logging',
                (
                    f'Hard-frame sampling: frames={len(getattr(train_sampler, "last_frame_losses", {}))}, '
                    f'sigma={float(getattr(train_sampler, "last_sigma", 0.0)):.8f}, '
                    f'repeated_next_epoch={len(hard_frames)}.'
                ),
            ])
        return epoch_stats

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _ActiveTrainProfiler, _RunContext

"""model_train_and_recognition: run train step."""
from __future__ import annotations
from typing import TYPE_CHECKING
import time
from collections.abc import Mapping
from contextlib import nullcontext
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    NON_FINITE_GRADIENT_SKIP_LIMIT,
    TRAINING_INSTABILITY_MESSAGE,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import TrainingInstabilityError

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _TrainStepResult


class RunTrainStepMixin:
    def _run_train_step(
        self,
        *,
        run_context: _RunContext,
        batch: _PreparedTrainBatch,
    ) -> _TrainStepResult | None:
        run_context.optimizer.zero_grad(set_to_none=True)
        input_issue = self._find_first_non_finite_input_tensor(batch.inputs)
        if input_issue is not None or not self._is_finite_tensor(batch.label):
            self._log_non_finite_training_state(stage='pre_forward', batch=batch)

        step_device = batch.image.device
        forward_start = time.perf_counter()
        step_events = None
        if run_context.enable_cuda_event_timing:
            step_events = self._create_cuda_timing_events(step_device, 4)
        if step_events is not None:
            forward_start_event, forward_end_event, backward_end_event, optimizer_end_event = step_events
            self._record_cuda_timing_event(forward_start_event, step_device)
        backward_denominator = float(max(1, int(batch.image.size(0))))
        with run_context.autocast_ctx():
            outputs = self._forward_model(batch.inputs)
            per_sample_loss = self._compute_per_sample_loss(
                outputs,
                batch.label,
                run_context.bce_criterion,
                apply_pixel_mining=True,
            )
            metric_loss = per_sample_loss.mean()
            loss = per_sample_loss.sum() / backward_denominator
        if (not self._is_finite_tensor(loss)) or (not self._is_finite_tensor(metric_loss)):
            # Retry in full precision to avoid occasional AMP overflow/underflow instability.
            with nullcontext():
                retry_inputs = (
                    batch.image.float()
                    if not isinstance(batch.inputs, Mapping)
                    else {
                        key: value.float() if torch.is_tensor(value) else value
                        for key, value in batch.inputs.items()
                    }
                )
                outputs = self._forward_model(retry_inputs)
                per_sample_loss = self._compute_per_sample_loss(
                    outputs,
                    batch.label,
                    run_context.bce_criterion,
                    apply_pixel_mining=True,
                )
                metric_loss = per_sample_loss.mean()
                loss = per_sample_loss.sum() / backward_denominator
            if (not self._is_finite_tensor(loss)) or (not self._is_finite_tensor(metric_loss)):
                self._bus.put(['logging', 'Training warning: non-finite loss detected, batch skipped.'])
                self._log_non_finite_training_state(stage='non_finite_loss', batch=batch, outputs=outputs)
                run_context.optimizer.zero_grad(set_to_none=True)
                return None
        if step_events is not None:
            self._record_cuda_timing_event(forward_end_event, step_device)
            forward_ms = 0.0
        else:
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

        backward_start = time.perf_counter()
        run_context.scaler.scale(loss).backward()
        run_context.scaler.unscale_(run_context.optimizer)
        self._synchronize_loss_weighter_gradients()
        if self._has_non_finite_gradients():
            self._bus.put(['logging', 'Training warning: non-finite gradients detected, optimizer step skipped.'])
            self._non_finite_gradient_skip_count = int(
                getattr(self, '_non_finite_gradient_skip_count', 0)
            ) + 1
            self._log_non_finite_training_state(stage='non_finite_gradients', batch=batch, outputs=outputs)
            run_context.optimizer.zero_grad(set_to_none=True)
            run_context.scaler.update()
            if self._non_finite_gradient_skip_count > NON_FINITE_GRADIENT_SKIP_LIMIT:
                raise TrainingInstabilityError(TRAINING_INSTABILITY_MESSAGE)
            if step_events is not None:
                self._record_cuda_timing_event(backward_end_event, step_device)
                backward_end_event.synchronize()
                forward_ms = self._elapsed_cuda_event_ms(forward_start_event, forward_end_event)
                backward_ms = self._elapsed_cuda_event_ms(forward_end_event, backward_end_event)
            else:
                backward_ms = (time.perf_counter() - backward_start) * 1000.0
            return _TrainStepResult(
                outputs=outputs,
                per_sample_loss=per_sample_loss,
                metric_loss=metric_loss.detach(),
                loss_components=dict(getattr(self, '_last_batch_loss_components', {})),
                batch_samples=int(batch.image.size(0)),
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                optimizer_ms=0.0,
            )
        torch.nn.utils.clip_grad_norm_(tuple(self._all_trainable_parameters()), 1.0)
        if step_events is not None:
            self._record_cuda_timing_event(backward_end_event, step_device)
            backward_ms = 0.0
        else:
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

        optimizer_start = time.perf_counter()
        run_context.scaler.step(run_context.optimizer)
        run_context.scaler.update()
        self._step_batch_schedulers(run_context)
        if step_events is not None:
            self._record_cuda_timing_event(optimizer_end_event, step_device)
            optimizer_end_event.synchronize()
            forward_ms = self._elapsed_cuda_event_ms(forward_start_event, forward_end_event)
            backward_ms = self._elapsed_cuda_event_ms(forward_end_event, backward_end_event)
            optimizer_ms = self._elapsed_cuda_event_ms(backward_end_event, optimizer_end_event)
        else:
            optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0

        return _TrainStepResult(
            outputs=outputs,
            per_sample_loss=per_sample_loss,
            metric_loss=metric_loss.detach(),
            loss_components=dict(getattr(self, '_last_batch_loss_components', {})),
            batch_samples=int(batch.image.size(0)),
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            optimizer_ms=optimizer_ms,
        )

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _PreparedTrainBatch, _RunContext

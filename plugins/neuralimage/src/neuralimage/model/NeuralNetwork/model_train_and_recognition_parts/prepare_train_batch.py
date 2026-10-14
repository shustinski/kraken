"""model_train_and_recognition: prepare train batch."""
from __future__ import annotations
import time
from typing import Any
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _PreparedTrainBatch


class PrepareTrainBatchMixin:
    def _prepare_train_batch(
        self,
        *,
        batch: Any,
        device: torch.device,
        data_wait_started_at: float,
        enable_cuda_event_timing: bool = True,
    ) -> tuple[_PreparedTrainBatch | None, int]:
        data, target, sample_indices = self._split_batch(batch)
        inputs = self._move_batch_input_to_device(data, device)
        label = self._move_batch_target_to_device(target, device)
        inputs, label, sample_indices, skipped_here, has_valid_samples = self._filter_uniform_batch_samples(
            inputs,
            label,
            sample_indices,
        )
        batch_start = data_wait_started_at
        data_wait_ms = (time.perf_counter() - data_wait_started_at) * 1000.0
        if not has_valid_samples:
            return None, skipped_here
        has_batch_augmentations = self._has_training_batch_augmentations()
        augmentation_ms = 0.0
        augmentation_events = None
        if has_batch_augmentations:
            augmentation_start = time.perf_counter()
            if enable_cuda_event_timing:
                augmentation_events = self._create_cuda_timing_events(device, 2)
            if augmentation_events is not None:
                augmentation_start_event, augmentation_end_event = augmentation_events
                self._record_cuda_timing_event(augmentation_start_event, device)
        inputs, label, mixup_pair_indices, mixup_lambdas = self._apply_training_batch_augmentations(
            inputs,
            label,
            sample_indices,
        )
        if augmentation_events is not None:
            self._record_cuda_timing_event(augmentation_end_event, device)
            augmentation_end_event.synchronize()
            augmentation_ms = self._elapsed_cuda_event_ms(augmentation_start_event, augmentation_end_event)
        elif has_batch_augmentations:
            augmentation_ms = (time.perf_counter() - augmentation_start) * 1000.0
        image = self._extract_local_image(inputs)
        context_image = self._extract_context_image(inputs)
        return (
            _PreparedTrainBatch(
                data=data,
                target=target,
                sample_indices=sample_indices,
                mixup_pair_indices=mixup_pair_indices,
                mixup_lambdas=mixup_lambdas,
                inputs=inputs,
                image=image,
                context_image=context_image,
                label=label,
                batch_start=batch_start,
                data_wait_ms=data_wait_ms,
                augmentation_ms=augmentation_ms,
            ),
            skipped_here,
        )

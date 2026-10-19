"""model_train_and_recognition: record random patch profile."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    RANDOM_PATCH_PROFILE_BATCHES,
    RANDOM_PATCH_PROFILE_LOG_EVERY,
)


class RecordRandomPatchProfileMixin:
    def _record_random_patch_profile_batch(
        self,
        *,
        epoch: int,
        batch_index: int,
        batch: _PreparedTrainBatch,
        step_result: _TrainStepResult,
        batch_total_ms: float,
    ) -> None:
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import RANDOM_PATCH_PROFILE_BATCHES

        if not self._random_patch_profiling_active():
            return
        rows = self._random_patch_profile_rows
        if len(rows) >= max(1, int(RANDOM_PATCH_PROFILE_BATCHES)):
            return
        height, width = (int(value) for value in batch.image.shape[-2:])
        batch_size = int(batch.image.shape[0])
        row: dict[str, int | float] = {
            'epoch': int(epoch + 1),
            'batch': int(batch_index + 1),
            'batch_size': batch_size,
            'width': width,
            'height': height,
            'pixels_per_object': width * height,
            'megapixels_per_batch': (batch_size * width * height) / 1_000_000.0,
            'data_wait_ms': float(batch.data_wait_ms),
            'augmentation_ms': float(batch.augmentation_ms),
            'forward_ms': float(step_result.forward_ms),
            'backward_ms': float(step_result.backward_ms),
            'optimizer_ms': float(step_result.optimizer_ms),
            'total_ms': float(batch_total_ms),
            'objects_per_second': (batch_size * 1000.0) / max(float(batch_total_ms), 1e-9),
            'cuda_allocated_mb': 0.0,
            'cuda_reserved_mb': 0.0,
        }
        if batch.image.is_cuda:
            device = batch.image.device
            row['cuda_allocated_mb'] = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            row['cuda_reserved_mb'] = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        rows.append(row)
        log_every = max(1, int(RANDOM_PATCH_PROFILE_LOG_EVERY))
        if len(rows) == 1 or len(rows) % log_every == 0:
            self._bus.put([
                'logging',
                (
                    f'Random-patch profile [{len(rows)}/{RANDOM_PATCH_PROFILE_BATCHES}]: '
                    f'{width}x{height}, batch={batch_size}, total={batch_total_ms:.2f} ms, '
                    f'data={batch.data_wait_ms:.2f}, aug={batch.augmentation_ms:.2f}, '
                    f'forward={step_result.forward_ms:.2f}, backward={step_result.backward_ms:.2f}, '
                    f'optimizer={step_result.optimizer_ms:.2f} ms.'
                ),
            ])
            self._save_random_patch_profile_csv()

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _PreparedTrainBatch, _TrainStepResult

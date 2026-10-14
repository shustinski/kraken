"""model_train_and_recognition: publish train batch runtime."""
from __future__ import annotations
from typing import Any, cast
import torch


class PublishTrainBatchRuntimeMixin:
    def _publish_train_batch_runtime(
        self,
        *,
        epoch: int,
        batch_index: int,
        run_context: _RunContext,
        batch: _PreparedTrainBatch,
        step_result: _TrainStepResult,
        batch_total_ms: float,
    ) -> None:
        self._publish_batch_preview(
            epoch=epoch,
            batch_index=batch_index,
            train_size=run_context.train_size,
            data=batch.image,
            target=batch.label,
            outputs=step_result.outputs,
            preview_stride=run_context.strides.preview,
            sample_indices=batch.sample_indices,
        )

        if (batch_index % run_context.strides.metric == 0) or (batch_index == run_context.train_size - 1):
            batch_loss = self._loss_value_to_float(cast(torch.Tensor | float, step_result.metric_loss))
            loss_components = dict(step_result.loss_components or getattr(self, '_last_batch_loss_components', {}))
            epoch_points = self._batch_points_by_epoch.setdefault(int(epoch + 1), [])
            epoch_points.append((float(batch_index + 1), batch_loss))
            train_batch_metric: dict[str, Any] = {
                'type': 'train_batch',
                'epoch': epoch + 1,
                'batch_index': batch_index + 1,
                'loss': batch_loss,
            }
            if loss_components:
                train_batch_metric['loss_components'] = loss_components
            self._bus.put(['metrics', train_batch_metric])
            if loss_components:
                self._bus.put([
                    'logging',
                    (
                        'Loss components: '
                        f'BCE={loss_components.get("bce", 0.0):.4f} '
                        f'Dice={loss_components.get("dice", 0.0):.4f} '
                        f'Topograph={loss_components.get("topograph", 0.0):.4f} '
                        f'Total={loss_components.get("total", batch_loss):.4f}'
                    ),
                ])
            self._bus.put([
                'metrics',
                {
                    'type': 'train_perf',
                    'epoch': epoch + 1,
                    'batch_index': batch_index + 1,
                    'data_wait_ms': float(batch.data_wait_ms),
                    'augmentation_ms': float(batch.augmentation_ms),
                    'forward_ms': float(step_result.forward_ms),
                    'backward_ms': float(step_result.backward_ms),
                    'optimizer_ms': float(step_result.optimizer_ms),
                    'total_ms': float(batch_total_ms),
                },
            ])

        if (batch_index % run_context.strides.progress == 0) or (batch_index == run_context.train_size - 1):
            self._bus.put([
                'metrics',
                {
                    'type': 'train_batch_progress',
                    'current': int(batch_index + 1),
                    'total': int(run_context.train_size),
                },
            ])

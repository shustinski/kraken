"""model_train_and_recognition: publish epoch load breakdown."""
from __future__ import annotations


class PublishEpochLoadBreakdownMixin:
    def _publish_epoch_train_metrics(
        self,
        *,
        epoch: int,
        run_context: _RunContext,
        epoch_stats: _EpochStats,
        global_train_loss: float,
        global_train_samples: float,
    ) -> None:
        if global_train_samples <= 0:
            return
        avg_train_loss = global_train_loss / global_train_samples
        self._train_epoch_history.append((float(epoch + 1), float(avg_train_loss)))
        self._bus.put(['metrics', {'type': 'train_epoch', 'epoch': epoch + 1, 'loss': float(avg_train_loss)}])
        avg_denom = max(1, run_context.train_size)
        self._bus.put([
            'metrics',
            {
                'type': 'train_perf_epoch',
                'epoch': epoch + 1,
                'data_wait_ms': float(epoch_stats.data_wait_ms / avg_denom),
                'augmentation_ms': float(epoch_stats.augmentation_ms / avg_denom),
                'forward_ms': float(epoch_stats.forward_ms / avg_denom),
                'backward_ms': float(epoch_stats.backward_ms / avg_denom),
                'optimizer_ms': float(epoch_stats.optimizer_ms / avg_denom),
                'total_ms': float(epoch_stats.total_ms / avg_denom),
            },
        ])
        if self._skip_uniform_labels and epoch_stats.skipped_uniform_count > 0:
            self._bus.put([
                'logging',
                f'Пропущено сэмплов из-за uniform label (0/1): {epoch_stats.skipped_uniform_count}',
            ])
        if epoch_stats.skipped_non_finite_count > 0:
            self._bus.put([
                'logging',
                (
                    'Training warning: '
                    f'skipped {epoch_stats.skipped_non_finite_count} batch(es) with non-finite loss/gradients.'
                ),
            ])

    def _publish_epoch_load_breakdown(
        self,
        *,
        epoch: int,
        epoch_stats: _EpochStats,
        train_stage_ms: float,
        validation_ms: float,
        checkpoint_ms: float,
    ) -> None:
        total_epoch_ms = max(1e-6, float(train_stage_ms + validation_ms + checkpoint_ms))
        parts: list[tuple[str, float]] = [
            ('data_wait', float(epoch_stats.data_wait_ms)),
            ('augmentation', float(epoch_stats.augmentation_ms)),
            ('forward', float(epoch_stats.forward_ms)),
            ('backward', float(epoch_stats.backward_ms)),
            ('optimizer', float(epoch_stats.optimizer_ms)),
            ('validation', float(validation_ms)),
            ('checkpoint', float(checkpoint_ms)),
        ]
        dominant_name, dominant_ms = max(parts, key=lambda x: x[1])
        breakdown = ' | '.join(
            f'{name}: {ms/1000.0:.2f}s ({(ms / total_epoch_ms) * 100.0:.1f}%)'
            for name, ms in parts
        )
        self._bus.put([
            'logging',
            (
                f'Профилирование по времени, эпоха {epoch + 1}: {breakdown}. '
                f'Наибольшая нагрузка: {dominant_name} ({dominant_ms/1000.0:.2f}s).'
            ),
        ])
        self._bus.put([
            'metrics',
            {
                'type': 'epoch_time_breakdown',
                'epoch': int(epoch + 1),
                'train_total_ms': float(train_stage_ms),
                'data_wait_ms': float(epoch_stats.data_wait_ms),
                'augmentation_ms': float(epoch_stats.augmentation_ms),
                'forward_ms': float(epoch_stats.forward_ms),
                'backward_ms': float(epoch_stats.backward_ms),
                'optimizer_ms': float(epoch_stats.optimizer_ms),
                'validation_ms': float(validation_ms),
                'checkpoint_ms': float(checkpoint_ms),
                'dominant_stage': dominant_name,
            },
        ])

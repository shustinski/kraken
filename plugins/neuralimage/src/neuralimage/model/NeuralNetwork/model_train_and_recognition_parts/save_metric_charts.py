"""model_train_and_recognition: save metric charts."""
from __future__ import annotations


class SaveMetricChartsMixin:
    def _save_metric_charts(self) -> None:
        artifact_dir = self._save_path.parent
        if self._train_epoch_history or self._val_epoch_history:
            loss_chart_series = [
                ('Train Loss', list(self._train_epoch_history), (80, 186, 255)),
                ('Val Loss', list(self._val_epoch_history), (255, 170, 92)),
            ]
            self._render_chart_image(
                save_path=artifact_dir / 'training_metrics_loss_by_epoch.png',
                title='Loss vs Epoch',
                x_label='Epoch',
                y_label='Loss',
                series=loss_chart_series,
            )
            self._save_chart_csv(
                save_path=artifact_dir / 'training_metrics_loss_by_epoch.csv',
                x_label='Epoch',
                series=loss_chart_series,
            )
        if self._val_iou_history or self._val_dice_history:
            validation_quality_series = [
                ('IoU', list(self._val_iou_history), (142, 210, 110)),
                ('Dice', list(self._val_dice_history), (255, 111, 145)),
            ]
            self._render_chart_image(
                save_path=artifact_dir / 'training_metrics_validation_quality.png',
                title='Validation IoU / Dice vs Epoch',
                x_label='Epoch',
                y_label='Score',
                series=validation_quality_series,
                y_formatter=lambda value: f'{float(value) * 100.0:.1f}%',
            )
            self._save_chart_csv(
                save_path=artifact_dir / 'training_metrics_validation_quality.csv',
                x_label='Epoch',
                series=validation_quality_series,
            )
        if self._batch_points_by_epoch:
            last_epoch = max(self._batch_points_by_epoch)
            batch_chart_series = [
                (
                    'Train Loss',
                    list(self._batch_points_by_epoch.get(last_epoch, [])),
                    (137, 228, 125),
                ),
            ]
            self._render_chart_image(
                save_path=artifact_dir / 'training_metrics_train_loss_by_batch.png',
                title=f'Train Loss vs Batch (Epoch {int(last_epoch)})',
                x_label='Batch',
                y_label='Loss',
                series=batch_chart_series,
            )
            self._save_chart_csv(
                save_path=artifact_dir / 'training_metrics_train_loss_by_batch.csv',
                x_label='Batch',
                series=batch_chart_series,
            )

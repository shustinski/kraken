"""model_train_and_recognition: publish validation metrics."""
from __future__ import annotations
import torch
import torch.distributed as dist

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _TrainingRuntimeState


class PublishValidationMetricsMixin:
    def _run_validation_if_main_process(
        self,
        *,
        epoch: int,
        is_main_process: bool,
        device: torch.device,
        run_context: _RunContext,
    ) -> dict[str, float] | None:
        if not is_main_process:
            return None
        return self._run_validation_epoch(
            epoch,
            device,
            run_context.bce_criterion,
            run_context.autocast_ctx,
        )

    def _publish_validation_metrics(self, *, epoch: int, validation_result: dict[str, float]) -> None:
        avg_val_loss = validation_result['loss']
        val_accuracy = validation_result['accuracy']
        val_iou = validation_result['iou']
        val_dice = validation_result['dice']
        val_f1 = validation_result['f1']
        best_threshold = float(validation_result.get('best_threshold', 0.5))
        best_threshold_iou = float(validation_result.get('best_threshold_iou', val_iou))
        best_threshold_dice = float(validation_result.get('best_threshold_dice', val_dice))
        best_threshold_f1 = float(validation_result.get('best_threshold_f1', val_f1))
        self._val_epoch_history.append((float(epoch + 1), float(avg_val_loss)))
        self._val_iou_history.append((float(epoch + 1), float(val_iou)))
        self._val_dice_history.append((float(epoch + 1), float(val_dice)))

        self._bus.put([
            'logging',
            (
                f'Epoch [{epoch + 1}/{self._epochs}] '
                f'Validation accuracy: {val_accuracy:.4%} | '
                f'IoU: {val_iou:.4%} | Dice: {val_dice:.4%} | F1: {val_f1:.4%} | '
                f'Best threshold: {best_threshold:.2f} '
                f'(IoU {best_threshold_iou:.4%}, Dice {best_threshold_dice:.4%}, F1 {best_threshold_f1:.4%})'
            ),
        ])
        self._bus.put([
            'metrics',
            {
                'type': 'val_epoch',
                'epoch': epoch + 1,
                'loss': float(avg_val_loss),
                'accuracy': float(val_accuracy),
                'iou': float(val_iou),
                'dice': float(val_dice),
                'f1': float(val_f1),
                'best_threshold': float(best_threshold),
                'best_threshold_iou': float(best_threshold_iou),
                'best_threshold_dice': float(best_threshold_dice),
                'best_threshold_f1': float(best_threshold_f1),
            },
        ])

    def _update_early_stopping_from_validation(
        self,
        *,
        epoch: int,
        validation_result: dict[str, float],
        early_stopping_state: _EarlyStoppingState,
        early_stopping_config: _EarlyStoppingConfig,
    ) -> None:
        if not early_stopping_config.enabled:
            return

        avg_val_loss = float(validation_result['loss'])
        improved = (
            early_stopping_state.best_loss is None
            or (early_stopping_state.best_loss - avg_val_loss) > early_stopping_config.min_delta
        )
        if improved:
            early_stopping_state.best_loss = avg_val_loss
            early_stopping_state.bad_epochs = 0
            early_stopping_state.best_epoch = int(epoch + 1)
            early_stopping_state.best_threshold = float(validation_result.get('best_threshold', self._recommended_inference_threshold))
            self._bus.put([
                'logging',
                f'Early stopping: new best result at epoch {epoch + 1} (val_loss={avg_val_loss:.6f}).',
            ])
            return

        early_stopping_state.bad_epochs += 1
        self._bus.put([
            'logging',
            f'Early stopping: no improvement {early_stopping_state.bad_epochs}/{early_stopping_config.patience}.',
        ])

    def _handle_validation(
        self,
        *,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        early_stopping_state: _EarlyStoppingState,
        early_stopping_config: _EarlyStoppingConfig,
        is_main_process: bool,
    ) -> dict[str, float] | None:
        validation_result = self._run_validation_if_main_process(
            epoch=epoch,
            is_main_process=is_main_process,
            device=device,
            run_context=run_context,
        )
        if validation_result is None:
            return None

        self._publish_validation_metrics(epoch=epoch, validation_result=validation_result)
        self._update_early_stopping_from_validation(
            epoch=epoch,
            validation_result=validation_result,
            early_stopping_state=early_stopping_state,
            early_stopping_config=early_stopping_config,
        )
        return validation_result

    def _save_epoch_artifacts(
        self,
        *,
        epoch: int,
        run_context: _RunContext,
        early_stopping_state: _EarlyStoppingState,
        is_main_process: bool,
    ) -> None:
        if not is_main_process:
            return
        self._save_model_artifact()
        self._save_checkpoint(
            epoch + 1,
            run_context.optimizer,
            run_context.scaler,
            run_context.warmup_scheduler,
            run_context.scheduler,
            early_stopping_best_loss=early_stopping_state.best_loss,
            early_stopping_bad_epochs=early_stopping_state.bad_epochs,
            early_stopping_best_epoch=early_stopping_state.best_epoch,
            early_stopping_best_model_state=early_stopping_state.best_model_state,
            early_stopping_best_threshold=early_stopping_state.best_threshold,
        )

    @staticmethod
    def _sync_early_stopping_signal(
        should_stop: bool,
        *,
        is_main_process: bool,
        distributed: bool,
        device: torch.device,
    ) -> bool:
        if not distributed:
            return bool(should_stop)
        stop_tensor = torch.tensor([1 if (should_stop and is_main_process) else 0], device=device, dtype=torch.int32)
        dist.broadcast(stop_tensor, src=0)
        return bool(int(stop_tensor.item()))

    def _initialize_training_runtime(
        self,
        *,
        device: torch.device,
        rank: int,
        distributed: bool,
    ) -> _TrainingRuntimeState:
        is_main_process = (rank == 0)
        run_context = self._create_run_context(device, is_main_process, distributed)
        start_epoch, early_stopping_state, early_stopping_config = self._restore_training_state(run_context)
        active_profiler = self._start_training_profiler(
            device=device,
            train_size=run_context.train_size,
            is_main_process=is_main_process,
            distributed=distributed,
        )
        return _TrainingRuntimeState(
            is_main_process=is_main_process,
            run_context=run_context,
            start_epoch=start_epoch,
            early_stopping_state=early_stopping_state,
            early_stopping_config=early_stopping_config,
            active_profiler=active_profiler,
        )

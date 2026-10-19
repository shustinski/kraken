"""model_train_and_recognition: create training process."""
from __future__ import annotations
from typing import TYPE_CHECKING
import time
from typing import Any, cast
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import _drain_process_queue


class CreateTrainingProcessMixin:
    def _validate_training_inputs(self) -> bool:
        if self._train_dataloader is not None and self._model is not None:
            return True
        self.error_message = 'Ошибка: отсутствуют необходимые данные для обучения модели.'
        self._bus.publish('error', self.error_message)
        return False

    def _create_training_process(self) -> 'TrainerProcess':
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess

        train_dataloader = cast(DataLoader, self._train_dataloader)
        val_dataloader = cast(DataLoader | None, self._val_dataloader)
        model = cast(nn.Module, self._model)
        return TrainerProcess(
            train_dataloader,
            val_dataloader,
            model,
            self._save_path,
            self._epochs,
            self.message_queue,
            optimizer_params=self._optimizer_params,
            mixed_precision=self._mixed_precision,
            loss_function=self._loss_function,
            loss_term_weights=self._loss_term_weights,
            dice_loss_weight=self._dice_loss_weight,
            iou_loss_weight=self._iou_loss_weight,
            hard_mining_params=self._hard_mining_params,
            cutout_params=self._cutout_params,
            random_artifacts_params=self._random_artifacts_params,
            mixup_params=self._mixup_params,
            early_stopping_params=self._early_stopping_params,
            warmup_params=self._warmup_params,
            scheduler_params=self._scheduler_params,
            skip_uniform_labels=self._skip_uniform_labels,
            resume_from_checkpoint=self._resume_from_checkpoint,
            extend_epochs_on_resume=self._extend_epochs_on_resume,
            use_multi_gpu=self._use_multi_gpu,
            multi_gpu_mode=self._multi_gpu_mode,
            show_batch_preview=self._show_batch_preview,
            log_update_frequency=self._log_update_frequency,
            save_validation_binary_images=self._save_validation_binary_images,
            control_dataloader=self._control_dataloader,
            pause_event=self._pause_event,
            loss_weighting_strategy=self._loss_weighting_strategy,
            mask_loss_weight_floor=self._mask_loss_weight_floor,
            topograph_enabled=self._topograph_enabled,
            topograph_loss_weight=self._topograph_loss_weight,
            topograph_debug_viz=self._topograph_debug_viz,
            topograph_num_processes=self._topograph_num_processes,
            topograph_use_c=self._topograph_use_c,
        )

    @staticmethod
    def _elapsed_suffix(since: float) -> str:
        elapsed_seconds = int(max(0.0, time.perf_counter() - since))
        elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
        elapsed_minutes, elapsed_secs = divmod(remainder, 60)
        return f' Прошло: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}'

    def _publish_training_message(
        self,
        message: Any,
        *,
        append_elapsed_suffix: bool,
        started_at: float,
    ) -> None:
        topic, payload = message[0], message[1]
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(started_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)

    def _drain_training_queue(self, *, append_elapsed_suffix: bool, started_at: float) -> None:
        def _publish(queued_message: Any) -> None:
            self._publish_training_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                started_at=started_at,
            )
        _drain_process_queue(self.message_queue, _publish)

    def _finalize_training_result(self, training_process: Any) -> bool:
        if self._stop_event.is_set():
            self.succeeded = False
            return False
        exit_code = training_process.exitcode if training_process is not None else 1
        if exit_code not in (0, None):
            if self.error_message is None:
                self.error_message = f'Training error: process exited with code {exit_code}.'
                self._bus.publish('error', self.error_message)
            self.succeeded = False
            return False
        self.succeeded = True
        if self.callback is not None:
            self.callback()
        print('Model saved successfully!')
        return True

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess

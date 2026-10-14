"""model_train_and_recognition: periodic validation due."""
from __future__ import annotations

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import VALIDATION_CHECK_INTERVAL_BATCHES


class PeriodicValidationDueMixin:
    def _validation_check_interval(self) -> int | None:
        if getattr(self, '_val_dataloader', None) is not None:
            return VALIDATION_CHECK_INTERVAL_BATCHES
        config = getattr(self, '_automatic_early_stopping_config', None)
        if config is None:
            return None
        return max(1, int(config.check_every_batches))

    @staticmethod
    def _periodic_validation_due(global_batch: int, last_check_batch: int | None, interval: int | None) -> bool:
        return bool(
            interval is not None
            and int(global_batch) > 0
            and int(global_batch) % int(interval) == 0
            and last_check_batch != int(global_batch)
        )

    @staticmethod
    def _epoch_end_validation_due(global_batch: int, last_check_batch: int | None, interval: int | None) -> bool:
        return bool(
            interval is not None
            and int(global_batch) > 0
            and last_check_batch != int(global_batch)
        )

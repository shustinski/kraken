"""model_train_and_recognition: restore training state."""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
from collections.abc import Mapping, Sized
from typing import cast
from neuralimage.model.NeuralNetwork.early_stopping import (
    EarlyStoppingConfigCalculator,
    EarlyStoppingPolicy,
    EarlyStoppingRuntimeConfig,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import VALIDATION_CHECK_INTERVAL_BATCHES

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import (
    _EarlyStoppingConfig,
    _EarlyStoppingState,
)


class RestoreTrainingStateMixin:
    def _restore_training_state(self, run_context: _RunContext) -> tuple[int, _EarlyStoppingState, _EarlyStoppingConfig]:
        start_epoch, checkpoint = self._load_checkpoint_if_available(
            run_context.optimizer,
            run_context.scaler,
            run_context.warmup_scheduler,
            run_context.scheduler,
        )
        early_stopping_state = _EarlyStoppingState(
            best_loss=(
                float(checkpoint['early_stopping_best_loss'])
                if checkpoint.get('early_stopping_best_loss') is not None
                else None
            ),
            bad_epochs=int(checkpoint.get('early_stopping_bad_epochs', 0)),
            best_epoch=int(checkpoint.get('early_stopping_best_epoch', 0)),
            best_model_state=checkpoint.get('early_stopping_best_model_state'),
            best_threshold=float(checkpoint.get('early_stopping_best_threshold', self._recommended_inference_threshold)),
        )

        early_stopping_config = _EarlyStoppingConfig(
            enabled=False,
            patience=0,
            min_delta=0.0,
        )
        if self._early_stopping_params.enabled:
            train_size = int(len(cast(Sized, self._train_dataloader.dataset)))
            batches_per_epoch = max(1, int(len(self._train_dataloader)))
            automatic_config = EarlyStoppingConfigCalculator.calculate(
                train_size=train_size,
                batches_per_epoch=batches_per_epoch,
            )
            self._global_batch = max(0, int(checkpoint.get('global_batch', start_epoch * batches_per_epoch)))
            self._epochs = int(automatic_config.max_epochs)
            if self._val_dataloader is not None:
                checks_per_epoch = max(1, math.ceil(batches_per_epoch / VALIDATION_CHECK_INTERVAL_BATCHES))
                automatic_config = EarlyStoppingRuntimeConfig(
                    train_size=automatic_config.train_size,
                    batches_per_epoch=automatic_config.batches_per_epoch,
                    control_size=automatic_config.control_size,
                    check_every_batches=VALIDATION_CHECK_INTERVAL_BATCHES,
                    checks_per_epoch=checks_per_epoch,
                    patience_checks=max(3, 2 * checks_per_epoch),
                    warmup_batches=automatic_config.warmup_batches,
                    trend_window=max(2, 2 * checks_per_epoch),
                    max_epochs=automatic_config.max_epochs,
                    single_check_min_improvement=automatic_config.single_check_min_improvement,
                    trend_min_improvement=automatic_config.trend_min_improvement,
                )
            self._automatic_early_stopping_config = automatic_config
            self._automatic_early_stopping_policy = EarlyStoppingPolicy(automatic_config)
            policy_state = checkpoint.get('automatic_early_stopping_state')
            if isinstance(policy_state, Mapping):
                self._automatic_early_stopping_policy.load_state_dict(policy_state)
            self._checkpoint_manager.best_loss = self._automatic_early_stopping_policy.best_actual_loss
            self._checkpoint_manager.best_global_batch = self._automatic_early_stopping_policy.best_actual_batch
            metric_source = 'frame-averaged validation loss' if self._val_dataloader is not None else 'train-control loss'
            self._bus.put([
                'logging',
                (
                    'Automatic early stopping enabled: '
                    f'metric={metric_source}, train_size={train_size}, '
                    f'batches_per_epoch={batches_per_epoch}, control_size={automatic_config.control_size}, '
                    f'check_every_batches={automatic_config.check_every_batches}, '
                    f'checks_per_epoch={automatic_config.checks_per_epoch}, '
                    f'patience_checks={automatic_config.patience_checks}, '
                    f'warmup_batches={automatic_config.warmup_batches}, '
                    f'trend_window={automatic_config.trend_window}, max_epochs={automatic_config.max_epochs}.'
                ),
            ])

        target_epochs = self._epochs if self._early_stopping_params.enabled else self._resolve_target_epochs(start_epoch)
        if target_epochs != self._epochs:
            self._bus.put([
                'logging',
                (
                    f'Режим дообучения: продолжение с эпохи {start_epoch}. '
                    f'Будет добавлено {self._epochs} эпох (до {target_epochs}).'
                ),
            ])
            self._epochs = target_epochs
            self._bus.put([
                'metrics',
                {'type': 'train_epoch_progress', 'current': int(start_epoch), 'total': int(self._epochs)},
            ])

        return start_epoch, early_stopping_state, early_stopping_config

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _RunContext

"""model_train_and_recognition: run automatic early stopping."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch


class RunAutomaticEarlyStoppingMixin:
    def _run_automatic_early_stopping_check(
        self,
        *,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        is_main_process: bool,
        distributed: bool,
    ) -> bool:
        policy = self._automatic_early_stopping_policy
        should_stop = False
        if is_main_process:
            validation_result = None
            if self._val_dataloader is not None:
                validation_result = self._run_validation_epoch(
                    epoch,
                    device,
                    run_context.bce_criterion,
                    run_context.autocast_ctx,
                )
                if validation_result is None:
                    raise RuntimeError('Validation produced no finite result for early stopping.')
                self._publish_validation_metrics(epoch=epoch, validation_result=validation_result)
                self._latest_validation_result = dict(validation_result)
                self._latest_validation_global_batch = int(self._global_batch)
                loss = float(validation_result['loss'])
            else:
                loss = self._evaluate_automatic_stop_metric(device=device, run_context=run_context)
            if policy is None:
                return self._sync_early_stopping_signal(
                    False,
                    is_main_process=is_main_process,
                    distributed=distributed,
                    device=device,
                )
            decision = policy.update(loss=loss, global_batch=self._global_batch)
            source = 'validation' if self._val_dataloader is not None else 'train-control'
            if decision.actual_best_improved:
                self._checkpoint_manager.save_best(
                    model=self._base_model,
                    loss=loss,
                    global_batch=self._global_batch,
                    extra={
                        'metric_source': source,
                        'recommended_inference_threshold': float(self._recommended_inference_threshold),
                    },
                )
            trend = 'n/a' if decision.trend_improvement is None else f'{decision.trend_improvement:.4%}'
            self._bus.put([
                'logging',
                (
                    f'Early stopping check: source={source}, global_batch={self._global_batch}, '
                    f'loss={loss:.8f}, significant={decision.significant_improvement}, '
                    f'bad_checks={decision.bad_checks}/{policy.config.patience_checks}, trend={trend}.'
                ),
            ])
            should_stop = decision.should_stop
            if should_stop:
                self._bus.put([
                    'logging',
                    f'Early stopping reason at global batch {self._global_batch}: {decision.reason}.',
                ])
        return self._sync_early_stopping_signal(
            should_stop,
            is_main_process=is_main_process,
            distributed=distributed,
            device=device,
        )

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _RunContext

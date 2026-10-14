"""model_train_and_recognition: create warmup scheduler."""
from __future__ import annotations
from typing import Any
from torch import optim
from neuralimage.lib.data_interfaces import (
    SchedulerName,
)


class CreateWarmupSchedulerMixin:
    def _resolved_scheduler_name(self) -> SchedulerName:
        scheduler_params = getattr(self, '_scheduler_params', None)
        scheduler_name = getattr(scheduler_params, 'name', SchedulerName.off)
        if isinstance(scheduler_name, SchedulerName):
            return scheduler_name
        try:
            return SchedulerName(str(scheduler_name or SchedulerName.off.value).strip().lower())
        except ValueError:
            return SchedulerName.off

    def _warmup_supported_with_scheduler(self) -> bool:
        return self._resolved_scheduler_name() != SchedulerName.one_cycle

    def _create_warmup_scheduler(self, optimizer, train_steps_per_epoch: int) -> tuple[Any, int]:
        warmup = self._warmup_params
        if not warmup.enabled:
            return None, 0
        if not self._warmup_supported_with_scheduler():
            self._bus.put([
                'logging',
                'Warmup disabled for OneCycleLR: use OneCycle pct_start to control the initial ramp phase.',
            ])
            return None, 0
        if train_steps_per_epoch <= 0:
            return None, 0

        warmup_epochs = max(1, int(warmup.epochs))
        warmup_steps = warmup_epochs * train_steps_per_epoch
        start_factor = float(min(max(warmup.start_factor, 0.0), 1.0))

        def lr_lambda(step: int) -> float:
            if step >= warmup_steps:
                return 1.0
            progress = (step + 1) / warmup_steps
            return start_factor + (1.0 - start_factor) * progress

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps

    @staticmethod
    def _resolve_one_cycle_anneal_strategy(value: str) -> str:
        normalized = str(value or 'cos').strip().lower()
        return normalized if normalized in {'cos', 'linear'} else 'cos'

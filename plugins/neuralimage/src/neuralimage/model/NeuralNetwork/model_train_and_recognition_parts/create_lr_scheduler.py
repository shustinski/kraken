"""model_train_and_recognition: create lr scheduler."""
from __future__ import annotations
from typing import Any
from torch import optim
from neuralimage.lib.data_interfaces import (
    SchedulerName,
    SchedulerParameters,
)


class CreateLrSchedulerMixin:
    def _create_lr_scheduler(
        self,
        optimizer,
        *,
        train_steps_per_epoch: int,
    ) -> tuple[Any, str]:
        scheduler_params = getattr(self, '_scheduler_params', None) or SchedulerParameters()
        scheduler_name = self._resolved_scheduler_name()
        if scheduler_name == SchedulerName.off:
            return None, 'none'
        if scheduler_name == SchedulerName.reduce_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(min(max(getattr(scheduler_params, 'plateau_factor', 0.5), 1e-4), 0.9999)),
                patience=max(0, int(getattr(scheduler_params, 'plateau_patience', 3))),
                threshold=max(0.0, float(getattr(scheduler_params, 'plateau_threshold', 1e-4))),
                min_lr=max(0.0, float(getattr(scheduler_params, 'plateau_min_lr', 1e-6))),
                cooldown=max(0, int(getattr(scheduler_params, 'plateau_cooldown', 0))),
            )
            return scheduler, 'plateau'
        if scheduler_name == SchedulerName.cosine_annealing:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(getattr(scheduler_params, 'cosine_t_max', 10))),
                eta_min=max(0.0, float(getattr(scheduler_params, 'cosine_eta_min', 1e-6))),
            )
            return scheduler, 'epoch'
        if scheduler_name == SchedulerName.one_cycle:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max(0.0, float(getattr(scheduler_params, 'one_cycle_max_lr', 1e-3))),
                epochs=max(1, int(self._epochs)),
                steps_per_epoch=max(1, int(train_steps_per_epoch)),
                pct_start=float(
                    min(max(getattr(scheduler_params, 'one_cycle_pct_start', 0.3), 0.0), 1.0)
                ),
                anneal_strategy=self._resolve_one_cycle_anneal_strategy(
                    getattr(scheduler_params, 'one_cycle_anneal_strategy', 'cos')
                ),
                div_factor=max(1.0, float(getattr(scheduler_params, 'one_cycle_div_factor', 25.0))),
                final_div_factor=max(
                    1.0,
                    float(getattr(scheduler_params, 'one_cycle_final_div_factor', 10000.0)),
                ),
                three_phase=bool(getattr(scheduler_params, 'one_cycle_three_phase', False)),
            )
            return scheduler, 'batch'
        if scheduler_name == SchedulerName.step_lr:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(getattr(scheduler_params, 'step_lr_step_size', 10))),
                gamma=float(min(max(getattr(scheduler_params, 'step_lr_gamma', 0.1), 1e-4), 1.0)),
            )
            return scheduler, 'epoch'
        return None, 'none'

"""model_train_and_recognition: log scheduler configuration."""
from __future__ import annotations
from typing import Any
from neuralimage.lib.data_interfaces import (
    SchedulerName,
    SchedulerParameters,
)


class LogSchedulerConfigurationMixin:
    def _log_scheduler_configuration(self, scheduler: Any, scheduler_step_mode: str) -> None:
        if scheduler is None:
            return
        scheduler_name = self._resolved_scheduler_name()
        scheduler_params = getattr(self, '_scheduler_params', None) or SchedulerParameters()
        if scheduler_name == SchedulerName.reduce_on_plateau:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: ReduceLROnPlateau '
                    f'(factor={float(getattr(scheduler_params, "plateau_factor", 0.5)):.4f}, '
                    f'patience={int(getattr(scheduler_params, "plateau_patience", 3))}, '
                    f'threshold={float(getattr(scheduler_params, "plateau_threshold", 1e-4)):.6f}, '
                    f'min_lr={float(getattr(scheduler_params, "plateau_min_lr", 1e-6)):.6f}, '
                    f'cooldown={int(getattr(scheduler_params, "plateau_cooldown", 0))}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.cosine_annealing:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: CosineAnnealingLR '
                    f'(T_max={int(getattr(scheduler_params, "cosine_t_max", 10))}, '
                    f'eta_min={float(getattr(scheduler_params, "cosine_eta_min", 1e-6)):.6f}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.one_cycle:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: OneCycleLR '
                    f'(max_lr={float(getattr(scheduler_params, "one_cycle_max_lr", 1e-3)):.6f}, '
                    f'pct_start={float(getattr(scheduler_params, "one_cycle_pct_start", 0.3)):.3f}, '
                    f'anneal={self._resolve_one_cycle_anneal_strategy(getattr(scheduler_params, "one_cycle_anneal_strategy", "cos"))}, '
                    f'div_factor={float(getattr(scheduler_params, "one_cycle_div_factor", 25.0)):.2f}, '
                    f'final_div_factor={float(getattr(scheduler_params, "one_cycle_final_div_factor", 10000.0)):.2f}, '
                    f'three_phase={bool(getattr(scheduler_params, "one_cycle_three_phase", False))}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.step_lr:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: StepLR '
                    f'(step_size={int(getattr(scheduler_params, "step_lr_step_size", 10))}, '
                    f'gamma={float(getattr(scheduler_params, "step_lr_gamma", 0.1)):.4f}, '
                    f'step_mode={scheduler_step_mode}).'
                ),
            ])

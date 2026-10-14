"""model_train_and_recognition: step epoch scheduler."""
from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
from neuralimage.lib.loss_config import resolve_loss_term_weights
from neuralimage.losses.topograph import (
    build_topograph_loss,
    compute_topograph_loss_per_sample,
)
from neuralimage.model.NeuralNetwork.blocks import extract_mask_outputs


class StepEpochSchedulerMixin:
    def _step_batch_schedulers(self, run_context: _RunContext) -> None:
        warmup_scheduler = run_context.warmup_scheduler
        if warmup_scheduler is not None and int(run_context.warmup_total_steps) > 0:
            completed_steps = int(getattr(warmup_scheduler, 'last_epoch', -1)) + 1
            if completed_steps < int(run_context.warmup_total_steps):
                warmup_scheduler.step()
                return
        if run_context.scheduler is not None and run_context.scheduler_step_mode == 'batch':
            run_context.scheduler.step()

    def _step_epoch_scheduler(
        self,
        *,
        run_context: _RunContext,
        validation_result: dict[str, float] | None,
        train_loss: float | None,
        distributed: bool,
        device: torch.device,
        is_main_process: bool,
    ) -> None:
        scheduler = run_context.scheduler
        if scheduler is None:
            return
        if run_context.scheduler_step_mode == 'epoch':
            scheduler.step()
            return
        if run_context.scheduler_step_mode != 'plateau':
            return

        metric = validation_result.get('loss') if validation_result is not None else train_loss
        try:
            metric_value = float(metric) if metric is not None else None
        except (TypeError, ValueError):
            metric_value = None
        if metric_value is None or not math.isfinite(metric_value):
            return
        if distributed:
            metric_tensor = torch.tensor(
                [metric_value if is_main_process else 0.0],
                device=device,
                dtype=torch.float64,
            )
            dist.broadcast(metric_tensor, src=0)
            metric_value = float(metric_tensor.item())
        scheduler.step(metric_value)

    def _resolve_target_epochs(self, start_epoch: int) -> int:
        """
        In fine-tuning mode the UI epoch value is treated as "additional epochs".
        For fresh training it is treated as total epochs.
        """
        if self._resume_from_checkpoint and self._extend_epochs_on_resume and start_epoch > 0:
            return start_epoch + int(self._epochs)
        return int(self._epochs)

    def _resolved_loss_function(self) -> str:
        if self._loss_function in (
            'bce',
            'dice',
            'cldice',
            'bce_dice',
            'iou',
            'bce_iou',
            'ce',
            'ce_dice',
            'focal_bce',
            'focal_dice',
            'focal_iou',
            'boundary',
            'focal_tversky',
        ):
            return self._loss_function
        return 'bce'

    def _resolved_dice_weight(self) -> float:
        return float(min(max(self._dice_loss_weight, 0.0), 1.0))

    def _resolved_iou_weight(self) -> float:
        return float(min(max(self._iou_loss_weight, 0.0), 1.0))

    def _resolved_loss_term_weights(self) -> dict[str, float]:
        return resolve_loss_term_weights(
            getattr(self, '_loss_term_weights', None),
            fallback_loss_function=self._resolved_loss_function(),
        )

    def _resolved_hard_pixel_keep_ratio(self) -> float:
        params = getattr(self, '_hard_mining_params', None)
        ratio = float(getattr(params, 'pixel_keep_ratio', 0.25))
        return float(min(max(ratio, 0.01), 1.0))

    def _hard_pixel_mining_enabled(self) -> bool:
        params = getattr(self, '_hard_mining_params', None)
        return bool(getattr(params, 'pixel_enabled', False))

    def _loss_supports_hard_pixel_mining(self, loss_mode: str) -> bool:
        return loss_mode not in ('dice', 'cldice', 'iou', 'boundary', 'focal_tversky')

    def _topograph_is_active(self) -> bool:
        return bool(
            getattr(self, '_topograph_enabled', False)
            and float(getattr(self, '_topograph_loss_weight', 0.0)) > 0.0
        )

    def _ensure_topograph_loss(self) -> Any:
        if getattr(self, '_topograph_loss_module', None) is None:
            self._topograph_loss_module = build_topograph_loss(
                num_processes=max(1, int(getattr(self, '_topograph_num_processes', 1))),
                use_c=bool(getattr(self, '_topograph_use_c', False)),
            )
        return self._topograph_loss_module

    def _compute_topograph_component_loss(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._topograph_is_active():
            return None
        if int(outputs.shape[1]) != 1:
            if not getattr(self, '_topograph_binary_skip_logged', False):
                self._bus.put([
                    'logging',
                    'Topograph loss skipped: binary single-channel outputs are required.',
                ])
                self._topograph_binary_skip_logged = True
            return None
        return compute_topograph_loss_per_sample(
            outputs,
            label,
            self._ensure_topograph_loss(),
        )

    def _update_batch_loss_components(
        self,
        *,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
        combined_loss: torch.Tensor,
        topograph_loss: torch.Tensor | None,
        apply_pixel_mining: bool,
    ) -> dict[str, float]:
        components: dict[str, float] = {'bce': 0.0, 'dice': 0.0, 'topograph': 0.0, 'total': 0.0}
        sanitized_outputs = self._sanitize_outputs_for_loss(outputs)
        sanitized_label = self._sanitize_labels_for_loss(label)
        for loss_mode, coefficient in self._resolved_loss_term_weights().items():
            if coefficient <= 0.0:
                continue
            single_loss = self._compute_single_per_sample_loss(
                loss_mode,
                sanitized_outputs,
                sanitized_label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
            value = self._loss_value_to_float(single_loss.mean())
            if loss_mode in ('bce', 'focal_bce'):
                components['bce'] = value
            elif loss_mode == 'dice':
                components['dice'] = value
        if topograph_loss is not None:
            components['topograph'] = self._loss_value_to_float(topograph_loss.mean())
        components['total'] = self._loss_value_to_float(combined_loss.mean())
        self._last_batch_loss_components = components
        return components

    @staticmethod
    def _is_finite_tensor(value: Any) -> bool:
        """Return whether every tensor contained in *value* is finite."""
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, Mapping):
            return all(ModelTrainer._is_finite_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(ModelTrainer._is_finite_tensor(item) for item in value)
        return True

    @staticmethod
    def _extract_primary_output_tensor(outputs: Any) -> torch.Tensor:
        outputs = extract_mask_outputs(outputs)
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, (list, tuple)) and outputs:
            primary = outputs[0]
            if torch.is_tensor(primary):
                return primary
        raise TypeError('Model must return a tensor or a non-empty sequence of tensors.')

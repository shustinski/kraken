"""model_train_and_recognition: evaluate automatic stop metric."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import torch
from neuralimage.model.NeuralNetwork.early_stopping import (
    MetricEvaluator,
)


class EvaluateAutomaticStopMetricMixin:
    def _evaluate_automatic_stop_metric(
        self,
        *,
        device: torch.device,
        run_context: _RunContext,
    ) -> float:
        dataloader = self._val_dataloader if self._val_dataloader is not None else self._control_dataloader
        if dataloader is None:
            raise RuntimeError('Early stopping requires validation or train-control dataloader')

        def adapt_batch(batch: Any, target_device: torch.device) -> tuple[Any, Any]:
            data, target, _sample_indices = self._split_batch(batch)
            return (
                self._move_batch_input_to_device(data, target_device),
                self._move_batch_target_to_device(target, target_device),
            )

        evaluator = MetricEvaluator()
        return evaluator.evaluate(
            model=self._base_model,
            dataloader=dataloader,
            device=device,
            batch_adapter=adapt_batch,
            forward_fn=lambda model, inputs: model(inputs),
            per_sample_loss_fn=lambda outputs, target: self._compute_per_sample_loss(
                outputs,
                target,
                run_context.bce_criterion,
            ),
            autocast_ctx=run_context.autocast_ctx,
        )

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _RunContext

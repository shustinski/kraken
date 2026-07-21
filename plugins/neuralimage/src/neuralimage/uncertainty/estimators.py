from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ConfidenceHeadEstimator:
    @staticmethod
    def from_outputs(outputs: Any) -> np.ndarray | None:
        if not isinstance(outputs, dict):
            return None
        confidence = outputs.get('confidence')
        if confidence is None:
            return None
        if torch.is_tensor(confidence):
            return torch.sigmoid(confidence).detach().cpu().numpy()
        return np.asarray(confidence, dtype=np.float32)


class MonteCarloDropoutEstimator:
    def __init__(self, *, samples: int = 8):
        self.samples = max(1, int(samples))

    def estimate(
        self,
        model: nn.Module,
        inputs: Any,
        *,
        forward_fn: Callable[[Any], Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        model.train()
        predictions: list[np.ndarray] = []
        for _ in range(self.samples):
            outputs = forward_fn(inputs)
            if isinstance(outputs, dict):
                mask = outputs.get('mask')
                if isinstance(mask, (list, tuple)):
                    mask = mask[0]
            else:
                mask = outputs
            predictions.append(torch.sigmoid(mask).detach().cpu().numpy())
        model.eval()
        stacked = np.stack(predictions, axis=0)
        mean = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        confidence = 1.0 - np.clip(variance, 0.0, 1.0)
        return mean.astype(np.float32), confidence.astype(np.float32)


class TTAVarianceEstimator:
    def __init__(self, *, use_flips: bool = True, use_rotations: bool = False):
        self.use_flips = bool(use_flips)
        self.use_rotations = bool(use_rotations)

    def _transform_variants(self, tensor: torch.Tensor) -> list[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
        variants: list[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = [(tensor, lambda value: value)]
        if self.use_flips:
            variants.append((torch.flip(tensor, dims=(-1,)), lambda value: torch.flip(value, dims=(-1,))))
            variants.append((torch.flip(tensor, dims=(-2,)), lambda value: torch.flip(value, dims=(-2,))))
        if self.use_rotations:
            variants.append((torch.rot90(tensor, k=1, dims=(-2, -1)), lambda value: torch.rot90(value, k=-1, dims=(-2, -1))))
        return variants

    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        *,
        forward_fn: Callable[[Any], Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        predictions: list[np.ndarray] = []
        for transformed, inverse in self._transform_variants(inputs):
            outputs = forward_fn(transformed)
            if isinstance(outputs, dict):
                mask = outputs.get('mask')
                if isinstance(mask, (list, tuple)):
                    mask = mask[0]
            else:
                mask = outputs
            restored = inverse(torch.sigmoid(mask))
            predictions.append(restored.detach().cpu().numpy())
        stacked = np.stack(predictions, axis=0)
        mean = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        confidence = 1.0 - np.clip(variance, 0.0, 1.0)
        return mean.astype(np.float32), confidence.astype(np.float32)


def estimate_uncertainty(
    *,
    method: str,
    model: nn.Module,
    inputs: Any,
    outputs: Any,
    forward_fn: Callable[[Any], Any],
    mc_samples: int = 8,
) -> np.ndarray | None:
    normalized = str(method or 'confidence_head').strip().lower()
    if normalized == 'confidence_head':
        return ConfidenceHeadEstimator.from_outputs(outputs)
    if normalized == 'mc_dropout':
        _, confidence = MonteCarloDropoutEstimator(samples=mc_samples).estimate(model, inputs, forward_fn=forward_fn)
        return confidence
    if normalized == 'tta_variance':
        _, confidence = TTAVarianceEstimator().estimate(model, inputs if torch.is_tensor(inputs) else inputs, forward_fn=forward_fn)
        return confidence
    return ConfidenceHeadEstimator.from_outputs(outputs)

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
        states = {module: module.training for module in model.modules()}
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.train(True)
        predictions: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for _ in range(self.samples):
                    outputs = forward_fn(inputs)
                    mask = _extract_mask_tensor(outputs)
                    predictions.append(torch.sigmoid(mask).detach().cpu().numpy())
        finally:
            for module, training in states.items():
                module.train(training)
        stacked = np.stack(predictions, axis=0)
        mean = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        confidence = 1.0 - np.clip(4.0 * variance, 0.0, 1.0)
        return mean.astype(np.float32), confidence.astype(np.float32)


class TTAVarianceEstimator:
    def __init__(self, *, use_flips: bool = True, use_rotations: bool = False):
        self.use_flips = bool(use_flips)
        self.use_rotations = bool(use_rotations)

    @staticmethod
    def _apply_to_images(inputs: Any, transform: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        if torch.is_tensor(inputs):
            return transform(inputs)
        if isinstance(inputs, dict):
            return {
                key: transform(value) if torch.is_tensor(value) and value.ndim >= 4 else value
                for key, value in inputs.items()
            }
        raise TypeError(f'Unsupported TTA input type: {type(inputs)!r}.')

    def _transform_variants(self, inputs: Any) -> list[tuple[Any, Callable[[torch.Tensor], torch.Tensor]]]:
        variants: list[tuple[Any, Callable[[torch.Tensor], torch.Tensor]]] = [(inputs, lambda value: value)]
        if self.use_flips:
            variants.append((self._apply_to_images(inputs, lambda value: torch.flip(value, dims=(-1,))), lambda value: torch.flip(value, dims=(-1,))))
            variants.append((self._apply_to_images(inputs, lambda value: torch.flip(value, dims=(-2,))), lambda value: torch.flip(value, dims=(-2,))))
        if self.use_rotations:
            variants.append((self._apply_to_images(inputs, lambda value: torch.rot90(value, k=1, dims=(-2, -1))), lambda value: torch.rot90(value, k=-1, dims=(-2, -1))))
        return variants

    def estimate(
        self,
        model: nn.Module,
        inputs: Any,
        *,
        forward_fn: Callable[[Any], Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        states = {module: module.training for module in model.modules()}
        model.eval()
        predictions: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for transformed, inverse in self._transform_variants(inputs):
                    restored = inverse(torch.sigmoid(_extract_mask_tensor(forward_fn(transformed))))
                    predictions.append(restored.detach().cpu().numpy())
        finally:
            for module, training in states.items():
                module.train(training)
        stacked = np.stack(predictions, axis=0)
        mean = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        confidence = 1.0 - np.clip(4.0 * variance, 0.0, 1.0)
        return mean.astype(np.float32), confidence.astype(np.float32)


def _extract_mask_tensor(outputs: Any) -> torch.Tensor:
    mask = outputs.get('mask') if isinstance(outputs, dict) else outputs
    if isinstance(mask, (list, tuple)):
        mask = mask[0]
    if not torch.is_tensor(mask):
        raise TypeError('Uncertainty estimation requires a tensor mask output.')
    return mask


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
        _, confidence = TTAVarianceEstimator().estimate(model, inputs, forward_fn=forward_fn)
        return confidence
    if normalized == 'combined':
        sources = [ConfidenceHeadEstimator.from_outputs(outputs)]
        _, mc_confidence = MonteCarloDropoutEstimator(samples=mc_samples).estimate(
            model, inputs, forward_fn=forward_fn
        )
        _, tta_confidence = TTAVarianceEstimator().estimate(model, inputs, forward_fn=forward_fn)
        sources.extend((mc_confidence, tta_confidence))
        available = [np.asarray(source, dtype=np.float32) for source in sources if source is not None]
        return np.mean(np.stack(available, axis=0), axis=0).astype(np.float32) if available else None
    return ConfidenceHeadEstimator.from_outputs(outputs)

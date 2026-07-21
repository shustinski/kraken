from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


REGRESSION_TARGETS: frozenset[str] = frozenset({'sdf', 'orientation', 'curvature', 'thickness'})
VECTOR_TARGETS: frozenset[str] = frozenset({'tangent'})


def resolve_auxiliary_head_weights(
    enabled_targets: Sequence[str],
    configured_weights: Mapping[str, float] | None = None,
    *,
    default_weight: float = 0.25,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    configured = dict(configured_weights or {})
    active = [name for name in enabled_targets if name != 'mask']
    if not active:
        return weights
    for name in active:
        weight = configured.get(name)
        if weight is None:
            weight = default_weight
        weights[name] = float(max(0.0, weight))
    total = sum(weights.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(active))
        return {name: uniform for name in active}
    return {name: value / total for name, value in weights.items()}


class DynamicLossWeighter:
    """Exponential moving average dynamic loss weighting across terms."""

    def __init__(self, term_names: Sequence[str], *, ema_alpha: float = 0.1, eps: float = 1e-6):
        self.term_names = tuple(term_names)
        self.ema_alpha = float(min(max(ema_alpha, 0.0), 1.0))
        self.eps = float(eps)
        self._ema: dict[str, float] = {name: 1.0 for name in self.term_names}

    def update(self, term_losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name in self.term_names:
            if name not in term_losses:
                continue
            value = float(term_losses[name].detach().mean().item())
            previous = self._ema.get(name, 1.0)
            self._ema[name] = previous * (1.0 - self.ema_alpha) + value * self.ema_alpha
        inverse = {name: 1.0 / (self._ema[name] + self.eps) for name in self.term_names if name in term_losses}
        total = sum(inverse.values())
        if total <= 0.0:
            return {name: 1.0 / max(1, len(inverse)) for name in inverse}
        for name, value in inverse.items():
            weights[name] = value / total
        return weights


class AuxiliaryHeadLoss:
    """Loss computation for auxiliary supervision heads."""

    @staticmethod
    def _resize_like(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-2:] == target.shape[-2:]:
            return prediction
        return F.interpolate(prediction, size=target.shape[-2:], mode='bilinear', align_corners=False)

    @classmethod
    def compute_single_head(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        head_name: str,
    ) -> torch.Tensor:
        prediction = cls._resize_like(prediction, target)
        if head_name in REGRESSION_TARGETS:
            loss_map = F.smooth_l1_loss(prediction, target, reduction='none')
            if loss_map.ndim == 4 and loss_map.shape[1] == 1:
                loss_map = loss_map[:, 0, :, :]
            return loss_map.view(loss_map.shape[0], -1).mean(dim=1)

        if head_name in VECTOR_TARGETS:
            if prediction.shape[1] != target.shape[1]:
                raise ValueError(f'Vector head {head_name!r} channel mismatch.')
            loss_map = F.smooth_l1_loss(prediction, target, reduction='none')
            return loss_map.view(loss_map.shape[0], -1).mean(dim=1)

        if target.shape[1] != 1 and target.ndim == 4:
            target = target[:, :1, :, :]
        loss_map = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        return loss_map.view(loss_map.shape[0], -1).mean(dim=1)


def compute_auxiliary_head_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    head_weights: Mapping[str, float],
) -> torch.Tensor | None:
    if not head_weights:
        return None
    combined: torch.Tensor | None = None
    for head_name, weight in head_weights.items():
        if weight <= 0.0:
            continue
        prediction = outputs.get(head_name)
        target = targets.get(head_name)
        if prediction is None or target is None:
            continue
        head_loss = AuxiliaryHeadLoss.compute_single_head(prediction, target, head_name=head_name)
        head_loss = torch.nan_to_num(head_loss, nan=1.0, posinf=50.0, neginf=0.0)
        weighted = head_loss * float(weight)
        combined = weighted if combined is None else (combined + weighted)
    return combined

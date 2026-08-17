from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


REGRESSION_TARGETS: frozenset[str] = frozenset({'sdf', 'distance_transform', 'curvature', 'thickness'})
VECTOR_TARGETS: frozenset[str] = frozenset({'orientation', 'tangent'})
SPARSE_TARGETS: frozenset[str] = frozenset({'boundary', 'skeleton', 'vertex', 'corner', 'endpoint', 'junction', 'topology'})


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


class HomoscedasticLossWeighter(nn.Module):
    """Learn task weights as log variances while retaining a mask-loss floor."""

    def __init__(self, term_names: Sequence[str], *, mask_term: str = 'mask', mask_weight_floor: float = 0.25):
        super().__init__()
        self.term_names = tuple(str(name) for name in term_names)
        self.mask_term = str(mask_term)
        self.mask_weight_floor = float(max(0.0, mask_weight_floor))
        self.log_variances = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.term_names}
        )

    def forward(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        combined: torch.Tensor | None = None
        for name in self.term_names:
            if name not in losses:
                continue
            log_variance = torch.clamp(self.log_variances[name], -6.0, 6.0)
            precision = torch.exp(-log_variance)
            if name == self.mask_term:
                precision = torch.clamp(precision, min=self.mask_weight_floor)
            weighted = precision * losses[name] + 0.5 * log_variance
            combined = weighted if combined is None else combined + weighted
        if combined is None:
            raise ValueError('No configured loss terms were provided.')
        return combined


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
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = cls._resize_like(prediction, target)
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        else:
            valid_mask = cls._resize_like(valid_mask, target)
            if valid_mask.shape[1] == 1 and target.shape[1] > 1:
                valid_mask = valid_mask.expand(-1, target.shape[1], -1, -1)
        valid_mask = torch.clamp(valid_mask.to(dtype=prediction.dtype), 0.0, 1.0)

        def reduce_valid(loss_map: torch.Tensor) -> torch.Tensor:
            weighted = loss_map * valid_mask
            numerator = weighted.flatten(1).sum(dim=1)
            denominator = valid_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            return numerator / denominator

        if head_name in REGRESSION_TARGETS:
            loss_map = F.smooth_l1_loss(prediction, target, reduction='none')
            return reduce_valid(loss_map)

        if head_name in VECTOR_TARGETS:
            if prediction.shape[1] != target.shape[1]:
                raise ValueError(f'Vector head {head_name!r} channel mismatch.')
            prediction_vector = F.normalize(prediction, dim=1, eps=1e-6)
            target_vector = F.normalize(target, dim=1, eps=1e-6)
            cosine_loss = 1.0 - (prediction_vector * target_vector).sum(dim=1, keepdim=True)
            vector_valid = valid_mask[:, :1]
            numerator = (cosine_loss * vector_valid).flatten(1).sum(dim=1)
            denominator = vector_valid.flatten(1).sum(dim=1).clamp_min(1.0)
            return numerator / denominator

        if target.ndim != 4:
            raise ValueError(f'Auxiliary target {head_name!r} must be BCHW.')
        if prediction.shape[1] != target.shape[1]:
            raise ValueError(
                f'Auxiliary head {head_name!r} channel mismatch: '
                f'{prediction.shape[1]} != {target.shape[1]}.'
            )
        bce = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        if head_name in SPARSE_TARGETS:
            probabilities = torch.sigmoid(prediction)
            pt = (probabilities * target) + ((1.0 - probabilities) * (1.0 - target))
            alpha = 0.75 if head_name == 'junction' else 0.5
            alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            bce = alpha_t * bce * torch.pow(1.0 - pt, 2.0)
            intersection = (probabilities * target * valid_mask).flatten(1).sum(dim=1)
            probability_mass = (probabilities * valid_mask).flatten(1).sum(dim=1)
            target_mass = (target * valid_mask).flatten(1).sum(dim=1)
            dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (probability_mass + target_mass + 1e-6))
            combined = (0.75 * reduce_valid(bce)) + (0.25 * dice_loss)
            if head_name == 'topology' and probabilities.shape[1] == 2:
                # Foreground/background critical maps must remain separated;
                # simultaneous activation is a differentiable bridge/hole risk.
                overlap = probabilities[:, :1] * probabilities[:, 1:2]
                overlap_valid = valid_mask[:, :1]
                separation = (overlap * overlap_valid).flatten(1).sum(dim=1) / overlap_valid.flatten(1).sum(dim=1).clamp_min(1.0)
                combined = combined + 0.2 * separation
            return combined
        return reduce_valid(bce)


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
        valid_mask = targets.get(f'{head_name}__valid')
        head_loss = AuxiliaryHeadLoss.compute_single_head(
            prediction,
            target,
            head_name=head_name,
            valid_mask=valid_mask,
        )
        head_loss = torch.nan_to_num(head_loss, nan=1.0, posinf=50.0, neginf=0.0)
        weighted = head_loss * float(weight)
        combined = weighted if combined is None else (combined + weighted)
    return combined

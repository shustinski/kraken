from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_distance_boundary_loss(
    logits: torch.Tensor,
    signed_distance_field: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample SDF-weighted boundary loss.

    The project SDF is positive inside the foreground and negative outside.
    This non-negative form is equivalent to distance-weighted classification:
    errors far from the reference boundary cost more than errors near it.
    """
    if logits.shape[-2:] != signed_distance_field.shape[-2:]:
        logits = F.interpolate(logits, size=signed_distance_field.shape[-2:], mode='bilinear', align_corners=False)
    sdf = torch.clamp(signed_distance_field.to(dtype=logits.dtype), -1.0, 1.0)
    probabilities = torch.sigmoid(logits)
    foreground_cost = (1.0 - sdf) * 0.5
    background_cost = (1.0 + sdf) * 0.5
    loss_map = probabilities * foreground_cost + (1.0 - probabilities) * background_cost
    if valid_mask is None:
        valid_mask = torch.ones_like(loss_map)
    elif valid_mask.shape[-2:] != loss_map.shape[-2:]:
        valid_mask = F.interpolate(valid_mask, size=loss_map.shape[-2:], mode='nearest')
    valid_mask = torch.clamp(valid_mask.to(dtype=loss_map.dtype), 0.0, 1.0)
    numerator = (loss_map * valid_mask).flatten(1).sum(dim=1)
    denominator = valid_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator

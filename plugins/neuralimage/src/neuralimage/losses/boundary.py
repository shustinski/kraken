from __future__ import annotations

import torch
import torch.nn.functional as F


BOUNDARY_LOSS_KERNEL_SIZE = 3


def compute_soft_boundary_map(
    mask: torch.Tensor,
    *,
    kernel_size: int = BOUNDARY_LOSS_KERNEL_SIZE,
) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return torch.clamp(torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)

    pad = kernel_size // 2
    padded_mask = F.pad(mask, (pad, pad, pad, pad), mode='constant', value=0.0)
    dilation = F.max_pool2d(padded_mask, kernel_size=kernel_size, stride=1)
    erosion = -F.max_pool2d(-padded_mask, kernel_size=kernel_size, stride=1)
    boundary_map = dilation - erosion
    boundary_map = torch.nan_to_num(boundary_map, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(boundary_map, min=0.0, max=1.0)


def compute_boundary_loss(
    outputs: torch.Tensor,
    label: torch.Tensor,
    *,
    kernel_size: int = BOUNDARY_LOSS_KERNEL_SIZE,
) -> torch.Tensor:
    """Improved boundary loss using soft boundary maps and Dice-style overlap."""
    probs = torch.sigmoid(outputs)
    label_bin = (label >= 0.5).to(dtype=probs.dtype)
    pred_boundary = compute_soft_boundary_map(probs, kernel_size=kernel_size)
    target_boundary = compute_soft_boundary_map(label_bin, kernel_size=kernel_size)
    pred_boundary_flat = pred_boundary.view(pred_boundary.shape[0], -1)
    target_boundary_flat = target_boundary.view(target_boundary.shape[0], -1)
    eps = 1e-6
    intersection = (pred_boundary_flat * target_boundary_flat).sum(dim=1)
    pred_boundary_mass = pred_boundary_flat.sum(dim=1)
    target_boundary_mass = target_boundary_flat.sum(dim=1)
    denom = pred_boundary_mass + target_boundary_mass
    boundary_loss = 1.0 - ((2.0 * intersection + eps) / (denom + eps))
    empty_target_mask = target_boundary_mass <= eps
    if bool(empty_target_mask.any()):
        pixel_count = max(1, int(pred_boundary_flat.shape[1]))
        empty_target_penalty = pred_boundary_mass / float(pixel_count)
        empty_target_penalty = torch.clamp(empty_target_penalty, min=0.0, max=1.0)
        boundary_loss = torch.where(empty_target_mask, empty_target_penalty, boundary_loss)
    boundary_loss = torch.nan_to_num(boundary_loss, nan=1.0, posinf=50.0, neginf=0.0)
    return torch.clamp(boundary_loss, min=0.0, max=50.0)

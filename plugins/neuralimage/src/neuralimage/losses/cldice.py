from __future__ import annotations

import torch
import torch.nn.functional as F

CLDICE_SKELETON_ITERATIONS = 10


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    eroded = torch.minimum(vertical, horizontal)
    eroded = torch.nan_to_num(eroded, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(eroded, min=0.0, max=1.0)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    dilated = torch.nan_to_num(dilated, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(dilated, min=0.0, max=1.0)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(
    mask: torch.Tensor,
    *,
    iterations: int = CLDICE_SKELETON_ITERATIONS,
) -> torch.Tensor:
    work = torch.clamp(torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
    skeleton = F.relu(work - _soft_open(work))
    for _ in range(max(0, int(iterations))):
        work = _soft_erode(work)
        delta = F.relu(work - _soft_open(work))
        skeleton = skeleton + F.relu(delta - (skeleton * delta))
    skeleton = torch.nan_to_num(skeleton, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(skeleton, min=0.0, max=1.0)


def compute_cldice_loss(
    outputs: torch.Tensor,
    label: torch.Tensor,
    *,
    iterations: int = CLDICE_SKELETON_ITERATIONS,
) -> torch.Tensor:
    probs = torch.sigmoid(outputs)
    target = (label >= 0.5).to(dtype=probs.dtype)
    pred_skeleton = soft_skeletonize(probs, iterations=iterations)
    target_skeleton = soft_skeletonize(target, iterations=iterations)
    pred_flat = probs.view(probs.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    pred_skeleton_flat = pred_skeleton.view(pred_skeleton.shape[0], -1)
    target_skeleton_flat = target_skeleton.view(target_skeleton.shape[0], -1)
    eps = 1e-6
    topology_precision = ((pred_skeleton_flat * target_flat).sum(dim=1) + eps) / (
        pred_skeleton_flat.sum(dim=1) + eps
    )
    topology_sensitivity = ((target_skeleton_flat * pred_flat).sum(dim=1) + eps) / (
        target_skeleton_flat.sum(dim=1) + eps
    )
    cldice = (2.0 * topology_precision * topology_sensitivity + eps) / (
        topology_precision + topology_sensitivity + eps
    )
    cldice_loss = 1.0 - cldice
    cldice_loss = torch.nan_to_num(cldice_loss, nan=1.0, posinf=50.0, neginf=0.0)
    return torch.clamp(cldice_loss, min=0.0, max=50.0)

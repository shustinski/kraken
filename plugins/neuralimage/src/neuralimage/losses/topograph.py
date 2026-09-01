from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_VENDOR_ROOT = Path(__file__).resolve().parents[3] / 'vendor' / 'topograph'
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from losses.topograph import (  # noqa: E402
    TopographLoss,
    create_graph,
    create_relabel_masks,
    get_critical_nodes,
)
from losses.utils import AggregationType, ThresholdDistribution, new_compute_diag_diffs, new_compute_diffs  # noqa: E402

__all__ = [
    'TopographLoss',
    'binary_logits_to_two_channel',
    'build_topograph_loss',
    'compute_topograph_loss_per_sample',
    'extract_critical_region_mask',
]


def binary_logits_to_two_channel(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert single-channel binary logits and labels to 2-channel one-hot format."""
    if logits.ndim != 4 or labels.ndim != 4:
        raise ValueError('Topograph expects 4D tensors with shape (batch, channel, height, width).')
    if logits.shape[1] != 1:
        raise ValueError('Topograph adapter supports only single-channel binary logits.')

    logits_two_class = torch.cat([-logits, logits], dim=1)
    soft_label = labels[:, 0, :, :]
    target_probs = torch.stack((1.0 - soft_label, soft_label), dim=1)
    return logits_two_class, target_probs


def build_topograph_loss(
    *,
    include_background: bool = False,
    eight_connectivity: bool = False,
    use_c: bool = False,
    num_processes: int = 1,
    softmax: bool = True,
    sphere: bool = False,
    aggregation: AggregationType = AggregationType.MEAN,
    thres_distr: ThresholdDistribution = ThresholdDistribution.NONE,
    thres_var: float = 0.0,
) -> TopographLoss:
    return TopographLoss(
        softmax=softmax,
        num_processes=max(1, int(num_processes)),
        include_background=include_background,
        use_c=use_c,
        sphere=sphere,
        eight_connectivity=eight_connectivity,
        aggregation=aggregation,
        thres_distr=thres_distr,
        thres_var=thres_var,
    )


def compute_topograph_loss_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_module: TopographLoss,
) -> torch.Tensor:
    """Compute Topograph loss and broadcast the batch scalar to per-sample shape."""
    prediction, target = binary_logits_to_two_channel(logits, labels)
    batch_size = int(prediction.shape[0])
    if batch_size == 0:
        return torch.zeros((0,), device=logits.device, dtype=logits.dtype)

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        topograph_scalar = loss_module(prediction.float(), target.float())

    topograph_scalar = torch.nan_to_num(topograph_scalar, nan=0.0, posinf=50.0, neginf=0.0)
    topograph_scalar = torch.clamp(topograph_scalar, min=0.0, max=50.0)
    return topograph_scalar.expand(batch_size)


def extract_critical_region_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_index: int = 0,
    class_index: int = 1,
    eight_connectivity: bool = False,
    use_c: bool = False,
) -> torch.Tensor:
    """Build a boolean mask of topologically critical regions for one sample."""
    import numpy as np

    prediction, target = binary_logits_to_two_channel(logits, labels)
    prediction = prediction.detach().float()
    target = target.detach().float()

    argmax_preds = torch.argmax(prediction, dim=1)
    argmax_gts = torch.argmax(target, dim=1)

    bin_preds = torch.zeros_like(argmax_preds)
    bin_gts = torch.zeros_like(argmax_gts)
    bin_preds[argmax_preds == class_index] = 1
    bin_gts[argmax_gts == class_index] = 1

    paired_imgs = bin_preds + 2 * bin_gts
    diag_val_1, diag_val_2 = (-4, 16) if eight_connectivity else (16, -4)
    paired_imgs = paired_imgs.clone()
    paired_imgs[paired_imgs == 0] = diag_val_1
    paired_imgs[paired_imgs == 3] = diag_val_2

    h_diff, v_diff = new_compute_diffs(paired_imgs)
    diagr, diagl, special_diag_r, special_diag_l = new_compute_diag_diffs(paired_imgs, th=7)

    sample_idx = int(sample_index)
    argmax_pred = bin_preds[sample_idx].cpu().numpy()
    argmax_gt = bin_gts[sample_idx].cpu().numpy()
    graph, labelled_regions = create_graph(
        argmax_pred,
        argmax_gt,
        h_diff[sample_idx].cpu().numpy(),
        v_diff[sample_idx].cpu().numpy(),
        diagr[sample_idx].cpu().numpy(),
        diagl[sample_idx].cpu().numpy(),
        special_diag_r[sample_idx].cpu().numpy(),
        special_diag_l[sample_idx].cpu().numpy(),
    )
    critical_nodes, cluster_lengths = get_critical_nodes(graph)
    if use_c:
        from losses.topograph import create_relabel_masks_c

        region_error_infos = create_relabel_masks_c(critical_nodes, cluster_lengths, labelled_regions)
    else:
        region_error_infos = create_relabel_masks(critical_nodes, cluster_lengths, labelled_regions)

    height, width = argmax_pred.shape
    critical_mask = np.zeros((height, width), dtype=bool)
    for region_indices in region_error_infos:
        critical_mask[region_indices[0], region_indices[1]] = True

    return torch.from_numpy(critical_mask)

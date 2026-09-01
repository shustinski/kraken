"""Segmentation loss functions and composite loss utilities."""

from neuralimage.losses.boundary import compute_boundary_loss, compute_soft_boundary_map
from neuralimage.losses.cldice import compute_cldice_loss, soft_skeletonize
from neuralimage.losses.topograph import (
    build_topograph_loss,
    compute_topograph_loss_per_sample,
    extract_critical_region_mask,
)
from neuralimage.losses.topograph_viz import render_critical_regions_overlay
from neuralimage.losses.composite import (
    AuxiliaryHeadLoss,
    DynamicLossWeighter,
    HomoscedasticLossWeighter,
    compute_auxiliary_head_loss,
    resolve_auxiliary_head_weights,
)

__all__ = [
    'AuxiliaryHeadLoss',
    'compute_auxiliary_head_loss',
    'compute_boundary_loss',
    'compute_soft_boundary_map',
    'compute_cldice_loss',
    'compute_distance_boundary_loss',
    'build_topograph_loss',
    'compute_topograph_loss_per_sample',
    'extract_critical_region_mask',
    'render_critical_regions_overlay',
    'soft_skeletonize',
    'DynamicLossWeighter',
    'HomoscedasticLossWeighter',
    'resolve_auxiliary_head_weights',
]

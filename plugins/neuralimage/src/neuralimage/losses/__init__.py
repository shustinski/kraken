"""Segmentation loss functions and composite loss utilities."""

from neuralimage.losses.boundary import compute_boundary_loss, compute_soft_boundary_map
from neuralimage.losses.cldice import compute_cldice_loss, soft_skeletonize
from neuralimage.losses.composite import (
    AuxiliaryHeadLoss,
    DynamicLossWeighter,
    compute_auxiliary_head_loss,
    resolve_auxiliary_head_weights,
)

__all__ = [
    'AuxiliaryHeadLoss',
    'compute_auxiliary_head_loss',
    'compute_boundary_loss',
    'compute_soft_boundary_map',
    'compute_cldice_loss',
    'soft_skeletonize',
    'DynamicLossWeighter',
    'resolve_auxiliary_head_weights',
]

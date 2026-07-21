"""Multi-target segmentation head bundles."""

from neuralimage.heads.multi_target import (
    MultiTargetHeadBundle,
    build_training_output_dict,
    extract_inference_mask,
    resolve_head_output_channels,
)

__all__ = [
    'MultiTargetHeadBundle',
    'build_training_output_dict',
    'extract_inference_mask',
    'resolve_head_output_channels',
]

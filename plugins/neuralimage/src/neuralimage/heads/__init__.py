"""Multi-target segmentation head bundles."""

from neuralimage.heads.multi_target import (
    HEAD_SPECS,
    HeadSpec,
    MultiTargetHeadBundle,
    build_training_output_dict,
    extract_inference_mask,
    resolve_head_output_channels,
    resolve_head_spec,
)

__all__ = [
    'HEAD_SPECS',
    'HeadSpec',
    'MultiTargetHeadBundle',
    'build_training_output_dict',
    'extract_inference_mask',
    'resolve_head_output_channels',
    'resolve_head_spec',
]

"""recognition_pipeline: cut image prepare."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from kraken_core.asset_source import AssetSource
from neuralimage.lib.images import open_image
from neuralimage.lib.image_processing import cut_image
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.model.NeuralNetwork.context_utils import (
    build_context_batch,
    build_global_context_image,
    build_patch_coordinate_batch,
)

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import _resolve_relative_frame_name

def cut_image_prepare(
    img_path: Path,
    segment_size: tuple[int, int, int],
    overlap: int,
    *,
    use_context_branch: bool = False,
    use_cross_attention: bool = True,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    compression_factor: int = 1,
    source_root: Path | None = None,
    preprocessing: PreprocessingConfig | None = None,
    asset_source: AssetSource | None = None,
) -> dict[str, Any]:
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_process import _apply_preprocessing_to_channel_first
    frame_name = _resolve_relative_frame_name(img_path, source_root)
    payload: dict[str, Any] = {
        'baseim_size': None,
        'segment_size': segment_size,
        'overlap': overlap,
        'cutted_image': None,
        'context_image': None,
        'global_image': None,
        'patch_coords_norm': None,
        'patch_coords_px': None,
        'source_size_hw': None,
        'name': frame_name,
        'source_path': img_path,
    }

    Image.MAX_IMAGE_PIXELS = None
    channels = segment_size[0]
    with open_image(img_path, asset_source) as source_image:
        payload['baseim_size'] = source_image.size
        compression_factor = max(1, int(compression_factor))
        if compression_factor > 1:
            original_size = source_image.size
            compressed_size = (
                max(1, int(round(original_size[0] / compression_factor))),
                max(1, int(round(original_size[1] / compression_factor))),
            )
            source_image = source_image.resize(compressed_size, resample=Image.Resampling.BILINEAR)
            source_image = source_image.resize(original_size, resample=Image.Resampling.BILINEAR)
        work_image = image_to_channel_first_float01(source_image, channels)

    work_image = _apply_preprocessing_to_channel_first(work_image, preprocessing)
    payload['cutted_image'] = cut_image(work_image, segment_size, overlap, normalize=False)
    if use_context_branch and context_crop_size is not None and context_input_size is not None:
        base_shape_hw = (int(work_image.shape[1]), int(work_image.shape[2]))
        if bool(use_cross_attention):
            global_image = build_global_context_image(
                work_image,
                output_size_xy=tuple(context_input_size),
                interpolation_mode='bilinear',
            ).astype(np.float32, copy=False)
            coords_px, coords_norm = build_patch_coordinate_batch(
                base_shape_hw,
                local_patch_size_xy=(int(segment_size[1]), int(segment_size[2])),
                overlap=overlap,
            )
            patch_count = int(payload['cutted_image'].shape[0])
            payload['global_image'] = global_image
            payload['context_image'] = np.repeat(global_image[None, :, :, :], patch_count, axis=0)
            payload['patch_coords_px'] = coords_px.astype(np.float32, copy=False)
            payload['patch_coords_norm'] = coords_norm.astype(np.float32, copy=False)
            payload['source_size_hw'] = np.asarray(base_shape_hw, dtype=np.float32)
        else:
            payload['context_image'] = build_context_batch(
                work_image,
                local_patch_size_xy=(int(segment_size[1]), int(segment_size[2])),
                overlap=overlap,
                context_crop_size_xy=tuple(context_crop_size),
                context_input_size_xy=tuple(context_input_size),
            ).astype(np.float32, copy=False)
    return payload

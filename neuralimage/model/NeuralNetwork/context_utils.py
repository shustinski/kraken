from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchWindow:
    """Patch window in source-image coordinates."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return int(self.left + self.width)

    @property
    def bottom(self) -> int:
        return int(self.top + self.height)

    @property
    def center_x(self) -> float:
        return float(self.left + (self.width / 2.0))

    @property
    def center_y(self) -> float:
        return float(self.top + (self.height / 2.0))


def normalize_size_pair(
    value: Sequence[int] | int | None,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """Normalize a width/height pair used by patch-based settings."""

    if value is None:
        return int(fallback[0]), int(fallback[1])
    if isinstance(value, int):
        normalized = max(1, int(value))
        return normalized, normalized
    if len(value) != 2:
        raise ValueError(f'Expected a pair of integers, got {value!r}.')
    return max(1, int(value[0])), max(1, int(value[1]))


def normalize_channel_sequence(
    value: Sequence[int] | int | None,
    *,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    """Normalize encoder channel settings into a non-empty tuple."""

    if value is None:
        return tuple(int(channel) for channel in fallback)
    if isinstance(value, int):
        channel = max(1, int(value))
        return channel, channel * 2, channel * 4, channel * 8

    channels = tuple(max(1, int(channel)) for channel in value)
    if not channels:
        raise ValueError('context_branch_channels must not be empty.')
    return channels


def resolve_sliding_windows(
    base_shape_hw: tuple[int, int],
    patch_size_xy: tuple[int, int],
    overlap: int,
) -> list[PatchWindow]:
    """Return patch windows in the same order as the existing cut/sew pipeline."""

    base_height, base_width = int(base_shape_hw[0]), int(base_shape_hw[1])
    patch_width, patch_height = int(patch_size_xy[0]), int(patch_size_xy[1])
    stride_height = max(1, int(patch_height - overlap))
    stride_width = max(1, int(patch_width - overlap))
    row_steps = int(base_height / stride_height) + 1
    column_steps = int(base_width / stride_width) + 1

    windows: list[PatchWindow] = []
    for row in range(row_steps):
        for col in range(column_steps):
            left = col * stride_width
            top = row * stride_height
            right = left + patch_width
            bottom = top + patch_height

            src_top = top if bottom <= base_height else max(0, base_height - patch_height)
            src_left = left if right <= base_width else max(0, base_width - patch_width)
            src_bottom = min(base_height, src_top + patch_height)
            src_right = min(base_width, src_left + patch_width)
            windows.append(
                PatchWindow(
                    left=int(src_left),
                    top=int(src_top),
                    width=max(1, int(src_right - src_left)),
                    height=max(1, int(src_bottom - src_top)),
                )
            )
    return windows


def extract_centered_crop(
    image_chw: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size_xy: tuple[int, int],
    output_size_xy: tuple[int, int],
    interpolation_mode: str = 'bilinear',
) -> np.ndarray:
    """Extract a padded crop around a center point and resize it if needed."""

    if image_chw.ndim != 3:
        raise ValueError(f'Expected CHW image, got shape {image_chw.shape!r}.')

    channels, base_height, base_width = (
        int(image_chw.shape[0]),
        int(image_chw.shape[1]),
        int(image_chw.shape[2]),
    )
    crop_width, crop_height = int(crop_size_xy[0]), int(crop_size_xy[1])
    target_width, target_height = int(output_size_xy[0]), int(output_size_xy[1])

    left = int(round(center_x - (crop_width / 2.0)))
    top = int(round(center_y - (crop_height / 2.0)))
    right = left + crop_width
    bottom = top + crop_height

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(base_width, right)
    src_bottom = min(base_height, bottom)

    dst_left = max(0, -left)
    dst_top = max(0, -top)
    dst_right = dst_left + max(0, src_right - src_left)
    dst_bottom = dst_top + max(0, src_bottom - src_top)

    crop = np.zeros((channels, crop_height, crop_width), dtype=image_chw.dtype)
    if src_right > src_left and src_bottom > src_top:
        crop[:, dst_top:dst_bottom, dst_left:dst_right] = image_chw[:, src_top:src_bottom, src_left:src_right]

    if crop_width == target_width and crop_height == target_height:
        return np.ascontiguousarray(crop)

    return resize_chw_image(
        crop,
        output_size_xy=(target_width, target_height),
        interpolation_mode=interpolation_mode,
    )


def resize_chw_image(
    image_chw: np.ndarray,
    *,
    output_size_xy: tuple[int, int],
    interpolation_mode: str = 'bilinear',
) -> np.ndarray:
    """Resize a CHW image using torch interpolation and preserve dtype."""

    if image_chw.ndim != 3:
        raise ValueError(f'Expected CHW image, got shape {image_chw.shape!r}.')

    target_width, target_height = int(output_size_xy[0]), int(output_size_xy[1])
    if image_chw.shape[-1] == target_width and image_chw.shape[-2] == target_height:
        return np.ascontiguousarray(image_chw)

    tensor = torch.from_numpy(np.ascontiguousarray(image_chw)).unsqueeze(0)
    if interpolation_mode == 'nearest':
        resized = F.interpolate(tensor, size=(target_height, target_width), mode='nearest')
    else:
        resized = F.interpolate(
            tensor,
            size=(target_height, target_width),
            mode=interpolation_mode,
            align_corners=False,
        )
    return resized.squeeze(0).cpu().numpy().astype(image_chw.dtype, copy=False)


def build_context_batch(
    image_chw: np.ndarray,
    *,
    local_patch_size_xy: tuple[int, int],
    overlap: int,
    context_crop_size_xy: tuple[int, int],
    context_input_size_xy: tuple[int, int],
) -> np.ndarray:
    """Build context inputs aligned with sliding-window local patches."""

    windows = resolve_sliding_windows(
        base_shape_hw=(int(image_chw.shape[1]), int(image_chw.shape[2])),
        patch_size_xy=local_patch_size_xy,
        overlap=overlap,
    )
    context_batch = [
        extract_centered_crop(
            image_chw,
            center_x=window.center_x,
            center_y=window.center_y,
            crop_size_xy=context_crop_size_xy,
            output_size_xy=context_input_size_xy,
        )
        for window in windows
    ]
    if not context_batch:
        channels = int(image_chw.shape[0])
        return np.zeros(
            (
                0,
                channels,
                int(context_input_size_xy[1]),
                int(context_input_size_xy[0]),
            ),
            dtype=image_chw.dtype,
        )
    return np.stack(context_batch, axis=0)

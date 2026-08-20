"""Viewport sampling for vector gradient-field arrows (image coordinates)."""

from __future__ import annotations

from math import ceil

import numpy as np

DEFAULT_VIEW_STEP_PX = 18.0
DEFAULT_ARROW_LENGTH_VIEW_PX = 14.0
DEFAULT_MIN_MAGNITUDE_FRACTION = 0.08
DEFAULT_MAX_ARROWS = 2500

ArrowSample = tuple[float, float, float, float]


def gradient_sample_step_image_px(
    scene_units_per_view_px: float,
    view_step_px: float = DEFAULT_VIEW_STEP_PX,
) -> float:
    """Image-space grid step for a viewport spacing of ``view_step_px`` screen pixels."""
    return max(float(view_step_px), 1e-6) * max(float(scene_units_per_view_px), 1e-12)


def sample_gradient_field_arrows(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    visible_rect: tuple[float, float, float, float],
    scene_units_per_view_px: float,
    *,
    view_step_px: float = DEFAULT_VIEW_STEP_PX,
    arrow_length_view_px: float = DEFAULT_ARROW_LENGTH_VIEW_PX,
    min_magnitude_fraction: float = DEFAULT_MIN_MAGNITUDE_FRACTION,
    max_arrows: int = DEFAULT_MAX_ARROWS,
    peak_magnitude: float | None = None,
) -> list[ArrowSample]:
    """Sample Sobel ``gx``/``gy`` on a viewport-spaced grid.

    ``scene_units_per_view_px`` is image/scene pixels per view pixel (``1 / zoom``).
    Each returned ``(x, y, dx, dy)`` is in scene/image coordinates; ``(dx, dy)`` has
    length ``arrow_length_view_px * scene_units_per_view_px``.
    """
    gx = np.asarray(gradient_x)
    gy = np.asarray(gradient_y)
    if gx.ndim != 2 or gy.shape != gx.shape or gx.size == 0:
        return []
    height, width = int(gx.shape[0]), int(gx.shape[1])
    left, top, right, bottom = (float(v) for v in visible_rect)
    vis_left = min(left, right)
    vis_right = max(left, right)
    vis_top = min(top, bottom)
    vis_bottom = max(top, bottom)
    clip_left = max(0.0, vis_left)
    clip_top = max(0.0, vis_top)
    clip_right = min(float(width), vis_right)
    clip_bottom = min(float(height), vis_bottom)
    if clip_right <= clip_left or clip_bottom <= clip_top:
        return []

    step = gradient_sample_step_image_px(scene_units_per_view_px, view_step_px)
    vis_w = clip_right - clip_left
    vis_h = clip_bottom - clip_top
    columns = max(1, int(ceil(vis_w / step)))
    rows = max(1, int(ceil(vis_h / step)))
    estimated = columns * rows
    if estimated > max(1, int(max_arrows)):
        step *= (estimated / float(max_arrows)) ** 0.5

    length_scene = max(float(arrow_length_view_px), 0.0) * max(float(scene_units_per_view_px), 1e-12)
    if length_scene <= 1e-12:
        return []

    xs = _grid_samples(clip_left, clip_right, step)
    ys = _grid_samples(clip_top, clip_bottom, step)
    if xs.size == 0 or ys.size == 0:
        return []

    ix = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
    iy = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
    grid_x, grid_y = np.meshgrid(ix, iy, indexing="xy")
    sample_x, sample_y = np.meshgrid(xs, ys, indexing="xy")
    dx = gx[grid_y, grid_x].astype(np.float64, copy=False)
    dy = gy[grid_y, grid_x].astype(np.float64, copy=False)
    mag_sq = dx * dx + dy * dy

    peak = float(peak_magnitude) if peak_magnitude is not None else peak_gradient_magnitude(gx, gy)
    if peak <= 1e-12:
        return []
    min_mag_sq = (float(min_magnitude_fraction) * peak) ** 2
    strong = mag_sq >= min_mag_sq
    if not np.any(strong):
        return []

    pixel_keys = grid_x[strong].astype(np.int64) * np.int64(height) + grid_y[strong].astype(np.int64)
    _, unique_idx = np.unique(pixel_keys, return_index=True)
    unique_idx.sort()

    dx_s = dx[strong].reshape(-1)[unique_idx]
    dy_s = dy[strong].reshape(-1)[unique_idx]
    mag = np.sqrt(dx_s * dx_s + dy_s * dy_s)
    mag = np.maximum(mag, 1e-12)
    scale = length_scene / mag
    out_dx = dx_s * scale
    out_dy = dy_s * scale
    out_x = sample_x[strong].reshape(-1)[unique_idx]
    out_y = sample_y[strong].reshape(-1)[unique_idx]

    arrows: list[ArrowSample] = [
        (float(px), float(py), float(adx), float(ady))
        for px, py, adx, ady in zip(out_x, out_y, out_dx, out_dy)
    ]
    if len(arrows) > int(max_arrows):
        stride = int(ceil(len(arrows) / float(max_arrows)))
        arrows = arrows[::stride][: int(max_arrows)]
    return arrows


def peak_gradient_magnitude(gradient_x: np.ndarray, gradient_y: np.ndarray) -> float:
    mag_sq = np.square(np.asarray(gradient_x, dtype=np.float64)) + np.square(
        np.asarray(gradient_y, dtype=np.float64)
    )
    if mag_sq.size == 0:
        return 0.0
    return float(np.sqrt(np.max(mag_sq)))


def _grid_samples(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 1e-12 or stop <= start:
        return np.empty(0, dtype=np.float64)
    origin = 0.5 * step
    first_index = int(ceil((start - origin) / step - 1e-12))
    first = origin + first_index * step
    if first < start:
        first += step
    if first >= stop:
        return np.empty(0, dtype=np.float64)
    return np.arange(first, stop, step, dtype=np.float64)

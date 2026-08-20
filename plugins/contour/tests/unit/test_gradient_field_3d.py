from __future__ import annotations

import cv2
import numpy as np

from contour.graphics.gradient_field_3d import (
    HEIGHT_MODE_INTENSITY,
    HEIGHT_MODE_MAGNITUDE,
    PREVIEW_MAX_SIDE,
    prepare_gradient_field_3d,
    project_points,
    render_gradient_field_3d_bgr,
)


def _ring_gradients(size: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-2.0, 2.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    height = np.exp(-((radius - 1.0) ** 2) / 0.18)
    gradient_y, gradient_x = np.gradient(height)
    return gradient_x.astype(np.float32), gradient_y.astype(np.float32), height.astype(np.float32)


def test_prepare_builds_surface_and_streamlines() -> None:
    gx, gy, height = _ring_gradients()
    model = prepare_gradient_field_3d(gx, gy, intensity=height, max_side=32, streamline_count=16)
    assert model.z.shape[0] >= 8
    assert model.z.shape == model.gx.shape
    assert float(model.z.max()) > float(model.z.min())
    assert model.streamlines


def test_height_mode_intensity_uses_source_values() -> None:
    gx, gy, height = _ring_gradients()
    mag = prepare_gradient_field_3d(gx, gy, intensity=height, height_mode=HEIGHT_MODE_MAGNITUDE, max_side=24)
    src = prepare_gradient_field_3d(gx, gy, intensity=height, height_mode=HEIGHT_MODE_INTENSITY, max_side=24)
    assert mag.z.shape == src.z.shape
    assert float(np.max(np.abs(mag.z - src.z))) > 1e-4


def test_render_gradient_field_3d_is_not_blank() -> None:
    gx, gy, height = _ring_gradients()
    model = prepare_gradient_field_3d(gx, gy, intensity=height, max_side=24, streamline_count=9)
    image = render_gradient_field_3d_bgr(model, width=240, height=180)
    assert image.shape == (180, 240, 3)
    assert image.dtype == np.uint8
    assert int(image.max()) > 80
    assert not np.array_equal(image, np.full_like(image, image[0, 0]))


def test_preview_render_skips_blank_canvas() -> None:
    gx, gy, height = _ring_gradients()
    model = prepare_gradient_field_3d(gx, gy, intensity=height, max_side=20, streamline_count=4)
    preview = render_gradient_field_3d_bgr(model, width=160, height=120, preview=True)
    full = render_gradient_field_3d_bgr(model, width=160, height=120, preview=False)
    assert preview.shape == full.shape
    assert int(preview.max()) > 80


def test_quality_mesh_is_finer_than_preview() -> None:
    gx, gy, height = _ring_gradients(80)
    preview = prepare_gradient_field_3d(
        gx, gy, intensity=height, max_side=PREVIEW_MAX_SIDE, streamline_count=0
    )
    quality = prepare_gradient_field_3d(gx, gy, intensity=height)
    assert quality.z.shape[0] > preview.z.shape[0]
    assert quality.z.shape[1] > preview.z.shape[1]
    assert len(quality.streamlines) > len(preview.streamlines)


def test_intensity_surface_is_smoothed() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.random((80, 80), dtype=np.float32)
    gx = np.ones_like(noisy)
    gy = np.zeros_like(noisy)
    model = prepare_gradient_field_3d(
        gx,
        gy,
        intensity=noisy,
        height_mode=HEIGHT_MODE_INTENSITY,
        max_side=40,
        streamline_count=0,
    )
    raw = cv2.resize(noisy, (int(model.z.shape[1]), int(model.z.shape[0])), interpolation=cv2.INTER_AREA)
    raw = (raw - float(raw.min())) / (float(raw.max() - raw.min()) + 1e-6)
    height = model.z / (float(np.max(model.z)) + 1e-6)
    lap_model = cv2.Laplacian(height.astype(np.float32), cv2.CV_32F)
    lap_raw = cv2.Laplacian(raw.astype(np.float32), cv2.CV_32F)
    assert float(np.mean(np.abs(lap_model))) < float(np.mean(np.abs(lap_raw)))


def test_projection_changes_with_azimuth() -> None:
    x = np.array([[-1.0, 1.0]], dtype=np.float32)
    y = np.array([[0.0, 0.0]], dtype=np.float32)
    z = np.array([[0.0, 0.0]], dtype=np.float32)
    sx_a, _, _ = project_points(x, y, z, -60.0, 30.0)
    sx_b, _, _ = project_points(x, y, z, 20.0, 30.0)
    assert float(np.max(np.abs(sx_a - sx_b))) > 1e-3


def test_render_fills_wide_canvas() -> None:
    gx, gy, height = _ring_gradients()
    model = prepare_gradient_field_3d(gx, gy, intensity=height, max_side=24, streamline_count=4)
    image = render_gradient_field_3d_bgr(model, width=420, height=160)
    background = np.array([22, 24, 32], dtype=np.int16)
    occupied_cols = np.any(np.abs(image.astype(np.int16) - background) > 12, axis=(0, 2))
    used = np.flatnonzero(occupied_cols)
    assert used.size > 0
    assert int(used[0]) < 24
    assert int(used[-1]) > 420 - 24
    occupied_rows = np.any(np.abs(image.astype(np.int16) - background) > 12, axis=(1, 2))
    used_rows = np.flatnonzero(occupied_rows)
    assert int(used_rows[0]) < 24
    assert int(used_rows[-1]) > 160 - 24

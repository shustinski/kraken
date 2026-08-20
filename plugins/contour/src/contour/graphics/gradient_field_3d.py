"""Software 3D preview of a Sobel gradient field (surface + streamlines)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin

import cv2
import numpy as np

DEFAULT_MAX_SIDE = 72
PREVIEW_MAX_SIDE = 36
DEFAULT_STREAMLINE_COUNT = 36
DEFAULT_AZIMUTH_DEG = -60.0
DEFAULT_ELEVATION_DEG = 28.0
HEIGHT_MODE_MAGNITUDE = "magnitude"
HEIGHT_MODE_INTENSITY = "intensity"
_BACKGROUND_BGR = (22, 24, 32)
_STREAM_BGR = (240, 240, 255)
_BOX_BGR = (210, 214, 230)
_TURBO_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_INFERNO).reshape(256, 3)


@dataclass(slots=True)
class GradientField3DModel:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    streamlines: tuple[np.ndarray, ...]
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    face_normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    face_color: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))


def prepare_gradient_field_3d(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    intensity: np.ndarray | None = None,
    height_mode: str = HEIGHT_MODE_MAGNITUDE,
    max_side: int = DEFAULT_MAX_SIDE,
    streamline_count: int = DEFAULT_STREAMLINE_COUNT,
) -> GradientField3DModel:
    gx = np.asarray(gradient_x, dtype=np.float32)
    gy = np.asarray(gradient_y, dtype=np.float32)
    if gx.ndim != 2 or gy.shape != gx.shape or gx.size == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return GradientField3DModel(empty, empty, empty, empty, empty, ())
    rows, cols = _fit_grid_shape(int(gx.shape[0]), int(gx.shape[1]), max(8, int(max_side)))
    intensity_mode = height_mode == HEIGHT_MODE_INTENSITY and intensity is not None
    gx_s = _lowpass_resize(gx, cols, rows, sigma=2.0)
    gy_s = _lowpass_resize(gy, cols, rows, sigma=2.0)
    magnitude = cv2.magnitude(gx_s, gy_s)
    if intensity_mode:
        source = np.asarray(intensity, dtype=np.float32)
        if source.ndim == 3:
            source = cv2.cvtColor(ensure_bgr_uint8(source), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if source.ndim == 2 and source.size:
            height = _lowpass_resize(source, cols, rows, sigma=3.2)
        else:
            height = magnitude
    else:
        height = magnitude
    height = _stretch_height(height)
    xs = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
    ys = np.linspace(1.0, -1.0, rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    span = max(cols / max(rows, 1), 1.0)
    grid_x = grid_x * span
    z = height.astype(np.float32, copy=False) * 0.38
    faces, normals, colors = _build_faces(grid_x, grid_y, z, height)
    lines: tuple[np.ndarray, ...] = ()
    if int(streamline_count) > 0:
        lines = tuple(
            _trace_streamlines(
                gx_s,
                gy_s,
                z,
                grid_x,
                grid_y,
                count=max(4, int(streamline_count)),
            )
        )
    return GradientField3DModel(
        grid_x,
        grid_y,
        z,
        gx_s,
        gy_s,
        tuple(lines),
        faces,
        normals,
        colors,
    )


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    if data.dtype != np.uint8:
        data = cv2.convertScaleAbs(data)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(data[..., :3])


@dataclass(slots=True)
class _CanvasMap:
    width: int
    height: int
    scale_x: float
    scale_y: float
    center_x: float
    center_y: float

    def apply(self, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        px = (sx - self.center_x) * self.scale_x + 0.5 * float(self.width)
        py = 0.5 * float(self.height) - (sy - self.center_y) * self.scale_y
        return np.stack((px, py), axis=-1)


def render_gradient_field_3d_bgr(
    model: GradientField3DModel,
    *,
    width: int = 960,
    height: int = 720,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    zoom: float = 1.0,
    preview: bool = False,
) -> np.ndarray:
    canvas_w = max(160, int(width))
    canvas_h = max(120, int(height))
    image = np.full((canvas_h, canvas_w, 3), _BACKGROUND_BGR, dtype=np.uint8)
    if model.z.size == 0:
        return image
    sx, sy, depth = _project_grid(model.x, model.y, model.z, azimuth_deg, elevation_deg)
    canvas = _fit_canvas_map(
        sx.reshape(-1),
        sy.reshape(-1),
        canvas_w,
        canvas_h,
        zoom=zoom,
    )
    mapped = canvas.apply(sx, sy)
    _draw_surface(image, mapped, depth, model, azimuth_deg, elevation_deg)
    _draw_box(image, model, azimuth_deg, elevation_deg, canvas, labels=not preview)
    if not preview:
        _draw_streamlines(image, model, azimuth_deg, elevation_deg, canvas)
    return image


def project_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _project_grid(x, y, z, azimuth_deg, elevation_deg)


def _stretch_height(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if hi - lo <= 1e-6:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi - lo <= 1e-6:
        return np.zeros_like(values)
    stretched = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.power(stretched, 0.85, dtype=np.float32)


def _lowpass_resize(values: np.ndarray, cols: int, rows: int, *, sigma: float) -> np.ndarray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    if min(int(data.shape[0]), int(data.shape[1])) >= 12:
        kernel = max(3, int(round(float(sigma) * 6.0)) | 1)
        data = cv2.GaussianBlur(data, (kernel, kernel), float(sigma))
    resized = cv2.resize(data, (int(cols), int(rows)), interpolation=cv2.INTER_AREA)
    fine = max(3, int(round(5.0)) | 1)
    return cv2.GaussianBlur(resized, (fine, fine), 1.15)


def _build_faces(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = grid_z.shape
    if rows < 2 or cols < 2:
        empty_i = np.zeros((0, 4), dtype=np.int32)
        empty_n = np.zeros((0, 3), dtype=np.float32)
        empty_c = np.zeros((0, 3), dtype=np.uint8)
        return empty_i, empty_n, empty_c
    row_i, col_i = np.mgrid[0 : rows - 1, 0 : cols - 1]
    i00 = row_i * cols + col_i
    faces = np.stack((i00, i00 + 1, i00 + cols + 1, i00 + cols), axis=-1).reshape(-1, 4).astype(np.int32)
    px = grid_x.reshape(-1)
    py = grid_y.reshape(-1)
    pz = grid_z.reshape(-1)
    p00 = np.stack((px[faces[:, 0]], py[faces[:, 0]], pz[faces[:, 0]]), axis=-1)
    p01 = np.stack((px[faces[:, 1]], py[faces[:, 1]], pz[faces[:, 1]]), axis=-1)
    p10 = np.stack((px[faces[:, 3]], py[faces[:, 3]], pz[faces[:, 3]]), axis=-1)
    normals = np.cross(p01 - p00, p10 - p00)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = (normals / norms).astype(np.float32)
    face_height = height.reshape(-1)[faces].mean(axis=1)
    lut_index = np.clip(np.rint(face_height * 255.0), 0, 255).astype(np.int32)
    colors = _TURBO_LUT[lut_index]
    return faces, normals, colors


def _fit_grid_shape(rows: int, cols: int, max_side: int) -> tuple[int, int]:
    longest = max(rows, cols, 1)
    scale = min(1.0, float(max_side) / float(longest))
    out_rows = max(8, int(round(rows * scale)))
    out_cols = max(8, int(round(cols * scale)))
    return out_rows, out_cols


def _project_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = radians(float(azimuth_deg))
    el = radians(float(elevation_deg))
    ca, sa = cos(az), sin(az)
    ce, se = cos(el), sin(el)
    xp = ca * x - sa * y
    yp = sa * x + ca * y
    screen_y = se * yp + ce * z
    depth = ce * yp - se * z
    return xp, screen_y, depth


def _fit_canvas_map(
    sx: np.ndarray,
    sy: np.ndarray,
    width: int,
    height: int,
    *,
    zoom: float,
) -> _CanvasMap:
    pad = 8.0
    usable_w = max(1.0, float(width) - 2.0 * pad)
    usable_h = max(1.0, float(height) - 2.0 * pad)
    span_x = float(np.max(sx) - np.min(sx)) if sx.size else 1.0
    span_y = float(np.max(sy) - np.min(sy)) if sy.size else 1.0
    zoom = max(0.35, min(2.8, float(zoom)))
    scale_x = usable_w / max(span_x, 1e-6) * zoom
    scale_y = usable_h / max(span_y, 1e-6) * zoom
    center_x = 0.5 * (float(np.max(sx) + np.min(sx))) if sx.size else 0.0
    center_y = 0.5 * (float(np.max(sy) + np.min(sy))) if sy.size else 0.0
    return _CanvasMap(width, height, scale_x, scale_y, center_x, center_y)


def _draw_surface(
    image: np.ndarray,
    mapped: np.ndarray,
    depth: np.ndarray,
    model: GradientField3DModel,
    azimuth_deg: float,
    elevation_deg: float,
) -> None:
    if model.faces.size == 0:
        return
    pts = mapped.reshape(-1, 2)
    depth_flat = depth.reshape(-1)
    face_depth = depth_flat[model.faces].mean(axis=1)
    shade = _face_shade(model.face_normal, azimuth_deg, elevation_deg)
    lit = np.clip(model.face_color.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(face_depth)
    polys = pts[model.faces].astype(np.int32)
    for index in order:
        cv2.fillConvexPoly(image, polys[index], (int(lit[index, 0]), int(lit[index, 1]), int(lit[index, 2])))


def _face_shade(normals: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = radians(float(azimuth_deg))
    el = radians(float(elevation_deg))
    ca, sa = cos(az), sin(az)
    ce, se = cos(el), sin(el)
    nx = ca * normals[:, 0] - sa * normals[:, 1]
    ny = sa * normals[:, 0] + ca * normals[:, 1]
    nz = normals[:, 2]
    cam_z = se * ny + ce * nz
    shade = 0.58 + 0.42 * np.abs(cam_z)
    return np.clip(shade, 0.45, 1.12).astype(np.float32)


def _model_corners(model: GradientField3DModel) -> np.ndarray:
    xmin, xmax = float(np.min(model.x)), float(np.max(model.x))
    ymin, ymax = float(np.min(model.y)), float(np.max(model.y))
    zmin, zmax = float(np.min(model.z)), float(np.max(model.z))
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float32,
    )


def _draw_box(
    image: np.ndarray,
    model: GradientField3DModel,
    azimuth_deg: float,
    elevation_deg: float,
    canvas: _CanvasMap,
    *,
    labels: bool,
) -> None:
    corners = _model_corners(model)
    sx, sy, _depth = _project_grid(corners[:, 0], corners[:, 1], corners[:, 2], azimuth_deg, elevation_deg)
    mapped = canvas.apply(sx, sy)
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for a, b in edges:
        p0 = (int(round(mapped[a, 0])), int(round(mapped[a, 1])))
        p1 = (int(round(mapped[b, 0])), int(round(mapped[b, 1])))
        cv2.line(image, p0, p1, _BOX_BGR, 1, cv2.LINE_AA)
    if not labels:
        return
    for index, text in ((1, "X"), (2, "Y"), (4, "Z")):
        px, py = int(round(mapped[index, 0])), int(round(mapped[index, 1]))
        cv2.putText(image, text, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 244, 255), 1, cv2.LINE_AA)


def _draw_streamlines(
    image: np.ndarray,
    model: GradientField3DModel,
    azimuth_deg: float,
    elevation_deg: float,
    canvas: _CanvasMap,
) -> None:
    for line in model.streamlines:
        if line.shape[0] < 2:
            continue
        sx, sy, _ = _project_grid(line[:, 0], line[:, 1], line[:, 2], azimuth_deg, elevation_deg)
        pts = canvas.apply(sx, sy).astype(np.int32)
        cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, _STREAM_BGR, 2, cv2.LINE_AA)
        step = max(7, pts.shape[0] // 6)
        for index in range(step, pts.shape[0], step):
            _draw_arrowhead(image, pts[max(0, index - 4) : index + 1], _STREAM_BGR)
        _draw_arrowhead(image, pts, _STREAM_BGR)


def _draw_arrowhead(image: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    if pts.shape[0] < 2:
        return
    end = pts[-1].astype(np.float64)
    start = pts[max(0, pts.shape[0] - 4)].astype(np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 4.0:
        return
    ux, uy = delta / length
    nx, ny = -uy, ux
    tip = 8.0
    left = end - ux * tip + nx * 4.0
    right = end - ux * tip - nx * 4.0
    poly = np.array([end, left, right], dtype=np.int32)
    cv2.fillConvexPoly(image, poly, color)


def _trace_streamlines(
    gx: np.ndarray,
    gy: np.ndarray,
    height: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    count: int,
) -> list[np.ndarray]:
    rows, cols = height.shape
    mag = cv2.magnitude(gx, gy)
    peak = float(np.max(mag)) if mag.size else 0.0
    if peak <= 1e-8:
        return []
    min_mag = 0.10 * peak
    seeds_n = max(3, int(round(count**0.5)))
    row_ids = np.linspace(1, rows - 2, seeds_n)
    col_ids = np.linspace(1, cols - 2, seeds_n)
    lines: list[np.ndarray] = []
    for row in row_ids:
        for col in col_ids:
            if float(mag[int(round(row)), int(round(col))]) < min_mag:
                continue
            forward = _integrate_line(gx, gy, height, grid_x, grid_y, col, row, 1.0, min_mag)
            backward = _integrate_line(gx, gy, height, grid_x, grid_y, col, row, -1.0, min_mag)
            merged = np.concatenate([backward[::-1], forward[1:]], axis=0) if backward.size and forward.size else forward
            if merged.shape[0] >= 6:
                lines.append(merged)
    return lines


def _integrate_line(
    gx: np.ndarray,
    gy: np.ndarray,
    height: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    start_col: float,
    start_row: float,
    direction: float,
    min_mag: float,
) -> np.ndarray:
    rows, cols = height.shape
    x_step = float(grid_x[0, 1] - grid_x[0, 0]) if cols > 1 else 1.0
    y_step = float(grid_y[1, 0] - grid_y[0, 0]) if rows > 1 else 1.0
    ds = 0.38 * max(abs(x_step), abs(y_step), 1e-6)
    col = float(start_col)
    row = float(start_row)
    points: list[list[float]] = []
    for _ in range(80):
        if col < 0.0 or row < 0.0 or col > cols - 1 or row > rows - 1:
            break
        vx = _sample_bilinear(gx, col, row)
        vy = _sample_bilinear(gy, col, row)
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < min_mag:
            break
        wx = _sample_bilinear(grid_x, col, row)
        wy = _sample_bilinear(grid_y, col, row)
        wz = _sample_bilinear(height, col, row) + 0.018
        points.append([wx, wy, wz])
        col += direction * (vx / speed) * (ds / max(abs(x_step), 1e-6))
        row += direction * (vy / speed) * (ds / max(abs(y_step), 1e-6))
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def _sample_bilinear(grid: np.ndarray, col: float, row: float) -> float:
    rows, cols = grid.shape
    c0 = int(np.floor(col))
    r0 = int(np.floor(row))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)
    c0 = min(max(c0, 0), cols - 1)
    r0 = min(max(r0, 0), rows - 1)
    tc = col - np.floor(col)
    tr = row - np.floor(row)
    v00 = float(grid[r0, c0])
    v10 = float(grid[r0, c1])
    v01 = float(grid[r1, c0])
    v11 = float(grid[r1, c1])
    return (1.0 - tc) * (1.0 - tr) * v00 + tc * (1.0 - tr) * v10 + (1.0 - tc) * tr * v01 + tc * tr * v11

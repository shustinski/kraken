from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .application.processing import DisplaySettings
from .domain import PolygonData
from .i18n import tr

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def is_visible_image_path(path: str | Path) -> bool:
    normalized = Path(path)
    return is_image_path(normalized) and not normalized.name.startswith("_")


def scan_image_files(directory: str | Path) -> list[str]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and is_visible_image_path(path)
    ]


def ensure_directory(path: str | Path) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    data = image.astype(np.float32)
    data = np.nan_to_num(data, copy=False)
    min_value = float(np.min(data)) if data.size else 0.0
    max_value = float(np.max(data)) if data.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _imread_unicode_safe(path: str | Path, flags: int) -> np.ndarray | None:
    if isinstance(path, str):
        normalized_text = path
        if not os.path.isfile(path):
            raise FileNotFoundError(tr("unable_to_load_image", path=path))
    else:
        normalized_path = Path(path)
        if not normalized_path.exists():
            raise FileNotFoundError(tr("unable_to_load_image", path=normalized_path))
        normalized_text = str(normalized_path)
    if normalized_text.isascii():
        image = cv2.imread(normalized_text, flags)
        if image is not None:
            return image
    try:
        raw_bytes = np.fromfile(normalized_text, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(tr("unable_to_read_image_bytes", path=normalized_text)) from exc
    if raw_bytes.size == 0:
        raise FileNotFoundError(tr("unable_to_read_image_bytes", path=normalized_text))
    return cv2.imdecode(raw_bytes, flags)


def imwrite_unicode_safe(path: str | Path, image: np.ndarray) -> None:
    normalized_path = Path(path)
    suffix = normalized_path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(tr("unable_to_encode_image", path=normalized_path))
    encoded.tofile(str(normalized_path))


def load_image_grayscale(path: str | Path) -> np.ndarray:
    image = _imread_unicode_safe(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(tr("unable_to_load_image", path=path))
    if image.ndim == 2:
        return ensure_uint8(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return ensure_uint8(image)


def load_image_color_thumbnail(path: str | Path, max_width: int, max_height: int, *, cover: bool = False) -> np.ndarray:
    """Load a color image scaled down for thumbnail grids (avoids full-resolution decode)."""

    target_w = max(1, int(max_width))
    target_h = max(1, int(max_height))
    target_max = max(target_w, target_h)
    if target_max <= 128:
        flags = cv2.IMREAD_REDUCED_COLOR_8
    elif target_max <= 512:
        flags = cv2.IMREAD_REDUCED_COLOR_4
    else:
        flags = cv2.IMREAD_REDUCED_COLOR_2
    image = _imread_unicode_safe(path, flags)
    if image is None:
        raise FileNotFoundError(tr("unable_to_load_image", path=path))
    if image.ndim == 2:
        image = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = ensure_uint8(image)
    h, w = image.shape[:2]
    if w > 0 and h > 0:
        if cover:
            scale = max(target_w / float(w), target_h / float(h))
        else:
            scale = min(target_w / float(w), target_h / float(h), 1.0)
        resized_w = max(1, round(w * scale))
        resized_h = max(1, round(h * scale))
        if resized_w != w or resized_h != h:
            image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    return image


def load_image_color(path: str | Path) -> np.ndarray:
    image = _imread_unicode_safe(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(tr("unable_to_load_image", path=path))
    if image.ndim == 2:
        gray = ensure_uint8(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return ensure_uint8(image)


def ensure_binary_mask(image: np.ndarray) -> np.ndarray:
    gray = ensure_uint8(image)
    if gray.ndim == 3 and gray.shape[2] == 4:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
    elif gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    unique_values = np.unique(gray)
    if unique_values.size <= 2:
        return np.where(gray > 0, 255, 0).astype(np.uint8)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def cv_to_qimage(image: np.ndarray | None):
    from .adapters.qt.image_conversion import cv_to_qimage as _cv_to_qimage

    return _cv_to_qimage(image)


def draw_polygon_overlay(
    image: np.ndarray,
    polygons: Iterable[PolygonData],
    display_settings: DisplaySettings,
) -> np.ndarray:
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(ensure_uint8(base), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    alpha = max(0.0, min(1.0, display_settings.fill_opacity))
    line_width = max(1, round(display_settings.line_width))

    for polygon in polygons:
        points = np.asarray(polygon.points, dtype=np.int32)
        if points.shape[0] < 3:
            continue
        color = hex_to_bgr(display_settings.hole_color if polygon.is_hole else display_settings.external_color)
        if _is_ellipse_display_polygon(polygon):
            center, axes = _ellipse_geometry_from_points(points)
            cv2.ellipse(overlay, center, axes, 0.0, 0.0, 360.0, color, thickness=line_width, lineType=cv2.LINE_AA)
            cv2.ellipse(overlay, center, axes, 0.0, 0.0, 360.0, color, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(overlay, [points], True, color, thickness=line_width, lineType=cv2.LINE_AA)
            cv2.fillPoly(overlay, [points], color)
        if display_settings.show_vertices and not _is_ellipse_display_polygon(polygon):
            for x_coord, y_coord in polygon.points:
                cv2.circle(
                    overlay,
                    (round(x_coord), round(y_coord)),
                    max(1, round(display_settings.vertex_size / 2)),
                    hex_to_bgr(display_settings.vertex_color),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        if display_settings.show_labels:
            cv2.putText(
                overlay,
                str(polygon.id),
                (int(points[:, 0].min()), int(points[:, 1].min()) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if alpha > 0:
        base = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    else:
        base = overlay
    return base


def _is_ellipse_display_polygon(polygon: PolygonData) -> bool:
    return polygon.shape_hint == "box" or polygon.category == "via"


def _ellipse_geometry_from_points(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    x_min = int(points[:, 0].min())
    x_max = int(points[:, 0].max())
    y_min = int(points[:, 1].min())
    y_max = int(points[:, 1].max())
    center = (round((x_min + x_max) / 2.0), round((y_min + y_max) / 2.0))
    axes = (max(1, round((x_max - x_min) / 2.0)), max(1, round((y_max - y_min) / 2.0)))
    return center, axes


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        return 0, 255, 0
    red = int(text[0:2], 16)
    green = int(text[2:4], 16)
    blue = int(text[4:6], 16)
    return blue, green, red

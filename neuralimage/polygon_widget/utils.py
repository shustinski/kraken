from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PyQt6.QtGui import QImage

from .i18n import tr
from .models import DisplaySettings, Point, PolygonData


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def scan_image_files(directory: str | Path) -> list[str]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and is_image_path(path)
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
    normalized_path = Path(path)
    image = cv2.imread(str(normalized_path), flags)
    if image is not None:
        return image
    if not normalized_path.exists():
        raise FileNotFoundError(tr("unable_to_load_image", path=normalized_path))
    try:
        raw_bytes = np.fromfile(str(normalized_path), dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(tr("unable_to_read_image_bytes", path=normalized_path)) from exc
    if raw_bytes.size == 0:
        raise FileNotFoundError(tr("unable_to_read_image_bytes", path=normalized_path))
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
    unique_values = np.unique(gray)
    if unique_values.size <= 2:
        return np.where(gray > 0, 255, 0).astype(np.uint8)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def cv_to_qimage(image: np.ndarray | None) -> QImage:
    if image is None:
        return QImage()
    data = ensure_uint8(image)
    if data.ndim == 2:
        height, width = data.shape
        qimage = QImage(data.data, width, height, data.strides[0], QImage.Format.Format_Grayscale8)
        return qimage.copy()
    if data.ndim == 3 and data.shape[2] == 3:
        rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        return qimage.copy()
    raise ValueError(tr("unsupported_qimage_shape", shape=data.shape))


def compute_polygon_metrics(points: Iterable[Point]) -> tuple[float, float, tuple[int, int, int, int]]:
    array = np.asarray(list(points), dtype=np.float32)
    if array.size == 0:
        return 0.0, 0.0, (0, 0, 0, 0)
    if array.shape[0] == 1:
        x_coord, y_coord = array[0]
        return 0.0, 0.0, (int(x_coord), int(y_coord), 1, 1)
    if array.shape[0] == 2:
        x0, y0 = array[0]
        x1, y1 = array[1]
        perimeter = float(np.linalg.norm(array[1] - array[0]) * 2.0)
        x_min = int(np.floor(min(x0, x1)))
        y_min = int(np.floor(min(y0, y1)))
        x_max = int(np.ceil(max(x0, x1)))
        y_max = int(np.ceil(max(y0, y1)))
        return 0.0, perimeter, (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    contour = array.reshape((-1, 1, 2))
    area = float(abs(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    x_coord, y_coord, width, height = cv2.boundingRect(contour)
    return area, perimeter, (int(x_coord), int(y_coord), int(width), int(height))


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
    line_width = max(1, int(round(display_settings.line_width)))

    for polygon in polygons:
        points = np.asarray(polygon.points, dtype=np.int32)
        if points.shape[0] < 3:
            continue
        color = hex_to_bgr(display_settings.hole_color if polygon.is_hole else display_settings.external_color)
        cv2.polylines(overlay, [points], True, color, thickness=line_width, lineType=cv2.LINE_AA)
        cv2.fillPoly(overlay, [points], color)
        if display_settings.show_vertices:
            for x_coord, y_coord in polygon.points:
                cv2.circle(
                    overlay,
                    (int(round(x_coord)), int(round(y_coord))),
                    max(1, int(round(display_settings.vertex_size / 2))),
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


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        return 0, 255, 0
    red = int(text[0:2], 16)
    green = int(text[2:4], 16)
    blue = int(text[4:6], 16)
    return blue, green, red

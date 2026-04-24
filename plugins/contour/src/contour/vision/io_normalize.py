"""Image load + canonical normalization to ``uint8`` grayscale (or 3ch BGR where needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ImageRef

try:
    from ..utils import ensure_uint8, load_image_color
except ImportError:  # pragma: no cover - package-relative edge cases
    ensure_uint8 = None
    load_image_color = None


def load_bgr(path: str | Path) -> np.ndarray:
    if load_image_color is None:
        import cv2

        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    return load_image_color(str(path))


def to_gray_u8(image: Any) -> np.ndarray:
    import cv2

    data = ensure_uint8(np.asarray(image)) if ensure_uint8 is not None else ensure_uint8_local(image)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[2] >= 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return data[:, :, 0]


def ensure_uint8_local(image: Any) -> np.ndarray:

    data = np.asarray(image)
    if data.dtype == np.uint8:
        return data
    if data.size == 0:
        return data.astype(np.uint8)
    data_f = data.astype(np.float32)
    minimum = float(data_f.min())
    maximum = float(data_f.max())
    if maximum <= minimum:
        return np.zeros_like(data_f, dtype=np.uint8)
    scaled = (data_f - minimum) * (255.0 / (maximum - minimum))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def make_image_ref(path: str | None, gray: np.ndarray) -> ImageRef:
    height, width = gray.shape[:2]
    return ImageRef(
        path=path,
        width=int(width),
        height=int(height),
        dtype="uint8",
        channels=1,
    )

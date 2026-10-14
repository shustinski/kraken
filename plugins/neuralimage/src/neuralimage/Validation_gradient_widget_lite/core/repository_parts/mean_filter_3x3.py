"""repository: mean filter 3x3."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

def _qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """Convert Qt image to contiguous grayscale ndarray."""

    grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pointer = grayscale.bits()
    pointer.setsize(grayscale.height() * grayscale.bytesPerLine())
    buffer = np.frombuffer(pointer, dtype=np.uint8).reshape((grayscale.height(), grayscale.bytesPerLine()))
    return buffer[:, : grayscale.width()].copy()

def _grayscale_array_to_qimage(array: np.ndarray) -> QImage:
    """Convert contiguous grayscale ndarray to Qt image."""

    contiguous = np.ascontiguousarray(array.astype(np.uint8))
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_Grayscale8)
    return image.copy()

def _load_grayscale_image_raw(path: Path) -> np.ndarray:
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"Unable to decode image: {path}")
    return _qimage_to_grayscale_array(image)

def load_grayscale_image(path: Path) -> np.ndarray:
    """Load one image as grayscale ndarray."""

    return _load_grayscale_image_raw(Path(path))

def resize_grayscale_image(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize grayscale image using Qt fast transformation."""

    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    source = np.asarray(array, dtype=np.uint8)
    if source.shape == (target_height, target_width):
        return source.copy()
    image = _grayscale_array_to_qimage(source)
    scaled = image.scaled(
        target_width,
        target_height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return _qimage_to_grayscale_array(scaled)

def _mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    """Compute 3x3 mean filter without SciPy dependency."""

    image_f = np.asarray(image, dtype=np.float32)
    padded = np.pad(image_f, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0

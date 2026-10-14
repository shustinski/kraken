"""repository: extract original frame features."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _OriginalFrameFeatures

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mean_filter_3x3 import _mean_filter_3x3

def extract_original_frame_features(image: np.ndarray | None) -> _OriginalFrameFeatures | None:
    """Extract scalar quality features from original grayscale frame."""

    if image is None:
        return None
    grayscale = np.asarray(image, dtype=np.uint8)
    image_f = grayscale.astype(np.float32)
    normalized = image_f / 255.0
    mean_brightness = float(normalized.mean())
    contrast = float(normalized.std())

    histogram = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    histogram /= max(1.0, histogram.sum())
    non_zero = histogram > 0.0
    entropy = float(-(histogram[non_zero] * np.log2(histogram[non_zero])).sum())

    padded = np.pad(image_f, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    )
    blur_score = float(laplacian.var() / (255.0 * 255.0))

    smoothed = _mean_filter_3x3(grayscale)
    noise_score = float(np.mean(np.abs(image_f - smoothed)) / 255.0)

    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    edge_density = float(np.mean(gradient_magnitude >= 20.0))

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    strict_peaks = np.logical_and.reduce([center > neighbor for neighbor in neighbors])
    local_peak_density = float(np.mean(strict_peaks))

    dynamic_range = float((float(grayscale.max()) - float(grayscale.min())) / 255.0)
    saturation_ratio = float(np.mean((grayscale <= 5) | (grayscale >= 250)))

    return _OriginalFrameFeatures(
        mean_brightness=mean_brightness,
        contrast=contrast,
        entropy=entropy,
        blur_score=blur_score,
        noise_score=noise_score,
        edge_density=edge_density,
        local_peak_density=local_peak_density,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
    )

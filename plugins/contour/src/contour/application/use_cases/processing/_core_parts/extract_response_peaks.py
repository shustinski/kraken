"""_core: extract response peaks."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from contour.utils import ensure_uint8

def _log_blob_response(gray: np.ndarray, expected_span: int, *, bright: bool) -> np.ndarray:
    data = ensure_uint8(gray).astype(np.float32)
    result = np.zeros_like(data, dtype=np.float32)
    sigmas = (
        max(0.65, expected_span * 0.20),
        max(0.85, expected_span * 0.34),
        max(1.10, expected_span * 0.50),
    )
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(data, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        local = (-laplacian if bright else laplacian) * float(sigma * sigma)
        local = np.maximum(local, 0.0)
        values = local[local > 0.0]
        if values.size == 0:
            continue
        # A confident bright via produces LoG(x)*sigma^2 on the order of the
        # intensity contrast. Do not normalize below ~40 intensity units,
        # otherwise weak noise blobs get amplified to near-saturation.
        scale = max(40.0, float(np.percentile(values, 99.5)))
        result = np.maximum(result, np.clip(local * (255.0 / scale), 0.0, 255.0))
    return result.astype(np.uint8)

def _ring_template_response(gradient: np.ndarray, radii: list[int]) -> np.ndarray:
    """Maximum response of circle-template matched against gradient elevation."""

    if gradient.size == 0 or not radii:
        return np.zeros_like(gradient, dtype=np.uint8)
    search = ensure_uint8(gradient).astype(np.float32) / 255.0
    response = np.zeros(gradient.shape, dtype=np.float32)
    for radius in radii:
        radius = max(2, int(radius))
        size = radius * 2 + 5
        if gradient.shape[0] < size or gradient.shape[1] < size:
            continue
        center = size // 2
        template = np.zeros((size, size), dtype=np.float32)
        thickness = max(1, round(radius * 0.22))
        cv2.circle(template, (center, center), radius, 1.0, thickness=thickness, lineType=cv2.LINE_AA)
        if float(template.sum()) <= 0.0:
            continue
        scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        scores = np.nan_to_num(scores, copy=False)
        if scores.size == 0:
            continue
        placed = np.zeros_like(response, dtype=np.float32)
        ph, pw = scores.shape
        placed[center : center + ph, center : center + pw] = scores
        response = np.maximum(response, placed)
    return np.clip(response * 255.0, 0.0, 255.0).astype(np.uint8)

def _extract_response_peaks(response: np.ndarray, *, min_distance: int) -> list[tuple[int, int]]:
    """Return list of ``(y, x)`` local maxima with NMS, sorted by descending value."""

    if response.size == 0:
        return []
    data = ensure_uint8(response)
    nonzero_values = data[data > 0]
    if nonzero_values.size == 0:
        return []
    # Require peaks to stand out against the local background. The old cutoff
    # (50 % of the 80th percentile) was dominated by noise on low-signal
    # images; raising both the floor and the percentile keeps the peak list
    # short and focused on genuine candidates.
    cutoff = max(30, int(np.percentile(nonzero_values, 90) * 0.65))
    kernel_size = max(3, int(min_distance) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(data, kernel)
    local_max = (data >= dilated) & (data >= cutoff)
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return []
    values = data[ys, xs]
    order = np.argsort(values)[::-1]
    used = np.zeros(data.shape, dtype=bool)
    selected: list[tuple[int, int]] = []
    radius = max(1, int(min_distance))
    max_peaks = 2000
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if used[y, x]:
            continue
        selected.append((y, x))
        if len(selected) >= max_peaks:
            break
        y0 = max(0, y - radius)
        y1 = min(data.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(data.shape[1], x + radius + 1)
        used[y0:y1, x0:x1] = True
    return selected

@dataclass(frozen=True, slots=True)
class _IsolationContext:
    """Labels / stats of the foreground mask used to reject trace-like shapes.

    ``active`` is ``False`` when the user-provided intensity range covers most
    of the image (so the connected-component gate would wrongly reject
    everything) or when there is nothing to anchor on.
    """

    active: bool
    labels: np.ndarray
    stats: np.ndarray

def _build_isolation_context(intensity_mask: np.ndarray, gray: np.ndarray, *, bright: bool) -> _IsolationContext:
    """Compute connected-component labels for the bright/dark anchor mask."""
    from contour.application.use_cases.processing._core import _EMPTY_ISOLATION


    if intensity_mask.size == 0:
        return _EMPTY_ISOLATION
    mask_area = float(cv2.countNonZero(intensity_mask))
    total_area = float(intensity_mask.size)
    if mask_area <= 0.0:
        return _EMPTY_ISOLATION
    ratio = mask_area / total_area
    if ratio > 0.55:
        # The user-supplied range covers most of the image; the component gate
        # would be meaningless. Fall back to an Otsu-like threshold derived
        # from the target polarity. If even that is too permissive we give up
        # on the isolation check for this scan.
        source = ensure_uint8(gray)
        try:
            threshold_value, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            return _EMPTY_ISOLATION
        del threshold_value
        mask = otsu if bright else cv2.bitwise_not(otsu)
        mask = cv2.bitwise_and(mask, intensity_mask)
        mask_area = float(cv2.countNonZero(mask))
        if mask_area <= 0.0 or mask_area / total_area > 0.55:
            return _EMPTY_ISOLATION
    else:
        mask = intensity_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return _EMPTY_ISOLATION
    return _IsolationContext(active=True, labels=labels, stats=stats)

def _component_extent(isolation: _IsolationContext, cx: int, cy: int) -> tuple[int, int, int] | None:
    """Return (area, width, height) of the component at (cx, cy), or ``None``."""

    if not isolation.active:
        return None
    labels = isolation.labels
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return None
    label = int(labels[cy, cx])
    if label == 0:
        return None
    stats = isolation.stats
    if label >= stats.shape[0]:
        return None
    return (
        int(stats[label, cv2.CC_STAT_AREA]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    )

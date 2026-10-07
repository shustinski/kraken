"""Mask components and their shape features, with no grid, rows or columns.

Every 8-connected piece of the binarized mask is one component. Its features describe
the piece alone (size, outline, holes, moments, distance-transform profile), so the
verdict on a cell never depends on where it stands or on its neighbours. Touching the
frame border is recorded as a flag; the analysis decides what it means.

``edge_softness`` reads the gray levels of the mask image itself: the share of mid-gray
pixels in a thin ring around the piece, per pixel of the piece. A network draws cells
with crisp edges (median 0.01 on real masks) and conductor grain with a wide gray fringe
(median 0.35). A true binary mask has no gray, and the value is 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .mask_normalization import MaskNormalizationConfig, NormalizedMask, normalize_mask


@dataclass(frozen=True, slots=True)
class CellComponent:
    label: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    filled_area: int
    perimeter: float
    solidity: float
    extent: float
    compactness: float
    hu: tuple[float, ...]
    inscribed_radius: float
    # Distance-transform quartiles over the component, in pixels.
    dt_profile: tuple[float, float, float]
    touches_border: bool
    border_sides: tuple[str, ...]
    contour: np.ndarray = field(repr=False, compare=False)
    edge_softness: float = 0.0

    @property
    def width(self) -> int:
        return int(self.bbox[2])

    @property
    def height(self) -> int:
        return int(self.bbox[3])

    @property
    def hole_area(self) -> int:
        return max(0, int(self.filled_area) - int(self.area))

    def crop(self, labels: np.ndarray, pad: int = 0) -> tuple[np.ndarray, int, int]:
        """Boolean pixel mask of this component around its box, and the box origin."""

        x, y, w, h = self.bbox
        rows, cols = labels.shape
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(cols, x + w + pad), min(rows, y + h + pad)
        return labels[y0:y1, x0:x1] == self.label, x0, y0


@dataclass(frozen=True, slots=True)
class ComponentSet:
    normalized: NormalizedMask
    labels: np.ndarray
    components: tuple[CellComponent, ...]
    # Mid-gray pixels of the mask image (uint8 0/1); None for a true binary mask.
    soft: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.normalized.shape

    def by_label(self) -> dict[int, CellComponent]:
        return {item.label: item for item in self.components}


def _hu(piece: np.ndarray) -> tuple[float, ...]:
    moments = cv2.moments(piece, binaryImage=True)
    values = cv2.HuMoments(moments).reshape(-1)
    return tuple(float(-math.copysign(1.0, v) * math.log10(abs(v))) if abs(v) > 1e-30 else 0.0 for v in values)


# Gray levels between these count as the soft fringe of a piece.
_SOFT_LOW, _SOFT_HIGH = 48, 208
_SOFT_RING = 2


def _edge_softness(soft: np.ndarray | None, labels: np.ndarray, label: int, x: int, y: int, w: int, h: int, area: int) -> float:
    if soft is None:
        return 0.0
    rows, cols = labels.shape
    x0, y0 = max(0, x - _SOFT_RING), max(0, y - _SOFT_RING)
    x1, y1 = min(cols, x + w + _SOFT_RING), min(rows, y + h + _SOFT_RING)
    piece = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
    ring = cv2.dilate(piece, np.ones((2 * _SOFT_RING + 1, 2 * _SOFT_RING + 1), np.uint8)) > 0
    return float(np.count_nonzero(soft[y0:y1, x0:x1][ring])) / float(max(1, area))


def _component(
    labels: np.ndarray,
    label: int,
    stats: np.ndarray,
    centroid: np.ndarray,
    shape: tuple[int, int],
    soft: np.ndarray | None = None,
) -> CellComponent:
    x, y, w, h, area = (int(value) for value in stats[:5])
    piece = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
    padded = np.pad(piece, 1)
    found = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    outer = max(contours, key=cv2.contourArea) if contours else np.zeros((0, 1, 2), dtype=np.int32)
    filled = np.zeros_like(padded)
    if len(outer):
        cv2.drawContours(filled, [outer], -1, 1, thickness=-1)
    filled_area = max(area, int(np.count_nonzero(filled)))
    perimeter = float(cv2.arcLength(outer, True)) if len(outer) >= 2 else float(2 * (w + h))
    contour_area = float(cv2.contourArea(outer)) if len(outer) >= 3 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(outer))) if len(outer) >= 3 else 0.0
    solidity = contour_area / hull_area if hull_area > 0.0 else 1.0
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1][piece > 0]
    quartiles = np.percentile(distance, (25, 50, 75)) if distance.size else np.zeros(3)
    rows, cols = shape
    sides = tuple(
        side
        for side, touching in (
            ("left", x <= 0),
            ("top", y <= 0),
            ("right", x + w >= cols),
            ("bottom", y + h >= rows),
        )
        if touching
    )
    contour = (outer.reshape(-1, 2) + np.array([x - 1, y - 1], dtype=np.int32)).astype(np.int32)
    return CellComponent(
        label=int(label),
        bbox=(x, y, w, h),
        centroid=(float(centroid[0]), float(centroid[1])),
        area=area,
        filled_area=filled_area,
        perimeter=perimeter,
        solidity=float(min(1.0, solidity)),
        extent=float(area) / float(max(1, w * h)),
        compactness=float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0)),
        hu=_hu(piece),
        inscribed_radius=float(distance.max()) if distance.size else 0.0,
        dt_profile=(float(quartiles[0]), float(quartiles[1]), float(quartiles[2])),
        touches_border=bool(sides),
        border_sides=sides,
        contour=contour.reshape(-1, 1, 2),
        edge_softness=_edge_softness(soft, labels, label, x, y, w, h, area),
    )


def extract_components(
    image: np.ndarray,
    *,
    normalization: MaskNormalizationConfig | None = None,
    normalized: NormalizedMask | None = None,
    min_area: int = 1,
) -> ComponentSet:
    """Components of one mask image (or of an already binarized mask)."""

    mask = normalized if normalized is not None else normalize_mask(image, normalization)
    binary = mask.binary.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    labels = labels.astype(np.int32, copy=False)
    soft = None
    if image is not None and mask.kind != "binary":
        from .mask_normalization import grayscale_levels

        gray = grayscale_levels(image)
        if gray.shape == binary.shape:
            soft = ((gray >= _SOFT_LOW) & (gray < _SOFT_HIGH)).astype(np.uint8)
    components = tuple(
        _component(labels, label, stats[label], centroids[label], binary.shape, soft)
        for label in range(1, int(count))
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area)
    )
    return ComponentSet(normalized=mask, labels=labels, components=components, soft=soft)

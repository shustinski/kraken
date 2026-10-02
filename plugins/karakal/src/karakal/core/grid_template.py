"""Cell shape template of one mask layer.

On one layer every cell has the same shape, whatever that shape is: a rectangle,
a triangle or anything else. The template is the mean silhouette of the typical
cells, aligned by their centroid. A cell is compared with it only outside a thin
band along the template contour, so the rounder or sharper edges the network
draws from cell to cell do not count; a bitten corner, a notch or a bulge does.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

# Seeds kept per run; more only costs time.
TEMPLATE_MAX_SEEDS = 600
# Seeds dropped as outliers before the final mean (share of the worst deviations).
TEMPLATE_OUTLIER_SHARE = 0.10
# Fewer seeds than this make no template.
TEMPLATE_MIN_SEEDS = 8
# Alignment search around the centroid, pixels.
_SHIFT = 1
_SHIFTS = tuple((dy, dx) for dy in range(-_SHIFT, _SHIFT + 1) for dx in range(-_SHIFT, _SHIFT + 1))


@dataclass(frozen=True, slots=True)
class CellShapeTemplate:
    canvas_width: int
    canvas_height: int
    mask: bytes
    band: float
    cell_width: float
    cell_height: float
    cell_area: float
    deviation_norm: float
    width_spread: float
    height_spread: float
    seed_count: int

    def array(self) -> np.ndarray:
        return np.frombuffer(self.mask, dtype=np.uint8).reshape(self.canvas_height, self.canvas_width).astype(bool)

    def cache_payload(self) -> dict[str, Any]:
        return {
            "mask_sha1": hashlib.sha1(self.mask).hexdigest(),
            "canvas": (int(self.canvas_width), int(self.canvas_height)),
            "band": round(float(self.band), 4),
            "cell": (round(float(self.cell_width), 3), round(float(self.cell_height), 3), round(float(self.cell_area), 2)),
            "deviation_norm": round(float(self.deviation_norm), 5),
            "spread": (round(float(self.width_spread), 5), round(float(self.height_spread), 5)),
            "seed_count": int(self.seed_count),
        }


def _contour_crop(points: Any) -> tuple[np.ndarray, int, int] | None:
    """Filled outer contour as a crop and its top-left corner."""

    if cv2 is None:
        return None
    array = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if array.shape[0] < 3:
        return None
    x0, y0 = int(array[:, 0].min()), int(array[:, 1].min())
    x1, y1 = int(array[:, 0].max()), int(array[:, 1].max())
    crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillPoly(crop, [(array - (x0, y0)).reshape(-1, 1, 2)], 1)
    return crop, x0, y0


def _to_size(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a piece to the cell size: shape is compared apart from size."""

    if crop.shape[1] == width and crop.shape[0] == height:
        return crop
    return cv2.resize(crop.astype(np.uint8), (max(1, int(width)), max(1, int(height))), interpolation=cv2.INTER_NEAREST)


def _place(crop: np.ndarray, canvas_height: int, canvas_width: int) -> np.ndarray | None:
    """Put a mask crop on the canvas with its centroid at the canvas center."""

    ys, xs = np.nonzero(crop)
    if ys.size == 0:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    top = int(round((canvas_height - 1) / 2.0 - cy))
    left = int(round((canvas_width - 1) / 2.0 - cx))
    canvas = np.zeros((canvas_height, canvas_width), dtype=bool)
    src_y0, src_x0 = max(0, -top), max(0, -left)
    dst_y0, dst_x0 = max(0, top), max(0, left)
    height = min(crop.shape[0] - src_y0, canvas_height - dst_y0)
    width = min(crop.shape[1] - src_x0, canvas_width - dst_x0)
    if height <= 0 or width <= 0:
        return None
    canvas[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = crop[src_y0 : src_y0 + height, src_x0 : src_x0 + width] > 0
    return canvas


def _band_mask(template: np.ndarray, band: float) -> np.ndarray:
    inside = cv2.distanceTransform(template.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~template).astype(np.uint8), cv2.DIST_L2, 3)
    return (template & (inside <= band)) | (~template & (outside <= band))


class CellTemplateMatcher:
    """Measures how far a mask piece is from the layer template."""

    def __init__(self, template: CellShapeTemplate) -> None:
        self.template = template
        self._mask = template.array()
        self._size = (int(round(float(template.cell_width))), int(round(float(template.cell_height))))
        self._outside_band = ~_band_mask(self._mask, float(template.band))
        self._area = max(1.0, float(self._mask.sum()))
        inside = cv2.distanceTransform(self._mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~self._mask).astype(np.uint8), cv2.DIST_L2, 3)
        # How far each canvas pixel lies from the template contour.
        self._depth = np.where(self._mask, inside, outside)
        # Padded copies: a shift is a slice, not a copy of the canvas.
        pad = _SHIFT
        self._pad_mask = np.pad(self._mask, pad)
        self._pad_outside_band = np.pad(self._outside_band, pad)
        self._pad_depth = np.pad(self._depth, pad)

    def measure_crop(self, crop: np.ndarray, depth_limit: float) -> tuple[float, int]:
        """(share of area differing beyond the band, pixels differing deeper than depth_limit).

        Rounder or sharper edges differ only near the contour; a bitten corner,
        a notch or a bulge reaches deep into or out of the cell.
        """

        crop = _to_size(crop, *self._size)
        placed = _place(crop, self.template.canvas_height, self.template.canvas_width)
        if placed is None:
            return 1.0, 10**6
        lost = max(0.0, float(np.count_nonzero(crop)) - float(np.count_nonzero(placed)))
        # One alignment for both measures: the shift with the smallest overall difference.
        # Taking each measure's own best shift let a shift hide a cut corner.
        height, width = placed.shape
        best = None
        for dy, dx in _SHIFTS:
            window = (slice(_SHIFT - dy, _SHIFT - dy + height), slice(_SHIFT - dx, _SHIFT - dx + width))
            diff = placed ^ self._pad_mask[window]
            total = int(np.count_nonzero(diff))
            if best is None or total < best[0]:
                best = (total, diff, window)
        _total, diff, window = best
        area = float(np.count_nonzero(diff & self._pad_outside_band[window]))
        deep = int(np.count_nonzero(diff & (self._pad_depth[window] > float(depth_limit))))
        return float((area + lost) / self._area), deep

    def measure_points(self, points: Any, depth_limit: float) -> tuple[float, int]:
        cropped = _contour_crop(points)
        if cropped is None:
            return 1.0, 10**6
        return self.measure_crop(cropped[0], depth_limit)

    def deviation_of_crop(self, crop: np.ndarray) -> float:
        """Share of the template area where the piece and the template differ beyond the band."""

        return self.measure_crop(crop, depth_limit=float("inf"))[0]

    def deviation_of_points(self, points: Any) -> float:
        cropped = _contour_crop(points)
        if cropped is None:
            return 1.0
        return self.deviation_of_crop(cropped[0])


def geometry_thresholds(
    template: CellShapeTemplate, geometry_sensitivity: int
) -> tuple[float, float, float, float]:
    """Shape deviation limit, width/height tolerances and depth limit for one slider position.

    The limits start from how much normal cells of the layer already differ from
    the template, so rounder or sharper edges of the network stay normal.
    """

    looseness = 1.0 - max(0.0, min(100.0, float(geometry_sensitivity))) / 100.0
    # Wide but shallow differences (slanted or wavy edges) are network variation; cuts and
    # notches are caught by depth, so the area limit stays loose.
    deviation_limit = float(template.deviation_norm) + 0.04 + 0.06 * looseness
    width_tolerance = max(0.12, 3.0 * float(template.width_spread)) + 0.10 * looseness
    height_tolerance = max(0.12, 3.0 * float(template.height_spread)) + 0.10 * looseness
    # Rounding of radius r reaches ~0.3*r into the cell; a cut corner or a notch goes deeper.
    depth_limit = max(2.0, (0.08 + 0.10 * looseness) * min(float(template.cell_width), float(template.cell_height)))
    return deviation_limit, width_tolerance, height_tolerance, depth_limit


def build_cell_shape_template(seeds: Sequence[tuple[Any, float, float, float]]) -> CellShapeTemplate | None:
    """Template from seed cells given as (outline points, bbox width, bbox height, area)."""

    if cv2 is None or len(seeds) < TEMPLATE_MIN_SEEDS:
        return None
    if len(seeds) > TEMPLATE_MAX_SEEDS:
        step = len(seeds) / float(TEMPLATE_MAX_SEEDS)
        seeds = [seeds[int(index * step)] for index in range(TEMPLATE_MAX_SEEDS)]
    widths = np.asarray([float(seed[1]) for seed in seeds], dtype=np.float64)
    heights = np.asarray([float(seed[2]) for seed in seeds], dtype=np.float64)
    cell_w, cell_h = float(np.median(widths)), float(np.median(heights))
    canvas_w = int(np.ceil(2.2 * cell_w)) + 4
    canvas_h = int(np.ceil(2.2 * cell_h)) + 4
    band = max(1.5, 0.08 * min(cell_w, cell_h))
    size = (int(round(cell_w)), int(round(cell_h)))
    crops = [_to_size(cropped[0], *size) for seed in seeds if (cropped := _contour_crop(seed[0])) is not None]
    placed = [canvas for crop in crops if (canvas := _place(crop, canvas_h, canvas_w)) is not None]
    if len(placed) < TEMPLATE_MIN_SEEDS:
        return None

    def mean_mask(items: Iterable[np.ndarray]) -> np.ndarray:
        stack = np.stack(list(items)).astype(np.float32)
        return stack.mean(axis=0) >= 0.5

    def deviations(mask: np.ndarray) -> np.ndarray:
        outside_band = np.pad(~_band_mask(mask, band), _SHIFT)
        padded = np.pad(mask, _SHIFT)
        area = max(1.0, float(mask.sum()))
        height, width = mask.shape
        values = []
        for item in placed:
            best = None
            for dy, dx in _SHIFTS:
                window = (slice(_SHIFT - dy, _SHIFT - dy + height), slice(_SHIFT - dx, _SHIFT - dx + width))
                diff = item ^ padded[window]
                total = int(np.count_nonzero(diff))
                if best is None or total < best[0]:
                    best = (total, float(np.count_nonzero(diff & outside_band[window])))
            values.append(best[1] / area)
        return np.asarray(values, dtype=np.float64)

    mask = mean_mask(placed)
    first = deviations(mask)
    # Defective cells in the sample must not shape the template.
    keep = first <= np.quantile(first, 1.0 - TEMPLATE_OUTLIER_SHARE)
    kept = [item for item, flag in zip(placed, keep) if flag]
    if len(kept) >= TEMPLATE_MIN_SEEDS:
        mask = mean_mask(kept)
    if not mask.any():
        return None
    final = deviations(mask)[keep] if len(kept) >= TEMPLATE_MIN_SEEDS else deviations(mask)

    def spread(values: np.ndarray) -> float:
        median = float(np.median(values))
        return float(1.4826 * np.median(np.abs(values - median)) / max(1.0, median))

    return CellShapeTemplate(
        canvas_width=int(canvas_w),
        canvas_height=int(canvas_h),
        mask=mask.astype(np.uint8).tobytes(),
        band=float(band),
        cell_width=cell_w,
        cell_height=cell_h,
        cell_area=float(np.median([float(seed[3]) for seed in seeds])),
        deviation_norm=float(np.quantile(final, 0.95)),
        width_spread=spread(widths),
        height_spread=spread(heights),
        seed_count=int(len(kept)),
    )

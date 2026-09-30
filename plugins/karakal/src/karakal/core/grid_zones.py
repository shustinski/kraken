"""Find conductor zones: topology outside the repeated cell lattice.

Cell-sized defects that sit on the lattice stay cell defects. Grain, buses and
pads that do not continue the lattice become one zone each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None


@dataclass(frozen=True, slots=True)
class ConductorZone:
    bbox: tuple[int, int, int, int]
    area_px: int
    fill: float = 1.0
    contour: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ConductorZoneMap:
    zones: tuple[ConductorZone, ...]
    tile: int
    occupied: np.ndarray
    # Pixel line at the top of the first lattice row. Negative means unused.
    boundary_y: float = -1.0
    boundary_x0: float = 0.0
    boundary_x1: float = -1.0

    def contains(self, x: float, y: float) -> bool:
        if self.boundary_y >= 0.0 and self.boundary_x0 <= float(x) <= self.boundary_x1:
            return float(y) < self.boundary_y
        tile = int(self.tile)
        occupied = self.occupied
        if tile <= 0 or occupied.size == 0:
            return False
        row = int(float(y) // tile)
        col = int(float(x) // tile)
        if row < 0 or col < 0 or row >= occupied.shape[0] or col >= occupied.shape[1]:
            return False
        return bool(occupied[row, col])


def _empty_map() -> ConductorZoneMap:
    return ConductorZoneMap(zones=(), tile=0, occupied=np.zeros((0, 0), dtype=bool))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close enclosed gaps so a damaged patch inside the lattice stays lattice."""

    if cv2 is None or mask.size == 0 or not np.any(mask):
        return mask
    work = np.pad(np.ascontiguousarray(mask.astype(np.uint8)), 1, constant_values=0)
    background = np.where(work > 0, np.uint8(0), np.uint8(255))
    cv2.floodFill(background, None, (0, 0), 128)
    holes = background[1:-1, 1:-1] == 255
    filled = mask.astype(bool).copy()
    filled[holes] = True
    return filled


def find_conductor_zones(
    *,
    width: int,
    height: int,
    candidates: list,
    columns: list[float] | None = None,
    rows: list[float] | None = None,
    pitch_x: float | None = None,
    pitch_y: float | None = None,
    modal_area: float = 0.0,
    modal_width: float = 0.0,
    modal_height: float = 0.0,
    confidence: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
) -> ConductorZoneMap:
    """Return connected off-lattice zones. An empty map means the frame is all lattice.

    ``confidence`` and ``foreground`` stay in the signature for callers. A soft
    confidence tile is not a zone by itself: on these frames the grain density
    is what separates conductors from a damaged lattice.
    """

    del confidence, foreground
    if cv2 is None or width < 8 or height < 8 or not candidates:
        return _empty_map()
    area_scale = float(modal_area) if modal_area > 1.0 else _fallback_area(candidates)
    pitch = _typical_pitch(pitch_x, pitch_y, modal_width, modal_height, area_scale)
    tile = int(np.clip(round(1.6 * pitch), 32, 96))
    cols = max(1, int(np.ceil(width / tile)))
    row_count = max(1, int(np.ceil(height / tile)))
    small_limit = min(250.0, max(36.0, 0.45 * area_scale))
    lattice = _lattice_tiles(
        candidates,
        columns=list(columns or []),
        rows=list(rows or []),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        modal_width=float(modal_width or pitch),
        modal_height=float(modal_height or pitch),
        area_scale=area_scale,
        tile=tile,
        shape=(row_count, cols),
    )
    small_counts = np.zeros((row_count, cols), dtype=np.int16)
    for item in candidates:
        area = float(getattr(item, "area", 0.0))
        if area >= small_limit:
            continue
        cx, cy = (float(value) for value in getattr(item, "centroid", (0.0, 0.0)))
        col = min(cols - 1, max(0, int(cx // tile)))
        row = min(row_count - 1, max(0, int(cy // tile)))
        small_counts[row, col] += 1
    threshold = max(3, int(round(4.0 * (tile * tile) / 10000.0)))
    if int(lattice.sum()) < 8:
        small_counts = np.zeros((row_count, cols), dtype=np.int16)
        for item in candidates:
            if float(getattr(item, "area", 0.0)) >= 200.0:
                continue
            cx, cy = (float(value) for value in getattr(item, "centroid", (0.0, 0.0)))
            col = min(cols - 1, max(0, int(cx // tile)))
            row = min(row_count - 1, max(0, int(cy // tile)))
            small_counts[row, col] += 1
        threshold = 4
    grain = small_counts >= threshold
    array = _array_mask(lattice, grain)
    if int(lattice.sum()) < 8:
        array = np.zeros_like(grain, dtype=bool)
        grain = _bbox_tiles(candidates, tile=tile, shape=(row_count, cols), min_area=40.0)
    # Grain outside the lattice is the conductor. A blob inside the lattice stays a cell defect.
    conductor = grain & ~array
    if not np.any(conductor):
        return _empty_map()
    # Two tiles of closing, then enclosed gaps, so cell-sized crumbs inside the grain
    # join the zone. Lattice tiles stay excluded on both sides of the fill.
    closed = cv2.morphologyEx(conductor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    closed = _fill_holes(closed.astype(bool) & ~array) & ~array
    if not np.any(closed):
        return _empty_map()
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    min_area = max(tile * tile, int(4.0 * pitch * pitch))
    occupied = np.zeros_like(closed, dtype=bool)
    zones: list[ConductorZone] = []
    for index in range(1, int(count)):
        tile_area = int(stats[index, cv2.CC_STAT_AREA]) * tile * tile
        if tile_area < min_area:
            continue
        component = labels == index
        # The visible zone is the component's span with the cell lattice punched out.
        # A rectangular span that only overlaps the corner array becomes the L, and
        # lattice cells stay searchable. Membership is this mask, never the raw rectangle.
        ys, xs = np.nonzero(component)
        if ys.size == 0:
            continue
        span = np.zeros_like(component)
        span[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1] = True
        # Enclosed conductor gaps join the zone. A lattice block that stays open
        # to the border, such as the crystal corner, remains a bite in the L.
        component = _fill_holes(span & ~array)
        if not np.any(component):
            continue
        occupied |= component
        left, top, box_w, box_h = _component_bbox(component, tile=tile, width=width, height=height)
        if box_w < tile or box_h < tile:
            continue
        tile_span = max(1, int(np.ceil(box_w / tile)) * int(np.ceil(box_h / tile)))
        fill = float(np.count_nonzero(component)) / float(tile_span)
        zones.append(
            ConductorZone(
                bbox=(left, top, box_w, box_h),
                area_px=int(np.count_nonzero(component) * tile * tile),
                fill=fill,
                contour=_zone_contour(component, tile=tile, width=width, height=height),
            )
        )
    if not zones:
        return _empty_map()
    boundary = _first_lattice_boundary(
        candidates,
        image_height=int(height),
        area_scale=float(area_scale),
        modal_width=float(modal_width or pitch),
        modal_height=float(modal_height or pitch),
    )
    if boundary is None:
        return ConductorZoneMap(zones=tuple(zones), tile=tile, occupied=occupied)
    boundary_y, boundary_x0, boundary_x1 = boundary
    occupied = _clip_zone_to_lattice_row(
        occupied,
        tile=tile,
        boundary_y=boundary_y,
        boundary_x0=boundary_x0,
        boundary_x1=boundary_x1,
    )
    rebuilt = _zones_from_occupied(occupied, tile=tile, width=width, height=height)
    if rebuilt:
        zones = tuple(rebuilt)
    return ConductorZoneMap(
        zones=zones,
        tile=tile,
        occupied=occupied,
        boundary_y=float(boundary_y),
        boundary_x0=float(boundary_x0),
        boundary_x1=float(boundary_x1),
    )


def _cluster_centers(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(float(value) for value in values)
    groups = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(group)) for group in groups if len(group) >= 3]


def _first_lattice_boundary(
    candidates: list,
    *,
    image_height: int,
    area_scale: float,
    modal_width: float,
    modal_height: float,
) -> tuple[float, float, float] | None:
    """Top of the first lattice row, and the column span that row belongs to.

    The row is continued from the dense array by the cell pitch, including a
    damaged edge row. Grain above that line is the zone. The row itself is not.
    """

    if area_scale <= 1.0 or image_height < 32 or modal_height <= 1.0:
        return None
    area_lo = 0.55 * float(area_scale)
    area_hi = 1.70 * float(area_scale)
    band = []
    every_y: list[float] = []
    for item in candidates:
        cx, cy = (float(value) for value in getattr(item, "centroid", (0.0, 0.0)))
        every_y.append(cy)
        area = float(getattr(item, "area", 0.0))
        if area_lo <= area <= area_hi:
            band.append((cx, cy))
    if len(band) < 12:
        return None
    low = [point for point in band if point[1] >= float(image_height) * 0.62]
    if len(low) < 12:
        return None
    points = np.asarray(low, dtype=np.float64)
    step_x, step_y = _cell_steps(points, modal_width=modal_width, modal_height=modal_height)
    if step_y <= 4.0:
        return None
    tolerance = max(6.0, 0.35 * float(step_y))
    row_centers = _cluster_centers([point[1] for point in low], tolerance)
    columns = _cluster_centers([point[0] for point in low], max(6.0, 0.35 * float(step_x or modal_width)))
    if len(row_centers) < 2 or len(columns) < 4:
        return None
    pitch = float(np.median(np.diff(sorted(row_centers)))) if len(row_centers) >= 2 else float(step_y)
    if pitch <= 4.0:
        pitch = float(step_y)
    column_reach = max(8.0, 0.55 * float(modal_width))
    needed = max(4, int(round(0.12 * len(columns))))
    accepted = float(min(row_centers))
    cursor = accepted
    crossed_gap = False
    matched: list[float] = []
    for _step in range(8):
        cursor -= pitch
        if cursor < float(modal_height):
            break
        aligned_y = [
            cy
            for cx, cy in band
            if abs(cy - cursor) <= 0.45 * pitch and min(abs(cx - column) for column in columns) <= column_reach
        ]
        if len(aligned_y) >= needed:
            accepted = cursor
            matched = aligned_y
            if crossed_gap:
                break
        else:
            crossed_gap = True
            if accepted - cursor > 6.5 * pitch:
                break
    if matched:
        boundary_y = float(min(matched)) - 0.5 * float(modal_height)
    else:
        # The dense array is separated from the grain by an empty gap. Stop at the
        # grain, not at the far lattice, so the undrawn rows in between stay array.
        above = [cy for cy in every_y if cy < accepted - 1.25 * pitch]
        boundary_y = float(max(above) + 0.5 * float(modal_height)) if above else float(accepted) - 0.5 * float(modal_height)
    return boundary_y, float(min(columns) - pitch), float(max(columns) + pitch)


def _clip_zone_to_lattice_row(
    occupied: np.ndarray,
    *,
    tile: int,
    boundary_y: float,
    boundary_x0: float,
    boundary_x1: float,
) -> np.ndarray:
    """Extend the zone down to the lattice row and keep that row out of it."""

    if occupied.size == 0 or tile <= 0:
        return occupied
    cleaned = occupied.copy()
    row_count, col_count = cleaned.shape
    col0 = min(col_count - 1, max(0, int(np.floor(boundary_x0 / tile))))
    col1 = min(col_count - 1, max(0, int(np.floor(boundary_x1 / tile))))
    if col1 < col0:
        return cleaned
    cut_row = int(np.floor(float(boundary_y) / tile))
    for row_index in range(row_count):
        if row_index < cut_row:
            cleaned[row_index, col0 : col1 + 1] = True
        else:
            cleaned[row_index, col0 : col1 + 1] = False
    return cleaned


def _zones_from_occupied(
    occupied: np.ndarray,
    *,
    tile: int,
    width: int,
    height: int,
) -> list[ConductorZone]:
    if cv2 is None or occupied.size == 0 or not np.any(occupied):
        return []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied.astype(np.uint8), connectivity=8)
    zones: list[ConductorZone] = []
    min_area = tile * tile
    for index in range(1, int(count)):
        if int(stats[index, cv2.CC_STAT_AREA]) * tile * tile < min_area:
            continue
        component = labels == index
        left, top, box_w, box_h = _component_bbox(component, tile=tile, width=width, height=height)
        if box_w < tile or box_h < tile:
            continue
        tile_span = max(1, int(np.ceil(box_w / tile)) * int(np.ceil(box_h / tile)))
        fill = float(np.count_nonzero(component)) / float(tile_span)
        zones.append(
            ConductorZone(
                bbox=(left, top, box_w, box_h),
                area_px=int(np.count_nonzero(component) * tile * tile),
                fill=fill,
                contour=_zone_contour(component, tile=tile, width=width, height=height),
            )
        )
    return zones


def _fallback_area(candidates: list) -> float:
    areas = [float(getattr(item, "area", 0.0)) for item in candidates if float(getattr(item, "area", 0.0)) > 1.0]
    if not areas:
        return 200.0
    return float(np.median(np.asarray(areas, dtype=np.float64)))


def _typical_pitch(
    pitch_x: float | None,
    pitch_y: float | None,
    modal_width: float,
    modal_height: float,
    area_scale: float,
) -> float:
    values = [float(value) for value in (pitch_x, pitch_y) if value is not None and float(value) > 4.0]
    if values:
        return float(np.median(values))
    if modal_width > 4.0 and modal_height > 4.0:
        return float(max(modal_width, modal_height) * 1.35)
    return float(max(16.0, np.sqrt(max(area_scale, 1.0)) * 1.6))


def _lattice_tiles(
    candidates: list,
    *,
    columns: list[float],
    rows: list[float],
    pitch_x: float | None,
    pitch_y: float | None,
    modal_width: float,
    modal_height: float,
    area_scale: float,
    tile: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Tiles whose cell-sized blobs repeat at the cell pitch, not at the grain pitch."""

    del columns, rows, pitch_x, pitch_y
    counts = np.zeros(shape, dtype=np.int16)
    area_lo = 0.45 * area_scale
    area_hi = 3.2 * area_scale
    band = [item for item in candidates if area_lo <= float(getattr(item, "area", 0.0)) <= area_hi]
    if len(band) < 8:
        return counts
    points = np.asarray([tuple(getattr(item, "centroid", (0.0, 0.0))) for item in band], dtype=np.float64)
    step_x, step_y = _cell_steps(points, modal_width=modal_width, modal_height=modal_height)
    if step_x <= 1.0 or step_y <= 1.0:
        return counts
    dx = np.abs(points[:, None, 0] - points[None, :, 0])
    dy = np.abs(points[:, None, 1] - points[None, :, 1])
    horizontal = (dx >= 0.55 * step_x) & (dx <= 1.45 * step_x) & (dy <= 0.45 * step_y)
    vertical = (dy >= 0.55 * step_y) & (dy <= 1.45 * step_y) & (dx <= 0.45 * step_x)
    np.fill_diagonal(horizontal, False)
    np.fill_diagonal(vertical, False)
    neighbors = (horizontal | vertical).sum(axis=1)
    for item, neighbor_count in zip(band, neighbors):
        if int(neighbor_count) < 3:
            continue
        cx, cy = (float(value) for value in getattr(item, "centroid", (0.0, 0.0)))
        col = min(shape[1] - 1, max(0, int(cx // tile)))
        row = min(shape[0] - 1, max(0, int(cy // tile)))
        counts[row, col] += 1
    return counts


def _cell_steps(points: np.ndarray, *, modal_width: float, modal_height: float) -> tuple[float, float]:
    """Horizontal and vertical pitch of repeated cells, ignoring tighter grain spacing."""

    dx = np.abs(points[:, None, 0] - points[None, :, 0])
    dy = np.abs(points[:, None, 1] - points[None, :, 1])
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    width_gate = max(6.0, 0.55 * float(modal_width or 0.0))
    height_gate = max(6.0, 0.55 * float(modal_height or 0.0))
    horizontal = np.where(dy < height_gate, dx, np.inf).min(axis=1)
    vertical = np.where(dx < width_gate, dy, np.inf).min(axis=1)
    width_floor = 0.8 * max(4.0, float(modal_width or 0.0))
    height_floor = 0.8 * max(4.0, float(modal_height or 0.0))
    horizontal = horizontal[(horizontal > width_floor) & np.isfinite(horizontal)]
    vertical = vertical[(vertical > height_floor) & np.isfinite(vertical)]
    step_x = float(np.median(horizontal)) if horizontal.size else _neighbor_step(points[:, 0])
    step_y = float(np.median(vertical)) if vertical.size else _neighbor_step(points[:, 1])
    return step_x, step_y


def _neighbor_step(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    gaps = np.diff(np.sort(values))
    gaps = gaps[(gaps > 4.0) & (gaps < 80.0)]
    if gaps.size == 0:
        return 0.0
    return float(np.median(gaps))


def _bbox_tiles(candidates: list, *, tile: int, shape: tuple[int, int], min_area: float) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    rows, cols = shape
    for item in candidates:
        if float(getattr(item, "area", 0.0)) < min_area:
            continue
        x, y, box_w, box_h = (int(value) for value in getattr(item, "bbox", (0, 0, 0, 0))[:4])
        col0 = min(cols - 1, max(0, x // tile))
        col1 = min(cols - 1, max(0, (x + max(1, box_w) - 1) // tile))
        row0 = min(rows - 1, max(0, y // tile))
        row1 = min(rows - 1, max(0, (y + max(1, box_h) - 1) // tile))
        covered[row0 : row1 + 1, col0 : col1 + 1] = True
    return covered


def _array_mask(lattice_counts: np.ndarray, grain: np.ndarray) -> np.ndarray:
    if cv2 is None or int(lattice_counts.sum()) < 8:
        return np.zeros_like(lattice_counts, dtype=bool)
    occupied = lattice_counts >= 1
    neighbor = cv2.filter2D(occupied.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
    dense = (lattice_counts >= 2) | ((lattice_counts >= 1) & (neighbor >= 3))
    if int(np.count_nonzero(dense)) < 2:
        return np.zeros_like(lattice_counts, dtype=bool)
    closed = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)).astype(bool)
    pure_grain = grain & (lattice_counts == 0)
    closed = closed & ~pure_grain
    filled = _fill_holes(closed)
    return filled & ~pure_grain


def _zone_contour(component: np.ndarray, *, tile: int, width: int, height: int) -> tuple[tuple[int, int], ...]:
    """Image-space outline of one zone. The shape follows the mask, not its bounding box."""

    if cv2 is None or component.size == 0 or not np.any(component):
        return ()
    padded = np.zeros((component.shape[0] + 2, component.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = component.astype(np.uint8)
    found, _hierarchy = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return ()
    contour = max(found, key=cv2.contourArea)
    points: list[tuple[int, int]] = []
    for point in contour.reshape(-1, 2):
        x = int(np.clip((int(point[0]) - 1) * tile, 0, max(0, width - 1)))
        y = int(np.clip((int(point[1]) - 1) * tile, 0, max(0, height - 1)))
        if points and points[-1] == (x, y):
            continue
        points.append((x, y))
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return tuple(points)


def _component_bbox(component: np.ndarray, *, tile: int, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(component)
    left = int(xs.min()) * tile
    top = int(ys.min()) * tile
    right = min(width, (int(xs.max()) + 1) * tile)
    bottom = min(height, (int(ys.max()) + 1) * tile)
    return left, top, max(1, right - left), max(1, bottom - top)

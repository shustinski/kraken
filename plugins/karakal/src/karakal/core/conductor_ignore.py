"""Conductor fields found without a grid: dense grain and large irregular masses.

A conductor shows in the mask as many small pieces close together, often around large
ragged masses. The field is found from local density in windows a few cells wide, so
it can lie on any side of the frame and needs no rows or columns.

Two stages (decision Н3):

- ``preliminary_zones`` is strict and cheap: everything doubtful is kept out of the
  normal-cell bank. It marks nothing as IGNORE.
- ``final_ignore`` runs after the bank is fixed: the zone loses every piece that looks
  like a cell (and a margin around it), so a normal cell next to a conductor is never
  swallowed. What stays is the IGNORE pixel mask and the pieces inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .cell_components import CellComponent, ComponentSet


@dataclass(frozen=True, slots=True)
class ConductorConfig:
    # A grain piece is smaller than this share of the typical cell area (and than max_grain_area).
    grain_area_ratio: float = 0.35
    max_grain_area: int = 400
    # A larger piece also counts as grain when its outline is ragged (cells have clean outlines).
    ragged_max_solidity: float = 0.85
    ragged_max_extent: float = 0.60
    # ... or when its size is far from the typical cell: an array repeats one size, grain does not.
    typical_area_range: tuple[float, float] = (0.6, 1.6)
    # ... or when the network drew it with a wide gray fringe (see CellComponent.edge_softness).
    soft_edge: float = 0.15
    # A mass is a ragged piece of at least this many cell areas.
    mass_area_ratio: float = 4.0
    mass_max_solidity: float = 0.60
    # Window side in typical cell sizes (sqrt of the cell area), clamped in pixels.
    window_cells: float = 3.5
    min_window: int = 40
    max_window: int = 160
    # Grain pieces per window (scaled to a 64 px window) that make a field.
    grain_per_window: float = 4.0
    # Conductor grain covers 12-30% of its field with mask; below this a field is sparse noise.
    min_foreground_density: float = 0.06
    # Zones smaller than this many windows are dropped.
    min_zone_windows: float = 1.0
    # A soft mask (JPEG or probability) shows conductor grain as a gray haze: windows with this
    # share of mid-gray pixels are a field (cell-only frames stay below 0.02 even in their worst
    # windows, grain fields reach 0.13-0.20). Such a field must cover this many windows, so one
    # blurred cell is not a conductor.
    soft_density: float = 0.03
    min_soft_zone_windows: float = 2.0
    # Final stage: a non-cell piece mostly inside the zone is ignored with it; one only partly
    # inside is an ambiguous border (checked as usual, kept out of the bank).
    ignore_share: float = 0.50
    # Margin kept free around protected cells, pixels.
    protect_margin: int = 2


@dataclass(frozen=True, slots=True)
class ConductorZones:
    """Zone pixel mask and the pieces it covers. Empty mask: no conductor in the frame."""

    zone: np.ndarray
    ignored_labels: tuple[int, ...] = ()
    protected_labels: tuple[int, ...] = ()
    near_labels: tuple[int, ...] = ()
    window: int = 0

    @property
    def found(self) -> bool:
        return bool(self.zone.any())


def typical_cell_area(components: ComponentSet) -> float:
    """Modal area of the cell-sized pieces of one frame (log-area histogram), 0 when there are none."""

    areas = np.asarray([item.area for item in components.components if item.area >= 40 and item.solidity >= 0.7])
    if areas.size < 4:
        return 0.0
    bins = np.floor(np.log(areas) / 0.25).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[int(np.argmax(counts))]
    return float(np.median(areas[bins == mode]))


def _window(cell_area: float, cfg: ConductorConfig) -> int:
    side = float(np.sqrt(max(cell_area, 64.0)))
    return int(np.clip(round(cfg.window_cells * side), cfg.min_window, cfg.max_window))


def _density_zone(components: ComponentSet, cell_area: float, cfg: ConductorConfig) -> tuple[np.ndarray, int]:
    rows, cols = components.shape
    window = _window(cell_area, cfg)
    area_scale = cell_area if cell_area > 0 else 200.0
    grain_limit = min(float(cfg.max_grain_area), cfg.grain_area_ratio * area_scale)
    seeds = np.zeros((rows, cols), dtype=np.float32)
    masses = np.zeros((rows, cols), dtype=bool)
    for item in components.components:
        mass = item.area >= cfg.mass_area_ratio * area_scale
        ragged = item.solidity < cfg.ragged_max_solidity or item.extent < cfg.ragged_max_extent
        low, high = cfg.typical_area_range
        atypical = cell_area > 0 and not low * cell_area <= item.area <= high * cell_area
        soft = item.edge_softness >= cfg.soft_edge
        if item.area < grain_limit or ((ragged or atypical or soft) and not mass):
            cx, cy = int(round(item.centroid[0])), int(round(item.centroid[1]))
            seeds[min(rows - 1, max(0, cy)), min(cols - 1, max(0, cx))] += 1.0
        elif mass and item.solidity < cfg.mass_max_solidity:
            piece, x0, y0 = item.crop(components.labels)
            masses[y0 : y0 + piece.shape[0], x0 : x0 + piece.shape[1]] |= piece
    kernel = (window, window)
    grain_count = cv2.boxFilter(seeds, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
    density = cv2.boxFilter(
        components.normalized.binary.astype(np.float32), -1, kernel, normalize=True, borderType=cv2.BORDER_CONSTANT
    )
    needed = cfg.grain_per_window * (window / 64.0) ** 2
    zone = (grain_count >= needed) & (density >= cfg.min_foreground_density)
    if masses.any():
        zone |= cv2.dilate(masses.astype(np.uint8), np.ones((window // 2 | 1, window // 2 | 1), np.uint8)) > 0
    if components.soft is not None:
        haze = cv2.boxFilter(components.soft.astype(np.float32), -1, kernel, normalize=True, borderType=cv2.BORDER_CONSTANT)
        soft_zone = (haze >= cfg.soft_density).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(soft_zone, connectivity=8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_soft_zone_windows * window * window
        zone |= keep[labels]
    if not zone.any():
        return zone, window
    closed = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((window // 2 | 1, window // 2 | 1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_zone_windows * window * window
    return keep[labels], window


def preliminary_zones(
    components: ComponentSet, *, cell_area: float | None = None, config: ConductorConfig | None = None
) -> ConductorZones:
    """Strict zone for building the bank: the field plus half a window around it."""

    cfg = config or ConductorConfig()
    area = float(cell_area) if cell_area else typical_cell_area(components)
    zone, window = _density_zone(components, area, cfg)
    if zone.any():
        zone = cv2.dilate(zone.astype(np.uint8), np.ones((window // 2 | 1, window // 2 | 1), np.uint8)) > 0
    return ConductorZones(zone=zone, window=window)


def excluded_from_bank(components: ComponentSet, zones: ConductorZones) -> set[int]:
    """Labels of pieces that touch the strict zone at all."""

    if not zones.found:
        return set()
    return {int(value) for value in np.unique(components.labels[zones.zone]) if value > 0}


def final_ignore(
    components: ComponentSet,
    *,
    cell_area: float,
    is_cell_like: Callable[[CellComponent], bool],
    config: ConductorConfig | None = None,
) -> ConductorZones:
    """IGNORE mask after the bank: cell-like pieces are cut out of the zone with a margin."""

    cfg = config or ConductorConfig()
    zone, window = _density_zone(components, float(cell_area), cfg)
    if not zone.any():
        return ConductorZones(zone=zone, window=window)
    labels = components.labels
    inside_counts = np.bincount(labels[zone], minlength=int(labels.max()) + 1)
    protected: list[int] = []
    ignored: list[int] = []
    near: list[int] = []
    protect = np.zeros(zone.shape, dtype=bool)
    for item in components.components:
        inside = int(inside_counts[item.label]) if item.label < inside_counts.size else 0
        if inside == 0:
            continue
        if is_cell_like(item):
            protected.append(item.label)
            piece, x0, y0 = item.crop(labels, pad=cfg.protect_margin)
            if cfg.protect_margin > 0:
                piece = cv2.dilate(piece.astype(np.uint8), np.ones((2 * cfg.protect_margin + 1,) * 2, np.uint8)) > 0
            protect[y0 : y0 + piece.shape[0], x0 : x0 + piece.shape[1]] |= piece
        elif inside >= cfg.ignore_share * item.area:
            ignored.append(item.label)
        else:
            near.append(item.label)
    if ignored:
        # The field is the ignored pieces and the gaps between them, not the whole density
        # window: debris a little way off the field is still checked.
        pieces = np.isin(labels, ignored)
        reach = max(3, window // 6) | 1
        zone = (zone & (cv2.dilate(pieces.astype(np.uint8), np.ones((reach, reach), np.uint8)) > 0)) | pieces
    else:
        zone = np.zeros_like(zone)
    zone &= ~protect
    return ConductorZones(
        zone=zone,
        ignored_labels=tuple(sorted(ignored)),
        protected_labels=tuple(sorted(protected)),
        near_labels=tuple(sorted(near)),
        window=window,
    )

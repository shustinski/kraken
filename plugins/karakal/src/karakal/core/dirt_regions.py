"""Dirty regions of a frame, read from the network's own output: its mask and, when there is one,
its confidence map. Qt-free; no cells and no lattice.

A clean frame is drawn by the network in crisp shapes. Dirt shows in the mask in four ways, and
each marks pixels:

- blur: mid-gray pixels farther than a couple of pixels from any crisp edge (the network drew a
  smear, not a shape);
- unsure: the confidence map is low away from the drawn edges (every edge has a thin ring of low
  confidence; that ring is not dirt);
- nicks: small holes and bites, the pixels a small closing fills;
- crumbs: small bits and burrs, the pixels a small opening removes.

Blur and unsure weigh 1, nicks and crumbs half that. Their share in a window of 2 x 2 cell pitches
is the dirtiness map. A region grows from a core (dirtiness at least ``core_level``) down to
``grow_level``; regions smaller than ``min_area_cells`` cells or with a mean below ``min_mean`` are
dropped. Regions closer than ``merge_gap_cells`` pitches are one cluster: its parts are joined
along the shortest links, a band one cell wide each, so the cluster has one outline.

Every region is given as a box and an exact outline, both in frame pixels from the top-left
corner, with its dirtiness from 0 to 1 and a level (light, medium, heavy).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None


DIRT_VERSION = "dirt.v1"

LEVEL_LIGHT = "light"
LEVEL_MEDIUM = "medium"
LEVEL_HEAVY = "heavy"
LEVELS = (LEVEL_LIGHT, LEVEL_MEDIUM, LEVEL_HEAVY)

# Raw dirtiness (share of dirty pixels, blur weighing 1) that reads as 100 %.
FULL_DIRTINESS = 0.12
# Normalized dirtiness from which a region is medium / heavy.
MEDIUM_FROM = 0.40
HEAVY_FROM = 0.65


@dataclass(frozen=True, slots=True)
class DirtConfig:
    # Cell pitch of the memory array, pixels; the window and all sizes are measured in it.
    pitch_x: float = 37.75
    pitch_y: float = 62.9
    window_cells: float = 2.0
    core_level: float = 0.050
    grow_level: float = 0.020
    min_area_cells: float = 2.0
    min_mean: float = 0.035
    merge_gap_cells: float = 3.0
    # Higher sensitivity lowers all three dirtiness thresholds in proportion.
    sensitivity: float = 1.0
    blur_margin: float = 2.5
    edge_ring: float = 3.0
    unsure_below: int = 200
    nick_size: int = 9
    crumb_size: int = 7
    defect_weight: float = 0.5

    def normalized(self) -> "DirtConfig":
        return replace(
            self,
            pitch_x=max(4.0, float(self.pitch_x)),
            pitch_y=max(4.0, float(self.pitch_y)),
            window_cells=max(0.5, float(self.window_cells)),
            min_area_cells=max(0.0, float(self.min_area_cells)),
            merge_gap_cells=max(0.0, float(self.merge_gap_cells)),
            sensitivity=min(4.0, max(0.25, float(self.sensitivity))),
            unsure_below=int(min(255, max(1, int(self.unsure_below)))),
            nick_size=max(3, int(self.nick_size)) | 1,
            crumb_size=max(3, int(self.crumb_size)) | 1,
        )

    @property
    def levels(self) -> tuple[float, float, float]:
        """Core, grow and mean thresholds after sensitivity."""

        s = float(self.sensitivity)
        return self.core_level / s, self.grow_level / s, self.min_mean / s

    def signature(self) -> tuple:
        cfg = self.normalized()
        return (
            DIRT_VERSION,
            round(cfg.pitch_x, 3),
            round(cfg.pitch_y, 3),
            round(cfg.window_cells, 3),
            round(cfg.core_level, 5),
            round(cfg.grow_level, 5),
            round(cfg.min_area_cells, 3),
            round(cfg.min_mean, 5),
            round(cfg.merge_gap_cells, 3),
            round(cfg.sensitivity, 3),
            round(cfg.blur_margin, 3),
            round(cfg.edge_ring, 3),
            cfg.unsure_below,
            cfg.nick_size,
            cfg.crumb_size,
            round(cfg.defect_weight, 3),
        )


@dataclass(frozen=True, slots=True)
class DirtRegion:
    id: int
    level: str
    dirtiness: float
    peak: float
    bbox: tuple[int, int, int, int]
    contour: tuple[tuple[int, int], ...]
    parts: int
    area_px: int
    dirty_area_px: int
    edge: bool
    blur_px: int = 0
    unsure_px: int = 0
    nick_px: int = 0
    crumb_px: int = 0
    part_contours: tuple[tuple[tuple[int, int], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class FrameDirtResult:
    frame_id: str
    mask_path: str
    confidence_path: str
    width: int
    height: int
    used_confidence: bool
    regions: tuple[DirtRegion, ...] = ()
    error: str = ""
    settings: dict = field(default_factory=dict)

    @property
    def max_dirtiness(self) -> float:
        return max((region.dirtiness for region in self.regions), default=0.0)

    @property
    def dirty_share(self) -> float:
        """Share of the frame inside regions."""

        total = max(1, int(self.width) * int(self.height))
        return min(1.0, sum(region.area_px for region in self.regions) / total)


@dataclass(frozen=True, slots=True)
class DirtLayers:
    """Per-pixel layers of one frame: what was marked dirty and the dirtiness map."""

    blur: np.ndarray
    unsure: np.ndarray
    nicks: np.ndarray
    crumbs: np.ndarray
    density: np.ndarray


def level_at_least(level: str, minimum: str) -> bool:
    """Whether ``level`` is ``minimum`` or dirtier (light < medium < heavy)."""

    if level not in LEVELS or minimum not in LEVELS:
        return False
    return LEVELS.index(level) >= LEVELS.index(minimum)


def worst_level(result: "FrameDirtResult | None") -> str:
    """Level of the frame's dirtiest region; empty for a clean or unknown frame."""

    if result is None or not result.regions:
        return ""
    return level_of(max(region.dirtiness for region in result.regions))


def level_of(dirtiness: float) -> str:
    if dirtiness >= HEAVY_FROM:
        return LEVEL_HEAVY
    if dirtiness >= MEDIUM_FROM:
        return LEVEL_MEDIUM
    return LEVEL_LIGHT


def read_gray(path: Path | str | None) -> np.ndarray | None:
    """Grayscale image from a path that may hold non-ASCII characters."""

    if not path or cv2 is None:
        return None
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if not data.size:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def confidence_certainty(image: np.ndarray) -> np.ndarray:
    """Certainty 0..255 (255 sure) from a confidence map, or from a class-probability map saved in
    its place (certainty |2p - 1|)."""

    from .grid_anomaly import classify_confidence_map_kind

    gray = np.asarray(image, dtype=np.uint8)
    if classify_confidence_map_kind(gray) == "class_probability":
        probability = gray.astype(np.float32) / 255.0
        return np.clip(np.abs(2.0 * probability - 1.0) * 255.0, 0, 255).astype(np.uint8)
    return gray


# Regions are shaped on a coarse grid: a cell pitch is COARSE_CELL x COARSE_CELL points there, so
# every shape operation is round and cheap. The dirtiness map is smooth over two pitches anyway.
COARSE_CELL = 8


def _disk(cells: float) -> np.ndarray:
    size = max(1, int(round(cells * COARSE_CELL))) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _round_kernel(radius: float) -> np.ndarray:
    """Every offset no farther than ``radius``: a pixel is farther than ``radius`` from a set exactly
    when the set dilated by this kernel does not reach it."""

    reach = int(np.floor(radius))
    y, x = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    return ((x * x + y * y) <= radius * radius).astype(np.uint8)


def _as_bool(flags: np.ndarray) -> np.ndarray:
    """A 0/255 OpenCV mask as a bool array (0/1 bytes viewed as bool, no extra pass of numpy logic)."""

    return np.ascontiguousarray(cv2.min(flags, 1)).view(bool)


def dirt_layers(mask: np.ndarray, confidence: np.ndarray | None = None, config: DirtConfig | None = None) -> DirtLayers:
    """Per-pixel layers of one frame.

    The same sets as plain numpy logic (``mid = (m > 55) & (m < 200)``, ``blur = mid & ~(near
    white & near black)``, ``nicks = close(hi) > hi``, ...), computed with OpenCV on 0/255 bytes:
    a frame holds millions of pixels and this runs for every frame of a run.
    """

    cfg = (config or DirtConfig()).normalized()
    m = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    # hi = m >= 128 (0/1), the binary the shape operations work on.
    hi = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)[1]
    mid = cv2.inRange(m, 56, 199)
    margin = _round_kernel(cfg.blur_margin)
    near_white = cv2.dilate(cv2.threshold(m, 199, 255, cv2.THRESH_BINARY)[1], margin)
    near_black = cv2.dilate(cv2.threshold(m, 55, 255, cv2.THRESH_BINARY_INV)[1], margin)
    blur = cv2.bitwise_and(mid, cv2.bitwise_not(cv2.bitwise_and(near_white, near_black)))
    nicks = cv2.compare(cv2.morphologyEx(hi, cv2.MORPH_CLOSE, np.ones((cfg.nick_size, cfg.nick_size), np.uint8)), hi, cv2.CMP_GT)
    crumbs = cv2.compare(hi, cv2.morphologyEx(hi, cv2.MORPH_OPEN, np.ones((cfg.crumb_size, cfg.crumb_size), np.uint8)), cv2.CMP_GT)
    if confidence is not None and np.asarray(confidence).shape == m.shape:
        certainty = np.ascontiguousarray(confidence_certainty(confidence))
        edge = cv2.morphologyEx(hi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        near_edge = cv2.dilate(edge, _round_kernel(cfg.edge_ring))
        below = cv2.threshold(certainty, float(cfg.unsure_below) - 1.0, 255, cv2.THRESH_BINARY_INV)[1]
        # near_edge > 0 is not «unsure»: below where the edge ring is zero.
        unsure = cv2.bitwise_and(below, cv2.compare(near_edge, 0, cv2.CMP_EQ))
    else:
        unsure = np.zeros(m.shape, dtype=np.uint8)
    strong = cv2.bitwise_or(blur, unsure)
    weak = cv2.bitwise_and(cv2.bitwise_or(nicks, crumbs), cv2.bitwise_not(strong))
    # dirt = 1 on strong pixels, defect_weight on weak ones (they never overlap), else 0; the same
    # float32 values the sum strong + defect_weight * weak gives.
    code = cv2.add(cv2.min(strong, 2), cv2.min(weak, 1))
    lut = np.zeros(256, dtype=np.float32)
    lut[1], lut[2] = float(cfg.defect_weight), 1.0
    dirt = cv2.LUT(code, lut)
    wx = int(round(cfg.window_cells * cfg.pitch_x)) | 1
    wy = int(round(cfg.window_cells * cfg.pitch_y)) | 1
    density = cv2.boxFilter(dirt, -1, (wx, wy), borderType=cv2.BORDER_REFLECT)
    return DirtLayers(blur=_as_bool(blur), unsure=_as_bool(unsure), nicks=_as_bool(nicks), crumbs=_as_bool(crumbs), density=density)


def _outline(mask: np.ndarray, offset: tuple[int, int], scale: tuple[float, float], size: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Outer outline of a coarse mask, in frame pixels."""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.5, True).reshape(-1, 2)
    ox, oy = offset
    sx, sy = scale
    width, height = size
    last_x, last_y = int(np.ceil(width / sx)) - 1, int(np.ceil(height / sy)) - 1

    def to_frame(value: int, step: float, last: int, limit: int) -> int:
        # Points on the outer coarse row or column lie on the frame edge itself.
        if value <= 0:
            return 0
        if value >= last:
            return limit - 1
        return int(min(limit - 1, max(0, round((value + 0.5) * step))))

    return tuple(
        (to_frame(int(x) + ox, sx, last_x, width), to_frame(int(y) + oy, sy, last_y, height)) for x, y in contour
    )


def _bridge(masks: list[np.ndarray]) -> np.ndarray:
    """One connected shape over a cluster (coarse grid): its parts joined along the shortest links
    (a spanning tree), each link a band one cell wide, smoothed by a one-cell closing."""

    shape = np.zeros(masks[0].shape, np.uint8)
    for part in masks:
        shape[part] = 1
    if len(masks) > 1:
        links = []
        for i, part in enumerate(masks):
            to_part = cv2.distanceTransform((~part).astype(np.uint8), cv2.DIST_L2, 5)
            for j in range(i + 1, len(masks)):
                ys, xs = np.nonzero(masks[j])
                k = int(np.argmin(to_part[ys, xs]))
                links.append((float(to_part[ys[k], xs[k]]), i, j, (int(xs[k]), int(ys[k]))))
        root = list(range(len(masks)))

        def find(a: int) -> int:
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a

        for _distance, i, j, end in sorted(links):
            a, b = find(i), find(j)
            if a == b:
                continue
            root[a] = b
            ys, xs = np.nonzero(masks[i])
            k = int(np.argmin((xs - end[0]) ** 2 + (ys - end[1]) ** 2))
            cv2.line(shape, (int(xs[k]), int(ys[k])), end, 1, COARSE_CELL)
    shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, _disk(1.0))
    return shape > 0


def _touches_edge(shape: np.ndarray, window: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
    x0, y0, x1, y1 = window
    cw, ch = size
    return bool(
        (x0 == 0 and shape[:, 0].any())
        or (y0 == 0 and shape[0, :].any())
        or (x1 == cw and shape[:, -1].any())
        or (y1 == ch and shape[-1, :].any())
    )


def find_dirt_regions(
    mask: np.ndarray,
    confidence: np.ndarray | None = None,
    config: DirtConfig | None = None,
    *,
    layers: DirtLayers | None = None,
) -> tuple[DirtRegion, ...]:
    cfg = (config or DirtConfig()).normalized()
    if cv2 is None:
        return ()
    m = np.asarray(mask, dtype=np.uint8)
    layers = layers or dirt_layers(m, confidence, cfg)
    height, width = m.shape
    sx, sy = cfg.pitch_x / COARSE_CELL, cfg.pitch_y / COARSE_CELL
    cw, ch = max(1, int(np.ceil(width / sx))), max(1, int(np.ceil(height / sy)))
    density = cv2.resize(layers.density, (cw, ch), interpolation=cv2.INTER_AREA)
    core_level, grow_level, min_mean = cfg.levels
    grown = cv2.morphologyEx((density >= grow_level).astype(np.uint8), cv2.MORPH_CLOSE, _disk(1.0))
    count, labels = cv2.connectedComponents(grown, connectivity=8)
    if count <= 1:
        return ()
    cores = np.unique(labels[density >= core_level])
    min_area = cfg.min_area_cells * COARSE_CELL * COARSE_CELL
    parts = np.zeros((ch, cw), np.int32)
    part_count = 0
    for label in cores[cores > 0]:
        region = labels == label
        if region.sum() < min_area or float(density[region].mean()) < min_mean:
            continue
        part_count += 1
        parts[region] = part_count
    if not part_count:
        return ()
    any_part = (parts > 0).astype(np.uint8)
    linked = cv2.dilate(any_part, _disk(cfg.merge_gap_cells)) if cfg.merge_gap_cells > 0 else any_part
    group_count, groups, group_stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    pad = COARSE_CELL
    found = []
    for group in range(1, group_count):
        gx, gy, gw, gh = (int(v) for v in group_stats[group, :4])
        x0, y0 = max(0, gx - pad), max(0, gy - pad)
        x1, y1 = min(cw, gx + gw + pad), min(ch, gy + gh + pad)
        crop_parts = parts[y0:y1, x0:x1]
        members = np.unique(crop_parts[(groups[y0:y1, x0:x1] == group) & (crop_parts > 0)])
        if not len(members):
            continue
        member_masks = [crop_parts == k for k in members]
        union = np.zeros(crop_parts.shape, dtype=bool)
        for part in member_masks:
            union |= part
        shape = _bridge(member_masks) | union
        contour = _outline(shape, (x0, y0), (sx, sy), (width, height))
        if len(contour) < 3:
            continue
        # Back to frame pixels for the numbers: the coarse cells of the crop cover this window.
        fx0, fy0 = int(round(x0 * sx)), int(round(y0 * sy))
        fx1, fy1 = min(width, int(round(x1 * sx))), min(height, int(round(y1 * sy)))
        size = (fx1 - fx0, fy1 - fy0)
        union_full = cv2.resize(union.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        shape_full = cv2.resize(shape.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        if not union_full.any():
            continue
        window = (slice(fy0, fy1), slice(fx0, fx1))
        values = layers.density[window][union_full]
        raw = float(values.mean())
        dirtiness = round(min(1.0, raw / FULL_DIRTINESS), 3)
        bx, by, bw, bh = cv2.boundingRect(np.asarray(contour, np.int32))
        found.append(
            (
                raw * float(union_full.sum()),
                dict(
                    level=level_of(dirtiness),
                    dirtiness=dirtiness,
                    peak=round(min(1.0, float(values.max()) / FULL_DIRTINESS), 3),
                    bbox=(int(bx), int(by), int(bw), int(bh)),
                    contour=contour,
                    parts=len(members),
                    area_px=int(shape_full.sum()),
                    dirty_area_px=int(union_full.sum()),
                    edge=_touches_edge(shape, (x0, y0, x1, y1), (cw, ch)),
                    blur_px=int(layers.blur[window][union_full].sum()),
                    unsure_px=int(layers.unsure[window][union_full].sum()),
                    nick_px=int(layers.nicks[window][union_full].sum()),
                    crumb_px=int(layers.crumbs[window][union_full].sum()),
                    part_contours=(
                        tuple(_outline(part, (x0, y0), (sx, sy), (width, height)) for part in member_masks)
                        if len(members) > 1
                        else ()
                    ),
                ),
            )
        )
    found.sort(key=lambda item: -item[0])
    return tuple(DirtRegion(id=index, **values) for index, (_weight, values) in enumerate(found, 1))


def analyze_dirt_frame(
    mask_path: Path | str,
    *,
    frame_id: str = "",
    confidence_path: Path | str | None = None,
    config: DirtConfig | None = None,
) -> FrameDirtResult:
    cfg = (config or DirtConfig()).normalized()
    mask = read_gray(mask_path)
    if mask is None:
        return FrameDirtResult(
            frame_id=str(frame_id),
            mask_path=str(mask_path or ""),
            confidence_path=str(confidence_path or ""),
            width=0,
            height=0,
            used_confidence=False,
            error="mask_unreadable",
        )
    confidence = read_gray(confidence_path) if confidence_path else None
    if confidence is not None and confidence.shape != mask.shape:
        confidence = cv2.resize(confidence, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    regions = find_dirt_regions(mask, confidence, cfg)
    return FrameDirtResult(
        frame_id=str(frame_id),
        mask_path=str(mask_path),
        confidence_path=str(confidence_path or "") if confidence is not None else "",
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        used_confidence=confidence is not None,
        regions=regions,
        settings=settings_payload(cfg),
    )


def settings_payload(config: DirtConfig) -> dict:
    cfg = config.normalized()
    return {
        "version": DIRT_VERSION,
        "pitch_px": [round(cfg.pitch_x, 3), round(cfg.pitch_y, 3)],
        "merge_gap_cells": round(cfg.merge_gap_cells, 3),
        "sensitivity": round(cfg.sensitivity, 3),
    }


def region_label_image(result: FrameDirtResult) -> np.ndarray:
    """Frame-sized image, each pixel the id of its region (0 outside). 8 bits while ids fit."""

    dtype = np.uint8 if len(result.regions) <= 255 else np.uint16
    label = np.zeros((max(1, result.height), max(1, result.width)), dtype)
    # Larger ids first so a smaller id wins where outlines touch.
    for region in sorted(result.regions, key=lambda item: -item.id):
        if len(region.contour) >= 3:
            cv2.fillPoly(label, [np.asarray(region.contour, np.int32)], int(region.id))
    return label


LEVEL_BGR = {LEVEL_LIGHT: (80, 200, 255), LEVEL_MEDIUM: (0, 140, 255), LEVEL_HEAVY: (40, 40, 230)}


def render_dirt_view(mask: np.ndarray, result: FrameDirtResult, layers: DirtLayers | None = None) -> np.ndarray:
    """A picture for people: the mask, dimmed, with the dirtiness map, the dirty pixels, every
    region's outline (its level's color), its box and a label."""

    m = np.asarray(mask, dtype=np.uint8)
    view = cv2.cvtColor((m.astype(np.float32) * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if layers is not None:
        heat = np.clip(layers.density / (FULL_DIRTINESS * 0.67), 0.0, 1.0)
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        alpha = (heat * 0.55)[..., None]
        view = (view * (1.0 - alpha) + colored * alpha).astype(np.uint8)
        view[layers.unsure] = (0, 255, 255)
        view[layers.blur] = (255, 200, 0)
        view[layers.nicks] = (0, 0, 255)
        view[layers.crumbs] = (255, 0, 255)
    scale = max(1.0, max(m.shape) / 1000.0)
    for region in result.regions:
        color = LEVEL_BGR.get(region.level, (0, 255, 0))
        x, y, w, h = region.bbox
        cv2.rectangle(view, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), max(1, int(round(scale))), cv2.LINE_AA)
        for part in region.part_contours:
            cv2.polylines(view, [np.asarray(part, np.int32)], True, (0, 140, 0), max(1, int(round(scale))), cv2.LINE_AA)
        cv2.polylines(view, [np.asarray(region.contour, np.int32)], True, color, max(2, int(round(2 * scale))), cv2.LINE_AA)
        text = "#%d %s %d%%" % (region.id, region.level, int(round(region.dirtiness * 100)))
        if region.parts > 1:
            text += " (%d)" % region.parts
        font_scale = 0.55 * scale
        origin = (x + 4, y - int(8 * scale) if y > 30 * scale else y + int(28 * scale))
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(3, int(5 * scale)), cv2.LINE_AA)
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return view

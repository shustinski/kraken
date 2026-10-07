"""Masks with known answers for tests and evaluation. Not used by the analysis itself.

A frame starts from normal cells, on a lattice with empty slots or scattered without one,
optionally cut by the frame edge. Defects are then drawn into the network mask while the
truth mask keeps what the operator would mark: bites, cut corners, short or wrongly sized
cells, debris, bridged and solid merges, split cells, conductor grain (IGNORE), and changes
that must stay normal (a moved cell, a removed cell). Each defect records which pixels of
the network mask it produced, so the expected class of every mask object is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from . import error_classes as ec

DEFECT_CLASSES: dict[str, int] = {
    "bite": ec.BAD_GEOMETRY,
    "cut_corner": ec.BAD_GEOMETRY,
    "short": ec.BAD_GEOMETRY,
    "wrong_size": ec.BAD_GEOMETRY,
    "debris": ec.DEBRIS,
    "bridge": ec.MERGE,
    "solid_merge": ec.MERGE,
    "split": ec.SPLIT,
    "conductor": ec.IGNORE,
    "moved": ec.OK,
    "removed": ec.MISSED,
}
GEOMETRY_KINDS = ("bite", "cut_corner", "short", "wrong_size")


@dataclass(frozen=True, slots=True)
class SyntheticLayout:
    width: int = 640
    height: int = 480
    cell_width: int = 18
    cell_height: int = 40
    pitch_x: int = 46
    pitch_y: int = 70
    # Share of lattice slots that hold a cell; the rest stay empty, which is not an error.
    fill: float = 0.8
    # "lattice" or "scatter" (cells at random places, no rows or columns).
    arrangement: str = "lattice"
    # Shift of the whole array, so cells can be cut by the frame edge.
    offset_x: int = 30
    offset_y: int = 30
    # Rounded corners like a network draws them.
    corner_radius: int = 2


@dataclass(frozen=True, slots=True)
class SyntheticDefect:
    kind: str
    error_class: int
    bbox: tuple[int, int, int, int]
    # Network-mask pixels this defect produced (for MISSED: the truth pixels of the lost cell).
    pixels: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=bool))


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    mask: np.ndarray
    truth: np.ndarray
    ignore: np.ndarray
    cells: tuple[tuple[int, int, int, int], ...]
    defects: tuple[SyntheticDefect, ...] = ()

    def expected_class_map(self) -> np.ndarray:
        """Class of every network-mask object (whole objects), misses, IGNORE = 255."""

        count, labels = cv2.connectedComponents((self.mask > 0).astype(np.uint8), connectivity=8)
        object_classes: dict[int, list[int]] = {}
        for defect in self.defects:
            if defect.error_class in (ec.MISSED, ec.IGNORE, ec.OK):
                continue
            for label in np.unique(labels[defect.pixels & (labels > 0)]):
                object_classes.setdefault(int(label), []).append(defect.error_class)
        lookup = np.zeros(count, dtype=np.uint8)
        for label, classes in object_classes.items():
            lookup[label] = ec.dominant_class(classes)
        class_map = lookup[labels]
        for defect in self.defects:
            if defect.error_class == ec.MISSED:
                class_map[defect.pixels & (self.mask == 0)] = ec.MISSED
        class_map[self.ignore] = ec.IGNORE
        return class_map

    def jpeg(self, quality: int = 80) -> "SyntheticFrame":
        """The same frame with the network mask stored as JPEG (gray edges, noise)."""

        ok, encoded = cv2.imencode(".jpg", self.mask, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:  # pragma: no cover - encoding a uint8 image does not fail
            raise RuntimeError("JPEG encoding failed")
        return replace(self, mask=cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE))


def _draw_cell(image: np.ndarray, box: tuple[int, int, int, int], radius: int) -> None:
    x, y, w, h = box
    height, width = image.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    cell = np.full((h, w), 255, dtype=np.uint8)
    r = max(0, min(int(radius), w // 2, h // 2))
    if r > 0:
        corner = np.zeros((r, r), dtype=np.uint8)
        cv2.circle(corner, (r - 1, r - 1), r, 255, -1)
        cell[:r, :r] = corner
        cell[:r, -r:] = corner[:, ::-1]
        cell[-r:, :r] = corner[::-1, :]
        cell[-r:, -r:] = corner[::-1, ::-1]
    image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], cell[y0 - y : y1 - y, x0 - x : x1 - x])


def normal_frame(layout: SyntheticLayout | None = None, *, seed: int = 0) -> SyntheticFrame:
    """Normal cells only: the truth equals the mask, empty slots are simply empty."""

    lay = layout or SyntheticLayout()
    rng = np.random.default_rng(seed)
    mask = np.zeros((lay.height, lay.width), dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    if lay.arrangement == "scatter":
        attempts = 0
        target = int(lay.fill * (lay.width // lay.pitch_x) * (lay.height // lay.pitch_y))
        while len(boxes) < target and attempts < target * 60:
            attempts += 1
            x = int(rng.integers(4, lay.width - lay.cell_width - 4))
            y = int(rng.integers(4, lay.height - lay.cell_height - 4))
            gap = 8
            if any(
                x < bx + bw + gap and bx < x + lay.cell_width + gap and y < by + bh + gap and by < y + lay.cell_height + gap
                for bx, by, bw, bh in boxes
            ):
                continue
            boxes.append((x, y, lay.cell_width, lay.cell_height))
    else:
        for y in range(lay.offset_y - lay.pitch_y, lay.height, lay.pitch_y):
            for x in range(lay.offset_x - lay.pitch_x, lay.width, lay.pitch_x):
                if x + lay.cell_width <= 0 or y + lay.cell_height <= 0:
                    continue
                if rng.random() > lay.fill:
                    continue
                boxes.append((x, y, lay.cell_width, lay.cell_height))
    for box in boxes:
        _draw_cell(mask, box, lay.corner_radius)
    return SyntheticFrame(mask=mask, truth=mask.copy(), ignore=np.zeros(mask.shape, dtype=bool), cells=tuple(boxes))


def _inside(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int = 3) -> bool:
    x, y, w, h = box
    return x >= margin and y >= margin and x + w <= shape[1] - margin and y + h <= shape[0] - margin


def _cell_pixels(frame: SyntheticFrame, box: tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    x, y, w, h = box
    region = np.zeros(frame.mask.shape, dtype=bool)
    region[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = True
    return region & (frame.mask > 0)


def _smooth_blob(rng: np.random.Generator, width: int, height: int, fill: float) -> np.ndarray:
    """Irregular blob: thresholded smooth noise (a cheap Perlin stand-in), kept as one piece."""

    noise = rng.random((height + 8, width + 8)).astype(np.float32)
    sigma = max(1.0, min(width, height) / 4.0)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)[4:-4, 4:-4]
    yy, xx = np.mgrid[0:height, 0:width]
    falloff = 1.0 - (((xx - (width - 1) / 2) / (width / 2)) ** 2 + ((yy - (height - 1) / 2) / (height / 2)) ** 2)
    field_values = smooth + 0.15 * falloff
    blob = field_values >= np.quantile(field_values, 1.0 - fill)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), connectivity=8)
    if count <= 1:
        blob[height // 2, width // 2] = True
        return blob
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _free_spot(
    frame: SyntheticFrame, rng: np.random.Generator, width: int, height: int, clearance: int
) -> tuple[int, int] | None:
    occupied = (frame.mask > 0) | frame.ignore | (frame.truth > 0)
    if cv2 is not None:
        occupied = cv2.dilate(occupied.astype(np.uint8), np.ones((2 * clearance + 1, 2 * clearance + 1), np.uint8)) > 0
    rows, cols = frame.mask.shape
    for _attempt in range(400):
        x = int(rng.integers(4, max(5, cols - width - 4)))
        y = int(rng.integers(4, max(5, rows - height - 4)))
        if not occupied[y : y + height, x : x + width].any():
            return x, y
    return None


def _usable(frame: SyntheticFrame, index: int, used: set[int]) -> bool:
    x, y, w, h = frame.cells[index]
    return (
        index not in used
        and _inside((x, y, w, h), frame.mask.shape, margin=8)
        and not frame.ignore[y : y + h, x : x + w].any()
    )


def _pick_cell(
    frame: SyntheticFrame, rng: np.random.Generator, used: set[int], *, with_neighbour: bool = False
) -> int | None:
    candidates = [
        index
        for index in range(len(frame.cells))
        if _usable(frame, index, used) and (not with_neighbour or _neighbour(frame, index, used) is not None)
    ]
    return None if not candidates else int(rng.choice(candidates))


def _neighbour(frame: SyntheticFrame, index: int, used: set[int]) -> int | None:
    x, y, w, h = frame.cells[index]
    best = None
    for other, (ox, oy, ow, oh) in enumerate(frame.cells):
        if other == index or not _usable(frame, other, used):
            continue
        same_row = abs(oy - y) <= 2 and ox > x
        gap = ox - (x + w)
        if same_row and 0 < gap <= 3 * w and (best is None or gap < best[0]):
            best = (gap, other)
    return None if best is None else best[1]


def add_defect(
    frame: SyntheticFrame,
    kind: str,
    *,
    seed: int = 0,
    used: set[int] | None = None,
) -> SyntheticFrame:
    """Frame with one more defect of this kind; unchanged when there is no room for it."""

    if kind not in DEFECT_CLASSES:
        raise ValueError(f"unknown defect kind: {kind}")
    rng = np.random.default_rng(seed)
    taken = used if used is not None else set()
    mask = frame.mask.copy()
    truth = frame.truth.copy()
    ignore = frame.ignore.copy()
    cells = list(frame.cells)
    error_class = DEFECT_CLASSES[kind]

    def finish(box: tuple[int, int, int, int], pixels: np.ndarray) -> SyntheticFrame:
        defect = SyntheticDefect(kind, error_class, tuple(int(v) for v in box), pixels)
        return SyntheticFrame(mask, truth, ignore, tuple(cells), frame.defects + (defect,))

    if kind == "debris":
        width = int(rng.integers(5, 13))
        height = int(rng.integers(5, 13))
        spot = _free_spot(frame, rng, width, height, clearance=5)
        if spot is None:
            return frame
        blob = _smooth_blob(rng, width, height, fill=float(rng.uniform(0.45, 0.7)))
        x, y = spot
        mask[y : y + height, x : x + width][blob] = 255
        pixels = np.zeros(mask.shape, dtype=bool)
        pixels[y : y + height, x : x + width] = blob
        return finish((x, y, width, height), pixels)

    if kind == "conductor":
        # Conductors lie where there are no cells: the cells under the field are not drawn at all.
        rows, cols = mask.shape
        width = int(rng.integers(cols // 6, cols // 3))
        height = int(rng.integers(rows // 6, rows // 3))
        spot = None
        for _attempt in range(200):
            x = int(rng.integers(0, cols - width))
            y = int(rng.integers(0, rows - height))
            if not any(
                bx < x + width + 6 and x - 6 < bx + bw and by < y + height + 6 and y - 6 < by + bh
                for index, (bx, by, bw, bh) in enumerate(cells)
                if index in taken
            ):
                spot = (x, y)
                break
        if spot is None:
            return frame
        x, y = spot
        for bx, by, bw, bh in cells:
            if bx < x + width + 6 and x - 6 < bx + bw and by < y + height + 6 and y - 6 < by + bh:
                mask[max(0, by) : by + bh, max(0, bx) : bx + bw] = 0
                truth[max(0, by) : by + bh, max(0, bx) : bx + bw] = 0
        grain = np.zeros((height, width), dtype=np.uint8)
        for _ in range(int(width * height * 0.03)):
            gx, gy = int(rng.integers(0, width)), int(rng.integers(0, height))
            cv2.circle(grain, (gx, gy), int(rng.integers(1, 4)), 255, -1)
        mask[y : y + height, x : x + width] = np.maximum(mask[y : y + height, x : x + width], grain)
        ignore[y : y + height, x : x + width] = True
        pixels = np.zeros(mask.shape, dtype=bool)
        pixels[y : y + height, x : x + width] = grain > 0
        return finish((x, y, width, height), pixels)

    index = _pick_cell(frame, rng, taken, with_neighbour=kind in ("bridge", "solid_merge"))
    if index is None:
        return frame
    x, y, w, h = cells[index]
    short_side = min(w, h)

    if kind in ("bridge", "solid_merge"):
        other = _neighbour(frame, index, taken)
        if other is None:
            return frame
        ox = cells[other][0]
        if kind == "bridge":
            thickness = max(3, int(round(short_side * float(rng.uniform(0.3, 0.6)))))
            top = y + int(rng.integers(h // 4, max(h // 4 + 1, 3 * h // 4 - thickness)))
            mask[top : top + thickness, x + w : ox] = 255
        else:
            mask[y : y + h, x + w : ox] = 255
        taken.update((index, other))
        pixels = _cell_pixels(SyntheticFrame(mask, truth, ignore, tuple(cells)), (x, y, ox + w - x, h))
        return finish((x, y, ox + w - x, h), pixels)

    taken.add(index)
    region = np.zeros(mask.shape, dtype=bool)
    region[y : y + h, x : x + w] = True

    if kind == "bite":
        depth = max(3, int(round(short_side * float(rng.uniform(0.3, 0.5)))))
        length = max(4, int(round(max(w, h) * float(rng.uniform(0.25, 0.4)))))
        start = y + int(rng.integers(2, max(3, h - length - 2)))
        if rng.random() < 0.5:
            mask[start : start + length, x : x + depth] = 0
        else:
            mask[start : start + length, x + w - depth : x + w] = 0
    elif kind == "cut_corner":
        leg = max(4, int(round(short_side * float(rng.uniform(0.5, 0.75)))))
        corner = int(rng.integers(0, 4))
        cx = x if corner in (0, 2) else x + w - 1
        cy = y if corner in (0, 1) else y + h - 1
        dx = leg if corner in (0, 2) else -leg
        dy = leg if corner in (0, 1) else -leg
        triangle = np.array([[cx, cy], [cx + dx, cy], [cx, cy + dy]], dtype=np.int32)
        cv2.fillPoly(mask, [triangle], 0)
    elif kind == "short":
        cut = int(round(h * float(rng.uniform(0.3, 0.45))))
        if rng.random() < 0.5:
            mask[y : y + cut, x : x + w] = 0
        else:
            mask[y + h - cut : y + h, x : x + w] = 0
    elif kind == "wrong_size":
        scale = float(rng.choice([0.7, 1.35]))
        nw, nh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
        mask[region] = 0
        nx, ny = x + (w - nw) // 2, y + (h - nh) // 2
        _draw_cell(mask, (nx, ny, nw, nh), 2)
        region[max(0, ny) : ny + nh, max(0, nx) : nx + nw] = True
    elif kind == "split":
        gap = int(rng.integers(2, 4))
        at = y + int(round(h * float(rng.uniform(0.35, 0.6))))
        mask[at : at + gap, x : x + w] = 0
    elif kind == "moved":
        spot = _free_spot(frame, rng, w, h, clearance=8)
        if spot is None:
            return frame
        mask[region] = 0
        truth[region] = 0
        nx, ny = spot
        _draw_cell(mask, (nx, ny, w, h), 2)
        _draw_cell(truth, (nx, ny, w, h), 2)
        cells[index] = (nx, ny, w, h)
        region = np.zeros(mask.shape, dtype=bool)
        region[ny : ny + h, nx : nx + w] = True
        return finish((nx, ny, w, h), region & (mask > 0))
    elif kind == "removed":
        pixels = region & (truth > 0)
        mask[region] = 0
        return finish((x, y, w, h), pixels)
    return finish((x, y, w, h), region & (mask > 0))


def synthetic_frame(
    defects: dict[str, int],
    *,
    layout: SyntheticLayout | None = None,
    seed: int = 0,
) -> SyntheticFrame:
    """Normal frame plus the given number of defects of each kind (each on its own cell)."""

    frame = normal_frame(layout, seed=seed)
    used: set[int] = set()
    step = 0
    # Conductor fields first: they clear the cells under them before any cell gets a defect.
    for kind in sorted(defects, key=lambda name: (name != "conductor", name)):
        for _ in range(int(defects[kind])):
            step += 1
            frame = add_defect(frame, kind, seed=seed * 1000 + step, used=used)
    return frame

"""Chip photos with known answers for the heuristic check's tests and evaluation. Not used by the analysis itself.

A layer is first drawn as a metal mask: horizontal and vertical tracks with designed cuts (every
gap at least ``gap`` px), a pour, optionally 45-degree tracks, tracks of two widths, or, for a via
layer, arrays of square vias. Defects are drawn into the mask (opens, necks, bridges, over- and
under-etch) or into the photo (particles, a residue film, a stain, a blur), each with its box and
the reason codes of the heuristic check that count as finding it. Then the mask becomes a photo:
metal brighter or darker than the substrate, uneven light, metal grain, rough edges, blur, noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

# What the heuristic check should say about each planted defect: reason codes that count as
# finding it, and the verdict of a frame that has only it (None: any verdict but "unreadable").
DEFECT_EXPECTATIONS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "clean": ((), "normal"),
    "open": (("open", "thin_metal"), "topology"),
    "neck": (("thin_metal", "open"), "topology"),
    "bridge": (("thin_metal", "narrow_gap"), "topology"),
    "particles": (("particle", "debris", "pinhole"), "dirt"),
    "residue": (("residue",), "dirt"),
    "stain": (("stain", "particle"), None),
    "over_etch": (("etch_over", "width_spread"), "etch_deviation"),
    "under_etch": (("etch_under", "width_spread"), "etch_deviation"),
    "blur": (("blur",), "unreadable"),
    # Gross defects: a hair across the frame, a crater field, a hole torn in the layer.
    "hair": (("hair", "damage"), "gross"),
    "craters": (("crater", "damage"), "gross"),
    "hole": (("crater", "damage"), "gross"),
}
DEFECT_KINDS = tuple(DEFECT_EXPECTATIONS)
# Defects that change the whole frame: they have no box and need the rest of the run to show.
FRAME_WIDE_KINDS = frozenset({"over_etch", "under_etch", "blur"})


@dataclass(frozen=True, slots=True)
class LayerStyle:
    size: int = 512
    width: int = 8
    gap: int = 16
    diagonals: bool = False
    mixed_widths: bool = False
    vias: bool = False


@dataclass(frozen=True, slots=True)
class PhotoStyle:
    metal_level: float = 180.0
    substrate_level: float = 60.0
    blur: float = 1.2
    noise: float = 4.0
    # Light falling off across the frame, as a share of the metal-substrate contrast.
    light_gradient: float = 0.0
    # Darkening of the corners, as a share of the brightness.
    vignette: float = 0.0
    # Metal texture: gray levels of smooth noise on the metal.
    grain: float = 0.0
    # Edge waviness in pixels.
    edge_roughness: float = 0.0


@dataclass(frozen=True, slots=True)
class PlantedDefect:
    kind: str
    # x, y, width, height in frame pixels; None for a defect of the whole frame.
    box: tuple[int, int, int, int] | None
    expected_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyntheticChipFrame:
    image: np.ndarray
    metal: np.ndarray
    kind: str
    defects: tuple[PlantedDefect, ...]
    expected_verdict: str | None
    layer: LayerStyle = field(default_factory=LayerStyle)
    photo: PhotoStyle = field(default_factory=PhotoStyle)


# --- Layers --------------------------------------------------------------------------------------


def _segments(rng: np.random.Generator, start: int, stop: int, cut: int) -> list[tuple[int, int]]:
    """Pieces of one track with designed cuts at least ``cut`` long."""

    runs: list[tuple[int, int]] = []
    position = start
    while True:
        end = min(stop, position + int(rng.integers(60, 160)))
        if end - position >= 30:
            runs.append((position, end))
        position = end + int(rng.integers(cut, 2 * cut))
        if position >= stop - 30:
            return runs


def _track_widths(rng: np.random.Generator, style: LayerStyle, count: int) -> list[int]:
    if not style.mixed_widths:
        return [style.width] * count
    return [int(rng.choice((style.width, 2 * style.width, 3 * style.width), p=(0.6, 0.25, 0.15))) for _ in range(count)]


def _horizontal_tracks(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle, y0: int, y1: int, x0: int, x1: int) -> None:
    y = y0
    for width in _track_widths(rng, style, 64):
        if y + width > y1:
            return
        for a, b in _segments(rng, x0, x1, style.width + style.gap):
            metal[y : y + width, a:b] = 1
        y += width + style.gap


def _vertical_tracks(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle, x0: int, x1: int, y0: int, y1: int) -> None:
    x = x0
    for width in _track_widths(rng, style, 64):
        if x + width > x1:
            return
        for a, b in _segments(rng, y0, y1, style.width + style.gap):
            metal[a:b, x : x + width] = 1
        x += width + style.gap


def _diagonal_tracks(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle, x0: int, x1: int, y0: int, y1: int) -> None:
    """45-degree tracks with flat ends inside the box, ``gap`` apart across the tracks."""

    region = np.zeros_like(metal)
    step = (style.width + style.gap) * np.sqrt(2.0)
    half = style.width / np.sqrt(2.0) / 2.0
    offset = -float(y1 - y0)
    while offset < float(x1 - x0):
        # Track along x - y = offset (in box coordinates), pieces along its length.
        for a, b in _segments(rng, 0, int((x1 - x0) * 1.42), int(style.width + style.gap)):
            ta, tb = a / np.sqrt(2.0), b / np.sqrt(2.0)
            centre_a = (x0 + offset + ta + half, y0 + ta - half)
            centre_b = (x0 + offset + tb + half, y0 + tb - half)
            normal = (half, -half)
            polygon = np.array(
                [
                    (centre_a[0] - normal[0], centre_a[1] - normal[1]),
                    (centre_b[0] - normal[0], centre_b[1] - normal[1]),
                    (centre_b[0] + normal[0], centre_b[1] + normal[1]),
                    (centre_a[0] + normal[0], centre_a[1] + normal[1]),
                ],
                dtype=np.float32,
            )
            cv2.fillConvexPoly(region, np.round(polygon).astype(np.int32), 1)
        offset += step
    # Keep only whole pieces inside the box: a track cut by the box edge would end in a slant.
    inside = np.zeros_like(metal)
    inside[y0:y1, x0:x1] = 1
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    for label in range(1, count):
        left, top = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        right, bottom = left + stats[label, cv2.CC_STAT_WIDTH], top + stats[label, cv2.CC_STAT_HEIGHT]
        if left >= x0 and top >= y0 and right <= x1 and bottom <= y1 and stats[label, cv2.CC_STAT_AREA] >= 6 * style.width**2:
            metal[labels == label] = 1


def _via_arrays(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle) -> None:
    """Blocks of square vias on the routing grid, some grid places left empty by design."""

    pitch = style.width + style.gap
    size = style.size
    for block_y in range(16, size - 64, 128):
        for block_x in range(16, size - 64, 128):
            rows, columns = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            for row in range(rows):
                for column in range(columns):
                    if rng.random() < 0.15:
                        continue
                    y, x = block_y + row * pitch, block_x + column * pitch
                    metal[y : y + style.width, x : x + style.width] = 1


def draw_layer(style: LayerStyle | None = None, seed: int = 0) -> np.ndarray:
    """A clean metal mask (0/1) of one layer."""

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for synthetic chip frames")
    style = style or LayerStyle()
    rng = np.random.default_rng(seed)
    size = style.size
    metal = np.zeros((size, size), dtype=np.uint8)
    if style.vias:
        _via_arrays(metal, rng, style)
        return metal
    scale = size / 512.0
    _horizontal_tracks(metal, rng, style, int(16 * scale), int(360 * scale), int(8 * scale), int(248 * scale))
    _vertical_tracks(metal, rng, style, int(272 * scale), size - int(8 * scale), int(8 * scale), int(368 * scale))
    metal[int(400 * scale) : int(480 * scale), int(16 * scale) : int(240 * scale)] = 1
    if style.diagonals:
        _diagonal_tracks(metal, rng, style, int(272 * scale), size - int(8 * scale), int(392 * scale), size - int(8 * scale))
    return metal


# --- Defects in the mask -------------------------------------------------------------------------


def _horizontal_runs(metal: np.ndarray, width: int) -> list[tuple[int, int, int]]:
    """(row, start, end) of horizontal pieces exactly ``width`` tall, long enough to break."""

    runs: list[tuple[int, int, int]] = []
    height = metal.shape[0]
    for y in range(1, height - width - 1):
        row = metal[y + width // 2]
        if not row.any():
            continue
        columns = np.flatnonzero(row)
        breaks = np.flatnonzero(np.diff(columns) > 1)
        starts = np.concatenate(([columns[0]], columns[breaks + 1]))
        ends = np.concatenate((columns[breaks], [columns[-1]]))
        for start, end in zip(starts.tolist(), ends.tolist()):
            if end - start < 60:
                continue
            middle = (start + end) // 2
            piece = metal[y - 1 : y + width + 1, middle - 6 : middle + 6]
            # Exactly this track's band: free above and below, full inside.
            if piece[0].any() or piece[-1].any() or not piece[1:-1].all():
                continue
            runs.append((y, start, end))
    return runs


def _box_around(x: int, y: int, w: int, h: int, margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(shape[1], x + w + margin), min(shape[0], y + h + margin)
    return (x0, y0, x1 - x0, y1 - y0)


def _plant_open(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle) -> tuple[int, int, int, int]:
    runs = _horizontal_runs(metal, style.width)
    y, start, end = runs[int(rng.integers(len(runs)))]
    gap = int(rng.integers(3, 7))
    x = int(rng.integers(start + 25, end - 25 - gap))
    metal[y : y + style.width, x : x + gap] = 0
    return _box_around(x, y, gap, style.width, 4, metal.shape)


def _plant_neck(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle) -> tuple[int, int, int, int]:
    runs = _horizontal_runs(metal, style.width)
    y, start, end = runs[int(rng.integers(len(runs)))]
    length = int(rng.integers(6, 14))
    x = int(rng.integers(start + 25, end - 25 - length))
    # A neck worth the name: under a third of the line is left.
    bite = (style.width - max(2, int(round(style.width * 0.3)))) // 2
    metal[y : y + bite, x : x + length] = 0
    metal[y + style.width - bite : y + style.width, x : x + length] = 0
    return _box_around(x, y, length, style.width, 4, metal.shape)


def _plant_bridge(metal: np.ndarray, rng: np.random.Generator, style: LayerStyle) -> tuple[int, int, int, int]:
    """A thin strand across the gap between two tracks, one above the other."""

    runs = _horizontal_runs(metal, style.width)
    rng.shuffle(runs)
    for y, start, end in runs:
        below = y + style.width + style.gap
        for x in range(start + 25, end - 25, 7):
            if below + 2 < metal.shape[0] and metal[below + 1, x - 4 : x + 6].all():
                strand = int(rng.integers(3, 5))
                metal[y + style.width : below, x : x + strand] = 1
                return _box_around(x, y + style.width, strand, style.gap, 4, metal.shape)
    raise ValueError("no two tracks to bridge")


def _free_spot(metal: np.ndarray, rng: np.random.Generator, *, on_metal: bool, clearance: float) -> tuple[int, int]:
    from .mask_primitives import _distance_transform

    region = metal.astype(bool) if on_metal else ~metal.astype(bool)
    distance = _distance_transform(region)
    margin = int(clearance) + 2
    distance[:margin, :] = 0
    distance[-margin:, :] = 0
    distance[:, :margin] = 0
    distance[:, -margin:] = 0
    ys, xs = np.nonzero(distance > clearance)
    if ys.size == 0:
        raise ValueError("no free spot")
    index = int(rng.integers(len(ys)))
    return int(xs[index]), int(ys[index])


# --- Photos --------------------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    noise = cv2.GaussianBlur(rng.normal(0.0, 1.0, shape).astype(np.float32), (0, 0), sigma)
    return noise / max(float(np.std(noise)), 1.0e-6)


def render_photo(metal: np.ndarray, style: PhotoStyle | None = None, seed: int = 0) -> np.ndarray:
    """The photo of a metal mask (uint8)."""

    style = style or PhotoStyle()
    rng = np.random.default_rng(seed + 1000)
    shape = metal.shape
    coverage = metal.astype(np.float32)
    if style.edge_roughness > 0.0:
        soft = cv2.GaussianBlur(coverage, (0, 0), 1.5)
        wave = _smooth_noise(rng, shape, 2.0) * (style.edge_roughness * 0.12)
        coverage = (soft + wave > 0.5).astype(np.float32)
    contrast = style.metal_level - style.substrate_level
    image = style.substrate_level + contrast * coverage
    if style.grain > 0.0:
        image += coverage * _smooth_noise(rng, shape, 1.2) * style.grain
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    if style.light_gradient:
        image += abs(contrast) * style.light_gradient * (xx / shape[1] - 0.5)
    if style.vignette:
        radius = ((xx - shape[1] / 2) ** 2 + (yy - shape[0] / 2) ** 2) / ((shape[0] / 2) ** 2 + (shape[1] / 2) ** 2)
        image *= 1.0 - style.vignette * radius
    image = cv2.GaussianBlur(image, (0, 0), style.blur)
    image += rng.normal(0.0, style.noise, shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _add_residue(image: np.ndarray, rng: np.random.Generator, style: PhotoStyle) -> tuple[int, int, int, int]:
    height, width = image.shape
    w, h = int(rng.integers(60, 110)), int(rng.integers(40, 70))
    x, y = int(rng.integers(16, width - w - 16)), int(rng.integers(16, int(height * 0.7) - h))
    lift = 0.45 * (style.metal_level - style.substrate_level)
    patch = image[y : y + h, x : x + w].astype(np.float32) + lift
    image[y : y + h, x : x + w] = np.clip(patch, 0, 255).astype(np.uint8)
    return (x, y, w, h)


def _add_stain(image: np.ndarray, metal: np.ndarray, rng: np.random.Generator, style: PhotoStyle) -> tuple[int, int, int, int]:
    try:
        x, y = _free_spot(metal, rng, on_metal=True, clearance=12.0)
    except ValueError:  # no wide metal (a via layer): the stain lies on the substrate
        x, y = _free_spot(metal, rng, on_metal=False, clearance=12.0)
    radius = 22.0
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(np.float32)
    bump = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * radius**2))
    shift = -0.3 * (style.metal_level - style.substrate_level)
    image[:] = np.clip(image.astype(np.float32) + shift * bump, 0, 255).astype(np.uint8)
    return _box_around(x, y, 0, 0, int(2 * radius), image.shape)


def _add_particles(image: np.ndarray, metal: np.ndarray, rng: np.random.Generator, style: PhotoStyle, count: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for index in range(count):
        on_metal = index % 2 == 0
        radius = int(rng.integers(3, 6))
        try:
            x, y = _free_spot(metal, rng, on_metal=on_metal, clearance=radius + 3.0)
        except ValueError:  # a via layer has no metal wide enough: the particle lands beside it
            on_metal = not on_metal
            x, y = _free_spot(metal, rng, on_metal=on_metal, clearance=radius + 3.0)
        level = style.substrate_level if on_metal else style.metal_level
        cv2.circle(image, (x, y), radius, int(np.clip(level, 0, 255)), -1)
        boxes.append(_box_around(x - radius, y - radius, 2 * radius, 2 * radius, 3, image.shape))
    return boxes


def _add_hair(image: np.ndarray, rng: np.random.Generator, style: PhotoStyle) -> tuple[int, int, int, int]:
    """A bright thin curve across much of the frame, over the pattern."""

    height, width = image.shape
    start = np.array([rng.uniform(0.1, 0.4) * width, rng.uniform(0.05, 0.2) * height])
    end = np.array([rng.uniform(0.6, 0.9) * width, rng.uniform(0.8, 0.95) * height])
    bend = (start + end) / 2.0 + np.array([rng.uniform(0.15, 0.25) * width, -rng.uniform(0.05, 0.15) * height])
    t = np.linspace(0.0, 1.0, 200)[:, None]
    points = (1 - t) ** 2 * start + 2 * (1 - t) * t * bend + t**2 * end
    level = int(np.clip(style.metal_level + 0.3 * (style.metal_level - style.substrate_level), 0, 255))
    cv2.polylines(image, [np.round(points).astype(np.int32)], False, level, thickness=int(rng.integers(3, 5)), lineType=cv2.LINE_AA)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    return _box_around(int(x0), int(y0), int(x1 - x0), int(y1 - y0), 4, image.shape)


def _crater(image: np.ndarray, x: float, y: float, radius: float, style: PhotoStyle) -> None:
    """A pit: black inside, a soft bright halo around (charging at its rim)."""

    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(np.float32)
    distance = np.hypot(xx - x, yy - y)
    contrast = style.metal_level - style.substrate_level
    halo = 0.6 * contrast * np.exp(-((distance - radius) ** 2) / (2.0 * (0.35 * radius) ** 2)) * (distance > radius)
    plane = image.astype(np.float32) + halo
    plane[distance <= radius] = min(style.metal_level, style.substrate_level) * 0.15
    image[:] = np.clip(plane, 0, 255).astype(np.uint8)


def _add_craters(image: np.ndarray, rng: np.random.Generator, style: PhotoStyle) -> tuple[int, int, int, int]:
    """A field of four to seven pits next to each other."""

    height, width = image.shape
    cx, cy = rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height
    xs, ys, rs = [], [], []
    for _index in range(int(rng.integers(4, 8))):
        radius = float(rng.uniform(10, 18))
        x, y = cx + rng.normal(0, 30), cy + rng.normal(0, 45)
        _crater(image, x, y, radius, style)
        xs += [x - radius, x + radius]
        ys += [y - radius, y + radius]
    x0, y0 = int(min(xs)), int(min(ys))
    return _box_around(x0, y0, int(max(xs)) - x0, int(max(ys)) - y0, 12, image.shape)


def _add_hole(image: np.ndarray, rng: np.random.Generator, style: PhotoStyle) -> tuple[int, int, int, int]:
    """A hole torn in the layer: an irregular dark region with a bright ragged rim."""

    height, width = image.shape
    cx, cy = rng.uniform(0.3, 0.7) * width, rng.uniform(0.3, 0.7) * height
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = rng.uniform(55, 95, size=angles.size)
    polygon = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1).astype(np.int32)
    inside = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(inside, [polygon], 1)
    contrast = style.metal_level - style.substrate_level
    rim = cv2.GaussianBlur((cv2.dilate(inside, np.ones((9, 9), np.uint8)) - inside).astype(np.float32), (0, 0), 3.0)
    plane = image.astype(np.float32) + 1.5 * contrast * rim
    dark = min(style.metal_level, style.substrate_level) * 0.3
    stripes = (np.arange(image.shape[1]) % 9 < 2).astype(np.float32)[None, :] * 0.15 * abs(contrast)
    plane = np.where(inside > 0, dark + stripes, plane)
    image[:] = np.clip(plane, 0, 255).astype(np.uint8)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return _box_around(int(x0), int(y0), int(x1 - x0), int(y1 - y0), 8, image.shape)


def make_frame(
    kind: str = "clean",
    seed: int = 0,
    layer: LayerStyle | None = None,
    photo: PhotoStyle | None = None,
) -> SyntheticChipFrame:
    """One frame with one kind of defect (or none) and its answer."""

    if kind not in DEFECT_EXPECTATIONS:
        raise ValueError(f"unknown defect kind: {kind}")
    layer = layer or LayerStyle()
    photo = photo or PhotoStyle()
    rng = np.random.default_rng(seed * 7919 + DEFECT_KINDS.index(kind))
    metal = draw_layer(layer, seed)
    codes, verdict = DEFECT_EXPECTATIONS[kind]
    boxes: list[tuple[int, int, int, int] | None] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    if kind == "open":
        boxes.append(_plant_open(metal, rng, layer))
    elif kind == "neck":
        boxes.append(_plant_neck(metal, rng, layer))
    elif kind == "bridge":
        boxes.append(_plant_bridge(metal, rng, layer))
    elif kind == "over_etch":
        metal = cv2.erode(metal, kernel)
        boxes.append(None)
    elif kind == "under_etch":
        metal = cv2.dilate(metal, kernel)
        boxes.append(None)
    render_style = photo
    if kind == "blur":
        from dataclasses import replace

        render_style = replace(photo, blur=max(4.0, layer.width / 2.0))
        boxes.append(None)
    image = render_photo(metal, render_style, seed)
    if kind == "residue":
        boxes.append(_add_residue(image, rng, photo))
    elif kind == "stain":
        boxes.append(_add_stain(image, metal, rng, photo))
    elif kind == "particles":
        boxes.extend(_add_particles(image, metal, rng, photo, count=int(rng.integers(6, 10))))
    elif kind == "hair":
        boxes.append(_add_hair(image, rng, photo))
    elif kind == "craters":
        boxes.append(_add_craters(image, rng, photo))
    elif kind == "hole":
        boxes.append(_add_hole(image, rng, photo))
    defects = tuple(PlantedDefect(kind, box, codes) for box in boxes)
    return SyntheticChipFrame(image, metal, kind, defects, verdict, layer, photo)


def box_overlaps(found: tuple[float, float, float, float], planted: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """Whether a finding's frame-relative box touches a planted box (frame pixels)."""

    height, width = shape
    fx, fy, fw, fh = found
    x0, y0, x1, y1 = fx * width, fy * height, (fx + fw) * width, (fy + fh) * height
    px, py, pw, ph = planted
    return x0 <= px + pw and px <= x1 and y0 <= py + ph and py <= y1


__all__ = [
    "DEFECT_EXPECTATIONS",
    "DEFECT_KINDS",
    "FRAME_WIDE_KINDS",
    "LayerStyle",
    "PhotoStyle",
    "PlantedDefect",
    "SyntheticChipFrame",
    "box_overlaps",
    "draw_layer",
    "make_frame",
    "render_photo",
]

"""Neural network outputs with known errors over synthetic chip photos, for testing validation checks.

Not used by the analysis itself. A frame has four layers:

- truth: the metal (or via) mask the chip really has (``synthetic_chip.draw_layer``);
- photo: what the microscope shows, optionally with a data defect (residue, particles, blur, ...)
  or shifted against the truth (a registration error);
- model masks: the truth with planted model errors, one mask per model (0/255);
- confidence maps: how sure each model is (0..255), with a planted behaviour: calibrated
  (unsure exactly where it is wrong and where the photo is bad), overconfident, underconfident, or
  unsure in flat places where nothing is wrong.

Every planted item carries its box in frame pixels (``None``: the whole frame), the diagnosis a
check should give it and the finding codes that count as finding it. ``synthetic_scorecard``
scores a check against these answers; the same truth can stand in for ground truth.

Diagnoses:

- ``model_error``: the model is wrong on a good photo;
- ``confident_error``: the model is wrong and sure of it (an overconfident model);
- ``bad_data``: the photo is bad here, whatever the model did;
- ``not_aligned``: the photo and the masks do not lie on each other;
- ``uncertain``: the model is right but unsure on a good photo;
- ``ok``: a harmless difference (boundary jitter) that no check should flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .findings import (
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    DIAG_OK,
    DIAG_UNCERTAIN,
)
from .synthetic_chip import (
    LayerStyle,
    PhotoStyle,
    _add_craters,
    _add_hair,
    _add_hole,
    _add_particles,
    _add_residue,
    _add_stain,
    _box_around,
    _free_spot,
    _plant_bridge,
    _plant_neck,
    _plant_open,
    _smooth_noise,
    draw_layer,
    render_photo,
)

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

Box = tuple[int, int, int, int]

# The diagnoses are the vocabulary of findings (core/findings.py); a synthetic frame never plants
# "unknown".
DIAGNOSES = (DIAG_OK, DIAG_MODEL_ERROR, DIAG_CONFIDENT_ERROR, DIAG_BAD_DATA, DIAG_NOT_ALIGNED, DIAG_UNCERTAIN)

# Model errors and the finding codes that count as finding them. The codes are the vocabulary the
# validation checks are going to speak; a check with other codes maps them (see the scorecard).
CONDUCTOR_ERROR_CODES: dict[str, tuple[str, ...]] = {
    "open": ("open",),
    "neck": ("neck", "open"),
    "bridge": ("bridge",),
    "missing_conductor": ("missing_object",),
    "extra_blob": ("extra_object",),
    # A window of wider or narrower metal: a shifted edge, seen up close as necks or bulges.
    "local_edge_shift": ("boundary_shift", "neck", "bulge"),
}
POINT_ERROR_CODES: dict[str, tuple[str, ...]] = {
    "missing_point": ("missing_point",),
    "extra_point": ("extra_point",),
    "merged_points": ("merged_points",),
    "shifted_point": ("shifted_point",),
}
# Errors of either layer.
COMMON_ERROR_CODES: dict[str, tuple[str, ...]] = {
    "edge_shift": ("boundary_shift",),
}
MODEL_ERROR_CODES: dict[str, tuple[str, ...]] = {**CONDUCTOR_ERROR_CODES, **POINT_ERROR_CODES, **COMMON_ERROR_CODES}
# Harmless differences between a mask and the truth: a check that flags them makes a false finding.
BENIGN_KINDS: dict[str, str] = {"boundary_jitter": "conductor", "point_jitter": "point"}
# Photo defects of synthetic_chip that leave the chip itself intact.
DATA_DEFECT_KINDS = ("residue", "stain", "particles", "hair", "craters", "hole", "blur")
DATA_DEFECT_CODES = ("bad_photo",)
MISALIGNED_CODES = ("not_aligned",)
UNCERTAIN_CODES = ("uncertain_region",)

BEHAVIOURS = ("calibrated", "overconfident", "underconfident", "flat_noise")
ENCODINGS = ("probability", "confidence")
REACTIONS = ("ignores", "follows")


@dataclass(frozen=True, slots=True)
class ModelError:
    """One planted error. ``shift``: px per side for edge shifts (positive grows the metal), px of the
    move for a shifted point; 0 picks one at random."""

    kind: str
    shift: int = 0


@dataclass(frozen=True, slots=True)
class ConfidenceStyle:
    """How a model's confidence map behaves and how it is written.

    ``encoding``: ``probability`` writes the probability of metal (unsure is mid gray);
    ``confidence`` writes |2p - 1| (white is sure, black is unsure).
    """

    behaviour: str = "calibrated"
    encoding: str = "probability"
    edge_softness: float = 1.5
    error_margin: int = 3


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """One model of a frame. Two plans with the same ``error_seed`` and errors make the same mistakes
    (correlated models); ``None`` gives every model its own."""

    model_id: str
    errors: tuple[ModelError, ...] = ()
    confidence: ConfidenceStyle = field(default_factory=ConfidenceStyle)
    error_seed: int | None = None


@dataclass(frozen=True, slots=True)
class PlantedItem:
    kind: str
    box: Box | None
    diagnosis: str
    codes: tuple[str, ...]
    # Models whose output carries the item; () for items of the photo.
    models: tuple[str, ...] = ()
    params: tuple[tuple[str, float | str], ...] = ()

    @property
    def is_positive(self) -> bool:
        """Whether a check should find it (a harmless difference should not be found)."""

        return self.diagnosis != DIAG_OK

    def param(self, name: str, default: float | str | None = None) -> float | str | None:
        return dict(self.params).get(name, default)


@dataclass(frozen=True, slots=True)
class SyntheticValidationFrame:
    truth: np.ndarray
    photo: np.ndarray
    masks: dict[str, np.ndarray]
    confidences: dict[str, np.ndarray]
    items: tuple[PlantedItem, ...]
    geometry: str
    layer: LayerStyle
    photo_style: PhotoStyle
    plans: tuple[ModelPlan, ...]
    case: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.truth.shape)

    def items_for(self, model_id: str | None) -> tuple[PlantedItem, ...]:
        """Items a check of one model's output is asked about: its own errors and the photo's.

        ``None``: every item; ``""``: the photo's only (a check that reads the photo, no mask).
        """

        if model_id is None:
            return self.items
        return tuple(item for item in self.items if not item.models or model_id in item.models)


# --- Model errors ----------------------------------------------------------------------------------


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(count), labels, stats


def _component_box(stats: np.ndarray, label: int, margin: int, shape: tuple[int, int]) -> Box:
    return _box_around(
        int(stats[label, cv2.CC_STAT_LEFT]),
        int(stats[label, cv2.CC_STAT_TOP]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
        margin,
        shape,
    )


def _union_box(first: Box, second: Box) -> Box:
    x0, y0 = min(first[0], second[0]), min(first[1], second[1])
    x1 = max(first[0] + first[2], second[0] + second[2])
    y1 = max(first[1] + first[3], second[1] + second[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _remove_component(mask: np.ndarray, rng: np.random.Generator, min_area: float, keep_largest: bool) -> Box:
    count, labels, stats = _components(mask)
    if count <= 1:
        raise ValueError("no object to remove")
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    candidates = [
        label
        for label in range(1, count)
        if areas[label - 1] >= min_area and not (keep_largest and label == largest and count > 2)
    ]
    if not candidates:
        raise ValueError("no object to remove")
    label = candidates[int(rng.integers(len(candidates)))]
    mask[labels == label] = 0
    return _component_box(stats, label, 3, mask.shape)


def _shift_edges(mask: np.ndarray, shift: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    if shift > 0:
        return cv2.dilate(mask, kernel, iterations=shift)
    if shift < 0:
        return cv2.erode(mask, kernel, iterations=-shift)
    return mask.copy()


def _local_window(mask: np.ndarray, rng: np.random.Generator, size: int) -> tuple[int, int]:
    height, width = mask.shape
    best: tuple[float, int, int] | None = None
    for _attempt in range(60):
        x = int(rng.integers(0, max(1, width - size)))
        y = int(rng.integers(0, max(1, height - size)))
        share = float(mask[y : y + size, x : x + size].mean())
        if 0.15 <= share <= 0.7:
            return x, y
        distance = abs(share - 0.4)
        if best is None or distance < best[0]:
            best = (distance, x, y)
    assert best is not None
    return best[1], best[2]


def _merge_points(mask: np.ndarray, rng: np.random.Generator, layer: LayerStyle) -> Box:
    count, _labels, stats = _components(mask)
    pitch = layer.width + layer.gap
    pairs: list[tuple[int, int]] = []
    for first in range(1, count):
        for second in range(1, count):
            dx = int(stats[second, cv2.CC_STAT_LEFT]) - int(stats[first, cv2.CC_STAT_LEFT])
            dy = abs(int(stats[second, cv2.CC_STAT_TOP]) - int(stats[first, cv2.CC_STAT_TOP]))
            if dy <= 1 and 0 < dx <= pitch + 1:
                pairs.append((first, second))
    if not pairs:
        raise ValueError("no two neighbouring points to merge")
    first, second = pairs[int(rng.integers(len(pairs)))]
    top = int(stats[first, cv2.CC_STAT_TOP])
    height = int(stats[first, cv2.CC_STAT_HEIGHT])
    left = int(stats[first, cv2.CC_STAT_LEFT])
    right = int(stats[second, cv2.CC_STAT_LEFT]) + int(stats[second, cv2.CC_STAT_WIDTH])
    mask[top : top + height, left:right] = 1
    return _box_around(left, top, right - left, height, 3, mask.shape)


def _move_point(mask: np.ndarray, rng: np.random.Generator, layer: LayerStyle, shift: int) -> tuple[Box, int]:
    count, labels, stats = _components(mask)
    if count <= 1:
        raise ValueError("no point to move")
    distance = int(shift) if shift > 0 else int(rng.integers(layer.width, 2 * layer.width + 1))
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    height, width = mask.shape
    for label in rng.permutation(np.arange(1, count)).tolist():
        own = labels == label
        others = mask.astype(bool) & ~own
        ys, xs = np.nonzero(own)
        for index in rng.permutation(len(directions)).tolist():
            ux, uy = directions[index]
            norm = float(np.hypot(ux, uy))
            dx, dy = int(round(distance * ux / norm)), int(round(distance * uy / norm))
            nx, ny = xs + dx, ys + dy
            if nx.min() < 2 or ny.min() < 2 or nx.max() >= width - 2 or ny.max() >= height - 2:
                continue
            target = np.zeros_like(own)
            target[ny, nx] = True
            clearance = cv2.dilate(target.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if np.any(clearance & others):
                continue
            mask[own] = 0
            mask[target] = 1
            old_box = _component_box(stats, label, 3, mask.shape)
            new_box = _box_around(int(nx.min()), int(ny.min()), int(np.ptp(nx)) + 1, int(np.ptp(ny)) + 1, 3, mask.shape)
            return _union_box(old_box, new_box), distance
    raise ValueError("no free place to move a point to")


def _jitter_points(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count, labels, _stats = _components(mask)
    result = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
        result[np.clip(ys + dy, 0, height - 1), np.clip(xs + dx, 0, width - 1)] = 1
    return result


def _jitter_boundary(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A wavy boundary within 1 px of the old one, with the same objects: nothing breaks or joins."""

    kernel = np.ones((3, 3), dtype=np.uint8)
    core = cv2.erode(mask, kernel)
    reach = cv2.dilate(mask, kernel)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    wave = _smooth_noise(rng, mask.shape, 2.0) * 0.12
    jittered = (core | ((soft + wave > 0.5).astype(np.uint8) & reach)).astype(np.uint8)
    count, labels, _stats = _components(jittered)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[core > 0])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def apply_model_error(
    mask: np.ndarray, error: ModelError, rng: np.random.Generator, layer: LayerStyle
) -> tuple[Box | None, dict[str, float]]:
    """Plant one error into a 0/1 mask in place: its box (None: the whole frame) and parameters."""

    kind = error.kind
    shape = mask.shape
    try:
        if kind == "open":
            return _plant_open(mask, rng, layer), {}
        if kind == "neck":
            return _plant_neck(mask, rng, layer), {}
        if kind == "bridge":
            return _plant_bridge(mask, rng, layer), {}
    except (ValueError, IndexError) as error_text:
        raise ValueError(f"cannot plant {kind}: {error_text}") from error_text
    if kind == "missing_conductor":
        return _remove_component(mask, rng, min_area=20.0 * layer.width, keep_largest=True), {}
    if kind == "missing_point":
        return _remove_component(mask, rng, min_area=0.5 * layer.width**2, keep_largest=False), {}
    if kind == "extra_blob":
        radius = int(rng.integers(max(3, layer.width // 2), layer.width + 2))
        x, y = _free_spot(mask, rng, on_metal=False, clearance=radius + layer.gap / 2.0)
        axes = (radius, max(2, int(round(radius * rng.uniform(0.6, 1.0)))))
        cv2.ellipse(mask, (x, y), axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        return _box_around(x - radius, y - radius, 2 * radius, 2 * radius, 3, shape), {"radius_px": float(radius)}
    if kind == "extra_point":
        side = layer.width
        x, y = _free_spot(mask, rng, on_metal=False, clearance=side + layer.gap / 2.0)
        x0, y0 = x - side // 2, y - side // 2
        mask[y0 : y0 + side, x0 : x0 + side] = 1
        return _box_around(x0, y0, side, side, 3, shape), {}
    if kind == "merged_points":
        return _merge_points(mask, rng, layer), {}
    if kind == "shifted_point":
        box, distance = _move_point(mask, rng, layer, error.shift)
        return box, {"shift_px": float(distance)}
    if kind == "edge_shift":
        shift = int(error.shift) if error.shift else int(rng.choice((-2, -1, 1, 2)))
        mask[:] = _shift_edges(mask, shift)
        return None, {"shift_px": float(shift)}
    if kind == "local_edge_shift":
        # Wider or narrower metal in one window, the objects themselves unchanged (no open, no bridge).
        shift = int(error.shift) if error.shift else int(rng.choice((-3, -2, 2, 3)))
        size = max(32, min(shape) // 4)
        shifted = _shift_edges(mask, shift)
        count = _components(mask)[0]
        for _attempt in range(20):
            x, y = _local_window(mask, rng, size)
            candidate = mask.copy()
            candidate[y : y + size, x : x + size] = shifted[y : y + size, x : x + size]
            if _components(candidate)[0] == count and np.any(candidate != mask):
                mask[:] = candidate
                return (x, y, size, size), {"shift_px": float(shift)}
        raise ValueError("no window to shift edges in without changing the objects")
    if kind == "boundary_jitter":
        mask[:] = _jitter_boundary(mask, rng)
        return None, {}
    if kind == "point_jitter":
        mask[:] = _jitter_points(mask, rng)
        return None, {}
    raise ValueError(f"unknown model error: {kind}")


# --- Photo -----------------------------------------------------------------------------------------


def _plant_data_defect(
    image: np.ndarray, truth: np.ndarray, kind: str, rng: np.random.Generator, photo: PhotoStyle
) -> list[Box | None]:
    if kind == "residue":
        return [_add_residue(image, rng, photo)]
    if kind == "stain":
        return [_add_stain(image, truth, rng, photo)]
    if kind == "particles":
        return list(_add_particles(image, truth, rng, photo, count=int(rng.integers(6, 10))))
    if kind == "hair":
        return [_add_hair(image, rng, photo)]
    if kind == "craters":
        return [_add_craters(image, rng, photo)]
    if kind == "hole":
        return [_add_hole(image, rng, photo)]
    if kind == "blur":
        return [None]
    raise ValueError(f"unknown data defect: {kind}")


def _shifted(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)


def _metal_seen_on_photo(image: np.ndarray, photo: PhotoStyle) -> np.ndarray:
    level = (photo.metal_level + photo.substrate_level) / 2.0
    if photo.metal_level >= photo.substrate_level:
        return (image > level).astype(np.uint8)
    return (image < level).astype(np.uint8)


# --- Confidence ------------------------------------------------------------------------------------


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    return np.where(mask > 0, inside - 0.5, -(outside - 0.5)).astype(np.float32)


def _confidence_map(
    mask: np.ndarray,
    unsure_regions: Sequence[np.ndarray],
    style: ConfidenceStyle,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Box]]:
    """The confidence map (uint8) and the boxes of planted unsure places in flat areas."""

    signed = _signed_distance(mask)
    softness = float(style.edge_softness) * (0.35 if style.behaviour == "overconfident" else 1.0)
    probability = 1.0 / (1.0 + np.exp(np.clip(-signed / max(1.0e-3, softness), -60.0, 60.0)))
    if style.behaviour in ("calibrated", "flat_noise"):
        for region in unsure_regions:
            probability[region] = 0.5 + (probability[region] - 0.5) * 0.15
    if style.behaviour == "underconfident":
        probability = 0.5 + (probability - 0.5) * 0.4
    flat_boxes: list[Box] = []
    if style.behaviour == "flat_noise":
        far = np.abs(signed) > 2.0 * softness + 3.0
        blotches = (_smooth_noise(rng, mask.shape, 12.0) > 1.6) & far
        count, labels, stats = _components(blotches.astype(np.uint8))
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < 120:
                blotches[labels == label] = False
                continue
            flat_boxes.append(_component_box(stats, label, 2, mask.shape))
        probability[blotches] = 0.5 + (probability[blotches] - 0.5) * 0.2
    # A slow wobble, as real networks have; smooth, so the PNG stays small at 4096 px.
    probability = np.clip(probability + 0.01 * _smooth_noise(rng, mask.shape, 6.0), 0.0, 1.0)
    if style.encoding == "confidence":
        values = np.abs(2.0 * probability - 1.0)
    elif style.encoding == "probability":
        values = probability
    else:
        raise ValueError(f"unknown confidence encoding: {style.encoding}")
    return np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8), flat_boxes


def _error_region(mask: np.ndarray, truth: np.ndarray, box: Box | None, margin: int) -> np.ndarray:
    wrong = (mask != truth).astype(np.uint8)
    if box is not None:
        x, y, w, h = box
        limited = np.zeros_like(wrong)
        limited[y : y + h, x : x + w] = wrong[y : y + h, x : x + w]
        wrong = limited
    if margin > 0:
        wrong = cv2.dilate(wrong, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return wrong.astype(bool)


def _box_region(shape: tuple[int, int], box: Box | None) -> np.ndarray:
    region = np.zeros(shape, dtype=bool)
    if box is None:
        region[:] = True
    else:
        x, y, w, h = box
        region[y : y + h, x : x + w] = True
    return region


# --- Frames ----------------------------------------------------------------------------------------


def _as_error(value: ModelError | str) -> ModelError:
    return value if isinstance(value, ModelError) else ModelError(str(value))


def _merge_items(items: list[PlantedItem]) -> tuple[PlantedItem, ...]:
    """One item for the same mistake made by several (correlated) models."""

    merged: dict[tuple[object, ...], PlantedItem] = {}
    for item in items:
        key = (item.kind, item.box, item.params, bool(item.models))
        existing = merged.get(key)
        if existing is None or not item.models:
            merged[key] = item
            continue
        merged[key] = replace(existing, models=tuple(dict.fromkeys(existing.models + item.models)))
    return tuple(merged.values())


def make_validation_frame(
    *,
    seed: int = 0,
    layer: LayerStyle | None = None,
    photo: PhotoStyle | None = None,
    models: Sequence[ModelPlan] = (ModelPlan("a"), ModelPlan("b")),
    data_defect: str | None = None,
    reaction: str = "ignores",
    misalignment: tuple[int, int] | None = None,
    case: str = "",
) -> SyntheticValidationFrame:
    """One frame: truth, photo, model masks and confidence maps, with every planted item."""

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for synthetic validation frames")
    if reaction not in REACTIONS:
        raise ValueError(f"unknown model reaction: {reaction}")
    layer = layer or LayerStyle()
    photo = photo or PhotoStyle()
    truth = draw_layer(layer, seed)
    geometry = "point" if layer.vias else "polygon"
    items: list[PlantedItem] = []

    photo_truth = truth
    if misalignment is not None:
        dx, dy = (int(value) for value in misalignment)
        photo_truth = _shifted(truth, dx, dy)
        items.append(PlantedItem("misaligned", None, DIAG_NOT_ALIGNED, MISALIGNED_CODES, (), (("dx", float(dx)), ("dy", float(dy)))))
    render_style = photo
    if data_defect == "blur":
        render_style = replace(photo, blur=max(4.0, layer.width / 2.0))
    image = render_photo(photo_truth, render_style, seed)
    defect_boxes: list[Box | None] = []
    if data_defect is not None:
        if data_defect not in DATA_DEFECT_KINDS:
            raise ValueError(f"unknown data defect: {data_defect}")
        defect_rng = np.random.default_rng(seed * 7919 + 17 + DATA_DEFECT_KINDS.index(data_defect))
        defect_boxes = _plant_data_defect(image, photo_truth, data_defect, defect_rng, photo)
        for box in defect_boxes:
            items.append(PlantedItem(data_defect, box, DIAG_BAD_DATA, DATA_DEFECT_CODES, (), (("reaction", reaction),)))
    seen = _metal_seen_on_photo(image, photo) if reaction == "follows" and defect_boxes else None

    masks: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    for index, plan in enumerate(models):
        error_seed = plan.error_seed if plan.error_seed is not None else 1000 + index
        rng = np.random.default_rng((int(seed), int(error_seed)))
        mask = truth.copy()
        model_items: list[tuple[PlantedItem, Box | None]] = []
        overconfident = plan.confidence.behaviour == "overconfident"
        for raw in plan.errors:
            error = _as_error(raw)
            box, params = apply_model_error(mask, error, rng, layer)
            if error.kind in BENIGN_KINDS:
                diagnosis, codes = DIAG_OK, ()
            else:
                diagnosis = DIAG_CONFIDENT_ERROR if overconfident else DIAG_MODEL_ERROR
                codes = MODEL_ERROR_CODES[error.kind]
            item = PlantedItem(error.kind, box, diagnosis, codes, (plan.model_id,), tuple(sorted(params.items())))
            model_items.append((item, box))
        if seen is not None:
            for box in defect_boxes:
                region = _box_region(mask.shape, box)
                mask[region] = seen[region]
        unsure: list[np.ndarray] = []
        for item, box in model_items:
            if item.is_positive:
                unsure.append(_error_region(mask, truth, box, plan.confidence.error_margin))
        for box in defect_boxes:
            unsure.append(_box_region(mask.shape, box))
        confidence_rng = np.random.default_rng((int(seed), int(error_seed), 7))
        confidence, flat_boxes = _confidence_map(mask, unsure, plan.confidence, confidence_rng)
        for box in flat_boxes:
            model_items.append((PlantedItem("uncertain_flat", box, DIAG_UNCERTAIN, UNCERTAIN_CODES, (plan.model_id,)), box))
        if plan.confidence.behaviour == "underconfident":
            # Unsure everywhere, also where it is right: the model, not any one place.
            model_items.append(
                (PlantedItem("underconfident", None, DIAG_UNCERTAIN, UNCERTAIN_CODES, (plan.model_id,)), None)
            )
        items.extend(item for item, _box in model_items)
        masks[plan.model_id] = (mask * 255).astype(np.uint8)
        confidences[plan.model_id] = confidence

    return SyntheticValidationFrame(
        truth=truth,
        photo=image,
        masks=masks,
        confidences=confidences,
        items=_merge_items(items),
        geometry=geometry,
        layer=layer,
        photo_style=photo,
        plans=tuple(models),
        case=case,
    )


# --- Cases -----------------------------------------------------------------------------------------


def cases_for(layer: LayerStyle) -> tuple[str, ...]:
    """Every single-problem case that makes sense for a layer, ``clean`` first."""

    if layer.vias:
        errors = (*POINT_ERROR_CODES, *COMMON_ERROR_CODES, "point_jitter")
    else:
        errors = (*CONDUCTOR_ERROR_CODES, *COMMON_ERROR_CODES, "boundary_jitter")
    return ("clean", *errors, *DATA_DEFECT_KINDS, "misaligned", "uncertain_flat")


def make_case(
    case: str,
    seed: int = 0,
    layer: LayerStyle | None = None,
    photo: PhotoStyle | None = None,
    *,
    model_count: int = 2,
    behaviour: str = "calibrated",
    encoding: str = "probability",
    shared: bool = False,
    reaction: str = "ignores",
) -> SyntheticValidationFrame:
    """A frame with one problem.

    A model error goes into model ``a`` only (model ``b`` is right there), or into every model when
    ``shared`` (correlated models: comparing them cannot see it). ``uncertain_flat`` makes model ``a``
    unsure in flat places; a data defect and ``misaligned`` leave the masks right.
    """

    if behaviour not in BEHAVIOURS:
        raise ValueError(f"unknown confidence behaviour: {behaviour}")
    if model_count < 1:
        raise ValueError("at least one model")
    layer = layer or LayerStyle()
    style = ConfidenceStyle(behaviour=behaviour, encoding=encoding)
    model_ids = tuple("abcdefgh"[index] for index in range(model_count))
    plans = [ModelPlan(model_id, confidence=style) for model_id in model_ids]
    data_defect = None
    misalignment = None
    if case in MODEL_ERROR_CODES or case in BENIGN_KINDS:
        carriers = model_ids if shared else model_ids[:1]
        plans = [
            replace(plan, errors=(ModelError(case),), error_seed=1 if shared else None) if plan.model_id in carriers else plan
            for plan in plans
        ]
    elif case in DATA_DEFECT_KINDS:
        data_defect = case
    elif case == "misaligned":
        rng = np.random.default_rng(seed + 31)
        magnitude = int(rng.integers(4, 9))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        misalignment = (int(round(magnitude * np.cos(angle))) or 1, int(round(magnitude * np.sin(angle))) or 1)
    elif case == "uncertain_flat":
        plans[0] = replace(plans[0], confidence=replace(style, behaviour="flat_noise"))
    elif case != "clean":
        raise ValueError(f"unknown case: {case}")
    return make_validation_frame(
        seed=seed,
        layer=layer,
        photo=photo,
        models=tuple(plans),
        data_defect=data_defect,
        reaction=reaction,
        misalignment=misalignment,
        case=case,
    )


def decode_probability(confidence: np.ndarray, encoding: str) -> np.ndarray:
    """The probability of metal (0..1) from a written map; for ``confidence`` only its distance from 0.5."""

    values = np.asarray(confidence, dtype=np.float32) / 255.0
    if encoding == "probability":
        return values
    return 0.5 + values / 2.0


__all__ = [
    "BEHAVIOURS",
    "BENIGN_KINDS",
    "CONDUCTOR_ERROR_CODES",
    "ConfidenceStyle",
    "DATA_DEFECT_CODES",
    "DATA_DEFECT_KINDS",
    "DIAGNOSES",
    "DIAG_BAD_DATA",
    "DIAG_CONFIDENT_ERROR",
    "DIAG_MODEL_ERROR",
    "DIAG_NOT_ALIGNED",
    "DIAG_OK",
    "DIAG_UNCERTAIN",
    "ENCODINGS",
    "MODEL_ERROR_CODES",
    "ModelError",
    "ModelPlan",
    "POINT_ERROR_CODES",
    "PlantedItem",
    "REACTIONS",
    "SyntheticValidationFrame",
    "apply_model_error",
    "cases_for",
    "decode_probability",
    "make_case",
    "make_validation_frame",
]

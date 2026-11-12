"""Does a network's mask agree with what the chip photo shows, and is the photo good enough to tell?

Two questions per frame, answered tile by tile so the answer has a place:

- Q, the photo: contrast against noise, how wide the edges are (blur), how much mid-tone lies away
  from any edge (a residue film, a stain). A tile with a poor photo is ``bad_photo``: whatever the
  model did there, the data is the problem (diagnosis ``bad_data``).
- A, the mask against the photo: the photo is turned into a metal mask of its own, with levels of
  metal and substrate taken locally (uneven light does not matter) from where the mask says metal
  and substrate are; the objects of the network's mask are then matched with it as in
  core/topology_events.py (reference "photo"). A disagreement on a good photo is the model's
  (``model_error``; ``confident_error`` when its confidence map is sure there, see stage 4).

Before any of that, the photo and the mask must lie on each other: a shift found by phase
correlation is reported as ``not_aligned`` and nothing else is said about that mask.

A small round blob of the other class away from every edge (a particle on the substrate, a pit in
the metal) is a dirty photo, not a missed or extra object.

Everything is read at full resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .findings import (
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    Finding,
    make_finding,
    sort_findings,
)
from .confidence_maps import build_model_uncertainty, normalize_model_confidence, resolve_confidence_encoding
from .findings import DIAG_UNCERTAIN
from .topology_events import TopologySettings, compare_to_reference

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

PHOTO_CHECK_VERSION = "photo_check.v1"
DETECTOR_PHOTO = "photo_check"
PHOTO_REFERENCE_ID = "photo"
PHOTO_QUALITY_RISK_METRIC = "photo_quality_risk"
PHOTO_MISMATCH_RISK_METRIC = "photo_mismatch_risk"
# How far the model's confidence is from how often it is right (against the photo, on good tiles),
# and the share of its disagreements with the photo it was sure of (0-100, higher is worse).
CONFIDENCE_ECE_METRIC = "confidence_ece"
CONFIDENT_ERROR_SHARE_METRIC = "confident_error_share"
UNCERTAIN_REGION_KIND = "uncertain_region"
BAD_PHOTO_KIND = "bad_photo"
NOT_ALIGNED_KIND = "not_aligned"


@dataclass(frozen=True, slots=True)
class PhotoCheckSettings:
    tiles: int = 8
    # A shift of the photo against the mask of at least this many pixels is a misalignment.
    misalign_px: float = 2.0
    misalign_min_response: float = 0.05
    # ... and only when the shifted mask agrees with the photo this much better (IoU): a periodic
    # layout or edges grown by a pixel or two give a correlation peak too, but no better fit.
    misalign_min_gain: float = 0.10
    # A correlation peak this strong is a shift by itself (small vias gain little IoU from it).
    misalign_sure_response: float = 0.4
    # Edge width (px from 0 to full contrast) above which a tile or the frame is blurred.
    blur_edge_width_px: float = 6.0
    # Contrast over noise below which a tile is too poor to read.
    min_snr: float = 4.0
    # Share of mid-tone pixels away from the edges above which a tile is dirty (a film, a stain).
    midtone_share: float = 0.10
    midtone_low: float = 0.25
    midtone_high: float = 0.75
    # A thin long piece of the photo's metal that the mask does not have (a hair): at most this
    # thick (px from its middle to its edge) and at least this long.
    hair_max_half_width: float = 3.5
    hair_min_length_px: int = 40
    quality_threshold: float = 0.5
    # Uncertainty (0..1) at an error under which the model was sure of it: a confident error.
    confident_uncertainty: float = 0.3
    # Unsure (uncertainty above this) where the mask and a good photo agree, away from edges, over
    # at least this many pixels: an unsure place worth a look (a new kind of object, poor training).
    unsure_flat_uncertainty: float = 0.5
    unsure_flat_min_area: int = 40
    calibration_bins: int = 10
    topology: TopologySettings = field(default_factory=TopologySettings)


@dataclass(slots=True)
class PhotoQuality:
    """What the photo of one frame is like, and its own metal mask."""

    shape: tuple[int, int]
    quality: np.ndarray  # tiles x tiles, 0 (unreadable) .. 1 (good)
    causes: list[list[str]]  # per tile index, why it is poor
    metal_brighter: bool
    edge_width_px: float
    snr: float
    photo_metal: np.ndarray  # bool, the metal the photo shows
    bad: np.ndarray  # bool, where the photo is poor
    shift: tuple[float, float]
    misaligned: bool
    findings: tuple[Finding, ...]

    @property
    def quality_risk(self) -> float:
        """0 (a good photo everywhere) .. 100 (unreadable everywhere)."""

        return float(100.0 * (1.0 - float(np.mean(self.quality)))) if self.quality.size else 0.0


def _tile_edges(size: int, tiles: int) -> np.ndarray:
    return np.linspace(0, size, tiles + 1).astype(int)


def _relative(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    x, y, w, h = box
    return (x / max(1, width), y / max(1, height), w / max(1, width), h / max(1, height))


def _registration(
    photo: np.ndarray, mask: np.ndarray, metal_brighter: bool, crop: int = 1024
) -> tuple[float, float, float]:
    """Shift (dx, dy) of the photo against the mask and the correlation response (the frame's middle)."""

    height, width = mask.shape
    y0, x0 = max(0, (height - crop) // 2), max(0, (width - crop) // 2)
    photo = photo[y0:y0 + crop, x0:x0 + crop]
    mask = mask[y0:y0 + crop, x0:x0 + crop]
    signal = photo.astype(np.float32)
    if not metal_brighter:
        signal = -signal
    signal = (signal - float(signal.mean())) / max(1e-6, float(signal.std()))
    reference = mask.astype(np.float32)
    reference = (reference - float(reference.mean())) / max(1e-6, float(reference.std()))
    window = cv2.createHanningWindow((mask.shape[1], mask.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(reference, signal, window)
    return float(dx), float(dy), float(response)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else float(np.count_nonzero(first & second)) / union


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST) > 0


def _edges(mask: np.ndarray) -> np.ndarray:
    return mask ^ (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)


def _far_from(edge: np.ndarray, distance: float) -> np.ndarray:
    """Pixels at least ``distance`` away from ``edge`` (a dilation by a disc: cheaper than distances)."""

    radius = max(0, int(math.ceil(distance)) - 1)
    if radius == 0:
        return ~edge
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(edge.astype(np.uint8), disc) == 0


_BLUR_SIGMAS = (0.6, 0.9, 1.3, 1.8, 2.5, 3.4, 4.6, 6.2, 8.4)


def _fit_blur_sigma(gray: np.ndarray, mask: np.ndarray, metal_brighter: bool, crop: int = 512) -> float:
    """The blur (Gaussian sigma, px) with which the mask correlates best with the photo (its middle).

    The mask's edges may lie a pixel or two off (a model that over- or under-segments): softening a
    shifted edge would also fit, so the mask is grown and shrunk by up to two pixels as well and the
    blur of the best pair is taken.
    """

    height, width = mask.shape
    y0, x0 = max(0, (height - crop) // 2), max(0, (width - crop) // 2)
    photo = gray[y0:y0 + crop, x0:x0 + crop].astype(np.float32)
    window = mask[y0:y0 + crop, x0:x0 + crop].astype(np.uint8)
    if not metal_brighter:
        photo = -photo
    photo = (photo - float(photo.mean())) / max(1e-6, float(photo.std()))
    kernel = np.ones((3, 3), np.uint8)
    shapes = [window]
    for step in (1, 2):
        shapes.append(cv2.dilate(window, kernel, iterations=step))
        shapes.append(cv2.erode(window, kernel, iterations=step))
    best_sigma, best = _BLUR_SIGMAS[0], -2.0
    for shape in shapes:
        shape_float = shape.astype(np.float32)
        for sigma in _BLUR_SIGMAS:
            model = cv2.GaussianBlur(shape_float, (0, 0), sigma)
            spread = float(model.std())
            if spread <= 1e-6:
                continue
            correlation = float(np.mean(photo * (model - float(model.mean())) / spread))
            if correlation > best:
                best_sigma, best = sigma, correlation
    return float(best_sigma)


def _level_maps(photo: np.ndarray, mask: np.ndarray, away: np.ndarray, tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Metal and substrate brightness of every pixel, from medians of tiles (each read with a margin)."""

    height, width = mask.shape
    ys, xs = _tile_edges(height, tiles), _tile_edges(width, tiles)
    metal_inside = mask & away
    substrate_inside = ~mask & away
    global_metal = float(np.median(photo[metal_inside])) if np.any(metal_inside) else float(np.median(photo[mask]))
    global_substrate = (
        float(np.median(photo[substrate_inside])) if np.any(substrate_inside) else float(np.median(photo[~mask]))
    )
    metal = np.full((tiles, tiles), global_metal, dtype=np.float32)
    substrate = np.full((tiles, tiles), global_substrate, dtype=np.float32)
    for row in range(tiles):
        for column in range(tiles):
            margin_y = (ys[row + 1] - ys[row]) // 2
            margin_x = (xs[column + 1] - xs[column]) // 2
            y0, y1 = max(0, ys[row] - margin_y), min(height, ys[row + 1] + margin_y)
            x0, x1 = max(0, xs[column] - margin_x), min(width, xs[column + 1] + margin_x)
            # A median does not need every pixel: about 64k of a window are plenty.
            step = max(1, int(math.sqrt((y1 - y0) * (x1 - x0) / 65536.0)))
            window = photo[y0:y1:step, x0:x1:step]
            inside = metal_inside[y0:y1:step, x0:x1:step]
            outside = substrate_inside[y0:y1:step, x0:x1:step]
            if np.count_nonzero(inside) >= 20:
                metal[row, column] = float(np.median(window[inside]))
            if np.count_nonzero(outside) >= 20:
                substrate[row, column] = float(np.median(window[outside]))
    size = (width, height)
    return (
        cv2.resize(metal, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(substrate, size, interpolation=cv2.INTER_LINEAR),
    )


def _tile_index(shape: tuple[int, int], tiles: int) -> np.ndarray:
    height, width = shape
    rows = (np.arange(height) * tiles // height)[:, None]
    columns = (np.arange(width) * tiles // width)[None, :]
    return (rows * tiles + columns).astype(np.int32)


def _per_tile(values: np.ndarray, where: np.ndarray, tiles_of: np.ndarray, count: int, reducer) -> np.ndarray:
    """``reducer`` of ``values`` over the ``where`` pixels of every tile (about 1M pixels sampled at most)."""

    result = np.full(count, np.nan, dtype=np.float64)
    step = max(1, int(math.sqrt(values.size / 1_048_576.0)))
    if step > 1:
        values, where, tiles_of = values[::step, ::step], where[::step, ::step], tiles_of[::step, ::step]
    picked_tiles = tiles_of[where]
    picked = values[where]
    if picked.size == 0:
        return result
    order = np.argsort(picked_tiles, kind="stable")
    picked_tiles, picked = picked_tiles[order], picked[order]
    starts = np.flatnonzero(np.r_[True, picked_tiles[1:] != picked_tiles[:-1]])
    stops = np.r_[starts[1:], picked.size]
    for start, stop in zip(starts.tolist(), stops.tolist()):
        result[int(picked_tiles[start])] = reducer(picked[start:stop])
    return result


def assess_photo(
    photo: np.ndarray, mask: np.ndarray, settings: PhotoCheckSettings | None = None
) -> PhotoQuality:
    """The photo of a frame read against a mask that is mostly right (one model's, or the models' majority)."""

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for the photo check")
    settings = settings or PhotoCheckSettings()
    gray = np.asarray(photo, dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    if gray.shape != mask_bool.shape:
        gray = cv2.resize(gray, (mask_bool.shape[1], mask_bool.shape[0]), interpolation=cv2.INTER_AREA)
    shape = mask_bool.shape
    tiles = max(1, int(settings.tiles))
    count = tiles * tiles
    if not np.any(mask_bool) or np.all(mask_bool):
        empty = np.ones((tiles, tiles), dtype=np.float32)
        return PhotoQuality(shape, empty, [[] for _ in range(count)], True, 0.0, 0.0, mask_bool.copy(),
                            np.zeros(shape, bool), (0.0, 0.0), False, ())
    metal_brighter = float(gray[mask_bool].mean()) >= float(gray[~mask_bool].mean())
    dx, dy, response = _registration(gray, mask_bool, metal_brighter)
    misaligned = False
    if math.hypot(dx, dy) >= settings.misalign_px and response >= settings.misalign_min_response:
        middle = 0.5 * (float(np.median(gray[mask_bool])) + float(np.median(gray[~mask_bool])))
        seen = gray > middle if metal_brighter else gray < middle
        gain = _iou(_shift_mask(mask_bool, dx, dy), seen) - _iou(mask_bool, seen)
        misaligned = gain >= settings.misalign_min_gain or (
            response >= settings.misalign_sure_response and gain >= 0.2 * settings.misalign_min_gain
        )
    edge = _edges(mask_bool)
    away = _far_from(edge, 3.0)
    metal_level, substrate_level = _level_maps(gray, mask_bool, away, tiles)
    contrast_map = metal_level - substrate_level
    safe = np.where(np.abs(contrast_map) < 1.0, np.where(contrast_map < 0, -1.0, 1.0), contrast_map)
    z = (gray - substrate_level) / safe
    # Away from the photo's own edges too: where the mask's edges are off by a pixel or two, the
    # photo's real edges (half-tone by nature) would otherwise look like a film.
    photo_edges = _edges(z > 0.5) & ~_far_from(edge, 5.0)
    away &= ~(cv2.dilate(photo_edges.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    tiles_of = _tile_index(shape, tiles)
    # Noise: spread of the flat parts around their own level.
    expected = np.where(mask_bool, metal_level, substrate_level)
    residual = np.abs(gray - expected)
    noise = 1.4826 * _per_tile(residual, away, tiles_of, count, np.median)
    contrast = _per_tile(np.abs(contrast_map), np.ones(shape, bool), tiles_of, count, np.mean)
    snr = contrast / np.maximum(noise, 0.5)
    # Blur: the Gaussian that makes the mask look most like the photo (works for small vias too).
    blur_sigma = _fit_blur_sigma(gray, mask_bool, metal_brighter)
    edge_width = np.full(count, blur_sigma * math.sqrt(2.0 * math.pi))
    # Mid-tones away from edges: neither metal nor substrate.
    midtone = (z > settings.midtone_low) & (z < settings.midtone_high)
    midtone_share = _per_tile(midtone.astype(np.float32), away, tiles_of, count, np.mean)

    def ramp(value: np.ndarray, good: float, bad: float) -> np.ndarray:
        return np.clip((value - bad) / (good - bad), 0.0, 1.0)

    blur_limit = float(settings.blur_edge_width_px)
    q_snr = np.where(np.isnan(snr), 1.0, ramp(snr, settings.min_snr * 1.5, settings.min_snr * 0.5))
    q_sharp = np.where(np.isnan(edge_width), 1.0, ramp(edge_width, blur_limit * 0.75, blur_limit * 1.25))
    q_clean = np.where(np.isnan(midtone_share), 1.0, ramp(midtone_share, settings.midtone_share * 0.5, settings.midtone_share * 1.5))
    # Blur is read for the whole frame (the median of the tiles): one tile of a wrong mask has edges
    # the photo has not, which would look blurred; a photo is blurred as a whole.
    del q_sharp
    quality = np.minimum(q_snr, q_clean).astype(np.float32)
    causes: list[list[str]] = []
    for index in range(count):
        tile_causes = []
        if q_snr[index] < settings.quality_threshold:
            tile_causes.append("low_contrast")
        if q_clean[index] < settings.quality_threshold:
            tile_causes.append("residue")
        causes.append(tile_causes)
    frame_edge_width = float(np.nanmedian(edge_width)) if np.any(~np.isnan(edge_width)) else 0.0
    frame_snr = float(np.nanmedian(snr)) if np.any(~np.isnan(snr)) else 0.0
    findings: list[Finding] = []
    bad = np.zeros(shape, dtype=bool)
    frame_causes = []
    if frame_edge_width > blur_limit:
        frame_causes.append("blur")
    if frame_snr < settings.min_snr:
        frame_causes.append("low_contrast")
    ys, xs = _tile_edges(shape[0], tiles), _tile_edges(shape[1], tiles)
    photo_metal = z > 0.5
    if misaligned:
        # Levels read from a mask that does not lie on the photo mean nothing: only the shift is said.
        bad[:] = True
    elif frame_causes:
        bad[:] = True
        findings.append(
            make_finding(BAD_PHOTO_KIND, DETECTOR_PHOTO, 1.0, None, diagnosis=DIAG_BAD_DATA, detector_version=PHOTO_CHECK_VERSION,
                         evidence={"cause": "+".join(frame_causes), "edge_width_px": round(frame_edge_width, 2), "snr": round(frame_snr, 2)})
        )
    else:
        poor = (quality < settings.quality_threshold).reshape(tiles, tiles).astype(np.uint8)
        groups, labels = cv2.connectedComponents(poor, connectivity=8)
        for group in range(1, groups):
            members = np.argwhere(labels == group)
            (row0, column0), (row1, column1) = members.min(axis=0), members.max(axis=0)
            box = (float(xs[column0]), float(ys[row0]), float(xs[column1 + 1] - xs[column0]), float(ys[row1 + 1] - ys[row0]))
            indices = [int(row * tiles + column) for row, column in members.tolist()]
            group_causes = sorted({cause for index in indices for cause in causes[index]})
            worst = float(min(quality[index] for index in indices))
            bad[ys[row0]:ys[row1 + 1], xs[column0]:xs[column1 + 1]] = True
            findings.append(
                make_finding(BAD_PHOTO_KIND, DETECTOR_PHOTO, 1.0 - worst, _relative(box, shape), diagnosis=DIAG_BAD_DATA,
                             detector_version=PHOTO_CHECK_VERSION, evidence={"cause": "+".join(group_causes), "quality": round(worst, 3)})
            )
        for box in _hairs(photo_metal, mask_bool, settings):
            x, y, w, h = box
            bad[y:y + h, x:x + w] = True
            findings.append(
                make_finding(BAD_PHOTO_KIND, DETECTOR_PHOTO, 0.8, _relative(tuple(float(v) for v in box), shape), diagnosis=DIAG_BAD_DATA,
                             detector_version=PHOTO_CHECK_VERSION, evidence={"cause": "hair"})
            )
        if np.any(bad) and not np.all(bad):
            # What lies next to a poor place is read with it: a crater field's rim reaches past its tile.
            reach = max(3, int(min(shape) / tiles / 4))
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
    return PhotoQuality(
        shape=shape,
        quality=quality.reshape(tiles, tiles),
        causes=causes,
        metal_brighter=metal_brighter,
        edge_width_px=frame_edge_width,
        snr=frame_snr,
        photo_metal=photo_metal,
        bad=bad,
        shift=(dx, dy),
        misaligned=bool(misaligned),
        findings=tuple(findings),
    )


def _hairs(photo_metal: np.ndarray, mask: np.ndarray, settings: PhotoCheckSettings) -> list[tuple[int, int, int, int]]:
    """Boxes of hairs and scratches: thin pieces of the photo's metal that the mask has not, crossing
    the substrate rather than running along the mask's edges, that line up into something long.

    A hair over a layer shows between the tracks as short thin strands; one strand is not enough.
    """

    extra = (photo_metal & ~mask).astype(np.uint8)
    if not np.any(extra):
        return []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(extra, connectivity=8)
    depth = cv2.distanceTransform(extra, cv2.DIST_L2, 3)
    touching = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    strands = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        if max(w, h) < 6:
            continue
        inside = labels[y:y + h, x:x + w] == label
        if float(depth[y:y + h, x:x + w][inside].max()) > settings.hair_max_half_width:
            continue
        # Along a mask edge (the mask's edge a pixel or two off) almost every pixel touches the mask.
        if float(np.mean(touching[y:y + h, x:x + w][inside])) > 0.5:
            continue
        strands[y:y + h, x:x + w][inside] = 1
    if not np.any(strands):
        return []
    reach = max(8, settings.hair_min_length_px // 4)
    chained = cv2.dilate(strands, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8))
    groups, group_labels, group_stats, _centroids = cv2.connectedComponentsWithStats(chained, connectivity=8)
    boxes = []
    for group in range(1, groups):
        own = (group_labels == group) & (strands > 0)
        ys, xs = np.nonzero(own)
        if ys.size == 0:
            continue
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if max(w, h) < 1.5 * settings.hair_min_length_px:
            continue
        pieces = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)[0] - 1
        if pieces < 2 and max(w, h) < 3 * settings.hair_min_length_px:
            continue
        boxes.append((int(xs.min()), int(ys.min()), w, h))
    return boxes


def _typical_width(mask: np.ndarray) -> float:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[mask]
    return float(2.0 * np.percentile(values, 75)) if values.size else 1.0


def _is_particle(finding: Finding, mask: np.ndarray, width: float) -> bool:
    """A small compact blob away from every edge of the mask: dirt on the photo, not a missed object."""

    if finding.kind not in {"missing_object", "missing_point", "bulge"} or finding.box is None:
        return False
    height, image_width = mask.shape
    x, y, w, h = finding.pixel_box(mask.shape)
    if max(w, h) > 3.0 * max(width, 4.0):
        return False
    x0, y0 = max(0, int(x) - 2), max(0, int(y) - 2)
    x1, y1 = min(image_width, int(math.ceil(x + w)) + 2), min(height, int(math.ceil(y + h)) + 2)
    window = mask[y0:y1, x0:x1]
    if finding.kind == "bulge":
        # A pit in the metal: the mask is metal all around it.
        return bool(window.all())
    # A speck on the substrate: no mask metal anywhere near it.
    return not bool(window.any())


def check_against_photo(
    mask: np.ndarray,
    quality: PhotoQuality,
    *,
    model_id: str,
    uncertainty: np.ndarray | None = None,
    point: bool = False,
    settings: PhotoCheckSettings | None = None,
) -> tuple[Finding, ...]:
    """Findings about one model's mask against the photo (whose quality is already read)."""

    settings = settings or PhotoCheckSettings()
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != quality.shape:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (quality.shape[1], quality.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if quality.misaligned:
        dx, dy = quality.shift
        return (
            make_finding(NOT_ALIGNED_KIND, DETECTOR_PHOTO, 1.0, None, diagnosis=DIAG_NOT_ALIGNED, model_id=model_id,
                         detector_version=PHOTO_CHECK_VERSION, evidence={"dx": round(dx, 2), "dy": round(dy, 2)}),
        )
    raw = compare_to_reference(
        mask_bool,
        quality.photo_metal,
        model_id=model_id,
        reference_id=PHOTO_REFERENCE_ID,
        point=point,
        diagnosis=DIAG_MODEL_ERROR,
        settings=settings.topology,
    )
    width = _typical_width(mask_bool)
    findings: list[Finding] = []
    particles = 0
    absorbed = 0
    for finding in raw:
        box = finding.pixel_box(quality.shape)
        if box is not None:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = max(x0 + 1, int(math.ceil(box[0] + box[2]))), max(y0 + 1, int(math.ceil(box[1] + box[3])))
            if np.any(quality.bad[y0:y1, x0:x1]):
                absorbed += 1  # the photo is poor here: its bad_photo finding speaks for it
                continue
        elif bool(quality.bad.all()):
            absorbed += 1
            continue
        if not point and _is_particle(finding, mask_bool, width):
            particles += 1
            findings.append(
                make_finding(BAD_PHOTO_KIND, DETECTOR_PHOTO, 0.5, finding.box, diagnosis=DIAG_BAD_DATA, model_id=model_id,
                             detector_version=PHOTO_CHECK_VERSION, evidence={"cause": "particle"})
            )
            continue
        diagnosis = DIAG_MODEL_ERROR
        evidence = dict(finding.evidence)
        if uncertainty is not None and box is not None:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(math.ceil(box[0] + box[2])), int(math.ceil(box[1] + box[3]))
            region = uncertainty[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            if region.size:
                # The surest quarter of the place: its edges are unsure for every model.
                unsure = float(np.percentile(region, 25))
                evidence["uncertainty"] = round(unsure, 3)
                if unsure < settings.confident_uncertainty:
                    diagnosis = DIAG_CONFIDENT_ERROR
        findings.append(
            make_finding(finding.kind, DETECTOR_PHOTO, finding.severity, finding.box, diagnosis=diagnosis, model_id=model_id,
                         detector_version=PHOTO_CHECK_VERSION, evidence=evidence)
        )
    return sort_findings(findings)


def confidence_against_photo(
    mask: np.ndarray,
    quality: PhotoQuality,
    uncertainty: np.ndarray,
    *,
    model_id: str,
    settings: PhotoCheckSettings | None = None,
) -> tuple[tuple[Finding, ...], dict[str, float]]:
    """Where a model is unsure though right, and how well its confidence is calibrated.

    The photo stands in for ground truth on its good tiles: the model is "right" where its mask and
    the photo's metal agree. With real ground truth the same reading applies to it.
    """

    settings = settings or PhotoCheckSettings()
    mask_bool = np.asarray(mask, dtype=bool)
    shape = quality.shape
    if mask_bool.shape != shape or uncertainty.shape != shape or quality.misaligned:
        return (), {}
    good = ~quality.bad
    agree = mask_bool == quality.photo_metal
    far = _far_from(_edges(mask_bool), 4.0) & _far_from(_edges(quality.photo_metal), 4.0)
    unsure = (uncertainty > settings.unsure_flat_uncertainty) & far & agree & good
    findings: list[Finding] = []
    flat_right = far & agree & good
    if np.count_nonzero(flat_right) and np.count_nonzero(unsure) > 0.5 * np.count_nonzero(flat_right):
        # Unsure almost everywhere it is right: one finding for the frame (an underconfident model).
        findings.append(
            make_finding(
                UNCERTAIN_REGION_KIND,
                DETECTOR_PHOTO,
                float(np.mean(uncertainty[flat_right])),
                None,
                diagnosis=DIAG_UNCERTAIN,
                model_id=model_id,
                detector_version=PHOTO_CHECK_VERSION,
                evidence={"share": round(float(np.count_nonzero(unsure)) / np.count_nonzero(flat_right), 3)},
            )
        )
        unsure = np.zeros_like(unsure)
    if np.any(unsure):
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(unsure.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < settings.unsure_flat_min_area:
                continue
            box = tuple(
                float(stats[label, key])
                for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
            )
            mean = float(uncertainty[labels == label].mean())
            findings.append(
                make_finding(
                    UNCERTAIN_REGION_KIND,
                    DETECTOR_PHOTO,
                    mean,
                    _relative(box, shape),
                    diagnosis=DIAG_UNCERTAIN,
                    model_id=model_id,
                    detector_version=PHOTO_CHECK_VERSION,
                    evidence={"area_px": float(area)},
                )
            )
    metrics: dict[str, float] = {}
    if np.any(good):
        confidence = 1.0 - uncertainty[good]
        # The chance the predicted class is right, as the map claims it (0.5 .. 1).
        claimed = 0.5 + 0.5 * confidence
        right = agree[good].astype(np.float32)
        bins = np.minimum((confidence * settings.calibration_bins).astype(np.int32), settings.calibration_bins - 1)
        counts = np.bincount(bins, minlength=settings.calibration_bins).astype(np.float64)
        claimed_sum = np.bincount(bins, weights=claimed, minlength=settings.calibration_bins)
        right_sum = np.bincount(bins, weights=right, minlength=settings.calibration_bins)
        used = counts > 0
        gap = np.abs(claimed_sum[used] / counts[used] - right_sum[used] / counts[used])
        metrics[f"{CONFIDENCE_ECE_METRIC}::{model_id}"] = float(100.0 * np.sum(gap * counts[used]) / max(1.0, counts.sum()))
        wrong = ~agree & good
        if np.count_nonzero(wrong) >= 20:
            sure = uncertainty[wrong] < settings.confident_uncertainty
            metrics[f"{CONFIDENT_ERROR_SHARE_METRIC}::{model_id}"] = float(100.0 * np.mean(sure))
    return sort_findings(findings), metrics


def photo_metrics(quality: PhotoQuality | None, findings_by_model: dict[str, tuple[Finding, ...]]) -> dict[str, float]:
    """Frame metrics: how poor the photo is, and how much each model disagrees with it (0-100)."""

    values: dict[str, float] = {}
    if quality is not None:
        values[PHOTO_QUALITY_RISK_METRIC] = quality.quality_risk
    worst = 0.0
    for model_id, findings in findings_by_model.items():
        weight = sum(item.severity for item in findings if item.diagnosis in {DIAG_MODEL_ERROR, DIAG_CONFIDENT_ERROR, DIAG_NOT_ALIGNED})
        risk = 100.0 * (1.0 - math.exp(-weight / 2.0))
        values[f"{PHOTO_MISMATCH_RISK_METRIC}::{model_id}"] = risk
        worst = max(worst, risk)
    if findings_by_model:
        values[PHOTO_MISMATCH_RISK_METRIC] = worst
    return values


def photo_findings(
    photo: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    confidences: dict[str, np.ndarray] | None = None,
    encoding: str = "auto",
    point: bool = False,
    settings: PhotoCheckSettings | None = None,
) -> tuple[tuple[Finding, ...], dict[str, float]]:
    """Every finding of the photo check of a frame and its metrics (see ``photo_check``)."""

    findings, metrics, _quality = photo_check(
        photo, masks, confidences=confidences, encoding=encoding, point=point, settings=settings
    )
    return findings, metrics


def photo_check(
    photo: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    confidences: dict[str, np.ndarray] | None = None,
    encoding: str = "auto",
    point: bool = False,
    settings: PhotoCheckSettings | None = None,
) -> tuple[tuple[Finding, ...], dict[str, float], PhotoQuality | None]:
    """Every finding of the photo check of a frame, its metrics and how the photo reads.

    The photo is read once, against the majority of the models' masks (the mask itself when there
    is one); every model's mask is then checked against it. ``confidences``: the models' confidence
    maps (0..1 or 0..255) written as ``encoding`` (``auto`` reads it from each map).
    """

    settings = settings or PhotoCheckSettings()
    items = {str(key): np.asarray(value, dtype=bool) for key, value in masks.items() if value is not None}
    if not items:
        return (), {}, None
    shape = next(iter(items.values())).shape
    items = {key: value for key, value in items.items() if value.shape == shape}
    votes = np.zeros(shape, dtype=np.uint16)
    for value in items.values():
        votes += value
    majority = votes * 2 > len(items) if len(items) > 1 else next(iter(items.values()))
    quality = assess_photo(photo, majority, settings)
    uncertainties: dict[str, np.ndarray] = {}
    encodings: dict[str, str] = {}
    for model_id, raw in (confidences or {}).items():
        if raw is None or model_id not in items:
            continue
        values = normalize_model_confidence(raw)
        if values.shape != shape:
            values = cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        encodings[model_id] = resolve_confidence_encoding(encoding, values, items[model_id])
        uncertainties[model_id] = build_model_uncertainty(values, encodings[model_id])
    by_model = {
        model_id: check_against_photo(
            mask, quality, model_id=model_id, uncertainty=uncertainties.get(model_id), point=point, settings=settings
        )
        for model_id, mask in items.items()
    }
    findings = list(quality.findings)
    for model_findings in by_model.values():
        findings.extend(model_findings)
    metrics = photo_metrics(quality, by_model)
    for model_id, uncertainty in uncertainties.items():
        unsure, calibration = confidence_against_photo(
            items[model_id], quality, uncertainty, model_id=model_id, settings=settings
        )
        findings.extend(unsure)
        metrics.update(calibration)
        metrics[f"confidence_encoding_is_confidence::{model_id}"] = 1.0 if encodings[model_id] == "confidence" else 0.0
    return sort_findings(findings), metrics, quality


__all__ = [
    "BAD_PHOTO_KIND",
    "DETECTOR_PHOTO",
    "NOT_ALIGNED_KIND",
    "PHOTO_CHECK_VERSION",
    "PHOTO_MISMATCH_RISK_METRIC",
    "PHOTO_QUALITY_RISK_METRIC",
    "PHOTO_REFERENCE_ID",
    "PhotoCheckSettings",
    "PhotoQuality",
    "CONFIDENCE_ECE_METRIC",
    "CONFIDENT_ERROR_SHARE_METRIC",
    "UNCERTAIN_REGION_KIND",
    "assess_photo",
    "confidence_against_photo",
    "check_against_photo",
    "photo_check",
    "photo_findings",
    "photo_metrics",
]

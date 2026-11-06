"""Disagreement of two networks that paint the same cells (ones and ones, zeros and zeros).

Both networks were asked for the same thing, so a cell only one of them found is a place to check.
Only cells count: what a network painted that is no cell (dirt, conductor grain) is left out on both
sides, by the bank of normal cells of its own layer. Comparing pixels would not do: two networks
never draw a contour alike, and the fringe around every cell would bury the real differences.

A cell of one network is found by the other when the other mask covers enough of it. That mask is
taken whole: a cell the other network drew raggedly, too ragged to pass as a cell, is still found.

Most frames have no disagreement, so the cheap part comes first: connected components of both masks
and how much of each piece the other mask covers. Only the pieces that are not covered go to the
bank, and frames without such pieces never reach it.

The run keeps every cell covered below :data:`AGREEMENT_RUN_MAX_COVER` with its cover. The operator's
limit only hides cells afterwards (:func:`filter_network_disagreement`), so changing it needs no new
run. Both masks are binarised by the same policy as the cell analysis of a layer.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import grid_anomaly
from .cache_utils import atomic_pickle_dump
from .cell_components import CellComponent, extract_components
from .cell_error_analysis import CellErrorConfig, cells_among
from .grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from .mask_normalization import NormalizedMask, normalize_mask
from .normal_bank import NormalBank
from .pair_conflict import PAIR_CONFLICT_LAYER, _empty_result, _recount

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None

_LOGGER = logging.getLogger(__name__)

AGREEMENT_REASON = "network_only"
AGREEMENT_ALGORITHM_VERSION = "pair_agreement.v1"
# Display default: a cell is found when the other mask covers this share of it.
AGREEMENT_MAX_COVER = 0.30
# The run keeps cells covered below this share; the display limit never goes above it.
AGREEMENT_RUN_MAX_COVER = 0.60
AGREEMENT_COVER_FEATURE = "agreement_cover"
AGREEMENT_AREA_FEATURE = "agreement_area_px"
# 0.0: only the first network has the cell, 1.0: only the second one.
AGREEMENT_SIDE_FEATURE = "agreement_side"
# A cell keeps the sides of a cell and at most this many shape limits from the normal shape. The
# looser cell-likeness of the defect search goes by area alone, and a round blob of a cell's area
# passes it (its shape is compared with the size taken out).
AGREEMENT_SHAPE_FACTOR = 2.0
# Pieces below this share of the normal cell are not worth asking the bank about.
_MIN_CANDIDATE_AREA = 0.15


@dataclass(frozen=True, slots=True)
class AgreementThresholds:
    max_cover: float = AGREEMENT_MAX_COVER

    def normalized(self) -> "AgreementThresholds":
        return AgreementThresholds(max_cover=float(np.clip(float(self.max_cover), 0.0, AGREEMENT_RUN_MAX_COVER)))


def _pieces(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(count), labels, stats


def _uncovered(count: int, labels: np.ndarray, stats: np.ndarray, other: np.ndarray, min_area: float) -> dict[int, float]:
    """{label: share of the piece the other mask covers} of the pieces it does not cover enough."""

    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    covered = np.bincount(labels[other], minlength=count).astype(np.float64)
    cover = covered / np.maximum(area, 1.0)
    found = np.flatnonzero((area >= min_area) & (cover < AGREEMENT_RUN_MAX_COVER))
    return {int(label): float(cover[label]) for label in found if label != 0}


def _finding(index: int, item: CellComponent, cover: float, side: int) -> GridCellAnalysisResult:
    return GridCellAnalysisResult(
        row=index,
        col=0,
        bbox=tuple(int(value) for value in item.bbox),
        centroid=(float(item.centroid[0]), float(item.centroid[1])),
        contour_id=int(item.label),
        status="broken",
        score=float(min(1.0, 0.7 + 0.3 * (1.0 - cover))),
        reasons=(AGREEMENT_REASON,),
        feature_snapshot=(
            (AGREEMENT_COVER_FEATURE, float(cover)),
            (AGREEMENT_AREA_FEATURE, float(item.area)),
            (AGREEMENT_SIDE_FEATURE, float(side)),
        ),
        outline=tuple((int(x), int(y)) for x, y in np.asarray(item.contour).reshape(-1, 2)),
    )


def _cells_without_partner(
    image: np.ndarray,
    normalized: NormalizedMask,
    candidates: dict[int, float],
    bank: NormalBank,
    config: CellErrorConfig,
) -> list[tuple[CellComponent, float]]:
    components = extract_components(image, normalized=normalized)
    cells = cells_among(components, bank, tuple(candidates), shape_factor=AGREEMENT_SHAPE_FACTOR, config=config)
    by_label = components.by_label()
    return [(by_label[label], candidates[label]) for label in sorted(cells)]


def analyze_network_disagreement(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_bank: NormalBank,
    second_bank: NormalBank,
    *,
    frame_id: str = "",
    frame_path: str = "",
    config: CellErrorConfig | None = None,
) -> GridFrameAnalysisResult:
    """Every cell of one network the other mask does not cover, with that cover.

    A mask with nothing on it is the plainest disagreement: the cells of the other one all count.
    """

    cfg = config or CellErrorConfig()
    first_mask = normalize_mask(first_image, cfg.normalization)
    second_mask = normalize_mask(second_image, cfg.normalization)
    first = np.asarray(first_mask.binary, dtype=bool)
    second = np.asarray(second_mask.binary, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Masks of a pair have different shapes: {first.shape} vs {second.shape}")
    height, width = first.shape
    if cv2 is None:
        return _empty_result(frame_id, frame_path, width, height)

    first_count, first_labels, first_stats = _pieces(first)
    second_count, second_labels, second_stats = _pieces(second)
    pieces = max(0, first_count - 1) + max(0, second_count - 1)
    first_open = _uncovered(
        first_count, first_labels, first_stats, second, _MIN_CANDIDATE_AREA * float(first_bank.cell_area)
    )
    second_open = _uncovered(
        second_count, second_labels, second_stats, first, _MIN_CANDIDATE_AREA * float(second_bank.cell_area)
    )
    found: list[tuple[CellComponent, float, int]] = []
    if first_open:
        found.extend(
            (item, cover, 0)
            for item, cover in _cells_without_partner(first_image, first_mask, first_open, first_bank, cfg)
        )
    if second_open:
        found.extend(
            (item, cover, 1)
            for item, cover in _cells_without_partner(second_image, second_mask, second_open, second_bank, cfg)
        )
    empty = _empty_result(frame_id, frame_path, width, height, pieces)
    if not found:
        return empty
    return _recount(empty, [_finding(index, *entry) for index, entry in enumerate(found)])


def agreement_features(cell: object) -> dict[str, float]:
    return {str(key): float(value) for key, value in (getattr(cell, "feature_snapshot", ()) or ())}


def agreement_side(cell: object) -> int:
    """0 when only the first network has the cell, 1 when only the second one."""

    return int(agreement_features(cell).get(AGREEMENT_SIDE_FEATURE, 0.0))


def filter_network_disagreement(
    result: GridFrameAnalysisResult | None,
    thresholds: AgreementThresholds,
    *,
    labels: tuple[str, str] = ("", ""),
) -> GridFrameAnalysisResult | None:
    """Keep the cells covered below the limit and recount the frame score.

    ``labels`` says on every kept cell which network has it (shown in lists and tooltips).
    """

    if not isinstance(result, GridFrameAnalysisResult):
        return result
    limit = thresholds.normalized().max_cover
    kept: list[GridCellAnalysisResult] = []
    for cell in result.per_cell_results or ():
        if agreement_features(cell).get(AGREEMENT_COVER_FEATURE, 0.0) >= limit:
            continue
        label = labels[agreement_side(cell)]
        kept.append(replace(cell, feature_cluster_label=label) if label else cell)
    return _recount(result, kept)


def _cache_key(
    path_a: Path, path_b: Path, frame_id: str, first_bank: NormalBank, second_bank: NormalBank, conductor_zone: bool
) -> tuple[object, ...]:
    return (
        AGREEMENT_ALGORITHM_VERSION,
        str(frame_id or ""),
        grid_anomaly._grid_cache_identity(path_a),
        grid_anomaly._grid_cache_identity(path_b),
        str(first_bank.bank_id),
        str(second_bank.bank_id),
        bool(conductor_zone),
    )


def analyze_network_disagreement_paths(
    first_path: Path | str,
    second_path: Path | str,
    first_bank: NormalBank,
    second_bank: NormalBank,
    *,
    frame_id: str = "",
    use_cache: bool = True,
    conductor_zone: bool = True,
) -> GridFrameAnalysisResult | None:
    """Load the two masks of a frame and find the cells only one network has."""

    path_a = Path(first_path)
    path_b = Path(second_path)
    cache_path = grid_anomaly._grid_damage_cache_path(
        _cache_key(path_a, path_b, frame_id, first_bank, second_bank, conductor_zone)
    )
    if use_cache and cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, GridFrameAnalysisResult):
                return payload
        except Exception as error:
            _LOGGER.warning("Ignoring corrupt network disagreement cache entry %s: %s", cache_path, error)
    image_a = grid_anomaly._load_cv2_grayscale_image(path_a)
    image_b = grid_anomaly._load_cv2_grayscale_image(path_b)
    if image_a is None or image_b is None:
        return None
    if image_a.shape != image_b.shape:
        image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = analyze_network_disagreement(
        image_a,
        image_b,
        first_bank,
        second_bank,
        frame_id=frame_id,
        frame_path=f"{path_a.name}|{path_b.name}",
        config=CellErrorConfig(conductor_zone=bool(conductor_zone)),
    )
    if use_cache:
        try:
            grid_anomaly.GRID_DAMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            atomic_pickle_dump(cache_path, result)
        except (OSError, pickle.PickleError, TypeError, ValueError) as error:
            _LOGGER.warning("Could not store network disagreement cache for %s: %s", frame_id, error)
    return result


def analyze_network_disagreement_chunk(
    entries: tuple[tuple[str, str, str], ...],
    *,
    first_bank: NormalBank,
    second_bank: NormalBank,
    use_cache: bool = True,
    conductor_zone: bool = True,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Disagreement of (frame key, first mask path, second mask path) entries."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    for key, first_path, second_path in entries:
        try:
            result = analyze_network_disagreement_paths(
                first_path,
                second_path,
                first_bank,
                second_bank,
                frame_id=key,
                use_cache=use_cache,
                conductor_zone=conductor_zone,
            )
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if result is None:
            errors[str(key)] = "decode_error"
            continue
        payloads[str(key)] = {PAIR_CONFLICT_LAYER: result}
    return payloads, errors

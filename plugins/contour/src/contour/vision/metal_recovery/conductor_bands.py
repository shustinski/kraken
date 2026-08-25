"""Transverse conductor-band inference from SEM structural evidence.

A ridge is a structural cue, not an instance.  A conductor band is the interval
between a persistent left/right boundary pair along the local normal.  Multiple
ridges inside that interval stay one logical seed unless a separating boundary
or valley splits them.

Ground truth, frame ids, and filenames are never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...utils import ensure_binary_mask

_MAX_HALF_EXTENT_PX = 18.0
_PROFILE_SAMPLES = 37
_LONGITUDINAL_OFFSETS = (-4.0, -2.0, 0.0, 2.0, 4.0)
_MIN_BOUNDARY_SCORE = 0.28
_MIN_BAND_WIDTH_PX = 2.5
_SEPARATOR_RELATIVE = 0.35


@dataclass(frozen=True, slots=True)
class BandEvidence:
    intensity: np.ndarray
    ridge_confidence: np.ndarray
    ridge_orientation: np.ndarray
    structure_orientation: np.ndarray
    coherence: np.ndarray
    persistent_edge: np.ndarray
    magnitude: np.ndarray
    rim_response: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray


@dataclass(frozen=True, slots=True)
class TransverseProfile:
    offsets: np.ndarray
    intensity: np.ndarray
    magnitude: np.ndarray
    ridge: np.ndarray
    rim: np.ndarray
    boundary: np.ndarray
    crossing: np.ndarray
    coherence: np.ndarray
    origin_row: float
    origin_col: float
    tangent_x: float
    tangent_y: float
    normal_x: float
    normal_y: float


@dataclass(frozen=True, slots=True)
class BoundaryPair:
    left_offset: float
    right_offset: float
    left_score: float
    right_score: float
    width: float
    confidence: float
    boundary_pair_confidence: float = 0.0
    interior_consistency: float = 0.0
    ridge_support: float = 0.0
    orientation_coherence: float = 0.0
    longitudinal_persistence: float = 0.0
    separator_absence: float = 0.0


@dataclass(frozen=True, slots=True)
class BandInferenceStats:
    raw_ridge_count: int = 0
    band_count: int = 0
    grouped_ridge_count: int = 0
    accepted_band_groups: int = 0
    rejected_band_groups: int = 0
    wide_kept_with_multi_ridge: int = 0
    wide_dropped_for_separator: int = 0


def evidence_from_consolidation(source: object) -> BandEvidence:
    return BandEvidence(
        intensity=np.asarray(getattr(source, "intensity")),
        ridge_confidence=np.asarray(getattr(source, "ridge_confidence")),
        ridge_orientation=np.asarray(getattr(source, "ridge_orientation")),
        structure_orientation=np.asarray(getattr(source, "structure_orientation")),
        coherence=np.asarray(getattr(source, "coherence")),
        persistent_edge=np.asarray(getattr(source, "persistent_edge")),
        magnitude=np.asarray(getattr(source, "magnitude")),
        rim_response=np.asarray(getattr(source, "rim_response")),
        gradient_x=np.asarray(getattr(source, "gradient_x")),
        gradient_y=np.asarray(getattr(source, "gradient_y")),
    )


def sample_transverse_profile(
    evidence: BandEvidence,
    row: float,
    col: float,
    tangent_x: float,
    tangent_y: float,
    *,
    half_extent: float = _MAX_HALF_EXTENT_PX,
    packed: np.ndarray | None = None,
) -> TransverseProfile:
    norm = max(float(np.hypot(tangent_x, tangent_y)), 1e-6)
    tx = float(tangent_x) / norm
    ty = float(tangent_y) / norm
    nx = -ty
    ny = tx
    offsets = np.linspace(-half_extent, half_extent, _PROFILE_SAMPLES).astype(np.float32)
    source = packed if packed is not None else _pack_evidence(evidence)
    lines: list[np.ndarray] = []
    for along in _LONGITUDINAL_OFFSETS:
        xs = col + along * tx + offsets * nx
        ys = row + along * ty + offsets * ny
        lines.append(_remap_line(source, ys, xs))
    median = np.median(np.stack(lines, axis=0), axis=0).astype(np.float32)
    gx = median[:, 6]
    gy = median[:, 7]
    return TransverseProfile(
        offsets=offsets,
        intensity=median[:, 0],
        magnitude=median[:, 1],
        ridge=median[:, 2],
        rim=median[:, 3],
        boundary=median[:, 4],
        crossing=np.abs(gx * tx + gy * ty).astype(np.float32),
        coherence=median[:, 5],
        origin_row=float(row),
        origin_col=float(col),
        tangent_x=tx,
        tangent_y=ty,
        normal_x=nx,
        normal_y=ny,
    )


def _pack_evidence(evidence: BandEvidence) -> np.ndarray:
    return np.stack(
        [
            evidence.intensity.astype(np.float32),
            evidence.magnitude,
            evidence.ridge_confidence,
            evidence.rim_response,
            evidence.persistent_edge,
            evidence.coherence,
            evidence.gradient_x,
            evidence.gradient_y,
        ],
        axis=-1,
    )


def band_score(profile: TransverseProfile) -> np.ndarray:
    score = _unit(profile.boundary) * 0.45 + _unit(profile.rim) * 0.35 + _unit(profile.magnitude) * 0.20
    return (score * (0.5 + 0.5 * profile.coherence)).astype(np.float32)


def detect_boundary_pair(profile: TransverseProfile) -> BoundaryPair | None:
    """Largest separator-free interval that still contains the sample origin."""

    score = band_score(profile)
    center = int(profile.offsets.size // 2)
    peaks = _score_peaks(score)
    left_peaks = [index for index in peaks if index < center]
    right_peaks = [index for index in peaks if index > center]
    if not left_peaks or not right_peaks:
        left = _first_outer_peak(score, start=0, end=center, take_left=True)
        right = _first_outer_peak(score, start=center + 1, end=int(score.size), take_left=False)
        if left is None or right is None:
            return None
        left_peaks = [left[0]]
        right_peaks = [right[0]]
    chosen: tuple[int, int] | None = None
    ranked = sorted(
        ((left_index, right_index) for left_index in left_peaks for right_index in right_peaks),
        key=lambda pair: float(profile.offsets[pair[1]] - profile.offsets[pair[0]]),
        reverse=True,
    )
    for left_index, right_index in ranked:
        left_offset = float(profile.offsets[left_index])
        right_offset = float(profile.offsets[right_index])
        if right_offset - left_offset < _MIN_BAND_WIDTH_PX:
            continue
        if has_separating_boundary(profile, left_offset, right_offset, score=score):
            continue
        chosen = (left_index, right_index)
        break
    if chosen is None:
        left_index = left_peaks[0]
        right_index = right_peaks[-1]
        if float(profile.offsets[right_index] - profile.offsets[left_index]) < _MIN_BAND_WIDTH_PX:
            return None
        chosen = (left_index, right_index)
    left_index, right_index = chosen
    left_score = float(score[left_index])
    right_score = float(score[right_index])
    if min(left_score, right_score) < _MIN_BOUNDARY_SCORE:
        return None
    width = float(profile.offsets[right_index] - profile.offsets[left_index])
    pair = BoundaryPair(
        left_offset=float(profile.offsets[left_index]),
        right_offset=float(profile.offsets[right_index]),
        left_score=left_score,
        right_score=right_score,
        width=width,
        confidence=float(np.clip(0.5 * (left_score + right_score), 0.0, 1.0)),
    )
    return _with_confidence(profile, pair)


def has_separating_boundary(
    profile: TransverseProfile,
    left_offset: float,
    right_offset: float,
    score: np.ndarray | None = None,
) -> bool:
    """True when two ridge groups in the interval are split by a structural gap.

    An extra ridge inside one conductor is not a separator.  A separator is a
    low-ridge run between ridge groups that also has a relative valley and a
    wall that is not itself a ridge.
    """

    if right_offset - left_offset <= _MIN_BAND_WIDTH_PX:
        return False
    inner = (profile.offsets > left_offset + 1.0) & (profile.offsets < right_offset - 1.0)
    if int(np.count_nonzero(inner)) < 3:
        return False
    ridge_peak = max(float(np.max(profile.ridge)), 1e-3)
    high = inner & (profile.ridge >= 0.28 * ridge_peak)
    padded = np.concatenate(([False], high, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    if score is None:
        score = band_score(profile)
    wall = score * (1.0 - _unit(profile.ridge))
    if starts.size >= 2:
        for start, end, next_start in zip(starts, ends, starts[1:]):
            width = int(next_start) - int(end)
            if width < 4:
                continue
            trim = max(1, width // 3)
            gap = slice(int(end) + trim, int(next_start) - trim)
            if gap.stop - gap.start < 2:
                continue
            if float(np.max(wall[gap])) < _MIN_BOUNDARY_SCORE:
                continue
            left_i = float(profile.intensity[max(int(start), 0)])
            right_i = float(profile.intensity[min(int(next_start), int(profile.intensity.size) - 1)])
            gap_i = float(np.min(profile.intensity[gap]))
            local_std = max(float(np.std(profile.intensity[gap])), 4.0)
            if (0.5 * (left_i + right_i) - gap_i) >= _SEPARATOR_RELATIVE * local_std + 8.0:
                return True
        return False
    if right_offset - left_offset <= 8.5:
        return False
    if float(np.max(wall[inner])) < _MIN_BOUNDARY_SCORE:
        return False
    left_index = int(np.argmin(np.abs(profile.offsets - left_offset)))
    right_index = int(np.argmin(np.abs(profile.offsets - right_offset)))
    flank = 0.5 * (
        float(profile.intensity[left_index]) + float(profile.intensity[right_index])
    )
    gap = float(np.min(profile.intensity[inner]))
    local_std = max(float(np.std(profile.intensity[inner])), 4.0)
    return (flank - gap) >= _SEPARATOR_RELATIVE * local_std + 8.0


def region_has_internal_separator(region: np.ndarray, evidence: BandEvidence) -> bool:
    """True when an eroded region is split by a persistent interior boundary."""

    binary = ensure_binary_mask(region)
    if not np.any(binary):
        return False
    eroded = cv2.erode(binary, np.ones((5, 5), np.uint8))
    if not np.any(eroded):
        return False
    interior_edge = evidence.persistent_edge[eroded > 0]
    local = float(np.median(interior_edge))
    spread = max(float(np.percentile(interior_edge, 80.0) - local), 1e-3)
    walls = np.zeros(binary.shape, dtype=np.uint8)
    walls[eroded > 0] = np.where(
        evidence.persistent_edge[eroded > 0] >= local + 1.8 * spread,
        255,
        0,
    ).astype(np.uint8)
    if not np.any(walls):
        return False
    walls = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    open_space = np.where((eroded > 0) & (walls == 0), 255, 0).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if count <= 2:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    eroded_area = max(int(np.count_nonzero(eroded)), 1)
    large = int(np.count_nonzero(areas >= max(48, 0.12 * eroded_area)))
    return large >= 2


def infer_band_at(
    evidence: BandEvidence,
    row: float,
    col: float,
) -> tuple[TransverseProfile, BoundaryPair | None]:
    tangent_x, tangent_y = _tangent_at(evidence, row, col)
    profile = sample_transverse_profile(evidence, row, col, tangent_x, tangent_y)
    return profile, detect_boundary_pair(profile)


def group_ridges_by_conductor_band(
    ridge_labels: np.ndarray,
    evidence: BandEvidence,
) -> tuple[np.ndarray, BandInferenceStats, dict[str, np.ndarray]]:
    """Union ridge fragments that sit inside one boundary pair with no separator."""

    ids = np.unique(ridge_labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        empty = np.zeros(ridge_labels.shape, dtype=np.int32)
        return empty, BandInferenceStats(), _empty_band_debug(ridge_labels.shape)
    fragment_count = int(ids.max())
    ys, xs = np.indices(ridge_labels.shape)
    weights = (ridge_labels > 0).astype(np.float64)
    counts = np.bincount(ridge_labels.ravel(), minlength=fragment_count + 1).astype(np.float64)
    sum_x = np.bincount(ridge_labels.ravel(), weights=xs.ravel() * weights.ravel(), minlength=fragment_count + 1)
    sum_y = np.bincount(ridge_labels.ravel(), weights=ys.ravel() * weights.ravel(), minlength=fragment_count + 1)
    centroids_x = np.divide(sum_x, np.maximum(counts, 1.0))
    centroids_y = np.divide(sum_y, np.maximum(counts, 1.0))
    parent = np.arange(fragment_count + 1, dtype=np.int32)
    accepted = 0
    rejected = 0
    packed = _pack_evidence(evidence)
    base = cv2.cvtColor(
        np.clip(evidence.intensity, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    dimmed = (base.astype(np.float32) * 0.45).astype(np.uint8)
    overlay = dimmed.copy()
    samples = dimmed.copy()
    accepted_view = dimmed.copy()
    rejected_view = dimmed.copy()
    bands: list[tuple[int, BoundaryPair, TransverseProfile]] = []
    for fragment_id in ids:
        row = float(centroids_y[int(fragment_id)])
        col = float(centroids_x[int(fragment_id)])
        tangent_x, tangent_y = _tangent_at(evidence, row, col)
        profile = sample_transverse_profile(
            evidence, row, col, tangent_x, tangent_y, packed=packed
        )
        pair = detect_boundary_pair(profile)
        bands.append((int(fragment_id), pair, profile))
        if int(fragment_id) % 8 == 0:
            _draw_sample_line(samples, profile, (180, 180, 40))
        if pair is not None:
            _draw_band_tick(overlay, profile, pair, (0, 180, 255))
            _draw_band_tick(samples, profile, pair, (0, 180, 255))

    cell = 8.0
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (fragment_id, _pair, _profile) in enumerate(bands):
        key = (
            int(centroids_x[fragment_id] // cell),
            int(centroids_y[fragment_id] // cell),
        )
        buckets.setdefault(key, []).append(index)

    for index, (fragment_id, pair, profile) in enumerate(bands):
        col = int(centroids_x[fragment_id] // cell)
        row = int(centroids_y[fragment_id] // cell)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other_index in buckets.get((col + dx, row + dy), ()):
                    if other_index <= index:
                        continue
                    other_id, other_pair, other_profile = bands[other_index]
                    point_a = np.array([centroids_x[fragment_id], centroids_y[fragment_id]])
                    point_b = np.array([centroids_x[other_id], centroids_y[other_id]])
                    if _same_conductor_band(profile, pair, other_profile, other_pair):
                        if _find(parent, fragment_id) != _find(parent, other_id):
                            _union(parent, fragment_id, other_id)
                            accepted += 1
                            _draw_group_line(overlay, point_a, point_b, (0, 220, 0))
                            _draw_group_line(accepted_view, point_a, point_b, (0, 220, 0))
                    elif _near_transverse_candidate(profile, pair, other_profile):
                        rejected += 1
                        _draw_group_line(overlay, point_a, point_b, (0, 0, 220))
                        _draw_group_line(rejected_view, point_a, point_b, (0, 0, 220))

    grouped = _remap_by_parent(ridge_labels, parent)
    stats_out = BandInferenceStats(
        raw_ridge_count=int(ids.size),
        band_count=int(sum(1 for _fragment_id, pair, _profile in bands if pair is not None)),
        grouped_ridge_count=_positive_count(grouped),
        accepted_band_groups=accepted,
        rejected_band_groups=rejected,
    )
    return grouped, stats_out, {
        "metal_structural_conductor_bands": overlay,
        "metal_structural_transverse_samples": samples,
        "metal_structural_band_groups_accepted": accepted_view,
        "metal_structural_band_groups_rejected": rejected_view,
    }


def _same_conductor_band(
    profile_a: TransverseProfile,
    pair_a: BoundaryPair | None,
    profile_b: TransverseProfile,
    pair_b: BoundaryPair | None,
) -> bool:
    angle = _acute(np.arctan2(profile_a.tangent_y, profile_a.tangent_x), np.arctan2(profile_b.tangent_y, profile_b.tangent_x))
    if angle > np.deg2rad(25.0):
        return False
    dx = profile_b.origin_col - profile_a.origin_col
    dy = profile_b.origin_row - profile_a.origin_row
    across = dx * profile_a.normal_x + dy * profile_a.normal_y
    along = dx * profile_a.tangent_x + dy * profile_a.tangent_y
    spatial = float(np.hypot(dx, dy))
    if spatial >= 2.0 and abs(across) < 1.2 and abs(along) >= 2.0:
        across, along = along, across
    if abs(along) > 12.0:
        return False
    if abs(across) < 1.0:
        return False
    across_limit = 14.0
    if pair_a is not None:
        across_limit = max(pair_a.width * 0.95, 10.0)
    if pair_b is not None:
        across_limit = max(across_limit, pair_b.width * 0.95)
    if abs(across) > across_limit:
        return False
    if pair_a is not None and (pair_a.left_offset - 1.0 <= across <= pair_a.right_offset + 1.0):
        pass
    elif pair_b is not None:
        across_b = -dx * profile_b.normal_x + -dy * profile_b.normal_y
        if not (pair_b.left_offset - 1.0 <= across_b <= pair_b.right_offset + 1.0):
            if abs(across) > 10.0:
                return False
    elif abs(across) > 10.0:
        return False
    return not has_separating_boundary(profile_a, min(0.0, across), max(0.0, across))


def _near_transverse_candidate(
    profile_a: TransverseProfile,
    pair_a: BoundaryPair | None,
    profile_b: TransverseProfile,
) -> bool:
    dx = profile_b.origin_col - profile_a.origin_col
    dy = profile_b.origin_row - profile_a.origin_row
    across = dx * profile_a.normal_x + dy * profile_a.normal_y
    along = dx * profile_a.tangent_x + dy * profile_a.tangent_y
    width = pair_a.width if pair_a is not None else 14.0
    if abs(along) > max(14.0, width):
        return False
    return 1.0 <= abs(across) <= width + 4.0


def corridor_has_transverse_separator(
    evidence: BandEvidence,
    row_a: int,
    col_a: int,
    row_b: int,
    col_b: int,
    *,
    edge_veto: float,
) -> bool:
    """Veto only a boundary that crosses the corridor, not a parallel rim."""

    length = max(int(round(np.hypot(col_b - col_a, row_b - row_a))), 1)
    if length <= 3:
        return False
    rows = np.linspace(row_a, row_b, length + 1)
    cols = np.linspace(col_a, col_b, length + 1)
    dy = float(row_b - row_a)
    dx = float(col_b - col_a)
    norm = max(float(np.hypot(dx, dy)), 1e-6)
    dir_x = dx / norm
    dir_y = dy / norm
    height, width = evidence.intensity.shape
    trim = max(1, int(round(0.15 * (length + 1))))
    ys = np.clip(np.round(rows).astype(np.int32), 0, height - 1)[trim:-trim]
    xs = np.clip(np.round(cols).astype(np.int32), 0, width - 1)[trim:-trim]
    if ys.size == 0:
        return False
    gx = evidence.gradient_x[ys, xs]
    gy = evidence.gradient_y[ys, xs]
    magnitude = np.maximum(evidence.magnitude[ys, xs], 1e-3)
    crossing = np.abs(gx * dir_x + gy * dir_y) / magnitude
    parallel = np.abs(gx * (-dir_y) + gy * dir_x) / magnitude
    crossing_edge = evidence.persistent_edge[ys, xs] * crossing
    return bool(
        float(np.max(crossing_edge)) >= edge_veto
        and float(np.max(crossing)) >= 0.55
        and float(np.max(crossing)) >= float(np.median(parallel)) + 0.12
    )


def _remap_line(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear sample along a 1D polyline. Avoids cv2.remap call overhead."""

    height, width = values.shape[:2]
    rows = np.clip(rows.astype(np.float32), 0.0, float(height - 1))
    cols = np.clip(cols.astype(np.float32), 0.0, float(width - 1))
    r0 = np.floor(rows).astype(np.int32)
    c0 = np.floor(cols).astype(np.int32)
    r1 = np.minimum(r0 + 1, height - 1)
    c1 = np.minimum(c0 + 1, width - 1)
    wr = rows - r0.astype(np.float32)
    wc = cols - c0.astype(np.float32)
    if values.ndim == 2:
        top = values[r0, c0] * (1.0 - wc) + values[r0, c1] * wc
        bottom = values[r1, c0] * (1.0 - wc) + values[r1, c1] * wc
        return (top * (1.0 - wr) + bottom * wr).astype(np.float32)
    w_exp = wc[:, None]
    r_exp = wr[:, None]
    top = values[r0, c0] * (1.0 - w_exp) + values[r0, c1] * w_exp
    bottom = values[r1, c0] * (1.0 - w_exp) + values[r1, c1] * w_exp
    return (top * (1.0 - r_exp) + bottom * r_exp).astype(np.float32)


def _unit(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (5.0, 95.0)) if values.size else (0.0, 1.0)
    span = float(high - low)
    if span <= 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - float(low)) / span, 0.0, 1.0).astype(np.float32)


def _score_peaks(score: np.ndarray) -> list[int]:
    if score.size < 3:
        return []
    floor = float(np.median(score)) + 0.55 * float(np.std(score))
    threshold = max(_MIN_BOUNDARY_SCORE, floor)
    peaks: list[int] = []
    for index in range(1, int(score.size) - 1):
        value = float(score[index])
        if value >= score[index - 1] and value >= score[index + 1] and value >= threshold:
            peaks.append(index)
    return peaks


def _first_outer_peak(
    score: np.ndarray,
    *,
    start: int,
    end: int,
    take_left: bool,
) -> tuple[int, float] | None:
    if end - start < 3:
        return None
    window = score[start:end]
    floor = float(np.median(window)) + 0.45 * float(np.std(window))
    threshold = max(_MIN_BOUNDARY_SCORE, floor)
    peaks: list[int] = []
    for local in range(1, window.size - 1):
        value = float(window[local])
        if value >= window[local - 1] and value >= window[local + 1] and value >= threshold:
            peaks.append(start + local)
    if not peaks:
        return None
    index = peaks[0] if take_left else peaks[-1]
    return index, float(score[index])


def _with_confidence(
    profile: TransverseProfile,
    pair: BoundaryPair,
) -> BoundaryPair:
    interior = (profile.offsets >= pair.left_offset) & (profile.offsets <= pair.right_offset)
    if int(np.count_nonzero(interior)) < 2:
        interior = np.ones(profile.offsets.shape, dtype=bool)
    ridge_peak = max(float(np.max(profile.ridge)), 1e-3)
    intensity = profile.intensity[interior]
    return BoundaryPair(
        left_offset=pair.left_offset,
        right_offset=pair.right_offset,
        left_score=pair.left_score,
        right_score=pair.right_score,
        width=pair.width,
        confidence=pair.confidence,
        boundary_pair_confidence=pair.confidence,
        interior_consistency=float(
            np.clip(1.0 - (np.std(intensity) / max(float(np.mean(intensity)), 8.0)), 0.0, 1.0)
        ),
        ridge_support=float(np.clip(np.max(profile.ridge[interior]) / ridge_peak, 0.0, 1.0)),
        orientation_coherence=float(np.mean(profile.coherence[interior])),
        longitudinal_persistence=1.0,
        separator_absence=0.0 if has_separating_boundary(profile, pair.left_offset, pair.right_offset) else 1.0,
    )


def _empty_band_debug(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    empty = np.zeros((*shape, 3), dtype=np.uint8)
    return {
        "metal_structural_conductor_bands": empty,
        "metal_structural_transverse_samples": empty,
        "metal_structural_band_groups_accepted": empty,
        "metal_structural_band_groups_rejected": empty,
    }


def _tangent_at(evidence: BandEvidence, row: float, col: float) -> tuple[float, float]:
    height, width = evidence.structure_orientation.shape
    y = int(np.clip(round(row), 0, height - 1))
    x = int(np.clip(round(col), 0, width - 1))
    along = float(evidence.structure_orientation[y, x]) + 0.5 * np.pi
    return float(np.cos(along)), float(np.sin(along))


def _acute(first: float, second: float) -> float:
    delta = abs((first - second + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(delta)


def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = int(parent[root])
    while parent[item] != item:
        nxt = int(parent[item])
        parent[item] = root
        item = nxt
    return int(root)


def _union(parent: np.ndarray, left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _remap_by_parent(labels: np.ndarray, parent: np.ndarray) -> np.ndarray:
    roots = np.arange(parent.size, dtype=np.int32)
    for index in range(1, parent.size):
        roots[index] = _find(parent, index)
    remapped = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    remapped[positive] = roots[labels[positive]]
    ids = np.unique(remapped)
    ids = ids[ids > 0]
    if ids.size == 0:
        return remapped
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, int(ids.size) + 1, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=np.int32)
    out[positive] = lookup[remapped[positive]]
    return out


def _positive_count(labels: np.ndarray) -> int:
    if not np.any(labels):
        return 0
    ids = np.unique(labels)
    return int(np.count_nonzero(ids > 0))


def _draw_band_tick(
    overlay: np.ndarray,
    profile: TransverseProfile,
    pair: BoundaryPair,
    color: tuple[int, int, int],
) -> None:
    left = (
        int(round(profile.origin_col + pair.left_offset * profile.normal_x)),
        int(round(profile.origin_row + pair.left_offset * profile.normal_y)),
    )
    right = (
        int(round(profile.origin_col + pair.right_offset * profile.normal_x)),
        int(round(profile.origin_row + pair.right_offset * profile.normal_y)),
    )
    cv2.line(overlay, left, right, color, 1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, left, 2, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, right, 2, color, -1, lineType=cv2.LINE_AA)


def _draw_sample_line(
    overlay: np.ndarray,
    profile: TransverseProfile,
    color: tuple[int, int, int],
) -> None:
    start = (
        int(round(profile.origin_col + float(profile.offsets[0]) * profile.normal_x)),
        int(round(profile.origin_row + float(profile.offsets[0]) * profile.normal_y)),
    )
    end = (
        int(round(profile.origin_col + float(profile.offsets[-1]) * profile.normal_x)),
        int(round(profile.origin_row + float(profile.offsets[-1]) * profile.normal_y)),
    )
    cv2.line(overlay, start, end, color, 1, lineType=cv2.LINE_AA)


def _draw_group_line(
    overlay: np.ndarray,
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    a = (int(round(centroid_a[0])), int(round(centroid_a[1])))
    b = (int(round(centroid_b[0])), int(round(centroid_b[1])))
    cv2.line(overlay, a, b, color, 1, lineType=cv2.LINE_AA)

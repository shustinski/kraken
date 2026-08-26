"""Overlap crops and FFT phase-correlation candidate generation (NumPy FFT, no OpenCV types)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cartograph.domain.coordinates import Translation2D
from cartograph.domain.registration import PhasePeak


Gray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class OverlapCrops:
    fixed: Gray
    moving: Gray
    integer_expected: Translation2D


def extract_overlap_crops(
    fixed: Gray,
    moving: Gray,
    expected: Translation2D,
    *,
    margin_px: int,
    min_overlap_px: int,
) -> OverlapCrops | None:
    """Crop the same world region from both tiles.

    Crop origins stay aligned with the rounded expected displacement, so a
    residual of (0, 0) means the expected shift was correct. Search uncertainty
    is applied by expanding the intersection and by masking peaks.
    """

    fixed_h, fixed_w = int(fixed.shape[0]), int(fixed.shape[1])
    moving_h, moving_w = int(moving.shape[0]), int(moving.shape[1])
    edx = int(round(expected.dx))
    edy = int(round(expected.dy))
    margin = max(0, int(margin_px))

    fx0 = max(0, edx) - margin
    fy0 = max(0, edy) - margin
    fx1 = min(fixed_w, edx + moving_w) + margin
    fy1 = min(fixed_h, edy + moving_h) + margin
    mx0 = fx0 - edx
    my0 = fy0 - edy
    mx1 = fx1 - edx
    my1 = fy1 - edy

    fx0, mx0, fx1, mx1 = _clip_aligned_interval(fx0, mx0, fx1, mx1, fixed_w, moving_w)
    fy0, my0, fy1, my1 = _clip_aligned_interval(fy0, my0, fy1, my1, fixed_h, moving_h)

    width = min(fx1 - fx0, mx1 - mx0)
    height = min(fy1 - fy0, my1 - my0)
    if width < min_overlap_px or height < min_overlap_px:
        return None
    return OverlapCrops(
        fixed=np.ascontiguousarray(fixed[fy0 : fy0 + height, fx0 : fx0 + width], dtype=np.float32),
        moving=np.ascontiguousarray(moving[my0 : my0 + height, mx0 : mx0 + width], dtype=np.float32),
        integer_expected=Translation2D(float(edx), float(edy)),
    )


def phase_correlation_map(fixed: Gray, moving: Gray) -> NDArray[np.float64]:
    window = _hanning_2d(fixed.shape)
    f_fft = np.fft.fft2(np.asarray(fixed, dtype=np.float64) * window)
    m_fft = np.fft.fft2(np.asarray(moving, dtype=np.float64) * window)
    cross = f_fft * np.conj(m_fft)
    magnitude = np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross / magnitude).real
    return np.asarray(correlation, dtype=np.float64)


def top_k_peaks(
    correlation: NDArray[np.float64],
    *,
    top_k: int,
    search_radius_px: float,
    suppression_radius: int = 2,
) -> tuple[PhasePeak, ...]:
    height, width = correlation.shape
    allowed = _shift_mask(height, width, search_radius_px)
    masked = np.where(allowed, correlation, -np.inf)
    order = np.argsort(masked, axis=None)[::-1]
    selected: list[tuple[int, int, float]] = []
    taken = np.zeros((height, width), dtype=bool)
    for flat in order:
        row, col = divmod(int(flat), width)
        if not np.isfinite(masked[row, col]) or taken[row, col]:
            continue
        selected.append((row, col, float(correlation[row, col])))
        _suppress(taken, row, col, suppression_radius)
        if len(selected) >= top_k:
            break
    if not selected:
        return ()
    first_value = selected[0][2]
    second_value = selected[1][2] if len(selected) > 1 else 0.0
    ratio = first_value / max(abs(second_value), 1e-12)
    peaks: list[PhasePeak] = []
    for row, col, value in selected:
        dx, dy = _index_to_shift(row, col, height, width)
        dx, dy = _refine_subpixel(correlation, row, col, dx, dy)
        peaks.append(
            PhasePeak(
                translation=Translation2D(dx, dy),
                phase_response=float(max(0.0, value)),
                peak_value=first_value,
                second_peak_value=second_value,
                peak_ratio=float(ratio),
            )
        )
    return tuple(peaks)


def _clip_aligned_interval(
    fixed0: int,
    moving0: int,
    fixed1: int,
    moving1: int,
    fixed_limit: int,
    moving_limit: int,
) -> tuple[int, int, int, int]:
    if fixed0 < 0:
        moving0 -= fixed0
        fixed0 = 0
    if moving0 < 0:
        fixed0 -= moving0
        moving0 = 0
    if fixed1 > fixed_limit:
        moving1 -= fixed1 - fixed_limit
        fixed1 = fixed_limit
    if moving1 > moving_limit:
        fixed1 -= moving1 - moving_limit
        moving1 = moving_limit
    return fixed0, moving0, fixed1, moving1


def _index_to_shift(row: int, col: int, height: int, width: int) -> tuple[float, float]:
    dy = float(row if row <= height // 2 else row - height)
    dx = float(col if col <= width // 2 else col - width)
    return dx, dy


def _refine_subpixel(
    correlation: NDArray[np.float64],
    row: int,
    col: int,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    height, width = correlation.shape

    def sample(r: int, c: int) -> float:
        return float(correlation[r % height, c % width])

    dx += _parabolic_offset(sample(row, col - 1), sample(row, col), sample(row, col + 1))
    dy += _parabolic_offset(sample(row - 1, col), sample(row, col), sample(row + 1, col))
    return dx, dy


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denom = 2.0 * center - left - right
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def _shift_mask(height: int, width: int, radius: float) -> NDArray[np.bool_]:
    rows = np.arange(height, dtype=np.int32)[:, None]
    cols = np.arange(width, dtype=np.int32)[None, :]
    dy = np.where(rows <= height // 2, rows, rows - height)
    dx = np.where(cols <= width // 2, cols, cols - width)
    return (np.abs(dx) <= radius) & (np.abs(dy) <= radius)


def _suppress(taken: NDArray[np.bool_], row: int, col: int, radius: int) -> None:
    height, width = taken.shape
    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            taken[(row + d_row) % height, (col + d_col) % width] = True


def _hanning_2d(shape: tuple[int, ...]) -> NDArray[np.float64]:
    height, width = int(shape[0]), int(shape[1])
    window_y = np.hanning(height).astype(np.float64)
    window_x = np.hanning(width).astype(np.float64)
    return np.outer(window_y, window_x)

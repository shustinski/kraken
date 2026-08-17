from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass
class PreprocessingConfig:
    """SEM-specific preprocessing shared by training and inference."""

    percentile_normalization: bool = False
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    illumination_correction: bool = False
    illumination_kernel_size: int = 51
    background_subtraction: bool = False
    background_blur_kernel: int = 31
    scan_line_suppression: bool = False
    scan_line_strength: float = 0.5
    scan_axis: str = 'rows'
    scan_profile_kernel: int = 31
    denoise: bool = False
    denoise_strength: float = 5.0
    operation_order: tuple[str, ...] = (
        'background_subtraction',
        'illumination_correction',
        'scan_line_suppression',
        'denoise',
        'percentile_normalization',
        'clahe',
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.percentile_low < self.percentile_high <= 100.0:
            raise ValueError('Percentile range must satisfy 0 <= low < high <= 100.')
        if self.clahe_clip_limit <= 0.0 or min(self.clahe_tile_grid_size) < 1:
            raise ValueError('CLAHE clip limit and tile grid sizes must be positive.')
        for name, value in (
            ('illumination_kernel_size', self.illumination_kernel_size),
            ('background_blur_kernel', self.background_blur_kernel),
            ('scan_profile_kernel', self.scan_profile_kernel),
        ):
            if int(value) < 3 or int(value) % 2 == 0:
                raise ValueError(f'{name} must be an odd integer >= 3.')
        if self.scan_axis not in {'rows', 'columns'}:
            raise ValueError('scan_axis must be "rows" or "columns".')
        if not 0.0 <= self.scan_line_strength <= 1.0:
            raise ValueError('scan_line_strength must be in [0, 1].')
        if self.denoise_strength < 0.0:
            raise ValueError('denoise_strength cannot be negative.')
        known = {
            'background_subtraction', 'illumination_correction', 'scan_line_suppression',
            'denoise', 'percentile_normalization', 'clahe',
        }
        if len(set(self.operation_order)) != len(self.operation_order) or set(self.operation_order) != known:
            raise ValueError('operation_order must contain every preprocessing operation exactly once.')

    def any_enabled(self) -> bool:
        return any(
            (
                self.percentile_normalization,
                self.clahe,
                self.illumination_correction,
                self.background_subtraction,
                self.scan_line_suppression,
                self.denoise,
            )
        )

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_preprocessing_config(raw: Mapping[str, Any] | None) -> PreprocessingConfig:
    if not isinstance(raw, Mapping):
        return PreprocessingConfig()
    tile_raw = raw.get('clahe_tile_grid_size', [8, 8])
    tile_grid = (8, 8)
    if isinstance(tile_raw, (list, tuple)) and len(tile_raw) == 2:
        tile_grid = (int(tile_raw[0]), int(tile_raw[1]))
    order_raw = raw.get('operation_order', PreprocessingConfig.operation_order)
    operation_order = tuple(str(value) for value in order_raw) if isinstance(order_raw, (list, tuple)) else PreprocessingConfig.operation_order
    return PreprocessingConfig(
        percentile_normalization=bool(raw.get('percentile_normalization', False)),
        percentile_low=float(raw.get('percentile_low', 1.0)),
        percentile_high=float(raw.get('percentile_high', 99.0)),
        clahe=bool(raw.get('clahe', False)),
        clahe_clip_limit=float(raw.get('clahe_clip_limit', 2.0)),
        clahe_tile_grid_size=tile_grid,
        illumination_correction=bool(raw.get('illumination_correction', False)),
        illumination_kernel_size=int(raw.get('illumination_kernel_size', 51)),
        background_subtraction=bool(raw.get('background_subtraction', False)),
        background_blur_kernel=int(raw.get('background_blur_kernel', 31)),
        scan_line_suppression=bool(raw.get('scan_line_suppression', False)),
        scan_line_strength=float(raw.get('scan_line_strength', 0.5)),
        scan_axis=str(raw.get('scan_axis', 'rows')).strip().lower(),
        scan_profile_kernel=int(raw.get('scan_profile_kernel', 31)),
        denoise=bool(raw.get('denoise', False)),
        denoise_strength=float(raw.get('denoise_strength', 5.0)),
        operation_order=operation_order,
    )

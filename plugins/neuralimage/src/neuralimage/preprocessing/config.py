from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


NORMALIZATION_MODES = ('none', 'per_image_percentile', 'dataset_zscore')


@dataclass
class PreprocessingConfig:
    """SEM-specific preprocessing shared by training and inference."""

    mode: str = 'none'
    percentile_normalization: bool = False
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    dataset_mean: float | None = None
    dataset_std: float | None = None
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
        self.mode = str(self.mode or 'none').strip().lower()
        if self.mode not in NORMALIZATION_MODES:
            raise ValueError(f'Unknown normalization mode: {self.mode!r}.')
        if not 0.0 <= self.percentile_low < self.percentile_high <= 100.0:
            raise ValueError('Percentile range must satisfy 0 <= low < high <= 100.')
        if self.dataset_mean is not None and not float('-inf') < float(self.dataset_mean) < float('inf'):
            raise ValueError('Dataset mean must be finite.')
        if self.dataset_std is not None and (
            not float('-inf') < float(self.dataset_std) < float('inf') or float(self.dataset_std) <= 0.0
        ):
            raise ValueError('Dataset standard deviation must be finite and positive.')
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
                self.mode != 'none',
                self.percentile_normalization,
                self.clahe,
                self.illumination_correction,
                self.background_subtraction,
                self.scan_line_suppression,
                self.denoise,
            )
        )

    def has_dataset_statistics(self) -> bool:
        return self.dataset_mean is not None and self.dataset_std is not None

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
    legacy_percentile = bool(raw.get('percentile_normalization', False))
    explicit_mode = raw.get('mode')
    mode = (
        str(explicit_mode).strip().lower()
        if explicit_mode is not None
        else ('per_image_percentile' if legacy_percentile else 'none')
    )
    return PreprocessingConfig(
        mode=mode,
        # Preserve non-normalization legacy operations for old artifacts, but
        # map the old percentile flag to the explicit normalization mode.
        percentile_normalization=False if explicit_mode is None else legacy_percentile,
        percentile_low=float(raw.get('percentile_low', 1.0)),
        percentile_high=float(raw.get('percentile_high', 99.0)),
        dataset_mean=(float(raw['dataset_mean']) if raw.get('dataset_mean') is not None else None),
        dataset_std=(float(raw['dataset_std']) if raw.get('dataset_std') is not None else None),
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

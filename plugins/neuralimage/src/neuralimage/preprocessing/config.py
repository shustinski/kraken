from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    denoise: bool = False
    denoise_strength: float = 5.0

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


def build_preprocessing_config(raw: Mapping[str, Any] | None) -> PreprocessingConfig:
    if not isinstance(raw, Mapping):
        return PreprocessingConfig()
    tile_raw = raw.get('clahe_tile_grid_size', [8, 8])
    tile_grid = (8, 8)
    if isinstance(tile_raw, (list, tuple)) and len(tile_raw) == 2:
        tile_grid = (int(tile_raw[0]), int(tile_raw[1]))
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
        denoise=bool(raw.get('denoise', False)),
        denoise_strength=float(raw.get('denoise_strength', 5.0)),
    )

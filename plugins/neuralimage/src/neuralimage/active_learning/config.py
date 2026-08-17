from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ActiveLearningConfig:
    enabled: bool = False
    export_dir: Path | None = None
    low_confidence_threshold: float = 0.35
    high_entropy_threshold: float = 0.65
    instability_threshold: float = 0.15
    disagreement_threshold: float = 0.2
    max_exports_per_run: int = 256
    max_rois_per_frame: int = 16
    min_roi_area: int = 64
    roi_padding: int = 16
    merge_distance: int = 8

    def __post_init__(self) -> None:
        for name in (
            'low_confidence_threshold', 'high_entropy_threshold',
            'instability_threshold', 'disagreement_threshold',
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f'{name} must be in [0, 1].')
        if self.max_exports_per_run <= 0 or self.max_rois_per_frame <= 0 or self.min_roi_area <= 0:
            raise ValueError('Active Learning export and ROI limits must be positive.')
        if self.roi_padding < 0 or self.merge_distance < 0:
            raise ValueError('Active Learning ROI padding and merge distance cannot be negative.')

    def resolved_export_dir(self, fallback: Path | None = None) -> Path:
        if self.export_dir is not None:
            return Path(self.export_dir)
        if fallback is not None:
            return Path(fallback) / 'NeedsAnnotation'
        return Path('NeedsAnnotation')


def build_active_learning_config(raw: Mapping[str, Any] | None) -> ActiveLearningConfig:
    if not isinstance(raw, Mapping):
        return ActiveLearningConfig()
    export_dir = raw.get('export_dir')
    return ActiveLearningConfig(
        enabled=bool(raw.get('enabled', False)),
        export_dir=Path(export_dir) if export_dir else None,
        low_confidence_threshold=float(raw.get('low_confidence_threshold', 0.35)),
        high_entropy_threshold=float(raw.get('high_entropy_threshold', 0.65)),
        instability_threshold=float(raw.get('instability_threshold', 0.15)),
        disagreement_threshold=float(raw.get('disagreement_threshold', 0.2)),
        max_exports_per_run=int(raw.get('max_exports_per_run', 256)),
        max_rois_per_frame=int(raw.get('max_rois_per_frame', 16)),
        min_roi_area=int(raw.get('min_roi_area', 64)),
        roi_padding=int(raw.get('roi_padding', 16)),
        merge_distance=int(raw.get('merge_distance', 8)),
    )

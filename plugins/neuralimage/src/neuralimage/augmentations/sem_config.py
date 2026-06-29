from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class SemAugmentationConfig:
    enabled: bool = False
    charging_artifacts: bool = True
    charging_probability: float = 0.15
    scan_drift: bool = True
    scan_drift_probability: float = 0.1
    local_focus_variation: bool = True
    focus_variation_probability: float = 0.12
    detector_noise: bool = True
    detector_noise_probability: float = 0.2
    brightness_gradients: bool = True
    brightness_gradient_probability: float = 0.15
    realistic_defects: bool = True
    realistic_defect_probability: float = 0.1


def build_sem_augmentation_config(raw: Mapping[str, Any] | None) -> SemAugmentationConfig:
    if not isinstance(raw, Mapping):
        return SemAugmentationConfig()
    return SemAugmentationConfig(
        enabled=bool(raw.get('enabled', False)),
        charging_artifacts=bool(raw.get('charging_artifacts', True)),
        charging_probability=float(raw.get('charging_probability', 0.15)),
        scan_drift=bool(raw.get('scan_drift', True)),
        scan_drift_probability=float(raw.get('scan_drift_probability', 0.1)),
        local_focus_variation=bool(raw.get('local_focus_variation', True)),
        focus_variation_probability=float(raw.get('focus_variation_probability', 0.12)),
        detector_noise=bool(raw.get('detector_noise', True)),
        detector_noise_probability=float(raw.get('detector_noise_probability', 0.2)),
        brightness_gradients=bool(raw.get('brightness_gradients', True)),
        brightness_gradient_probability=float(raw.get('brightness_gradient_probability', 0.15)),
        realistic_defects=bool(raw.get('realistic_defects', True)),
        realistic_defect_probability=float(raw.get('realistic_defect_probability', 0.1)),
    )

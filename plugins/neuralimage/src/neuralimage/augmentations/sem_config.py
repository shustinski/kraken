from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class SemAugmentationConfig:
    enabled: bool = False
    plan: str = 'legacy_v1'
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
    charging_strength: float = 0.25
    drift_max_pixels: float = 3.0
    focus_sigma_max: float = 2.5
    detector_peak_electrons: float = 80.0
    read_noise_sigma: float = 0.015
    gain_field_strength: float = 0.2

    def __post_init__(self) -> None:
        if self.plan not in {'legacy_v1', 'sem_v2'}:
            raise ValueError('SEM augmentation plan must be legacy_v1 or sem_v2.')
        probability_names = (
            'charging_probability', 'scan_drift_probability', 'focus_variation_probability',
            'detector_noise_probability', 'brightness_gradient_probability', 'realistic_defect_probability',
        )
        for name in probability_names:
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f'{name} must be in [0, 1].')
        for name in (
            'charging_strength', 'drift_max_pixels', 'focus_sigma_max',
            'read_noise_sigma', 'gain_field_strength',
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f'{name} cannot be negative.')
        if self.detector_peak_electrons <= 0.0:
            raise ValueError('detector_peak_electrons must be positive.')


def build_sem_augmentation_config(raw: Mapping[str, Any] | None) -> SemAugmentationConfig:
    if not isinstance(raw, Mapping):
        return SemAugmentationConfig()
    return SemAugmentationConfig(
        enabled=bool(raw.get('enabled', False)),
        plan=str(raw.get('plan', 'legacy_v1')).strip().lower(),
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
        charging_strength=float(raw.get('charging_strength', 0.25)),
        drift_max_pixels=float(raw.get('drift_max_pixels', 3.0)),
        focus_sigma_max=float(raw.get('focus_sigma_max', 2.5)),
        detector_peak_electrons=float(raw.get('detector_peak_electrons', 80.0)),
        read_noise_sigma=float(raw.get('read_noise_sigma', 0.015)),
        gain_field_strength=float(raw.get('gain_field_strength', 0.2)),
    )

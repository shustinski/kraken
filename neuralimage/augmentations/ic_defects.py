from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from augmentations.pcb_defects import PCBDefectAugmentor
from lib.data_interfaces import ICDefectParameters, PCBDefectParameters, build_ic_defect_parameters


@dataclass(frozen=True)
class _ICDelegateMapping:
    pcb_name: str
    severity_scale: float = 1.0


class ICDefectAugmentor:
    """Apply IC-specific synthetic defects on top of a mask/image pair."""

    _DEFECT_ORDER: tuple[str, ...] = (
        'line_break',
        'bridge',
        'necking',
        'missing_metal',
        'spur',
        'pinhole',
        'via_open',
        'line_shift',
    )
    _MAPPINGS: dict[str, _ICDelegateMapping] = {
        'line_break': _ICDelegateMapping('break', severity_scale=1.0),
        'bridge': _ICDelegateMapping('short', severity_scale=1.0),
        'necking': _ICDelegateMapping('missing_copper', severity_scale=0.6),
        'missing_metal': _ICDelegateMapping('missing_copper', severity_scale=1.0),
        'spur': _ICDelegateMapping('excess_copper', severity_scale=0.8),
        'pinhole': _ICDelegateMapping('pinhole', severity_scale=1.0),
        'via_open': _ICDelegateMapping('missing_copper', severity_scale=0.7),
        'line_shift': _ICDelegateMapping('misalignment', severity_scale=1.0),
    }

    def __init__(self, config: ICDefectParameters | dict[str, Any] | Any):
        self.config = build_ic_defect_parameters(config)
        self._weights = self._build_weight_vector()

    def _build_weight_vector(self) -> tuple[tuple[str, ...], np.ndarray]:
        names: list[str] = []
        weights: list[float] = []
        for name in self._DEFECT_ORDER:
            weight = max(0.0, float(self.config.defect_probabilities.get(name, 0.0)))
            if weight <= 0.0:
                continue
            names.append(name)
            weights.append(weight)
        if not names:
            names = list(self._DEFECT_ORDER)
            weights = [1.0] * len(names)
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        return tuple(names), probabilities

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        seed: int | None = None,
        return_debug: bool = False,
        return_augmented_mask: bool = False,
    ):
        if not self.config.enabled:
            delegate = PCBDefectAugmentor(self._build_delegate_config('line_break'))
            return delegate(
                image,
                mask,
                seed=seed,
                return_debug=return_debug,
                return_augmented_mask=return_augmented_mask,
            )

        rng = np.random.default_rng(seed)
        if float(rng.random()) >= float(self.config.defect_probability):
            delegate = PCBDefectAugmentor(self._build_delegate_config('line_break', enabled=False))
            return delegate(
                image,
                mask,
                seed=seed,
                return_debug=return_debug,
                return_augmented_mask=return_augmented_mask,
            )

        current_image = np.asarray(image).copy()
        current_mask = None if mask is None else np.asarray(mask).copy()
        original_image = np.asarray(image).copy()
        combined_defect_mask: np.ndarray | None = None
        applied = 0
        min_defects = max(1, int(self.config.min_defects))
        max_defects = max(min_defects, int(self.config.max_defects))
        target_count = int(rng.integers(min_defects, max_defects + 1))
        max_attempts = max(1, int(self.config.max_attempts_per_defect)) * max(1, target_count)

        for _attempt_index in range(max_attempts):
            if applied >= target_count:
                break
            defect_name = str(rng.choice(self._weights[0], p=self._weights[1]))
            delegate = PCBDefectAugmentor(self._build_delegate_config(defect_name))
            augmented_image, defect_mask, augmented_mask = delegate(
                current_image,
                current_mask,
                seed=int(rng.integers(0, 2**31 - 1)),
                return_augmented_mask=True,
            )
            defect_mask_array = np.asarray(defect_mask)
            if np.count_nonzero(defect_mask_array) <= 0 and np.array_equal(np.asarray(augmented_image), current_image):
                continue
            current_image = np.asarray(augmented_image).copy()
            current_mask = np.asarray(augmented_mask).copy()
            combined_defect_mask = (
                defect_mask_array.copy()
                if combined_defect_mask is None
                else np.maximum(combined_defect_mask, defect_mask_array)
            )
            applied += 1

        if combined_defect_mask is None:
            delegate = PCBDefectAugmentor(self._build_delegate_config('line_break', enabled=False))
            return delegate(
                image,
                mask,
                seed=seed,
                return_debug=return_debug,
                return_augmented_mask=return_augmented_mask,
            )

        if return_debug:
            if return_augmented_mask:
                return original_image, current_image, combined_defect_mask, current_mask
            return original_image, current_image, combined_defect_mask
        if return_augmented_mask:
            return current_image, combined_defect_mask, current_mask
        return current_image, combined_defect_mask

    def _build_delegate_config(
        self,
        defect_name: str,
        *,
        enabled: bool | None = None,
    ) -> PCBDefectParameters:
        mapping = self._MAPPINGS.get(defect_name, self._MAPPINGS['line_break'])
        severity = float(
            min(
                1.0,
                max(0.0, float(self.config.defect_severities.get(defect_name, 0.5)) * mapping.severity_scale),
            )
        )
        probabilities = {
            'break': 0.0,
            'short': 0.0,
            'missing_copper': 0.0,
            'excess_copper': 0.0,
            'pinhole': 0.0,
            'spurious_copper': 0.0,
            'via': 0.0,
            'misalignment': 0.0,
        }
        probabilities[mapping.pcb_name] = 1.0
        severities = {name: 0.5 for name in probabilities}
        severities[mapping.pcb_name] = severity
        return PCBDefectParameters(
            enabled=bool(self.config.enabled if enabled is None else enabled),
            defect_probability=1.0 if enabled is None else (1.0 if enabled else 0.0),
            min_defects=1,
            max_defects=1,
            max_attempts_per_defect=max(1, int(self.config.max_attempts_per_defect)),
            min_component_area=max(1, int(self.config.min_component_area)),
            break_width_range=tuple(int(v) for v in self.config.line_break_width_range),
            short_bridge_distance_range=tuple(int(v) for v in self.config.bridge_distance_range),
            missing_copper_radius_range=tuple(int(v) for v in self.config.missing_metal_radius_range),
            excess_copper_radius_range=tuple(int(v) for v in self.config.spur_radius_range),
            pinhole_radius_range=tuple(int(v) for v in self.config.pinhole_radius_range),
            spurious_copper_radius_range=tuple(int(v) for v in self.config.spur_radius_range),
            via_hole_area_range=tuple(int(v) for v in self.config.via_hole_area_range),
            via_shift_range=tuple(int(v) for v in self.config.via_shift_range),
            via_size_delta_range=tuple(int(v) for v in self.config.via_size_delta_range),
            misalignment_shift_range=tuple(int(v) for v in self.config.line_shift_range),
            misalignment_roi_scale_range=tuple(float(v) for v in self.config.line_shift_roi_scale_range),
            use_input_mask=bool(self.config.use_input_mask),
            use_defect_mask_as_label=False,
            defect_probabilities=probabilities,
            defect_severities=severities,
        )

"""data_interfaces: ic defect parameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from neuralimage.lib.data_interfaces_parts.pcb_defect_parameters import PCBDefectParameters

@dataclass
class ICDefectParameters:
    enabled: bool = False
    defect_probability: float = 0.5
    min_defects: int = 1
    max_defects: int = 3
    max_attempts_per_defect: int = 8
    min_component_area: int = 12
    line_break_width_range: tuple[int, int] = (1, 4)
    bridge_distance_range: tuple[int, int] = (1, 12)
    necking_radius_range: tuple[int, int] = (1, 3)
    missing_metal_radius_range: tuple[int, int] = (2, 5)
    spur_radius_range: tuple[int, int] = (1, 4)
    pinhole_radius_range: tuple[int, int] = (1, 2)
    via_hole_area_range: tuple[int, int] = (8, 220)
    via_shift_range: tuple[int, int] = (1, 3)
    via_size_delta_range: tuple[int, int] = (1, 2)
    line_shift_range: tuple[int, int] = (1, 3)
    line_shift_roi_scale_range: tuple[float, float] = (0.12, 0.3)
    use_input_mask: bool = True
    use_defect_mask_as_label: bool = False
    defect_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            'line_break': 1.0,
            'bridge': 1.0,
            'necking': 1.0,
            'missing_metal': 1.0,
            'spur': 1.0,
            'pinhole': 1.0,
            'via_open': 1.0,
            'line_shift': 1.0,
        }
    )
    defect_severities: dict[str, float] = field(
        default_factory=lambda: {
            'line_break': 0.5,
            'bridge': 0.5,
            'necking': 0.5,
            'missing_metal': 0.5,
            'spur': 0.5,
            'pinhole': 0.5,
            'via_open': 0.5,
            'line_shift': 0.5,
        }
    )

@dataclass
class SyntheticDefectGeneratorParameters:
    enabled: bool = False
    epoch_size_factor: float = 1.0
    topology_domain: str = 'pcb'
    topology_family: str = 'pcb_mixed'
    image_size_xy: tuple[int, int] = (1024, 1024)
    trace_count_range: tuple[int, int] = (5, 5)
    segment_count_range: tuple[int, int] = (4, 4)
    trace_half_width_range: tuple[int, int] = (2, 2)
    margin: int = 12
    background_noise_sigma_range: tuple[float, float] = (0.02, 0.02)
    trace_noise_sigma_range: tuple[float, float] = (0.01, 0.01)
    pcb_defects: PCBDefectParameters = field(
        default_factory=lambda: PCBDefectParameters(enabled=True)
    )
    ic_defects: ICDefectParameters = field(
        default_factory=lambda: ICDefectParameters(enabled=True)
    )
    defects: Any = field(default_factory=lambda: PCBDefectParameters(enabled=True))

    @property
    def trace_count(self) -> int:
        return int(self.trace_count_range[1])

    @property
    def segment_count(self) -> int:
        return int(self.segment_count_range[1])

    @property
    def trace_half_width(self) -> int:
        return int(self.trace_half_width_range[1])

    @property
    def background_noise_sigma(self) -> float:
        return float(self.background_noise_sigma_range[1])

    @property
    def trace_noise_sigma(self) -> float:
        return float(self.trace_noise_sigma_range[1])

def _coerce_probability(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return float(min(max(resolved, 0.0), 1.0))

def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(int(minimum), resolved)

def _coerce_positive_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return max(float(minimum), resolved)

def _coerce_size_pair(
    value: Any,
    default: tuple[int, int],
    *,
    minimum: int = 1,
) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return default
    return max(int(minimum), width), max(int(minimum), height)

def _coerce_range(
    value: Any,
    default: tuple[int, int] | tuple[float, float],
    *,
    cast_type,
) -> tuple[int, int] | tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    try:
        low = cast_type(value[0])
        high = cast_type(value[1])
    except (TypeError, ValueError):
        return default
    if low > high:
        low, high = high, low
    return low, high

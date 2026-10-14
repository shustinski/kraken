"""data_interfaces: pcb defect parameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from neuralimage.lib.data_interfaces_parts.module_state import RANDOM_ARTIFACT_TYPES

@dataclass
class HardMiningParameters:
    enabled: bool = False
    strength: float = 2.0
    pixel_enabled: bool = False
    pixel_keep_ratio: float = 0.25
    mode: str = 'online'
    geometry_weight: float = 0.5
    loss_weight: float = 0.5
    exploration_floor: float = 0.1
    ema_alpha: float = 0.1
    score_clip: float = 5.0
    refresh_epochs: int = 1
    offline_manifest: Path | None = None

@dataclass
class CutoutParameters:
    enabled: bool = False
    probability: float = 1.0
    holes: int = 1
    size_ratio: float = 0.25

@dataclass
class RandomArtifactsParameters:
    enabled: bool = False
    probability: float = 1.0
    count: int = 1
    size_ratio: float = 0.25
    dust_enabled: bool = True
    resist_residue_enabled: bool = True
    etch_residue_enabled: bool = True
    particle_cluster_enabled: bool = True
    flake_enabled: bool = True

    def enabled_types(self) -> tuple[str, ...]:
        enabled_names: list[str] = []
        for artifact_name in RANDOM_ARTIFACT_TYPES:
            if bool(getattr(self, f'{artifact_name}_enabled', True)):
                enabled_names.append(artifact_name)
        return tuple(enabled_names)

@dataclass
class MixupParameters:
    enabled: bool = False
    probability: float = 1.0
    alpha: float = 0.2

@dataclass
class TechGlobalWidthVariationParameters:
    probability: float = 0.45
    kernel_size_range: tuple[int, int] = (2, 3)
    erosion_probability: float = 0.5

@dataclass
class TechScaleRethresholdParameters:
    probability: float = 0.35
    scale_range: tuple[float, float] = (0.82, 1.18)
    threshold: float = 0.5

@dataclass
class TechBlurThresholdParameters:
    probability: float = 0.3
    blur_radius_range: tuple[float, float] = (0.6, 1.8)
    threshold: float = 0.5

@dataclass
class TechBoundaryAwareVariationParameters:
    probability: float = 0.7
    band_width_range: tuple[int, int] = (2, 4)
    noise_cell_size_range: tuple[int, int] = (3, 10)
    add_probability: float = 0.34
    remove_probability: float = 0.2
    min_addition_support: int = 1
    min_removal_support: int = 4
    smoothing_kernel_size: int = 2

@dataclass
class TechLocalMorphologyParameters:
    probability: float = 0.35
    roi_count_range: tuple[int, int] = (2, 4)
    roi_size_ratio_range: tuple[float, float] = (0.18, 0.4)
    kernel_size_range: tuple[int, int] = (2, 3)
    erosion_probability: float = 0.5

@dataclass
class TechGapVariationParameters:
    probability: float = 0.3
    kernel_size_range: tuple[int, int] = (2, 3)
    opening_probability: float = 0.4
    max_bridge_neighbor_count: int = 5
    min_gap_neighbor_count: int = 1

@dataclass
class TechAugmentationParameters:
    enabled: bool = False
    min_operations: int = 1
    max_operations: int = 3
    debug_return_pair: bool = False
    binarization_threshold: float = 0.5
    binary_tolerance: float = 0.15
    max_changed_pixels_ratio: float = 0.32
    max_foreground_ratio_delta: float = 0.2
    global_width: TechGlobalWidthVariationParameters = field(
        default_factory=TechGlobalWidthVariationParameters
    )
    scale_rethreshold: TechScaleRethresholdParameters = field(
        default_factory=TechScaleRethresholdParameters
    )
    blur_threshold: TechBlurThresholdParameters = field(
        default_factory=TechBlurThresholdParameters
    )
    boundary_aware: TechBoundaryAwareVariationParameters = field(
        default_factory=TechBoundaryAwareVariationParameters
    )
    local_morphology: TechLocalMorphologyParameters = field(
        default_factory=TechLocalMorphologyParameters
    )
    gap_variation: TechGapVariationParameters = field(
        default_factory=TechGapVariationParameters
    )

@dataclass
class PCBDefectParameters:
    enabled: bool = False
    defect_probability: float = 0.5
    min_defects: int = 1
    max_defects: int = 3
    max_attempts_per_defect: int = 8
    min_component_area: int = 12
    break_width_range: tuple[int, int] = (1, 5)
    short_bridge_distance_range: tuple[int, int] = (2, 24)
    missing_copper_radius_range: tuple[int, int] = (2, 8)
    excess_copper_radius_range: tuple[int, int] = (2, 7)
    pinhole_radius_range: tuple[int, int] = (1, 3)
    spurious_copper_radius_range: tuple[int, int] = (1, 4)
    via_hole_area_range: tuple[int, int] = (12, 400)
    via_shift_range: tuple[int, int] = (1, 4)
    via_size_delta_range: tuple[int, int] = (1, 3)
    misalignment_shift_range: tuple[int, int] = (1, 5)
    misalignment_roi_scale_range: tuple[float, float] = (0.2, 0.45)
    use_input_mask: bool = True
    use_defect_mask_as_label: bool = False
    defect_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            'break': 1.0,
            'short': 1.0,
            'missing_copper': 1.0,
            'excess_copper': 1.0,
            'pinhole': 1.0,
            'spurious_copper': 1.0,
            'via': 1.0,
            'misalignment': 1.0,
        }
    )
    defect_severities: dict[str, float] = field(
        default_factory=lambda: {
            'break': 0.5,
            'short': 0.5,
            'missing_copper': 0.5,
            'excess_copper': 0.5,
            'pinhole': 0.5,
            'spurious_copper': 0.5,
            'via': 0.5,
            'misalignment': 0.5,
        }
    )

"""detector: metal recovery config."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from contour.vision.metal_recovery.strategy_registry import MetalStrategyConfigs

@dataclass(slots=True)
class MetalRecoveryConfig:
    """Segmentation + contour extraction parameters exposed in the UI."""

    min_contrast: float = 50.0
    min_object_source_contrast: float = 12.0
    min_object_rim_contrast: float = 36.0
    min_object_rim_area_fraction: float = 0.001
    contrast_bias: float = 0.0  # Deprecated compatibility field; ignored by segmentation.
    min_hole_source_contrast: float = 8.0
    min_hole_source_contrast_fraction: float = 0.35
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    min_width_px: float = 8.0
    max_width_px: float | None = None
    min_length_px: float = 8.0
    min_area: float = 60.0
    max_area: float | None = None
    min_perimeter: float = 32.0
    max_perimeter: float | None = None
    epsilon_simplify: float = 2.0
    min_points: int = 4
    min_polygon_angle_deg: float = 0.0
    approximation_enabled: bool = True
    retrieval_external_only: bool = False
    border_mode: str = "mark"
    min_inner_hole_area: float = 100.0
    min_component_area: float = 60.0
    preset_name: str = "standard"
    retrieval_mode: str = "RETR_TREE"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    # Legacy / ignored at runtime (filters tab handles preprocessing).
    noise_suppression: int = 0
    segmentation_strategy: str = "legacy_otsu"
    auto_contrast_step: float = 10.0
    auto_source_contrast_step: float = 4.0
    auto_directional_gap_bridge_px: int = 3
    auto_directional_gap_min_source_intensity: float = 45.0
    use_wide_conductor_gradient: bool = False
    watershed_smoothing_sigma: float = 1.2
    watershed_core_margin: float = 23.0
    watershed_groove_margin: float = 19.0
    watershed_rim_probe_px: int = 2
    watershed_seed_speckle_px: int = 1
    watershed_valley_span_px: int = 5
    watershed_valley_depth: float = 65.0
    adaptive_block_size: int = 0
    adaptive_c: float = 0.0
    adaptive_method: str = "gaussian"
    random_walker_beta: float = 90.0
    random_walker_iterations: int = 160
    graph_cut_iterations: int = 5
    reconstruction_erode_px: int = 0
    boundary_relief: float = 16.0
    boundary_background_sigma: float = 12.0
    structural_variant: str = "s2"
    strategy_configs: MetalStrategyConfigs = field(default_factory=MetalStrategyConfigs)

    def __post_init__(self) -> None:
        if isinstance(self.strategy_configs, MetalStrategyConfigs):
            return
        if isinstance(self.strategy_configs, Mapping):
            self.strategy_configs = MetalStrategyConfigs.from_mapping(self.strategy_configs)
            return
        raise TypeError("strategy_configs must be MetalStrategyConfigs or a mapping")

    def to_snapshot(self) -> dict[str, Any]:
        snapshot = {
            field.name: getattr(self, field.name)
            for field in MetalRecoveryConfig.__dataclass_fields__.values()
        }
        snapshot["strategy_configs"] = self.strategy_configs.to_dict()
        return snapshot

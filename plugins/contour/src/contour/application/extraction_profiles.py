from __future__ import annotations

from .processing import ContourExtractionSettings


def default_contour_settings_profiles() -> dict[str, ContourExtractionSettings]:
    return {
        "conductors": ContourExtractionSettings(
            algorithm_backend="legacy",
            sem_noise_level="medium",
            extraction_profile="conductors",
            object_type="conductor",
            output_mode="polygon",
            min_polygon_angle=0.0,
            retrieval_mode="RETR_EXTERNAL",
            epsilon=1.0,
            min_area=95.0,
            min_perimeter=10.0,
            min_points=3,
            min_polygon_width_px=4.0,
            metal_structural_pipeline=True,
            metal_gap_bridge_px=0,
            metal_speckle_removal_px=3,
            metal_hierarchy_mode="full",
            metal_min_area=95.0,
            metal_min_perimeter=10.0,
            metal_min_trace_width_px=4.0,
        ),
        "vias": ContourExtractionSettings(
            algorithm_backend="sem",
            sem_noise_level="medium",
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_search_mode="heuristic",
            via_heuristic_polarity="bright",
            via_size_mode="range",
            bright_via_diameter_min=8,
            bright_via_diameter_max=8,
            via_output_diameter=8,
            bright_via_use_metal_mask=False,
            min_solidity=0.6,
            min_extent=0.5,
            min_aspect_ratio=0.5,
            max_aspect_ratio=2.0,
        ),
    }

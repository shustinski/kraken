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
            min_polygon_angle=30.0,
            retrieval_mode="RETR_TREE",
            epsilon=2.0,
            min_area=70.0,
            min_perimeter=32.0,
            min_points=4,
            min_polygon_width_px=4.0,
            metal_structural_pipeline=True,
        ),
        "vias": ContourExtractionSettings(
            algorithm_backend="sem",
            sem_noise_level="medium",
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_search_mode="bright_tophat_dog",
            min_solidity=0.6,
            min_extent=0.5,
            min_aspect_ratio=0.5,
            max_aspect_ratio=2.0,
        ),
    }

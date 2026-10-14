"""_core: metal preview batch result."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import replace
from typing import Any
import numpy as np
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData
from contour.application.processing import (
    RECOGNITION_MODE_VIA,
    BatchImageResult,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)

from contour.application.use_cases.processing._core_parts.filter_vias_by_conductor import _build_metalization_mask

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import _via_grayscale

def _metal_preview_batch_result(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    source: Any,
    preprocessed: Any,
    include_images_in_result: bool,
    output_directory: str | None,
    save_options: SaveOptions | None,
    display_settings: DisplaySettings | None,
    save_bundle: Callable[..., dict[str, str]],
) -> BatchImageResult:
    gray = _via_grayscale(preprocessed)
    mask = _build_metalization_mask(gray, contour_settings)
    min_area = max(0.0, float(getattr(contour_settings, "metal_min_object_area", 30.0)))
    metal_settings = replace(
        contour_settings,
        min_area=min_area,
        object_type="conductor",
        recognition_mode=RECOGNITION_MODE_VIA,
    )
    show_contours = bool(getattr(contour_settings, "metal_display_show_contours", True))
    polygons: list[PolygonData] = extract_polygons(mask, metal_settings) if show_contours else []
    debug_gradient_maps: dict[str, np.ndarray] = {}
    if bool(getattr(contour_settings, "metal_display_show_mask", False)) and include_images_in_result:
        debug_gradient_maps["metal_mask"] = mask
    saved_files: dict[str, str] = {}
    if output_directory:
        saved_files = save_bundle(
            output_directory=output_directory,
            image_path=image_path,
            polygons=polygons,
            source_image=source,
            display_settings=display_settings or DisplaySettings(),
            save_options=save_options or SaveOptions(),
            metadata={
                "contour_settings": contour_settings.to_dict(),
                "pipeline": pipeline_config,
            },
        )
    return BatchImageResult(
        image_path=image_path,
        source_image=source if include_images_in_result else None,
        preprocessed_image=preprocessed if include_images_in_result else None,
        pipeline_config=dict(pipeline_config),
        mask_image=mask if include_images_in_result else None,
        polygons=polygons,
        debug_candidates=[],
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
        saved_files=saved_files,
    )

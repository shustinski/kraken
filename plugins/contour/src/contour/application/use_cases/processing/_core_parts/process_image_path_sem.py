"""_core: process image path sem."""
from __future__ import annotations
from collections.abc import Callable
from time import perf_counter
from typing import Any
import numpy as np
from contour.utils import ensure_binary_mask, ensure_uint8
from contour.application.preview_cancellation import raise_if_preview_cancelled
from contour.application.processing import (
    BatchFrameTiming,
    BatchImageResult,
    ContourDebugCandidate,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import _recognition_image

from contour.application.use_cases.processing._core_parts.debug_candidates_via_debug import (
    _debug_candidates_from_via_debug,
    _debug_candidates_from_via_hits,
)

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import _via_grayscale

from contour.application.use_cases.processing._core_parts.render_polygon_mask_polygons import _render_polygon_mask_from_polygons

def _process_image_path_sem_backend(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions | None,
    display_settings: DisplaySettings | None,
    source: Any,
    preprocessed: Any,
    save_bundle: Callable[..., dict[str, str]],
    include_images_in_result: bool,
    timing: BatchFrameTiming | None = None,
    frame_started: float | None = None,
) -> BatchImageResult:
    from .....vision.integration import (
        contour_output_to_polygons,
        output_kind_from_text,
        run_contour_filled_mask,
        run_via_detection,
        via_output_to_polygons,
    )

    output_kind = output_kind_from_text(contour_settings.output_mode)
    debug_candidates: list[ContourDebugCandidate] = []
    debug_gradient_maps: dict[str, np.ndarray] = {}
    vision_json: dict[str, Any]

    raise_if_preview_cancelled()
    recognition_image = _recognition_image(source, preprocessed)
    if contour_settings.object_type == "via":
        phase_started = perf_counter()
        via_output = run_via_detection(
            recognition_image,
            image_path=image_path,
            output_kind=output_kind,
            legacy_settings=contour_settings,
        )
        polygons = via_output_to_polygons(via_output)
        mask = _render_polygon_mask_from_polygons(recognition_image, polygons)
        if timing is not None:
            timing.contour_extraction_ms += (perf_counter() - phase_started) * 1000.0
        vision_json = via_output.to_json_dict()
        if contour_settings.debug_enabled:
            debug_candidates = _debug_candidates_from_via_hits(via_output.hits)
            debug_candidates.extend(_debug_candidates_from_via_debug(via_output.debug))
        if contour_settings.debug_gradient_map_enabled:
            debug_gradient_maps = dict(getattr(via_output, "debug", {}) or {})
    else:
        raise_if_preview_cancelled()
        phase_started = perf_counter()
        contour_output = run_contour_filled_mask(
            recognition_image,
            image_path=image_path,
            output_kind=output_kind,
            noise_level=str(getattr(contour_settings, "sem_noise_level", "medium") or "medium"),
            legacy_settings=contour_settings,
        )
        polygons = contour_output_to_polygons(contour_output)
        mask = contour_output.filled_mask
        if timing is not None:
            timing.contour_extraction_ms += (perf_counter() - phase_started) * 1000.0
        vision_json = contour_output.to_json_dict()
        preprocessed = contour_output.debug.get("preprocessed", preprocessed)
        if contour_settings.debug_gradient_map_enabled:
            debug_gradient_maps = {
                "source_gray": _via_grayscale(source),
                "preprocessed": ensure_uint8(preprocessed),
                "mask": ensure_binary_mask(mask),
            }

    saved_files: dict[str, str] = {}
    if output_directory:
        phase_started = perf_counter()
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
                "vision_backend": vision_json,
            },
        )
        if timing is not None:
            timing.saving_ms += (perf_counter() - phase_started) * 1000.0

    if timing is not None and frame_started is not None:
        timing.total_frame_ms = (perf_counter() - frame_started) * 1000.0
    return BatchImageResult(
        image_path=image_path,
        source_image=source if include_images_in_result else None,
        preprocessed_image=preprocessed if include_images_in_result else None,
        pipeline_config=dict(pipeline_config),
        mask_image=mask if include_images_in_result else None,
        polygons=polygons,
        debug_candidates=debug_candidates if contour_settings.debug_enabled else [],
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
        saved_files=saved_files,
    )

"""_core: process image path impl."""
from __future__ import annotations
from collections.abc import Callable
from time import perf_counter
from typing import Any
import numpy as np
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData
from .....domain.polygon_offset import offset_conductor_polygons
from contour.pipeline import PreprocessingPipeline
from contour.serializers import save_result_bundle
from contour.utils import ensure_binary_mask, ensure_uint8, load_image_color
from contour.application.preview_cancellation import raise_if_preview_cancelled
from contour.application.processing import (
    RECOGNITION_MODE_DISABLED,
    BatchFrameTiming,
    BatchImageResult,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
    normalize_recognition_mode,
)

from contour.application.use_cases.processing._core_parts.build_detection_debug_maps import build_detection_debug_maps

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import (
    build_conductor_vectorization_mask,
    build_via_vectorization_mask,
)

from contour.application.use_cases.processing._core_parts.filter_vias_by_conductor import (
    _build_via_settings_for_dual_branch,
    _filter_vias_by_conductor_linearity,
    _merge_dual_branch_polygons,
    _should_run_sem_dual_branch,
)

from contour.application.use_cases.processing._core_parts.process_image_path_sem import _process_image_path_sem_backend

from contour.application.use_cases.processing._core_parts.run_structural_metal_recovery import (
    _run_structural_metal_recovery,
    _use_structural_metal_recovery,
)

def _process_image_path_impl(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    source_image: Any | None = None,
    preprocessed_image: Any | None = None,
    pipeline: PreprocessingPipeline | None = None,
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
    passthrough_polygons: list[PolygonData] | None = None,
    timing: BatchFrameTiming | None = None,
) -> BatchImageResult:
    frame_started = perf_counter()
    raise_if_preview_cancelled()
    phase_started = perf_counter()
    source = source_image if source_image is not None else image_loader(image_path)
    if timing is not None and source_image is None:
        timing.image_loading_ms += (perf_counter() - phase_started) * 1000.0
    raise_if_preview_cancelled()
    if preprocessed_image is not None:
        preprocessed = preprocessed_image
    else:
        active_pipeline = pipeline or PreprocessingPipeline.from_dict(pipeline_config)
        if timing is None:
            preprocessed = active_pipeline.apply(source)
        else:
            preprocessed, pipeline_timings = active_pipeline.apply_with_timing(source)
            timing.threshold_ms += float(pipeline_timings.get("threshold_ms", 0.0))
            timing.morphology_ms += float(pipeline_timings.get("morphology_ms", 0.0))
            timing.postprocessing_ms += float(pipeline_timings.get("postprocessing_ms", 0.0))
    raise_if_preview_cancelled()

    rec = normalize_recognition_mode(getattr(contour_settings, "recognition_mode", "via"))
    if rec == RECOGNITION_MODE_DISABLED:
        saved_files: dict[str, str] = {}
        if output_directory:
            phase_started = perf_counter()
            saved_files = save_bundle(
                output_directory=output_directory,
                image_path=image_path,
                polygons=list(passthrough_polygons) if passthrough_polygons else [],
                source_image=source,
                display_settings=display_settings or DisplaySettings(),
                save_options=save_options or SaveOptions(),
                metadata={
                    "contour_settings": contour_settings.to_dict(),
                    "pipeline": pipeline_config,
                },
            )
            if timing is not None:
                timing.saving_ms += (perf_counter() - phase_started) * 1000.0
        polys = list(passthrough_polygons) if passthrough_polygons else []
        if timing is not None:
            timing.total_frame_ms = (perf_counter() - frame_started) * 1000.0
        return BatchImageResult(
            image_path=image_path,
            source_image=source if include_images_in_result else None,
            preprocessed_image=preprocessed if include_images_in_result else None,
            pipeline_config=dict(pipeline_config),
            mask_image=ensure_uint8(preprocessed) if include_images_in_result else None,
            polygons=polys,
            debug_candidates=[],
            debug_gradient_maps={},
            saved_files=saved_files,
        )
    if str(getattr(contour_settings, "algorithm_backend", "legacy")).lower() == "sem":
        return _process_image_path_sem_backend(
            image_path=image_path,
            pipeline_config=pipeline_config,
            contour_settings=contour_settings,
            output_directory=output_directory,
            save_options=save_options,
            display_settings=display_settings,
            source=source,
            preprocessed=preprocessed,
            save_bundle=save_bundle,
            include_images_in_result=include_images_in_result,
            timing=timing,
            frame_started=frame_started,
        )

    metal_overlays: dict[str, list[PolygonData]] = {}
    metal_debug_extra: dict[str, np.ndarray] = {}
    if contour_settings.object_type == "via" or contour_settings.output_mode == "box":
        phase_started = perf_counter()
        mask, debug_candidates = build_via_vectorization_mask(preprocessed, contour_settings)
        if timing is not None:
            timing.postprocessing_ms += (perf_counter() - phase_started) * 1000.0
        raise_if_preview_cancelled()
        phase_started = perf_counter()
        polygons = extract_polygons(mask, contour_settings)
        if timing is not None:
            timing.contour_extraction_ms += (perf_counter() - phase_started) * 1000.0
        raise_if_preview_cancelled()
    elif _use_structural_metal_recovery(contour_settings):
        raise_if_preview_cancelled()
        phase_started = perf_counter()
        polygons, mask, metal_debug_extra, metal_overlays = _run_structural_metal_recovery(
            source, preprocessed, contour_settings
        )
        if timing is not None:
            timing.contour_extraction_ms += (perf_counter() - phase_started) * 1000.0
        raise_if_preview_cancelled()
        debug_candidates = []
    else:
        phase_started = perf_counter()
        mask = build_conductor_vectorization_mask(source, preprocessed, contour_settings)
        if timing is not None:
            timing.postprocessing_ms += (perf_counter() - phase_started) * 1000.0
        raise_if_preview_cancelled()
        debug_candidates = []
        phase_started = perf_counter()
        polygons = extract_polygons(mask, contour_settings)
        if timing is not None:
            timing.contour_extraction_ms += (perf_counter() - phase_started) * 1000.0
        if _should_run_sem_dual_branch(contour_settings):
            raise_if_preview_cancelled()
            phase_started = perf_counter()
            via_settings = _build_via_settings_for_dual_branch(contour_settings)
            via_mask, via_debug_candidates = build_via_vectorization_mask(preprocessed, via_settings)
            via_polygons = extract_polygons(via_mask, via_settings)
            via_polygons = _filter_vias_by_conductor_linearity(via_polygons, mask)
            polygons = _merge_dual_branch_polygons(polygons, via_polygons)
            debug_candidates.extend(via_debug_candidates)
            if timing is not None:
                timing.postprocessing_ms += (perf_counter() - phase_started) * 1000.0
    if not contour_settings.debug_enabled:
        debug_candidates = []
    base_metal_maps: dict[str, np.ndarray] = (
        dict(metal_debug_extra) if metal_debug_extra and include_images_in_result else {}
    )
    debug_gradient_maps: dict[str, np.ndarray] = dict(base_metal_maps)
    if contour_settings.debug_gradient_map_enabled or (
        contour_settings.debug_enabled and contour_settings.debug_gradient_map_enabled
    ):
        raise_if_preview_cancelled()
        try:
            extra_maps = build_detection_debug_maps(source, preprocessed, contour_settings)
            debug_gradient_maps = {**base_metal_maps, **extra_maps}
        except Exception:  # pragma: no cover - defensive: debug never breaks processing
            debug_gradient_maps = dict(base_metal_maps)
    if metal_debug_extra and "mask" not in debug_gradient_maps:
        debug_gradient_maps["mask"] = ensure_binary_mask(mask)
    recognition_base = [polygon.clone() for polygon in polygons]
    polygons = offset_conductor_polygons(
        recognition_base,
        float(getattr(contour_settings, "metal_conductor_size_offset_px", 0.0) or 0.0),
    )
    saved_files = {}
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
            },
        )
        if timing is not None:
            timing.saving_ms += (perf_counter() - phase_started) * 1000.0
    result_source = source if include_images_in_result else None
    result_preprocessed = preprocessed if include_images_in_result else None
    result_mask = mask if include_images_in_result else None
    if timing is not None:
        timing.total_frame_ms = (perf_counter() - frame_started) * 1000.0
    return BatchImageResult(
        image_path=image_path,
        source_image=result_source,
        preprocessed_image=result_preprocessed,
        pipeline_config=dict(pipeline_config),
        mask_image=result_mask,
        polygons=polygons,
        recognition_base_polygons=recognition_base,
        debug_candidates=debug_candidates,
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
        metal_overlay_polygons=metal_overlays,
        saved_files=saved_files,
    )

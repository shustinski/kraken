"""_core: process image path."""
from __future__ import annotations
import cProfile
import io
import pstats
from collections.abc import Callable
from typing import Any
import numpy as np
from contour.domain import PolygonData
from .....infrastructure.profiling import (
    processing_profiling_enabled,
    processing_top_lines,
    try_disable_profiler,
    try_enable_profiler,
    write_profile_report,
)
from contour.pipeline import PreprocessingPipeline
from contour.serializers import save_result_bundle
from contour.utils import load_image_color
from contour.application.processing import (
    BatchFrameTiming,
    BatchImageResult,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)

from contour.application.use_cases.processing._core_parts.process_image_path_impl import _process_image_path_impl

def process_image_path(
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    *,
    source_image: Any | None = None,
    preprocessed_image: Any | None = None,
    pipeline: PreprocessingPipeline | None = None,
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
    passthrough_polygons: list[PolygonData] | None = None,
) -> BatchImageResult:
    if processing_profiling_enabled():
        profiler = cProfile.Profile()
        profiler_enabled = try_enable_profiler(profiler)
        try:
            return _process_image_path_impl(
                image_path=image_path,
                pipeline_config=pipeline_config,
                contour_settings=contour_settings,
                output_directory=output_directory,
                save_options=save_options,
                display_settings=display_settings,
                source_image=source_image,
                preprocessed_image=preprocessed_image,
                pipeline=pipeline,
                image_loader=image_loader,
                save_bundle=save_bundle,
                include_images_in_result=include_images_in_result,
                passthrough_polygons=passthrough_polygons,
            )
        finally:
            if not profiler_enabled:
                write_profile_report(
                    f"[contour profiling] image={image_path} skipped=yes reason=cprofile_already_active"
                )
            else:
                try_disable_profiler(profiler)
                top_lines = processing_top_lines()
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats(top_lines)
                write_profile_report(
                    f"[contour profiling] image={image_path} top={top_lines}",
                    stream.getvalue(),
                )

    return _process_image_path_impl(
        image_path=image_path,
        pipeline_config=pipeline_config,
        contour_settings=contour_settings,
        output_directory=output_directory,
        save_options=save_options,
        display_settings=display_settings,
        source_image=source_image,
        preprocessed_image=preprocessed_image,
        pipeline=pipeline,
        image_loader=image_loader,
        save_bundle=save_bundle,
        include_images_in_result=include_images_in_result,
        passthrough_polygons=passthrough_polygons,
    )

def process_image_path_timed(
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    *,
    pipeline: PreprocessingPipeline | None = None,
    image_loader: Callable[[str], np.ndarray] = load_image_color,
) -> tuple[BatchImageResult, BatchFrameTiming]:
    timing = BatchFrameTiming()
    result = _process_image_path_impl(
        image_path=image_path,
        pipeline_config=pipeline_config,
        contour_settings=contour_settings,
        output_directory=output_directory,
        save_options=save_options,
        display_settings=display_settings,
        pipeline=pipeline,
        image_loader=image_loader,
        include_images_in_result=False,
        timing=timing,
    )
    return result, timing

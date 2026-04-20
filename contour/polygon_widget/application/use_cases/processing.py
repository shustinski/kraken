from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..processing import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from ...contour_extractor import extract_polygons
from ...pipeline import PreprocessingPipeline
from ...serializers import save_result_bundle
from ...utils import ensure_binary_mask, load_image_color


@dataclass(frozen=True, slots=True)
class PreviewProcessingRequest:
    image_path: str
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings
    source_image: Any | None = None
    preprocessed_image: Any | None = None


@dataclass(frozen=True, slots=True)
class PreparedImageRequest:
    image_path: str
    source_image: Any
    pipeline_config: dict[str, Any]


def build_preview_request_signature(request: PreviewProcessingRequest) -> tuple[str, str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
        json.dumps(request.contour_settings.to_dict(), ensure_ascii=False, sort_keys=True),
    )


def build_prepared_image_signature(request: PreparedImageRequest) -> tuple[str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
    )


def prepare_image_for_preview(source_image: Any, pipeline_config: dict[str, Any]) -> Any:
    return PreprocessingPipeline.from_dict(pipeline_config).apply(source_image)


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
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
) -> BatchImageResult:
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    source = source_image if source_image is not None else image_loader(image_path)
    preprocessed = preprocessed_image if preprocessed_image is not None else pipeline.apply(source)
    mask = ensure_binary_mask(preprocessed)
    polygons = extract_polygons(mask, contour_settings)
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
    result_source = source if include_images_in_result else None
    result_preprocessed = preprocessed if include_images_in_result else None
    result_mask = mask if include_images_in_result else None
    return BatchImageResult(
        image_path=image_path,
        source_image=result_source,
        preprocessed_image=result_preprocessed,
        pipeline_config=dict(pipeline_config),
        mask_image=result_mask,
        polygons=polygons,
        saved_files=saved_files,
    )

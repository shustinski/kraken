from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..processing import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from ...contour_extractor import extract_polygons
from ...pipeline import PreprocessingPipeline
from ...serializers import save_result_bundle
from ...utils import ensure_binary_mask, load_image_grayscale


@dataclass(frozen=True, slots=True)
class PreviewProcessingRequest:
    image_path: str
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings


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
    image_loader: Callable[[str], Any] = load_image_grayscale,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
) -> BatchImageResult:
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    source_image = image_loader(image_path)
    preprocessed = pipeline.apply(source_image)
    mask = ensure_binary_mask(preprocessed)
    polygons = extract_polygons(mask, contour_settings)
    saved_files: dict[str, str] = {}
    if output_directory:
        saved_files = save_bundle(
            output_directory=output_directory,
            image_path=image_path,
            polygons=polygons,
            source_image=source_image,
            display_settings=display_settings or DisplaySettings(),
            save_options=save_options or SaveOptions(),
            metadata={
                "contour_settings": contour_settings.to_dict(),
                "pipeline": pipeline_config,
            },
        )
    return BatchImageResult(
        image_path=image_path,
        source_image=source_image,
        preprocessed_image=preprocessed,
        mask_image=mask,
        polygons=polygons,
        saved_files=saved_files,
    )

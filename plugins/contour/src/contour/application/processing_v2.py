"""Versioned, serializable processing boundary for preview and batch.

The legacy :class:`ContourExtractionSettings` remains an internal adapter.  New
configuration files must use this module so settings from unrelated algorithms
cannot silently affect one another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from time import perf_counter
from typing import Any, Literal, cast

from ..domain import PolygonData
from .processing import ContourExtractionSettings, DisplaySettings, PipelineStepConfig

MetalSegmentationStrategyV2 = Literal[
    "auto",
    "global_otsu",
    "local_adaptive",
    "gradient_watershed",
    "random_walker",
    "graph_cut",
    "reconstruction",
    "closed_boundary",
]


@dataclass(slots=True)
class CommonContourSettings:
    retrieval_mode: str = "RETR_TREE"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    epsilon: float = 2.0
    epsilon_relative: bool = False
    preserve_corners: bool = False
    min_polygon_angle: float = 0.0
    min_area: float = 0.0
    max_area: float | None = None
    min_perimeter: float = 10.0
    min_points: int = 3


@dataclass(slots=True)
class MetalRecoverySettings:
    kind: Literal["metal"] = "metal"
    segmentation_strategy: MetalSegmentationStrategyV2 = "auto"
    min_contrast: float = 50.0
    min_object_source_contrast: float = 12.0
    # Deprecated schema-v2 field retained for compatibility with saved requests.
    contrast_bias: float = 0.0
    min_hole_source_contrast: float = 8.0
    min_hole_source_contrast_fraction: float = 0.35
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    min_trace_width_px: float = 8.0
    max_trace_width_px: float | None = None
    min_trace_length_px: float = 8.0
    min_object_area: float = 60.0
    hierarchy_mode: Literal["full", "external"] = "full"
    border_handling: Literal["mark", "ignore", "accept"] = "mark"
    use_wide_conductor_gradient: bool = False
    watershed_smoothing_sigma: float = 1.0
    watershed_core_margin: float = 8.0
    watershed_groove_margin: float = 16.0
    watershed_rim_probe_px: int = 6
    watershed_seed_speckle_px: int = 4
    watershed_valley_span_px: int = 5
    watershed_valley_depth: float = 45.0
    random_walker_beta: float = 90.0
    random_walker_iterations: int = 160
    graph_cut_iterations: int = 5
    reconstruction_erode_px: int = 0
    boundary_relief: float = 16.0
    boundary_background_sigma: float = 12.0

    def __post_init__(self) -> None:
        self.min_contrast = max(1.0, min(255.0, float(self.min_contrast)))
        self.min_object_source_contrast = max(
            0.0,
            min(255.0, float(self.min_object_source_contrast)),
        )


@dataclass(slots=True)
class ViaDetectionSettings:
    kind: Literal["via"] = "via"
    search_mode: Literal["heuristic", "bright_tophat_dog", "template", "hybrid"] = "heuristic"
    polarity: Literal["auto", "bright", "dark", "ring_light_ring", "ring_dark_ring"] = "auto"
    size_mode: Literal["range", "fixed"] = "range"
    candidate_diameter_min: int = 8
    candidate_diameter_max: int = 8
    output_diameter: int = 8
    min_width: int = 3
    max_width: int = 80
    min_height: int = 3
    max_height: int = 80
    fixed_widths: list[int] = field(default_factory=list)
    fixed_heights: list[int] = field(default_factory=list)
    white_range: tuple[int, int] = (180, 255)
    black_range: tuple[int, int] = (0, 75)
    min_score: float = 0.0
    nms_iou: float = 0.35
    template_boost: float = 0.15
    heuristic_parameters: dict[str, float | bool] = field(default_factory=dict)


RecognitionSettingsV2 = MetalRecoverySettings | ViaDetectionSettings


@dataclass(slots=True)
class SettingsMigrationReport:
    migrated_fields: list[str] = field(default_factory=list)
    dropped_fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_v2_metal_strategy(
    value: Any,
    *,
    use_wide_conductor_gradient: bool = False,
) -> MetalSegmentationStrategyV2:
    from ..vision.metal_recovery.segmentation import (
        SEEDED_SEGMENTATION_STRATEGIES,
        resolve_metal_segmentation_strategy,
    )

    normalized = resolve_metal_segmentation_strategy(
        value,
        use_wide_conductor_gradient=use_wide_conductor_gradient,
    )
    result = "global_otsu" if normalized == "legacy_otsu" else normalized
    allowed: set[str] = {"auto", "global_otsu", "local_adaptive"} | set(SEEDED_SEGMENTATION_STRATEGIES)
    if result not in allowed:
        result = "auto"
    return cast(MetalSegmentationStrategyV2, result)


@dataclass(slots=True)
class ProcessingRequestV2:
    input_id: str
    input_version: str
    recognition: RecognitionSettingsV2
    preprocessing: list[PipelineStepConfig] = field(default_factory=list)
    contour: CommonContourSettings = field(default_factory=CommonContourSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    schema_version: Literal[2] = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "input_id": self.input_id,
            "input_version": self.input_version,
            "preprocessing": [step.to_dict() for step in self.preprocessing],
            "contour": asdict(self.contour),
            "recognition": asdict(self.recognition),
            "display": self.display.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessingRequestV2:
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("ProcessingRequestV2 requires schema_version=2")
        recognition_payload = dict(payload.get("recognition") or {})
        kind = str(recognition_payload.get("kind") or "")
        if kind == "metal":
            if "min_contrast" not in recognition_payload and "contrast_bias" in recognition_payload:
                recognition_payload["min_contrast"] = max(
                    0.0,
                    float(recognition_payload.get("contrast_bias", 0.0)),
                )
            recognition_payload["segmentation_strategy"] = _normalized_v2_metal_strategy(
                recognition_payload.get("segmentation_strategy", "auto"),
                use_wide_conductor_gradient=bool(recognition_payload.get("use_wide_conductor_gradient", False)),
            )
            recognition: RecognitionSettingsV2 = MetalRecoverySettings(**recognition_payload)
        elif kind == "via":
            for key in ("white_range", "black_range"):
                if key in recognition_payload:
                    recognition_payload[key] = tuple(recognition_payload[key])
            recognition = ViaDetectionSettings(**recognition_payload)
        else:
            raise ValueError("recognition.kind must be 'metal' or 'via'")
        return cls(
            input_id=str(payload.get("input_id") or ""),
            input_version=str(payload.get("input_version") or ""),
            preprocessing=[PipelineStepConfig.from_dict(item) for item in payload.get("preprocessing", [])],
            contour=CommonContourSettings(**dict(payload.get("contour") or {})),
            recognition=recognition,
            display=DisplaySettings.from_dict(dict(payload.get("display") or {})),
        )

    def to_legacy_settings(self) -> ContourExtractionSettings:
        payload = asdict(self.contour)
        if isinstance(self.recognition, MetalRecoverySettings):
            metal_mode = self.recognition
            payload.update(
                recognition_mode="conductors",
                metal_segmentation_strategy=(
                    "legacy_otsu"
                    if metal_mode.segmentation_strategy == "global_otsu"
                    else metal_mode.segmentation_strategy
                ),
                metal_min_contrast=metal_mode.min_contrast,
                metal_min_object_source_contrast=metal_mode.min_object_source_contrast,
                metal_contrast_bias=metal_mode.contrast_bias,
                metal_min_hole_source_contrast=metal_mode.min_hole_source_contrast,
                metal_min_hole_source_contrast_fraction=metal_mode.min_hole_source_contrast_fraction,
                metal_gap_bridge_px=metal_mode.gap_bridge_px,
                metal_speckle_removal_px=metal_mode.speckle_removal_px,
                metal_min_trace_width_px=metal_mode.min_trace_width_px,
                metal_max_trace_width_px=metal_mode.max_trace_width_px,
                metal_min_trace_length_px=metal_mode.min_trace_length_px,
                metal_min_object_area=metal_mode.min_object_area,
                metal_hierarchy_mode=metal_mode.hierarchy_mode,
                metal_border_handling=metal_mode.border_handling,
                metal_use_wide_conductor_gradient=metal_mode.use_wide_conductor_gradient,
                metal_watershed_smoothing_sigma=metal_mode.watershed_smoothing_sigma,
                metal_watershed_core_margin=metal_mode.watershed_core_margin,
                metal_watershed_groove_margin=metal_mode.watershed_groove_margin,
                metal_watershed_rim_probe_px=metal_mode.watershed_rim_probe_px,
                metal_watershed_seed_speckle_px=metal_mode.watershed_seed_speckle_px,
                metal_watershed_valley_span_px=metal_mode.watershed_valley_span_px,
                metal_watershed_valley_depth=metal_mode.watershed_valley_depth,
                metal_random_walker_beta=metal_mode.random_walker_beta,
                metal_random_walker_iterations=metal_mode.random_walker_iterations,
                metal_graph_cut_iterations=metal_mode.graph_cut_iterations,
                metal_reconstruction_erode_px=metal_mode.reconstruction_erode_px,
                metal_boundary_relief=metal_mode.boundary_relief,
                metal_boundary_background_sigma=metal_mode.boundary_background_sigma,
            )
        else:
            via_mode = self.recognition
            payload.update(
                recognition_mode="via",
                via_search_mode=via_mode.search_mode,
                via_heuristic_polarity=via_mode.polarity,
                via_size_mode=via_mode.size_mode,
                bright_via_diameter_min=via_mode.candidate_diameter_min,
                bright_via_diameter_max=via_mode.candidate_diameter_max,
                via_output_diameter=via_mode.output_diameter,
                min_via_width=via_mode.min_width,
                max_via_width=via_mode.max_width,
                min_via_height=via_mode.min_height,
                max_via_height=via_mode.max_height,
                fixed_via_widths=via_mode.fixed_widths,
                fixed_via_heights=via_mode.fixed_heights,
                via_white_range_min=via_mode.white_range[0],
                via_white_range_max=via_mode.white_range[1],
                via_black_range_min=via_mode.black_range[0],
                via_black_range_max=via_mode.black_range[1],
                **dict(via_mode.heuristic_parameters),
            )
        return ContourExtractionSettings.from_dict(payload)

    @classmethod
    def migrate_legacy(
        cls,
        payload: dict[str, Any],
        *,
        input_id: str = "",
        input_version: str = "legacy",
    ) -> tuple[ProcessingRequestV2, SettingsMigrationReport]:
        legacy = ContourExtractionSettings.from_dict(payload)
        report = SettingsMigrationReport()
        known = {item.name for item in fields(ContourExtractionSettings)}
        explicitly_retired = {
            "metal_segmentation_method",
            "metal_sensitivity",
            "metal_sensitivity_0_100",
            "metal_morph_close_radius",
            "metal_morph_open_radius",
        }
        for key, value in payload.items():
            if key in known or key in explicitly_retired:
                report.migrated_fields.append(key)
            else:
                report.dropped_fields[key] = value
        if report.dropped_fields:
            report.warnings.append("Dropped unsupported legacy fields: " + ", ".join(sorted(report.dropped_fields)))
        common = CommonContourSettings(
            **{item.name: getattr(legacy, item.name) for item in fields(CommonContourSettings)}
        )
        if legacy.recognition_mode == "via":
            recognition: RecognitionSettingsV2 = ViaDetectionSettings(
                search_mode=legacy.via_search_mode,
                polarity=legacy.via_heuristic_polarity,
                size_mode=legacy.via_size_mode,
                candidate_diameter_min=legacy.bright_via_diameter_min,
                candidate_diameter_max=legacy.bright_via_diameter_max,
                output_diameter=legacy.via_output_diameter,
                min_width=legacy.min_via_width,
                max_width=legacy.max_via_width or 80,
                min_height=legacy.min_via_height,
                max_height=legacy.max_via_height or 80,
                fixed_widths=list(legacy.fixed_via_widths),
                fixed_heights=list(legacy.fixed_via_heights),
                white_range=(legacy.via_white_range_min, legacy.via_white_range_max),
                black_range=(legacy.via_black_range_min, legacy.via_black_range_max),
                heuristic_parameters={
                    item.name: getattr(legacy, item.name)
                    for item in fields(ContourExtractionSettings)
                    if item.name.startswith("heuristic_")
                },
            )
        else:
            recognition = MetalRecoverySettings(
                segmentation_strategy=_normalized_v2_metal_strategy(legacy.metal_segmentation_strategy),
                min_contrast=legacy.metal_min_contrast,
                min_object_source_contrast=legacy.metal_min_object_source_contrast,
                contrast_bias=legacy.metal_contrast_bias,
                min_hole_source_contrast=legacy.metal_min_hole_source_contrast,
                min_hole_source_contrast_fraction=legacy.metal_min_hole_source_contrast_fraction,
                gap_bridge_px=legacy.metal_gap_bridge_px,
                speckle_removal_px=legacy.metal_speckle_removal_px,
                min_trace_width_px=legacy.metal_min_trace_width_px,
                max_trace_width_px=legacy.metal_max_trace_width_px,
                min_trace_length_px=legacy.metal_min_trace_length_px,
                min_object_area=legacy.metal_min_object_area,
                hierarchy_mode=legacy.metal_hierarchy_mode,
                border_handling=legacy.metal_border_handling,
                use_wide_conductor_gradient=legacy.metal_use_wide_conductor_gradient,
                watershed_smoothing_sigma=legacy.metal_watershed_smoothing_sigma,
                watershed_core_margin=legacy.metal_watershed_core_margin,
                watershed_groove_margin=legacy.metal_watershed_groove_margin,
                watershed_rim_probe_px=legacy.metal_watershed_rim_probe_px,
                watershed_seed_speckle_px=legacy.metal_watershed_seed_speckle_px,
                watershed_valley_span_px=legacy.metal_watershed_valley_span_px,
                watershed_valley_depth=legacy.metal_watershed_valley_depth,
                random_walker_beta=legacy.metal_random_walker_beta,
                random_walker_iterations=legacy.metal_random_walker_iterations,
                graph_cut_iterations=legacy.metal_graph_cut_iterations,
                reconstruction_erode_px=legacy.metal_reconstruction_erode_px,
                boundary_relief=legacy.metal_boundary_relief,
                boundary_background_sigma=legacy.metal_boundary_background_sigma,
            )
        return (
            cls(
                input_id=input_id,
                input_version=input_version,
                preprocessing=[],
                contour=common,
                recognition=recognition,
            ),
            report,
        )


@dataclass(slots=True)
class ProcessingResultV2:
    polygons: list[PolygonData]
    hierarchy: dict[int, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: Literal[2] = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "polygons": [polygon.to_dict() for polygon in self.polygons],
            "hierarchy": {str(root): list(children) for root, children in self.hierarchy.items()},
            "warnings": list(self.warnings),
            "stage_timings_ms": {key: float(value) for key, value in self.stage_timings_ms.items()},
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessingResultV2:
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("ProcessingResultV2 requires schema_version=2")
        return cls(
            polygons=[PolygonData.from_dict(item) for item in payload.get("polygons", [])],
            hierarchy={
                int(root): [int(child) for child in children] for root, children in payload.get("hierarchy", {}).items()
            },
            warnings=[str(item) for item in payload.get("warnings", [])],
            stage_timings_ms={str(key): float(value) for key, value in payload.get("stage_timings_ms", {}).items()},
            provenance=dict(payload.get("provenance") or {}),
        )


def process_request_v2(
    request: ProcessingRequestV2,
    *,
    source_image: Any | None = None,
) -> ProcessingResultV2:
    """Execute the same request contract in interactive preview or batch code."""
    from .use_cases.processing import process_image_path, process_image_path_timed

    pipeline_payload = {"steps": [step.to_dict() for step in request.preprocessing]}
    if source_image is None:
        result, timing = process_image_path_timed(
            request.input_id,
            pipeline_payload,
            request.to_legacy_settings(),
            display_settings=request.display,
        )
        timing_payload = timing.to_dict()
    else:
        started = perf_counter()
        result = process_image_path(
            request.input_id,
            pipeline_payload,
            request.to_legacy_settings(),
            display_settings=request.display,
            source_image=source_image,
        )
        timing_payload = {"total_frame_ms": (perf_counter() - started) * 1000.0}
    hierarchy: dict[int, list[int]] = {}
    for polygon in result.polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            hierarchy.setdefault(int(polygon.parent_id), []).append(int(polygon.id))
        elif not polygon.is_hole:
            hierarchy.setdefault(int(polygon.id), [])
    warnings = [result.error] if result.error else []
    recognition = request.recognition
    provenance = {
        "recognition_kind": recognition.kind,
        "algorithm": (
            recognition.segmentation_strategy
            if isinstance(recognition, MetalRecoverySettings)
            else recognition.search_mode
        ),
        "input_version": request.input_version,
    }
    return ProcessingResultV2(
        polygons=[polygon.clone() for polygon in result.polygons],
        hierarchy=hierarchy,
        warnings=warnings,
        stage_timings_ms=timing_payload,
        provenance=provenance,
    )


__all__ = [
    "CommonContourSettings",
    "MetalRecoverySettings",
    "ProcessingRequestV2",
    "ProcessingResultV2",
    "SettingsMigrationReport",
    "ViaDetectionSettings",
    "process_request_v2",
]

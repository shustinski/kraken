from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain import PolygonData

VIA_SIZE_MODE_RANGE = "range"
VIA_SIZE_MODE_FIXED = "fixed"
VIA_SEARCH_MODE_HYBRID = "hybrid"
VIA_SEARCH_MODE_BLOB = "blob"
VIA_SEARCH_MODE_TEMPLATE = "template"
VIA_CHANNEL_MODE_COLUMNS = "columns"
VIA_CHANNEL_MODE_GRAYSCALE = "grayscale"
VIA_CHANNEL_MODE_RED_BLUE = "red_blue"
ALGORITHM_BACKEND_LEGACY = "legacy"
ALGORITHM_BACKEND_SEM = "sem"


def normalize_via_size_mode(value: Any) -> str:
    return VIA_SIZE_MODE_FIXED if str(value).strip().lower() == VIA_SIZE_MODE_FIXED else VIA_SIZE_MODE_RANGE


def normalize_via_search_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == VIA_SEARCH_MODE_BLOB:
        return VIA_SEARCH_MODE_BLOB
    if text == VIA_SEARCH_MODE_TEMPLATE:
        return VIA_SEARCH_MODE_TEMPLATE
    return VIA_SEARCH_MODE_HYBRID


def normalize_via_channel_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text == VIA_CHANNEL_MODE_COLUMNS:
        return VIA_CHANNEL_MODE_COLUMNS
    if text in {"gray", "grey", "grayscale"}:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text in {"rb", "red_blue", "red-blue", "redblue"}:
        return VIA_CHANNEL_MODE_RED_BLUE
    return VIA_CHANNEL_MODE_GRAYSCALE


def normalize_algorithm_backend(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {ALGORITHM_BACKEND_SEM, "new", "sem_auto", "auto_sem"}:
        return ALGORITHM_BACKEND_SEM
    if text == "legacy_via":
        return "legacy_via"
    return ALGORITHM_BACKEND_LEGACY


def parse_integer_value_list(payload: Any) -> list[int]:
    if payload in (None, ""):
        return []
    if isinstance(payload, str):
        raw_values = re.split(r"[\s,;]+", payload.strip())
    elif isinstance(payload, (list, tuple, set)):
        raw_values = list(payload)
    else:
        raw_values = [payload]

    values: set[int] = set()
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        try:
            parsed = int(float(text))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.add(parsed)
    return sorted(values)


def _serialize_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    serialized: list[Any] = []
    for item in payload:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, list):
            serialized.append(item)
    return serialized


def _parse_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, list)]


@dataclass(frozen=True, slots=True)
class OperationParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 3
    options: list[str] = field(default_factory=list)
    tooltip: str = ""


@dataclass(slots=True)
class PipelineStepConfig:
    operation: str
    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> PipelineStepConfig:
        return PipelineStepConfig(
            operation=self.operation,
            name=self.name,
            enabled=self.enabled,
            parameters=dict(self.parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "name": self.name,
            "enabled": self.enabled,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineStepConfig:
        return cls(
            operation=str(payload["operation"]),
            name=str(payload.get("name") or payload["operation"]),
            enabled=bool(payload.get("enabled", True)),
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(slots=True)
class ContourDebugCandidate:
    contour_index: int
    bbox: tuple[int, int, int, int]
    area: float = 0.0
    perimeter: float = 0.0
    roundness: float = 0.0
    accepted: bool = False
    reason: str = ""
    source: str = ""
    score: float = 0.0


@dataclass(slots=True)
class ContourExtractionSettings:
    algorithm_backend: str = ALGORITHM_BACKEND_LEGACY
    sem_noise_level: str = "medium"
    sem_polarity: str = "auto"
    sem_preserve_hierarchy: bool = True
    extraction_profile: str = "conductors"
    object_type: str = "conductor"
    output_mode: str = "polygon"
    retrieval_mode: str = "RETR_EXTERNAL"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    epsilon: float = 2.0
    epsilon_relative: bool = False
    preserve_corners: bool = False
    min_polygon_angle: float = 0.0
    min_area: float = 10.0
    max_area: float | None = None
    min_perimeter: float = 10.0
    max_perimeter: float | None = None
    min_points: int = 3
    min_bbox_width: int = 0
    max_bbox_width: int | None = None
    min_bbox_height: int = 0
    max_bbox_height: int | None = None
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float | None = None
    exclude_border_touching: bool = False
    min_solidity: float = 0.0
    min_extent: float = 0.0
    min_via_width: int = 0
    max_via_width: int | None = None
    min_via_height: int = 0
    max_via_height: int | None = None
    via_size_mode: str = VIA_SIZE_MODE_RANGE
    via_search_mode: str = VIA_SEARCH_MODE_HYBRID
    fixed_via_widths: list[int] = field(default_factory=list)
    fixed_via_heights: list[int] = field(default_factory=list)
    via_channel_mode: str = VIA_CHANNEL_MODE_GRAYSCALE
    via_white_range_enabled: bool = True
    via_white_range_min: int = 200
    via_white_range_max: int = 255
    via_black_range_enabled: bool = False
    via_black_range_min: int = 0
    via_black_range_max: int = 30
    via_min_roundness: float = 40.0
    via_min_score: float = 0.35
    via_min_contrast: float = 14.0
    via_min_edge_coverage: float = 0.45
    via_spot_line_suppression: float = 0.65
    via_template_min_score: float = 0.35
    via_template_images: list[Any] = field(default_factory=list)
    debug_enabled: bool = False
    debug_gradient_map_enabled: bool = False
    min_hierarchy_depth: int = 0
    max_hierarchy_depth: int | None = None
    max_hole_area_ratio: float | None = None
    conductor_gradient_enabled: bool = False
    conductor_gradient_min_strength: float = 18.0
    conductor_gradient_band_radius: int = 3
    edge_method: str = "sobel"
    via_gradient_edge_method: str = ""
    conductor_gradient_edge_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_profile": self.extraction_profile,
            "algorithm_backend": normalize_algorithm_backend(self.algorithm_backend),
            "sem_noise_level": self.sem_noise_level,
            "sem_polarity": self.sem_polarity,
            "sem_preserve_hierarchy": self.sem_preserve_hierarchy,
            "object_type": self.object_type,
            "output_mode": self.output_mode,
            "retrieval_mode": self.retrieval_mode,
            "approximation_mode": self.approximation_mode,
            "epsilon": self.epsilon,
            "epsilon_relative": self.epsilon_relative,
            "preserve_corners": self.preserve_corners,
            "min_polygon_angle": self.min_polygon_angle,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_perimeter": self.min_perimeter,
            "min_points": self.min_points,
            "max_perimeter": self.max_perimeter,
            "min_bbox_width": self.min_bbox_width,
            "max_bbox_width": self.max_bbox_width,
            "min_bbox_height": self.min_bbox_height,
            "max_bbox_height": self.max_bbox_height,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "exclude_border_touching": self.exclude_border_touching,
            "min_solidity": self.min_solidity,
            "min_extent": self.min_extent,
            "min_via_width": self.min_via_width,
            "max_via_width": self.max_via_width,
            "min_via_height": self.min_via_height,
            "max_via_height": self.max_via_height,
            "via_size_mode": normalize_via_size_mode(self.via_size_mode),
            "via_search_mode": normalize_via_search_mode(self.via_search_mode),
            "fixed_via_widths": list(self.fixed_via_widths),
            "fixed_via_heights": list(self.fixed_via_heights),
            "via_channel_mode": normalize_via_channel_mode(self.via_channel_mode),
            "via_white_range_enabled": self.via_white_range_enabled,
            "via_white_range_min": self.via_white_range_min,
            "via_white_range_max": self.via_white_range_max,
            "via_black_range_enabled": self.via_black_range_enabled,
            "via_black_range_min": self.via_black_range_min,
            "via_black_range_max": self.via_black_range_max,
            "via_min_roundness": self.via_min_roundness,
            "via_min_score": self.via_min_score,
            "via_min_contrast": self.via_min_contrast,
            "via_min_edge_coverage": self.via_min_edge_coverage,
            "via_spot_line_suppression": self.via_spot_line_suppression,
            "via_template_min_score": self.via_template_min_score,
            "via_template_images": _serialize_template_images(self.via_template_images),
            "debug_enabled": self.debug_enabled,
            "debug_gradient_map_enabled": self.debug_gradient_map_enabled,
            "min_hierarchy_depth": self.min_hierarchy_depth,
            "max_hierarchy_depth": self.max_hierarchy_depth,
            "max_hole_area_ratio": self.max_hole_area_ratio,
            "conductor_gradient_enabled": self.conductor_gradient_enabled,
            "conductor_gradient_min_strength": self.conductor_gradient_min_strength,
            "conductor_gradient_band_radius": self.conductor_gradient_band_radius,
            "edge_method": self.edge_method,
            "via_gradient_edge_method": self.via_gradient_edge_method,
            "conductor_gradient_edge_method": self.conductor_gradient_edge_method,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContourExtractionSettings:
        max_area = payload.get("max_area")
        max_perimeter = payload.get("max_perimeter")
        max_bbox_width = payload.get("max_bbox_width")
        max_bbox_height = payload.get("max_bbox_height")
        max_aspect_ratio = payload.get("max_aspect_ratio")
        max_via_width = payload.get("max_via_width")
        max_via_height = payload.get("max_via_height")
        max_hierarchy_depth = payload.get("max_hierarchy_depth")
        max_hole_area_ratio = payload.get("max_hole_area_ratio")
        white_range_enabled = payload.get("via_white_range_enabled", payload.get("via_white_threshold_enabled", True))
        white_range_min = payload.get("via_white_range_min", payload.get("via_white_threshold", 200))
        white_range_max = payload.get("via_white_range_max", 255)
        black_range_enabled = payload.get("via_black_range_enabled", payload.get("via_black_threshold_enabled", False))
        black_range_min = payload.get("via_black_range_min", 0)
        black_range_max = payload.get("via_black_range_max", payload.get("via_black_threshold", 30))
        if payload.get("via_threshold_range_enabled", False) and "via_white_range_min" not in payload:
            white_range_enabled = True
            white_range_min = payload.get("via_threshold_range_min", white_range_min)
            white_range_max = payload.get("via_threshold_range_max", white_range_max)
        return cls(
            extraction_profile=str(payload.get("extraction_profile", "conductors")),
            algorithm_backend=normalize_algorithm_backend(payload.get("algorithm_backend", ALGORITHM_BACKEND_LEGACY)),
            sem_noise_level=str(payload.get("sem_noise_level", "medium") or "medium"),
            sem_polarity=str(payload.get("sem_polarity", "auto") or "auto"),
            sem_preserve_hierarchy=bool(payload.get("sem_preserve_hierarchy", True)),
            object_type=str(payload.get("object_type", "conductor")),
            output_mode=str(payload.get("output_mode", "polygon")),
            retrieval_mode=str(payload.get("retrieval_mode", "RETR_EXTERNAL")),
            approximation_mode=str(payload.get("approximation_mode", "CHAIN_APPROX_SIMPLE")),
            epsilon=float(payload.get("epsilon", 2.0)),
            epsilon_relative=bool(payload.get("epsilon_relative", False)),
            preserve_corners=bool(payload.get("preserve_corners", False)),
            min_polygon_angle=max(0.0, min(180.0, float(payload.get("min_polygon_angle", 0.0)))),
            min_area=float(payload.get("min_area", 10.0)),
            max_area=None if max_area in (None, "", 0, 0.0) else float(max_area),
            min_perimeter=float(payload.get("min_perimeter", 10.0)),
            min_points=max(3, int(payload.get("min_points", 3))),
            max_perimeter=None if max_perimeter in (None, "", 0, 0.0) else float(max_perimeter),
            min_bbox_width=max(0, int(payload.get("min_bbox_width", 0))),
            max_bbox_width=None if max_bbox_width in (None, "", 0, 0.0) else max(1, int(max_bbox_width)),
            min_bbox_height=max(0, int(payload.get("min_bbox_height", 0))),
            max_bbox_height=None if max_bbox_height in (None, "", 0, 0.0) else max(1, int(max_bbox_height)),
            min_aspect_ratio=max(0.0, float(payload.get("min_aspect_ratio", 0.0))),
            max_aspect_ratio=None if max_aspect_ratio in (None, "", 0, 0.0) else float(max_aspect_ratio),
            exclude_border_touching=bool(payload.get("exclude_border_touching", False)),
            min_solidity=max(0.0, float(payload.get("min_solidity", 0.0))),
            min_extent=max(0.0, float(payload.get("min_extent", 0.0))),
            min_via_width=max(0, int(payload.get("min_via_width", 0))),
            max_via_width=None if max_via_width in (None, "", 0, 0.0) else max(1, int(max_via_width)),
            min_via_height=max(0, int(payload.get("min_via_height", 0))),
            max_via_height=None if max_via_height in (None, "", 0, 0.0) else max(1, int(max_via_height)),
            via_size_mode=normalize_via_size_mode(payload.get("via_size_mode", VIA_SIZE_MODE_RANGE)),
            via_search_mode=normalize_via_search_mode(payload.get("via_search_mode", VIA_SEARCH_MODE_HYBRID)),
            fixed_via_widths=parse_integer_value_list(payload.get("fixed_via_widths")),
            fixed_via_heights=parse_integer_value_list(payload.get("fixed_via_heights")),
            via_channel_mode=normalize_via_channel_mode(payload.get("via_channel_mode", VIA_CHANNEL_MODE_GRAYSCALE)),
            via_white_range_enabled=bool(white_range_enabled),
            via_white_range_min=max(0, min(255, int(white_range_min))),
            via_white_range_max=max(0, min(255, int(white_range_max))),
            via_black_range_enabled=bool(black_range_enabled),
            via_black_range_min=max(0, min(255, int(black_range_min))),
            via_black_range_max=max(0, min(255, int(black_range_max))),
            via_min_roundness=max(0.0, float(payload.get("via_min_roundness", 40.0))),
            via_min_score=max(0.0, min(1.0, float(payload.get("via_min_score", 0.35)))),
            via_min_contrast=max(0.0, min(255.0, float(payload.get("via_min_contrast", 14.0)))),
            via_min_edge_coverage=max(0.0, min(1.0, float(payload.get("via_min_edge_coverage", 0.45)))),
            via_template_min_score=max(0.0, min(1.0, float(payload.get("via_template_min_score", 0.35)))),
            via_template_images=_parse_template_images(payload.get("via_template_images", [])),
            via_spot_line_suppression=max(0.0, min(1.0, float(payload.get("via_spot_line_suppression", 0.65)))),
            debug_enabled=bool(payload.get("debug_enabled", False)),
            debug_gradient_map_enabled=bool(payload.get("debug_gradient_map_enabled", False)),
            min_hierarchy_depth=max(0, int(payload.get("min_hierarchy_depth", 0))),
            max_hierarchy_depth=None if max_hierarchy_depth in (None, "", 0, 0.0) else max(0, int(max_hierarchy_depth)),
            max_hole_area_ratio=None
            if max_hole_area_ratio in (None, "", 0, 0.0)
            else max(0.0, float(max_hole_area_ratio)),
            conductor_gradient_enabled=bool(payload.get("conductor_gradient_enabled", False)),
            conductor_gradient_min_strength=max(
                0.0, min(255.0, float(payload.get("conductor_gradient_min_strength", 18.0)))
            ),
            conductor_gradient_band_radius=max(0, min(25, int(payload.get("conductor_gradient_band_radius", 3)))),
            edge_method=str(payload.get("edge_method", "sobel") or "sobel"),
            via_gradient_edge_method=str(payload.get("via_gradient_edge_method", "") or ""),
            conductor_gradient_edge_method=str(payload.get("conductor_gradient_edge_method", "") or ""),
        )


@dataclass(slots=True)
class DisplaySettings:
    external_color: str = "#28C76F"
    hole_color: str = "#FF9F43"
    selected_color: str = "#00CFE8"
    vertex_color: str = "#FF4D6D"
    line_width: float = 2.0
    vertex_size: float = 7.0
    fill_opacity: float = 0.18
    show_vertices: bool = True
    show_labels: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_color": self.external_color,
            "hole_color": self.hole_color,
            "selected_color": self.selected_color,
            "vertex_color": self.vertex_color,
            "line_width": self.line_width,
            "vertex_size": self.vertex_size,
            "fill_opacity": self.fill_opacity,
            "show_vertices": self.show_vertices,
            "show_labels": self.show_labels,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DisplaySettings:
        return cls(
            external_color=str(payload.get("external_color", "#28C76F")),
            hole_color=str(payload.get("hole_color", "#FF9F43")),
            selected_color=str(payload.get("selected_color", "#00CFE8")),
            vertex_color=str(payload.get("vertex_color", "#FF4D6D")),
            line_width=float(payload.get("line_width", 2.0)),
            vertex_size=float(payload.get("vertex_size", 7.0)),
            fill_opacity=float(payload.get("fill_opacity", 0.18)),
            show_vertices=bool(payload.get("show_vertices", True)),
            show_labels=bool(payload.get("show_labels", False)),
        )


@dataclass(slots=True)
class SaveOptions:
    save_cif: bool = True
    save_json: bool = False
    save_csv: bool = False
    save_txt: bool = False
    save_svg: bool = False
    save_preview: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "save_cif": self.save_cif,
            "save_json": self.save_json,
            "save_csv": self.save_csv,
            "save_txt": self.save_txt,
            "save_svg": self.save_svg,
            "save_preview": self.save_preview,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SaveOptions:
        return cls(
            save_cif=bool(payload.get("save_cif", True)),
            save_json=bool(payload.get("save_json", False)),
            save_csv=bool(payload.get("save_csv", False)),
            save_txt=bool(payload.get("save_txt", False)),
            save_svg=bool(payload.get("save_svg", False)),
            save_preview=bool(payload.get("save_preview", True)),
        )


@dataclass(slots=True)
class ImageProcessingState:
    image_path: str
    source_image: Any | None = None
    preprocessed_image: Any | None = None
    pipeline_config: dict[str, Any] | None = None
    mask_image: Any | None = None
    polygons: list[PolygonData] = field(default_factory=list)
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    loaded_cif_path: str | None = None
    reference_polygons: list[PolygonData] = field(default_factory=list)


@dataclass(slots=True)
class BatchImageResult:
    image_path: str
    source_image: Any | None
    preprocessed_image: Any | None
    pipeline_config: dict[str, Any] | None
    mask_image: Any | None
    polygons: list[PolygonData]
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class BatchProcessingOptions:
    max_workers: int = 4
    output_directory: str | None = None
    save_options: SaveOptions = field(default_factory=SaveOptions)


def base_name_from_path(path: str) -> str:
    return Path(path).stem

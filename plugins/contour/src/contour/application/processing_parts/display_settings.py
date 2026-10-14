"""processing: display settings."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from contour.domain import PolygonData

from contour.application.processing_parts.module_state import VIA_DISPLAY_MODE_CIRCLE

from contour.application.processing_parts.pipeline_step_config import normalize_via_display_mode

@dataclass(slots=True)
class DisplaySettings:
    external_color: str = "#28C76F"
    hole_color: str = "#FF9F43"
    selected_color: str = "#00CFE8"
    via_selection_color: str = "#FACC15"
    conductor_hover_highlight_color: str = "#FB923C"
    vertex_color: str = "#FF4D6D"
    line_width: float = 2.0
    vertex_size: float = 7.0
    fill_opacity: float = 0.18
    show_vertices: bool = True
    show_labels: bool = False
    via_display_mode: str = VIA_DISPLAY_MODE_CIRCLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_color": self.external_color,
            "hole_color": self.hole_color,
            "selected_color": self.selected_color,
            "via_selection_color": self.via_selection_color,
            "conductor_hover_highlight_color": self.conductor_hover_highlight_color,
            "vertex_color": self.vertex_color,
            "line_width": self.line_width,
            "vertex_size": self.vertex_size,
            "fill_opacity": self.fill_opacity,
            "show_vertices": self.show_vertices,
            "show_labels": self.show_labels,
            "via_display_mode": normalize_via_display_mode(self.via_display_mode),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DisplaySettings:
        return cls(
            external_color=str(payload.get("external_color", "#28C76F")),
            hole_color=str(payload.get("hole_color", "#FF9F43")),
            selected_color=str(payload.get("selected_color", "#00CFE8")),
            via_selection_color=str(payload.get("via_selection_color", "#FACC15")),
            conductor_hover_highlight_color=str(payload.get("conductor_hover_highlight_color", "#FB923C")),
            vertex_color=str(payload.get("vertex_color", "#FF4D6D")),
            line_width=float(payload.get("line_width", 2.0)),
            vertex_size=float(payload.get("vertex_size", 7.0)),
            fill_opacity=float(payload.get("fill_opacity", 0.18)),
            show_vertices=bool(payload.get("show_vertices", True)),
            show_labels=bool(payload.get("show_labels", False)),
            via_display_mode=normalize_via_display_mode(payload.get("via_display_mode", VIA_DISPLAY_MODE_CIRCLE)),
        )

@dataclass(slots=True)
class SaveOptions:
    save_cif: bool = True
    save_cv: bool = False
    save_json: bool = False
    save_csv: bool = False
    save_txt: bool = False
    save_svg: bool = False
    save_preview: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "save_cif": self.save_cif,
            "save_cv": self.save_cv,
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
            save_cv=bool(payload.get("save_cv", False)),
            save_json=bool(payload.get("save_json", False)),
            save_csv=bool(payload.get("save_csv", False)),
            save_txt=bool(payload.get("save_txt", False)),
            save_svg=bool(payload.get("save_svg", False)),
            save_preview=bool(payload.get("save_preview", False)),
        )

@dataclass(slots=True)
class ImageProcessingState:
    image_path: str
    source_image: Any | None = None
    preprocessed_image: Any | None = None
    pipeline_config: dict[str, Any] | None = None
    mask_image: Any | None = None
    polygons: list[PolygonData] = field(default_factory=list)
    recognition_base_polygons: list[PolygonData] = field(default_factory=list)
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    metal_overlay_polygons: dict[str, list[PolygonData]] = field(default_factory=dict)
    loaded_cif_path: str | None = None
    loaded_cif_sha256: str | None = None
    reference_polygons: list[PolygonData] = field(default_factory=list)
    #: Cached ``image_has_changes``; ``None`` means recompute on next query.
    polygons_dirty: bool | None = None
    #: Prefetch/worker result of ``polygons_needing_repair``; ``None`` means not scanned yet.
    polygons_needing_repair: dict[int, list[str]] | None = None

@dataclass(slots=True)
class BatchImageResult:
    image_path: str
    source_image: Any | None
    preprocessed_image: Any | None
    pipeline_config: dict[str, Any] | None
    mask_image: Any | None
    polygons: list[PolygonData]
    recognition_base_polygons: list[PolygonData] = field(default_factory=list)
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    metal_overlay_polygons: dict[str, list[PolygonData]] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None

@dataclass(slots=True)
class BatchFrameTiming:
    image_loading_ms: float = 0.0
    morphology_ms: float = 0.0
    threshold_ms: float = 0.0
    contour_extraction_ms: float = 0.0
    postprocessing_ms: float = 0.0
    saving_ms: float = 0.0
    total_frame_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "image_loading_ms": float(self.image_loading_ms),
            "morphology_ms": float(self.morphology_ms),
            "threshold_ms": float(self.threshold_ms),
            "contour_extraction_ms": float(self.contour_extraction_ms),
            "postprocessing_ms": float(self.postprocessing_ms),
            "saving_ms": float(self.saving_ms),
            "total_frame_ms": float(self.total_frame_ms),
        }

@dataclass(slots=True)
class BatchImageMetadata:
    image_path: str
    polygon_count: int = 0
    saved_files: dict[str, str] = field(default_factory=dict)
    timing: BatchFrameTiming = field(default_factory=BatchFrameTiming)
    worker_pid: int | None = None
    error: str | None = None

@dataclass(slots=True)
class BatchProcessingOptions:
    max_workers: int = 4
    chunk_size: int = 16
    output_directory: str | None = None
    save_options: SaveOptions = field(default_factory=SaveOptions)

def base_name_from_path(path: str) -> str:
    return Path(path).stem

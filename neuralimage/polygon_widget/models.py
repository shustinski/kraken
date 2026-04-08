from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Point = tuple[float, float]


@dataclass(slots=True)
class PolygonData:
    id: int
    points: list[Point]
    is_hole: bool = False
    parent_id: int | None = None
    area: float = 0.0
    perimeter: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    def clone(self) -> "PolygonData":
        return PolygonData(
            id=self.id,
            points=[(float(x), float(y)) for x, y in self.points],
            is_hole=self.is_hole,
            parent_id=self.parent_id,
            area=float(self.area),
            perimeter=float(self.perimeter),
            bbox=tuple(int(v) for v in self.bbox),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "points": [[float(x), float(y)] for x, y in self.points],
            "is_hole": self.is_hole,
            "parent_id": self.parent_id,
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "bbox": [int(v) for v in self.bbox],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolygonData":
        return cls(
            id=int(payload["id"]),
            points=[(float(x), float(y)) for x, y in payload.get("points", [])],
            is_hole=bool(payload.get("is_hole", False)),
            parent_id=payload.get("parent_id"),
            area=float(payload.get("area", 0.0)),
            perimeter=float(payload.get("perimeter", 0.0)),
            bbox=tuple(int(v) for v in payload.get("bbox", (0, 0, 0, 0))),
        )


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

    def clone(self) -> "PipelineStepConfig":
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
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineStepConfig":
        return cls(
            operation=str(payload["operation"]),
            name=str(payload.get("name") or payload["operation"]),
            enabled=bool(payload.get("enabled", True)),
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(slots=True)
class ContourExtractionSettings:
    retrieval_mode: str = "RETR_EXTERNAL"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    epsilon: float = 2.0
    epsilon_relative: bool = False
    min_area: float = 10.0
    max_area: float | None = None
    min_perimeter: float = 10.0
    min_points: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "approximation_mode": self.approximation_mode,
            "epsilon": self.epsilon,
            "epsilon_relative": self.epsilon_relative,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_perimeter": self.min_perimeter,
            "min_points": self.min_points,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContourExtractionSettings":
        max_area = payload.get("max_area")
        return cls(
            retrieval_mode=str(payload.get("retrieval_mode", "RETR_EXTERNAL")),
            approximation_mode=str(payload.get("approximation_mode", "CHAIN_APPROX_SIMPLE")),
            epsilon=float(payload.get("epsilon", 2.0)),
            epsilon_relative=bool(payload.get("epsilon_relative", False)),
            min_area=float(payload.get("min_area", 10.0)),
            max_area=None if max_area in (None, "", 0, 0.0) else float(max_area),
            min_perimeter=float(payload.get("min_perimeter", 10.0)),
            min_points=max(3, int(payload.get("min_points", 3))),
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
    show_labels: bool = True

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
    def from_dict(cls, payload: dict[str, Any]) -> "DisplaySettings":
        return cls(
            external_color=str(payload.get("external_color", "#28C76F")),
            hole_color=str(payload.get("hole_color", "#FF9F43")),
            selected_color=str(payload.get("selected_color", "#00CFE8")),
            vertex_color=str(payload.get("vertex_color", "#FF4D6D")),
            line_width=float(payload.get("line_width", 2.0)),
            vertex_size=float(payload.get("vertex_size", 7.0)),
            fill_opacity=float(payload.get("fill_opacity", 0.18)),
            show_vertices=bool(payload.get("show_vertices", True)),
            show_labels=bool(payload.get("show_labels", True)),
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
    def from_dict(cls, payload: dict[str, Any]) -> "SaveOptions":
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
    mask_image: Any | None = None
    polygons: list[PolygonData] = field(default_factory=list)


@dataclass(slots=True)
class BatchImageResult:
    image_path: str
    source_image: Any | None
    preprocessed_image: Any | None
    mask_image: Any | None
    polygons: list[PolygonData]
    saved_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class BatchProcessingOptions:
    max_workers: int = 4
    output_directory: str | None = None
    save_options: SaveOptions = field(default_factory=SaveOptions)


def base_name_from_path(path: str) -> str:
    return Path(path).stem

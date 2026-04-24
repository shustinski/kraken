"""Unified internal result types (dataclasses; JSON-serializable via ``to_json_dict``)."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

Point2 = tuple[float, float]


class OutputShapeKind(StrEnum):
    """How geometry is represented in the UI and export."""

    POLYGON = "polygon"
    AXIS_ALIGNED_BOX = "box"
    ROTATED_BOX = "rbox"


class AppMode(StrEnum):
    """High-level user-facing processing mode (mutually exclusive workflows)."""

    CONTOUR = "contour"  # conductors / general topology extraction
    VIA = "via"  # via / transition hole detection only


class SemPolarity(StrEnum):
    """Whether foreground is darker or brighter than the local background."""

    AUTO = "auto"
    DARK_FOREGROUND = "dark_fg"
    BRIGHT_FOREGROUND = "bright_fg"


@dataclass(frozen=True, slots=True)
class ImageRef:
    """Logical link to the source image (file or in-memory)."""

    path: str | None
    width: int
    height: int
    dtype: str = "uint8"
    channels: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class RotatedBox:
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HierarchicalComponent:
    """One polygon component with parent/child links (OpenCV ``RETR_TREE`` ordering)."""

    id: int
    contour_index: int
    parent_id: int | None
    depth: int
    is_hole: bool
    points: list[Point2]
    area: float
    bbox_xywh: tuple[int, int, int, int]
    rotated_box: RotatedBox | None = None
    score: float = 1.0
    source_strategy: str = "mask"


@dataclass(slots=True)
class ViaHit:
    """Single via detection (center + extent + quality scores)."""

    center_x: float
    center_y: float
    width: float
    height: float
    score: float
    strategy: str
    contrast: float = 0.0
    edge_strength: float = 0.0
    annulus_coverage: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_axis_aligned_box(self) -> tuple[int, int, int, int]:
        x0 = round(self.center_x - self.width * 0.5)
        y0 = round(self.center_y - self.height * 0.5)
        return (x0, y0, max(1, round(self.width)), max(1, round(self.height)))


@dataclass(slots=True)
class ContourExtractionOutput:
    """Result of mask + vectorization; supports polygon / box and hierarchy metadata."""

    image: ImageRef
    mode: AppMode
    output_kind: OutputShapeKind
    filled_mask: Any  # np.ndarray uint8, kept as Any to avoid import cycles in typing-only paths
    components: list[HierarchicalComponent]
    strategy_used: str
    quality_notes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "image": {
                "id": self.image.id,
                "path": self.image.path,
                "width": self.image.width,
                "height": self.image.height,
            },
            "mode": str(self.mode),
            "output_kind": str(self.output_kind),
            "filled_mask": None,
            "components": [self._component_dict(c) for c in self.components],
            "strategy_used": self.strategy_used,
            "quality_notes": list(self.quality_notes),
        }

    @staticmethod
    def _component_dict(component: HierarchicalComponent) -> dict[str, Any]:
        return {
            "id": component.id,
            "contour_index": component.contour_index,
            "parent_id": component.parent_id,
            "depth": component.depth,
            "is_hole": component.is_hole,
            "points": [[float(x), float(y)] for x, y in component.points],
            "area": float(component.area),
            "bbox": [int(v) for v in component.bbox_xywh],
            "rotated_box": None if component.rotated_box is None else component.rotated_box.to_dict(),
            "score": float(component.score),
            "source_strategy": component.source_strategy,
        }


@dataclass(slots=True)
class ViaDetectionOutput:
    image: ImageRef
    mode: AppMode
    output_kind: OutputShapeKind
    hits: list[ViaHit]
    selected_strategy: str
    attempt_log: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "image": {
                "id": self.image.id,
                "path": self.image.path,
                "width": self.image.width,
                "height": self.image.height,
            },
            "mode": "via",
            "output_kind": str(self.output_kind),
            "selected_strategy": self.selected_strategy,
            "attempt_log": list(self.attempt_log),
            "vias": [
                {
                    "center": [h.center_x, h.center_y],
                    "size": [h.width, h.height],
                    "score": h.score,
                    "strategy": h.strategy,
                    "contrast": h.contrast,
                    "edge_strength": h.edge_strength,
                    "annulus_coverage": h.annulus_coverage,
                    "box_xywh": list(ViaHit.to_axis_aligned_box(h)),
                    "extra": dict(h.extra),
                }
                for h in self.hits
            ],
        }


def polygon_area(points: list[Point2]) -> float:
    if len(points) < 3:
        return 0.0
    accum = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        accum += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(accum) * 0.5


def min_area_rect_angle_deg(contour: list[Point2]) -> float | None:
    """Optional rotated box: OpenCV minAreaRect on float32 Nx1x2; angle in image coords."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if len(contour) < 3:
        return None
    arr = np.array([[[float(x), float(y)]] for x, y in contour], dtype=np.float32)
    (_cx, _cy), (_w, _h), angle = cv2.minAreaRect(arr)
    if not math.isfinite(angle):
        return None
    return float(angle)

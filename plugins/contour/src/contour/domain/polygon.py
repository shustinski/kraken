from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any

Point = tuple[float, float]


def integer_coord(value: float) -> int:
    return int(round(float(value)))


def _coord_is_integer(value: object) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        rounded = round(value)
        return rounded == value or abs(value - rounded) < 1e-9
    return False


def _points_are_integer(points: list[Point]) -> bool:
    if not points:
        return True
    return all(_coord_is_integer(x_coord) and _coord_is_integer(y_coord) for x_coord, y_coord in points)


def integer_point(point: Point) -> tuple[int, int]:
    x_coord, y_coord = point
    if isinstance(x_coord, int) and isinstance(y_coord, int):
        return x_coord, y_coord
    return integer_coord(x_coord), integer_coord(y_coord)


def integer_points(points: list[Point]) -> list[tuple[int, int]]:
    if _points_are_integer(points):
        return [(int(x_coord), int(y_coord)) for x_coord, y_coord in points]
    return [integer_point(point) for point in points]


@dataclass(slots=True)
class PolygonData:
    id: int
    points: list[Point]
    is_hole: bool = False
    parent_id: int | None = None
    category: str = "conductor"
    shape_hint: str = "polygon"
    area: float = 0.0
    perimeter: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    #: Metal recovery / debug only; not written to CIF by default.
    reject_reason: str = ""
    _points_normalized: InitVar[bool] = False

    def __post_init__(self, _points_normalized: bool) -> None:
        if _points_normalized:
            self.points = list(self.points)
            return
        self.points = integer_points(self.points)

    def clone(self) -> PolygonData:
        return PolygonData(
            id=self.id,
            points=list(self.points),
            is_hole=self.is_hole,
            parent_id=self.parent_id,
            category=str(self.category),
            shape_hint=str(self.shape_hint),
            area=float(self.area),
            perimeter=float(self.perimeter),
            bbox=(int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])),
            reject_reason=str(self.reject_reason),
            _points_normalized=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "points": [[integer_coord(x_coord), integer_coord(y_coord)] for x_coord, y_coord in self.points],
            "is_hole": self.is_hole,
            "parent_id": self.parent_id,
            "category": self.category,
            "shape_hint": self.shape_hint,
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "bbox": [int(value) for value in self.bbox],
            **({"reject_reason": self.reject_reason} if str(self.reject_reason).strip() else {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolygonData:
        raw_bbox = payload.get("bbox", (0, 0, 0, 0))
        bbox_values = [int(value) for value in raw_bbox]
        while len(bbox_values) < 4:
            bbox_values.append(0)
        bbox: tuple[int, int, int, int] = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        return cls(
            id=int(payload["id"]),
            points=integer_points([(float(x_coord), float(y_coord)) for x_coord, y_coord in payload.get("points", [])]),
            is_hole=bool(payload.get("is_hole", False)),
            parent_id=payload.get("parent_id"),
            category=str(payload.get("category", "conductor")),
            shape_hint=str(payload.get("shape_hint", "polygon")),
            area=float(payload.get("area", 0.0)),
            perimeter=float(payload.get("perimeter", 0.0)),
            bbox=bbox,
            reject_reason=str(payload.get("reject_reason", "") or ""),
            _points_normalized=True,
        )

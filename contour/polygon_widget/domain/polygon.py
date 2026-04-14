from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Point = tuple[float, float]


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

    def clone(self) -> "PolygonData":
        return PolygonData(
            id=self.id,
            points=[(float(x_coord), float(y_coord)) for x_coord, y_coord in self.points],
            is_hole=self.is_hole,
            parent_id=self.parent_id,
            category=str(self.category),
            shape_hint=str(self.shape_hint),
            area=float(self.area),
            perimeter=float(self.perimeter),
            bbox=tuple(int(value) for value in self.bbox),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "points": [[float(x_coord), float(y_coord)] for x_coord, y_coord in self.points],
            "is_hole": self.is_hole,
            "parent_id": self.parent_id,
            "category": self.category,
            "shape_hint": self.shape_hint,
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "bbox": [int(value) for value in self.bbox],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolygonData":
        return cls(
            id=int(payload["id"]),
            points=[(float(x_coord), float(y_coord)) for x_coord, y_coord in payload.get("points", [])],
            is_hole=bool(payload.get("is_hole", False)),
            parent_id=payload.get("parent_id"),
            category=str(payload.get("category", "conductor")),
            shape_hint=str(payload.get("shape_hint", "polygon")),
            area=float(payload.get("area", 0.0)),
            perimeter=float(payload.get("perimeter", 0.0)),
            bbox=tuple(int(value) for value in payload.get("bbox", (0, 0, 0, 0))),
        )

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
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


_REASON_UNSET = object()


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
    #: Recognition confidence for detected objects, normally 0..100.
    recognition_score: float | None = None
    #: Metal recovery / debug only; not written to CIF by default.
    reject_reason: str = ""
    #: Authored self-touching CIF ring for KLayout-compatible cv2 fill display.
    cif_paint_ring: list[Point] = field(default_factory=list)
    _description_invalid: bool | None = field(default=None, compare=False, repr=False)
    _description_invalid_reason: object = field(default=_REASON_UNSET, compare=False, repr=False)
    _points_normalized: InitVar[bool] = False

    def __setattr__(self, name: str, value: object) -> None:
        if name == "points":
            try:
                current = object.__getattribute__(self, "points")
            except AttributeError:
                current = None
            if current is not None and list(current) == list(value):
                if current is not value:
                    object.__setattr__(self, "points", list(value))
                return
        elif name in {"category", "shape_hint"}:
            try:
                current = object.__getattribute__(self, name)
            except AttributeError:
                current = None
            if current == value:
                return

        object.__setattr__(self, name, value)
        if name in {"points", "category", "shape_hint"}:
            object.__setattr__(self, "_description_invalid", None)
            object.__setattr__(self, "_description_invalid_reason", _REASON_UNSET)

    def __post_init__(self, _points_normalized: bool) -> None:
        if _points_normalized:
            self.points = list(self.points)
            return
        self.points = integer_points(self.points)

    def description_is_invalid(self) -> bool:
        """True for truly invalid outlines (e.g. self-crossing), not CIF keyhole bridges.

        ``repeated_vertex`` keyholes remain diagnosable via ``description_invalid_reason``
        but are valid CIF and must not trigger red marking / auto-repair offers.
        """

        cached = self._description_invalid
        if cached is not None:
            return cached
        reason = self.description_invalid_reason()
        return reason is not None and reason != "repeated_vertex"

    def description_invalid_reason(self) -> str | None:
        """Stable reason code for outline diagnostics, or ``None`` when the ring is simple.

        ``repeated_vertex`` is retained for keyhole tooling; use ``description_is_invalid``
        to decide whether the description itself is considered broken.
        """

        cached_reason = self._description_invalid_reason
        if cached_reason is not _REASON_UNSET:
            return None if cached_reason is None else str(cached_reason)
        if str(self.category) == "via" or str(self.shape_hint) == "box":
            object.__setattr__(self, "_description_invalid", False)
            object.__setattr__(self, "_description_invalid_reason", None)
            return None
        from .polygon_ring import closed_ring_description_invalid_reason

        reason = closed_ring_description_invalid_reason(self.points)
        object.__setattr__(
            self,
            "_description_invalid",
            reason is not None and reason != "repeated_vertex",
        )
        object.__setattr__(self, "_description_invalid_reason", reason)
        return reason

    def clone(self) -> PolygonData:
        cloned = PolygonData(
            id=self.id,
            points=list(self.points),
            is_hole=self.is_hole,
            parent_id=self.parent_id,
            category=str(self.category),
            shape_hint=str(self.shape_hint),
            area=float(self.area),
            perimeter=float(self.perimeter),
            bbox=(int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])),
            recognition_score=(
                None if self.recognition_score is None else float(self.recognition_score)
            ),
            reject_reason=str(self.reject_reason),
            cif_paint_ring=list(self.cif_paint_ring),
            _points_normalized=True,
        )
        object.__setattr__(cloned, "_description_invalid", self._description_invalid)
        object.__setattr__(cloned, "_description_invalid_reason", self._description_invalid_reason)
        return cloned

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
            **(
                {"recognition_score": float(self.recognition_score)}
                if self.recognition_score is not None
                else {}
            ),
            **({"reject_reason": self.reject_reason} if str(self.reject_reason).strip() else {}),
            **({"cif_paint_ring": [[integer_coord(x_coord), integer_coord(y_coord)] for x_coord, y_coord in self.cif_paint_ring]} if self.cif_paint_ring else {}),
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
            recognition_score=(
                float(payload["recognition_score"])
                if payload.get("recognition_score") is not None
                else None
            ),
            reject_reason=str(payload.get("reject_reason", "") or ""),
            cif_paint_ring=integer_points(
                [(float(x_coord), float(y_coord)) for x_coord, y_coord in payload.get("cif_paint_ring", [])]
            ),
            _points_normalized=True,
        )

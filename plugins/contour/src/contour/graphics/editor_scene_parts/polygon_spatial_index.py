"""editor_scene: polygon spatial index."""
from __future__ import annotations
from math import ceil, floor
from contour.domain import PolygonData

from contour.graphics.editor_scene_parts.module_state import _POLYGON_SPATIAL_CELL_SIZE

class _PolygonSpatialIndex:
    """Incremental bbox index for pointer and local-geometry queries."""

    def __init__(self, cell_size: float = _POLYGON_SPATIAL_CELL_SIZE) -> None:
        self._cell_size = max(1.0, float(cell_size))
        self._cells: dict[tuple[int, int], set[int]] = {}
        self._polygon_cells: dict[int, tuple[tuple[int, int], ...]] = {}

    def _cells_for_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        padding: float = 0.0,
    ):
        x_coord, y_coord, width, height = bbox
        pad = max(0.0, float(padding))
        left = floor((float(x_coord) - pad) / self._cell_size)
        right = floor((float(x_coord) + float(width) + pad) / self._cell_size)
        top = floor((float(y_coord) - pad) / self._cell_size)
        bottom = floor((float(y_coord) + float(height) + pad) / self._cell_size)
        for cell_y in range(top, bottom + 1):
            for cell_x in range(left, right + 1):
                yield cell_x, cell_y

    def clear(self) -> None:
        self._cells.clear()
        self._polygon_cells.clear()

    def upsert(self, polygon: PolygonData) -> None:
        self.remove(polygon.id)
        cells = tuple(self._cells_for_bbox(polygon.bbox))
        self._polygon_cells[polygon.id] = cells
        for cell in cells:
            self._cells.setdefault(cell, set()).add(polygon.id)

    def remove(self, polygon_id: int) -> None:
        for cell in self._polygon_cells.pop(polygon_id, ()):
            polygon_ids = self._cells.get(cell)
            if polygon_ids is None:
                continue
            polygon_ids.discard(polygon_id)
            if not polygon_ids:
                self._cells.pop(cell, None)

    def query_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        padding: float = 0.0,
    ) -> set[int]:
        result: set[int] = set()
        for cell in self._cells_for_bbox(bbox, padding=padding):
            result.update(self._cells.get(cell, ()))
        return result

    def query_point(self, x_coord: float, y_coord: float, *, padding: float = 0.0) -> set[int]:
        pad = max(0.0, float(padding))
        return self.query_bbox(
            (
                floor(float(x_coord) - pad),
                floor(float(y_coord) - pad),
                max(1, ceil(pad * 2.0)),
                max(1, ceil(pad * 2.0)),
            )
        )

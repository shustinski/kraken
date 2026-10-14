"""editor_scene: contact spatial index."""
from __future__ import annotations
from math import floor
from PyQt6.QtCore import QRectF

from contour.graphics.editor_scene_parts.module_state import _CONTACT_PASTE_SPATIAL_CELL_SIZE

class _ContactSpatialIndex:
    def __init__(self, minimum_distance: float) -> None:
        self._minimum_distance = max(0.0, float(minimum_distance))
        self._minimum_distance_sq = self._minimum_distance * self._minimum_distance
        self._entries: list[tuple[float, float, QRectF]] = []
        self._cell_entries: dict[tuple[int, int], list[int]] = {}

    @staticmethod
    def _cells_for_rect(rect: QRectF):
        normalized = rect.normalized()
        cell_size = _CONTACT_PASTE_SPATIAL_CELL_SIZE
        left = floor(float(normalized.left()) / cell_size)
        right = floor(float(normalized.right()) / cell_size)
        top = floor(float(normalized.top()) / cell_size)
        bottom = floor(float(normalized.bottom()) / cell_size)
        for cell_y in range(top, bottom + 1):
            for cell_x in range(left, right + 1):
                yield cell_x, cell_y

    def add(self, center_x: float, center_y: float, rect: QRectF) -> None:
        entry_index = len(self._entries)
        stored_rect = QRectF(rect)
        self._entries.append((float(center_x), float(center_y), stored_rect))
        for cell in self._cells_for_rect(stored_rect):
            self._cell_entries.setdefault(cell, []).append(entry_index)

    def conflicts(self, center_x: float, center_y: float, rect: QRectF) -> bool:
        search_rect = QRectF(rect)
        if self._minimum_distance > 0.0:
            radius = self._minimum_distance
            search_rect = search_rect.united(
                QRectF(
                    float(center_x) - radius,
                    float(center_y) - radius,
                    radius * 2.0,
                    radius * 2.0,
                )
            )
        visited: set[int] = set()
        for cell in self._cells_for_rect(search_rect):
            for entry_index in self._cell_entries.get(cell, ()):
                if entry_index in visited:
                    continue
                visited.add(entry_index)
                existing_x, existing_y, existing_rect = self._entries[entry_index]
                if rect.intersects(existing_rect):
                    return True
                if (
                    self._minimum_distance_sq > 0.0
                    and (
                        (float(center_x) - existing_x) ** 2
                        + (float(center_y) - existing_y) ** 2
                    )
                    < self._minimum_distance_sq
                ):
                    return True
        return False

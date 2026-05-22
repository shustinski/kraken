"""Virtualized frame path list backed by QAbstractListModel (no per-row QWidget allocation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt

from .item_status_painting import FRAME_STATUS_ROLE


class FramePathListModel(QAbstractListModel):
    def __init__(self, widget: Any | None = None) -> None:
        super().__init__(widget)
        self._widget = widget
        self._paths: list[str] = []
        self._stems: list[str] = []
        self._path_to_row: dict[str, int] = {}
        self._role_cache: dict[tuple[int, int], object] = {}

    def set_paths(self, paths: list[str]) -> None:
        self.beginResetModel()
        self._paths = [str(Path(path)) for path in paths]
        self._stems = [Path(path).stem for path in self._paths]
        self._path_to_row = {path: row for row, path in enumerate(self._paths)}
        self._role_cache.clear()
        self.endResetModel()

    def paths(self) -> tuple[str, ...]:
        return tuple(self._paths)

    def index_for_path(self, path: str | Path) -> int | None:
        normalized = str(Path(path))
        return self._path_to_row.get(normalized)

    def path_at(self, row: int) -> str | None:
        if row < 0 or row >= len(self._paths):
            return None
        return self._paths[row]

    def invalidate_path(self, path: str | Path) -> None:
        row = self.index_for_path(path)
        if row is None:
            return
        for key in [key for key in self._role_cache if key[0] == row]:
            self._role_cache.pop(key, None)
        top_left = self.index(row, 0)
        self.dataChanged.emit(
            top_left,
            top_left,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.ForegroundRole,
                Qt.ItemDataRole.BackgroundRole,
                FRAME_STATUS_ROLE,
            ],
        )

    def invalidate_all_rows(self) -> None:
        if not self._paths:
            return
        self._role_cache.clear()
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._paths) - 1, 0)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.ForegroundRole,
                Qt.ItemDataRole.BackgroundRole,
                FRAME_STATUS_ROLE,
            ],
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._paths)

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)):
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= len(self._paths):
            return None
        path = self._paths[row]
        widget = self._widget
        if role == int(Qt.ItemDataRole.DisplayRole):
            return self._stems[row]
        if role == int(Qt.ItemDataRole.ToolTipRole):
            if widget is not None and getattr(widget, "_ui_language", "en") == "ru":
                return f"Путь к файлу: {path}"
            return f"File path: {path}"
        if role == int(Qt.ItemDataRole.UserRole):
            return path
        if role in (
            int(Qt.ItemDataRole.BackgroundRole),
            int(Qt.ItemDataRole.ForegroundRole),
            FRAME_STATUS_ROLE,
        ):
            cache_key = (row, int(role))
            if cache_key in self._role_cache:
                return self._role_cache[cache_key]
            if widget is not None and hasattr(widget, "_image_list_model_item_data"):
                value = widget._image_list_model_item_data(path, role)
                self._role_cache[cache_key] = value
                return value
        return None


class FramePathFilterProxyModel(QSortFilterProxyModel):
    """Optional match-only filter over :class:`FramePathListModel`."""

    def __init__(self, widget: Any | None = None) -> None:
        super().__init__(widget)
        self._widget = widget

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        del source_parent
        widget = self._widget
        if widget is None or not bool(getattr(widget, "_asset_filter_match_only", False)):
            return True
        model = getattr(widget, "_image_list_model", None)
        if model is None:
            return True
        path = model.path_at(source_row)
        if not path:
            return False
        return bool(widget._image_path_has_matching_vector(path))

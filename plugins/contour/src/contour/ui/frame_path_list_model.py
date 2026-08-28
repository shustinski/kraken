"""Virtualized frame path list backed by QAbstractListModel (no per-row QWidget allocation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtWidgets import QListView, QListWidgetItem

from .item_status_painting import FRAME_STATUS_ROLE

_UNSET = object()
_ROLE_DISPLAY = int(Qt.ItemDataRole.DisplayRole)
_ROLE_TOOLTIP = int(Qt.ItemDataRole.ToolTipRole)
_ROLE_USER = int(Qt.ItemDataRole.UserRole)
_ROLE_BACKGROUND = int(Qt.ItemDataRole.BackgroundRole)
_ROLE_FOREGROUND = int(Qt.ItemDataRole.ForegroundRole)
_ROLE_FRAME_STATUS = int(FRAME_STATUS_ROLE)
_PAINT_ROLES = (_ROLE_BACKGROUND, _ROLE_FOREGROUND, _ROLE_FRAME_STATUS)
_PAINT_ROLE_SET = frozenset(_PAINT_ROLES)


class FramePathListModel(QAbstractListModel):
    def __init__(self, widget: Any | None = None) -> None:
        super().__init__(widget)
        self._widget = widget
        self._paths: list[str] = []
        self._stems: list[str] = []
        self._tooltips_ru: list[str] = []
        self._tooltips_en: list[str] = []
        self._path_to_row: dict[str, int] = {}
        self._count = 0
        self._paint_status: list[object] = []
        self._paint_background: list[object] = []
        self._paint_foreground: list[object] = []

    def set_paths(self, paths: list[str]) -> None:
        self.beginResetModel()
        # Callers pass already-normalized paths from the widget index; keep as-is.
        self._paths = list(paths)
        self._count = len(self._paths)
        self._stems = [Path(path).stem for path in self._paths]
        self._tooltips_ru = [f"Путь к файлу: {path}" for path in self._paths]
        self._tooltips_en = [f"File path: {path}" for path in self._paths]
        self._path_to_row = {path: row for row, path in enumerate(self._paths)}
        self._reset_paint_caches()
        self.endResetModel()

    def _reset_paint_caches(self) -> None:
        count = self._count
        self._paint_status = [_UNSET] * count
        self._paint_background = [_UNSET] * count
        self._paint_foreground = [_UNSET] * count

    def paths(self) -> tuple[str, ...]:
        return tuple(self._paths)

    def index_for_path(self, path: str | Path) -> int | None:
        if isinstance(path, str):
            row = self._path_to_row.get(path)
            if row is not None:
                return row
        return self._path_to_row.get(str(Path(path)))

    def path_at(self, row: int) -> str | None:
        if row < 0 or row >= self._count:
            return None
        return self._paths[row]

    def invalidate_path(self, path: str | Path) -> None:
        row = self.index_for_path(path)
        if row is None:
            return
        self._paint_status[row] = _UNSET
        self._paint_background[row] = _UNSET
        self._paint_foreground[row] = _UNSET
        top_left = self.index(row, 0)
        self.dataChanged.emit(top_left, top_left, list(_PAINT_ROLES))

    def invalidate_row_range(self, first_row: int, last_row: int) -> None:
        if self._count <= 0:
            return
        first = max(0, int(first_row))
        last = min(self._count - 1, int(last_row))
        if last < first:
            return
        for row in range(first, last + 1):
            self._paint_status[row] = _UNSET
            self._paint_background[row] = _UNSET
            self._paint_foreground[row] = _UNSET
        self.dataChanged.emit(self.index(first, 0), self.index(last, 0), list(_PAINT_ROLES))

    def invalidate_visible_rows(self, view: QListView | None) -> None:
        """Refresh only rows intersecting the viewport (large lists)."""

        if view is None or self._count <= 0:
            return
        viewport = view.viewport()
        if viewport is None:
            self.invalidate_all_rows()
            return
        rect = viewport.rect()
        top = view.indexAt(rect.topLeft())
        bottom = view.indexAt(rect.bottomLeft())
        if not top.isValid():
            return
        first = top.row()
        last = bottom.row() if bottom.isValid() else first
        model = view.model()
        if isinstance(model, QSortFilterProxyModel):
            source_first = model.mapToSource(top).row()
            source_last = model.mapToSource(bottom).row() if bottom.isValid() else source_first
            if source_first < 0:
                return
            if source_last < source_first:
                source_first, source_last = source_last, source_first
            self.invalidate_row_range(source_first, source_last)
            return
        if last < first:
            first, last = last, first
        self.invalidate_row_range(first, last)

    def invalidate_all_rows(self) -> None:
        if self._count <= 0:
            return
        self._reset_paint_caches()
        top_left = self.index(0, 0)
        bottom_right = self.index(self._count - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, list(_PAINT_ROLES))

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._count

    def _fill_paint_roles(self, row: int) -> None:
        widget = self._widget
        if widget is None or not hasattr(widget, "_image_list_model_row_paint_roles"):
            self._paint_status[row] = None
            self._paint_background[row] = None
            self._paint_foreground[row] = None
            return
        roles = widget._image_list_model_row_paint_roles(self._paths[row])
        self._paint_status[row] = roles.get(_ROLE_FRAME_STATUS)
        self._paint_background[row] = roles.get(_ROLE_BACKGROUND)
        self._paint_foreground[row] = roles.get(_ROLE_FOREGROUND)

    def data(self, index: QModelIndex, role: int = _ROLE_DISPLAY):
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= self._count:
            return None
        role_value = int(role)
        if role_value == _ROLE_DISPLAY:
            return self._stems[row]
        if role_value == _ROLE_USER:
            return self._paths[row]
        if role_value == _ROLE_TOOLTIP:
            if self._widget is not None and getattr(self._widget, "_ui_language", "en") == "ru":
                return self._tooltips_ru[row]
            return self._tooltips_en[row]
        if role_value not in _PAINT_ROLE_SET:
            return None
        if self._paint_status[row] is _UNSET:
            self._fill_paint_roles(row)
        if role_value == _ROLE_FRAME_STATUS:
            return self._paint_status[row]
        if role_value == _ROLE_BACKGROUND:
            return self._paint_background[row]
        return self._paint_foreground[row]


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


class FramePathListView(QListView):
    """QListView with the legacy QListWidget methods Contour still calls in tests and shims."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._manual_items: list[QListWidgetItem] = []

    def clear_manual_items(self) -> None:
        self._manual_items.clear()

    def count(self) -> int:
        if self._manual_items:
            return len(self._manual_items)
        model = self.model()
        return 0 if model is None else int(model.rowCount())

    def item(self, row: int) -> QListWidgetItem | None:
        row = int(row)
        if self._manual_items:
            return self._manual_items[row] if 0 <= row < len(self._manual_items) else None
        model = self.model()
        if model is None or row < 0 or row >= model.rowCount():
            return None
        index = model.index(row, 0)
        item = QListWidgetItem(str(model.data(index, Qt.ItemDataRole.DisplayRole) or ""))
        for role in (
            Qt.ItemDataRole.ToolTipRole,
            Qt.ItemDataRole.UserRole,
            Qt.ItemDataRole.BackgroundRole,
            Qt.ItemDataRole.ForegroundRole,
        ):
            value = model.data(index, role)
            if value is not None:
                item.setData(role, value)
        return item

    def clear(self) -> None:
        self._manual_items.clear()
        model = self.model()
        source = model.sourceModel() if isinstance(model, QSortFilterProxyModel) else model
        if isinstance(source, FramePathListModel):
            source.set_paths([])

    def addItem(self, item: QListWidgetItem) -> None:
        self._manual_items.append(item)
        paths = [str(existing.data(Qt.ItemDataRole.UserRole) or existing.text()) for existing in self._manual_items]
        model = self.model()
        source = model.sourceModel() if isinstance(model, QSortFilterProxyModel) else model
        if isinstance(source, FramePathListModel):
            source.set_paths(paths)

    def currentRow(self) -> int:
        index = self.currentIndex()
        return int(index.row()) if index.isValid() else -1

    def setCurrentRow(self, row: int) -> None:
        model = self.model()
        if model is None or row < 0 or row >= model.rowCount():
            self.clearSelection()
            self.setCurrentIndex(QModelIndex())
            return
        self.setCurrentIndex(model.index(int(row), 0))

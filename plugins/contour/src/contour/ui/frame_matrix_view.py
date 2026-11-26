"""Contour frame matrix on the shared Kraken matrix engine.

``FrameMatrixGraphicsView`` is a ``FrameMatrixWidget``, like the Kraken layer
matrix; it differs in the frame actions it declares. For Contour's callers it
keeps the small ``QListWidget`` subset they use: list index ``i`` is the cell
``(i % columns + 1, i // columns + 1)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from PyQt6.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QWidget

from kraken_core.frame_matrix.actions import FrameMatrixAction
from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
from kraken_core.frame_matrix.models import MatrixSession, MatrixViewportRequest
from kraken_core.frame_matrix.qt import FrameContext, FrameMatrixWidget, FrameSelection

from ..application.frame_lod import PyramidFrameStore
from .frame_matrix_sources import ContourFrameAssetSource, ContourFrameEntry, ContourMatrixDataSource
from .frame_matrix_view_parts.module_state import (
    _FRAME_ID_ROLE,
    _STATUS_MARKER_COLORS,
    _THUMBNAIL_LOD_LEVELS,
    _THUMBNAIL_PIXMAP_ROLE,
)
from .frame_matrix_view_parts.trailing_frame_number import _trailing_frame_number
from .item_status_painting import FRAME_STATUS_ROLE
from .large_dataset import clamp_thumbnail_source_size

_PATH_ROLE = int(Qt.ItemDataRole.UserRole)
_DEFAULT_CELL = QSize(64, 48)

Translator = Callable[[str], str]


class FrameMatrixGraphicsView(FrameMatrixWidget):
    """Contour matrix: frame navigation plus "delete frame" / "clear frame"."""

    itemClicked = pyqtSignal(QListWidgetItem)
    thumbnailLodChanged = pyqtSignal(int, int)
    frameNavigationRequested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._items: list[QListWidgetItem] = []
        self._columns = 1
        self._current_row = -1
        self._icon_size = QSize(_DEFAULT_CELL)
        self._grid_size = QSize(_DEFAULT_CELL)
        self._overlap_pixels_x = 0
        self._overlap_pixels_y = 0
        self._pyramid_store: PyramidFrameStore | None = None
        self._pyramid_generation = 0
        self._navigator_lods: tuple[int, ...] = ()
        self._image_cache: dict[int, tuple[int, QImage]] = {}
        self._emitted_thumbnail_source_size = QSize()
        self._translate: Translator = lambda key: key
        self._suppress_matrix_refresh = False
        self._session_revision = 0
        self._matrix_data_source = ContourMatrixDataSource()
        self._matrix_asset_source = ContourFrameAssetSource(self._matrix_data_source)
        super().__init__(
            1,
            1,
            parent=parent,
            data_source=self._matrix_data_source,
            asset_source=self._matrix_asset_source,
            thumbnail_store=MemoryThumbnailStore(),
        )
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.setInterval(0)
        self._publish_timer.timeout.connect(self._publish_now)
        self.set_status_colors({status: color.name() for status, color in _STATUS_MARKER_COLORS.items()})
        self._apply_cell_geometry()
        self.lodChanged.connect(self._emit_thumbnail_lod_if_needed)
        self.viewportChanged.connect(self._emit_thumbnail_lod_if_needed)
        self._publish_now()

    # -- actions ---------------------------------------------------------

    def set_action_translator(self, translate: Translator) -> None:
        self._translate = translate

    def frame_actions(self, context: FrameContext) -> Sequence[FrameMatrixAction]:
        return (
            FrameMatrixAction("delete_frame", lambda _ctx: self._translate("delete_frame_action"), self._is_frame),
            FrameMatrixAction("clear_frame", lambda _ctx: self._translate("clear_frame_action"), self._is_frame),
        )

    def _is_frame(self, context: FrameContext) -> bool:
        index = self.index_for_coordinate(context.x, context.y)
        return index >= 0 and not self._items[index].isHidden()

    # -- coordinates -----------------------------------------------------

    def columns(self) -> int:
        return self._columns

    def setColumns(self, columns: int) -> None:
        columns = max(1, int(columns))
        if columns != self._columns:
            self._columns = columns
            self._schedule_publish()

    def coordinate_for_index(self, index: int) -> tuple[int, int]:
        return index % self._columns + 1, index // self._columns + 1

    def index_for_coordinate(self, x: int, y: int) -> int:
        if not 1 <= int(x) <= self._columns:
            return -1
        index = (int(y) - 1) * self._columns + (int(x) - 1)
        return index if 0 <= index < len(self._items) else -1

    def frame_id_for_coordinate(self, x: int, y: int) -> int:
        index = self.index_for_coordinate(x, y)
        if index < 0:
            return -1
        return self._frame_id_for_item(self._items[index], index)

    @staticmethod
    def _frame_id_for_item(item: QListWidgetItem, index: int) -> int:
        value = item.data(_FRAME_ID_ROLE)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return index

    # -- QListWidget subset used by Contour -------------------------------

    def widget(self) -> FrameMatrixGraphicsView:
        return self

    def setWidgetResizable(self, _enabled: bool) -> None:
        return

    def setViewMode(self, _mode: QListWidget.ViewMode) -> None:
        return

    def setResizeMode(self, _mode: QListWidget.ResizeMode) -> None:
        return

    def setMovement(self, _movement: QListWidget.Movement) -> None:
        return

    def setSelectionMode(self, _mode: QAbstractItemView.SelectionMode) -> None:
        return

    def setUniformItemSizes(self, _enabled: bool) -> None:
        return

    def setWrapping(self, _enabled: bool) -> None:
        return

    def setSpacing(self, _spacing: int) -> None:
        return

    def setFixedSize(self, *_args) -> None:  # type: ignore[override]
        """Contour sized its old list to the layout extent; this view scrolls itself."""

    def setIconSize(self, size: QSize) -> None:
        self._icon_size = QSize(size)

    def setGridSize(self, size: QSize) -> None:
        self._grid_size = QSize(size)
        self._apply_cell_geometry()

    def setFrameOverlapPixels(self, pixels: int, pixels_y: int | None = None) -> None:
        self._overlap_pixels_x = max(0, int(pixels))
        self._overlap_pixels_y = self._overlap_pixels_x if pixels_y is None else max(0, int(pixels_y))
        self._apply_cell_geometry()

    def _apply_cell_geometry(self) -> None:
        self.set_cell_geometry(
            max(1, self._grid_size.width()),
            max(1, self._grid_size.height()),
            self._overlap_pixels_x,
            self._overlap_pixels_y,
        )

    def count(self) -> int:
        return len(self._items)

    def item(self, index: int) -> QListWidgetItem | None:
        if 0 <= int(index) < len(self._items):
            return self._items[int(index)]
        return None

    def addItem(self, item: QListWidgetItem) -> None:
        self._items.append(item)
        self._schedule_publish()

    def clear(self) -> None:
        self._items.clear()
        self._image_cache.clear()
        self._current_row = -1
        self.clear_selection()
        self._schedule_publish()

    def currentRow(self) -> int:
        return self._current_row

    def currentItem(self) -> QListWidgetItem | None:
        return self.item(self._current_row)

    def setCurrentRow(self, row: int) -> None:
        row = int(row)
        self._current_row = row if 0 <= row < len(self._items) else -1
        self._sync_selection()

    def clearSelection(self) -> None:
        self._current_row = -1
        self.clear_selection()

    def setCurrentFrameId(self, frame_id: int | None) -> None:
        if frame_id is None:
            self.clearSelection()
            return
        self.setCurrentRow(int(frame_id))

    def visualItemRect(self, item: QListWidgetItem) -> QRect:
        try:
            index = self._items.index(item)
        except ValueError:
            return QRect()
        x, y = self.coordinate_for_index(index)
        if not self._published(x, y):
            pitch_x, pitch_y = self.cell_pitch()
            cell_w, cell_h = self.cell_size()
            return QRect(round((x - 1) * pitch_x), round((y - 1) * pitch_y), round(cell_w), round(cell_h))
        return self.scene_rect_for_frame(x, y).toRect()

    def ensureRowVisible(self, row: int) -> None:
        item = self.item(row)
        if item is None:
            return
        x, y = self.coordinate_for_index(int(row))
        if not self._published(x, y):
            self._publish_now()
        self.ensureVisible(self.scene_rect_for_frame(x, y), 0, 0)

    def setUpdatesEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        super().setUpdatesEnabled(enabled)
        if enabled and not self._suppress_matrix_refresh:
            self._schedule_publish()

    def doItemsLayout(self) -> None:
        self._schedule_publish()

    def refreshItems(self) -> None:
        self._schedule_publish()

    def refreshVisibleRegion(self) -> None:
        self._schedule_visible_update()

    def refreshThumbnailIndexes(self, indexes: list[int]) -> None:
        self.updateThumbnailPixmaps(indexes)

    def updateThumbnailPixmaps(self, indexes: list[int]) -> None:
        if indexes:
            self._schedule_publish()

    # -- pyramid LOD --------------------------------------------------------

    def setPyramidFrameStore(self, store: PyramidFrameStore | None) -> None:
        if store is self._pyramid_store:
            self.refreshVisibleRegion()
            return
        self._pyramid_store = store if store is not None and store.has_lod() else None
        self._navigator_lods = ()
        if self._pyramid_store is not None:
            try:
                lod_source = (
                    self._pyramid_store.available_lods_hint()
                    if self._pyramid_store.__class__ is PyramidFrameStore
                    else self._pyramid_store.available_lods()
                )
                lods = tuple(sorted(int(lod) for lod in lod_source))
            except Exception:
                self._pyramid_store = None
            else:
                self._navigator_lods = lods[-3:]
        self._pyramid_generation += 1
        self._matrix_asset_source.set_pyramid_store(self._pyramid_store)
        self._schedule_publish()

    def navigatorLods(self) -> tuple[int, ...]:
        return tuple(self._navigator_lods)

    def shutdownPyramidLoading(self) -> None:
        self._pyramid_generation += 1
        self._matrix_asset_source.set_pyramid_store(None)
        self._publish_timer.stop()

    # -- thumbnail size follows zoom ----------------------------------------

    def _thumbnail_lod(self) -> float:
        zoom = max(1.0e-6, self.zoom_factor())
        for level in _THUMBNAIL_LOD_LEVELS:
            if zoom <= level:
                return level
        return _THUMBNAIL_LOD_LEVELS[-1]

    def thumbnailSourceSize(self) -> QSize:
        lod = self._thumbnail_lod()
        width, height = clamp_thumbnail_source_size(
            round(self._icon_size.width() * lod),
            round(self._icon_size.height() * lod),
        )
        return QSize(width, height)

    def _emit_thumbnail_lod_if_needed(self, *_args) -> None:
        size = self.thumbnailSourceSize()
        if size == self._emitted_thumbnail_source_size:
            return
        self._emitted_thumbnail_source_size = QSize(size)
        self.thumbnailLodChanged.emit(size.width(), size.height())

    def _current_viewport_request(self) -> MatrixViewportRequest:
        request = super()._current_viewport_request()
        size = self.thumbnailSourceSize()
        return MatrixViewportRequest(
            request.bounds,
            lod=request.lod,
            generation=request.generation,
            asset_width=max(1, size.width()),
            asset_height=max(1, size.height()),
        )

    # -- publishing the frame list to the engine ----------------------------

    def _schedule_publish(self) -> None:
        if self._suppress_matrix_refresh or not self.updatesEnabled():
            return
        self._publish_timer.start()

    def _publish_now(self) -> None:
        self._publish_timer.stop()
        entries = tuple(self._entries())
        self._matrix_data_source.replace(entries)
        width = self._columns
        height = max(1, math.ceil(len(self._items) / float(self._columns)))
        session = self.session()
        if session is None or (session.width, session.height) != (width, height):
            self._session_revision += 1
            self.set_session(
                MatrixSession(
                    namespace=f"contour-matrix-{id(self)}",
                    width=width,
                    height=height,
                    source_revision=str(self._session_revision),
                )
            )
        else:
            self.set_data_source(self._matrix_data_source)
        self._sync_selection()

    def _entries(self):
        live_ids = set()
        for index, item in enumerate(self._items):
            if item.isHidden():
                continue
            live_ids.add(id(item))
            x, y = self.coordinate_for_index(index)
            path = str(item.data(_PATH_ROLE) or "")
            text = item.text() or (Path(item.toolTip()).stem if item.toolTip() else "")
            image, revision = self._entry_image(item)
            yield ContourFrameEntry(
                index=self._frame_id_for_item(item, index),
                x=x,
                y=y,
                path=path,
                status=str(item.data(FRAME_STATUS_ROLE) or ""),
                label=_trailing_frame_number(text),
                tooltip=item.toolTip(),
                image=image,
                revision=revision,
            )
        for key in tuple(self._image_cache):
            if key not in live_ids:
                self._image_cache.pop(key, None)

    def _entry_image(self, item: QListWidgetItem):
        if self._pyramid_store is not None:
            return None, f"pyramid-{self._pyramid_generation}"
        pixmap = item.data(_THUMBNAIL_PIXMAP_ROLE)
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            self._image_cache.pop(id(item), None)
            return None, ""
        cached = self._image_cache.get(id(item))
        if cached is None or cached[0] != pixmap.cacheKey():
            cached = (pixmap.cacheKey(), pixmap.toImage())
            self._image_cache[id(item)] = cached
        return cached[1], f"pixmap-{cached[0]}"

    def _sync_selection(self) -> None:
        row = self._current_row
        if row < 0 or row >= len(self._items) or self._items[row].isHidden():
            self.clear_selection()
            return
        x, y = self.coordinate_for_index(row)
        if not self._published(x, y):
            # The next publish brings the matrix to this size and selects again.
            return
        self.set_selection(FrameSelection.single(x, y))

    def _published(self, x: int, y: int) -> bool:
        return x <= self.matrix_width and y <= self.matrix_height

    # -- mouse: single-frame navigation --------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        coordinate = self.frame_at_viewport_pos(event.position())
        index = -1 if coordinate is None else self.index_for_coordinate(*coordinate)
        if index < 0 or self._items[index].isHidden():
            super().mousePressEvent(event)
            return
        item = self._items[index]
        self.setCurrentRow(index)
        self.itemClicked.emit(item)
        self.frameNavigationRequested.emit(self._frame_id_for_item(item, index))
        event.accept()



__all__ = ["FrameMatrixGraphicsView"]

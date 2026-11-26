from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QSize, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QListWidgetItem

from contour.application.frame_lod import PyramidFrameStore
from contour.ui.frame_matrix_sources import (
    ContourFrameAssetSource,
    ContourFrameEntry,
    ContourMatrixDataSource,
    choose_pyramid_lod,
)
from contour.ui.frame_matrix_view import FrameMatrixGraphicsView
from kraken_core.frame_matrix.models import MatrixAssetRef, MatrixBounds, MatrixViewportRequest
from kraken_core.frame_matrix.qt import FrameContext, FrameMatrixWidget, FrameSelection

_PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 1001


class _Pyramid(PyramidFrameStore):
    def __init__(self) -> None:
        super().__init__([])

    def has_lod(self) -> bool:
        return True

    def available_lods(self) -> tuple[int, ...]:
        return (0, 1, 2)

    def get_frame_size(self, frame_id: int, lod: int = 0) -> tuple[int, int]:
        scale = 2 ** int(lod)
        return (256 // scale, 192 // scale)

    def get_thumbnail(self, frame_id: int, lod: int = 0, max_size: int = 256) -> np.ndarray:
        width, height = self.get_frame_size(frame_id, lod)
        return np.full((height, width, 3), 200, dtype=np.uint8)


def _matrix(count: int, columns: int) -> FrameMatrixGraphicsView:
    view = FrameMatrixGraphicsView()
    view.setColumns(columns)
    for index in range(count):
        item = QListWidgetItem(f"frame_{index:03d}")
        item.setData(Qt.ItemDataRole.UserRole, f"frame_{index:03d}.png")
        view.addItem(item)
    view._publish_now()
    return view


def _click(view: FrameMatrixGraphicsView, x: int, y: int, button=Qt.MouseButton.LeftButton) -> None:
    position = QPointF(view.mapFromScene(view.scene_rect_for_frame(x, y).center()))
    view.mousePressEvent(
        QMouseEvent(QEvent.Type.MouseButtonPress, position, button, button, Qt.KeyboardModifier.NoModifier)
    )


def test_contour_matrix_shares_the_kraken_matrix_class(_qt_application) -> None:
    assert issubclass(FrameMatrixGraphicsView, FrameMatrixWidget)


def test_list_index_maps_to_cells_by_columns(_qt_application) -> None:
    view = _matrix(5, 2)

    assert view.matrix_size() == (2, 3)
    assert [(entry.x, entry.y) for entry in view._matrix_data_source.entries()] == [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (1, 3),
    ]
    assert view.index_for_coordinate(2, 2) == 3
    assert view.index_for_coordinate(2, 3) == -1


def test_empty_slot_after_last_frame_has_no_actions(_qt_application) -> None:
    view = _matrix(3, 2)

    context = FrameContext(2, 2, view.cell_data(2, 2), FrameSelection.single(2, 2))
    assert view.visible_frame_actions(context) == ()


def test_click_navigates_to_the_frame(_qt_application) -> None:
    view = _matrix(4, 2)
    view.resize(400, 300)
    view.show()
    _qt_application.processEvents()
    clicked: list[QListWidgetItem] = []
    navigated: list[object] = []
    view.itemClicked.connect(clicked.append)
    view.frameNavigationRequested.connect(navigated.append)

    _click(view, 2, 1)

    assert navigated == [1]
    assert clicked == [view.item(1)]
    assert view.currentRow() == 1
    assert view.selection() == FrameSelection.single(2, 1)


def test_contour_actions_are_delete_and_clear(_qt_application) -> None:
    view = _matrix(2, 2)
    view.set_action_translator(lambda key: {"delete_frame_action": "Удалить", "clear_frame_action": "Очистить"}[key])
    context = FrameContext(1, 1, view.cell_data(1, 1), FrameSelection.single(1, 1))

    assert [(action.id, title) for action, title in view.visible_frame_actions(context)] == [
        ("delete_frame", "Удалить"),
        ("clear_frame", "Очистить"),
    ]


def test_cells_use_contour_geometry_and_overlap(_qt_application) -> None:
    view = _matrix(5, 4)
    view.setGridSize(QSize(64, 48))
    view.setFrameOverlapPixels(12, 9)

    assert view.visualItemRect(view.item(1)).x() == 64 - 12
    assert view.visualItemRect(view.item(4)).y() == 48 - 9


def test_asset_source_uses_loaded_thumbnail(_qt_application) -> None:
    image = QImage(64, 64, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    data = ContourMatrixDataSource([ContourFrameEntry(0, 1, 1, path="a.png", image=image, revision="r")])
    assets = ContourFrameAssetSource(data)

    payload = assets.load_asset(
        MatrixAssetRef("a.png", "r", metadata={"frame_id": 0}),
        width=64,
        height=48,
    )

    loaded = QImage.fromData(payload)
    assert (loaded.width(), loaded.height()) == (64, 48)


def test_asset_source_prefers_the_pyramid_level_that_covers_the_cell(_qt_application) -> None:
    store = _Pyramid()
    assets = ContourFrameAssetSource(ContourMatrixDataSource())
    assets.set_pyramid_store(store)

    assert choose_pyramid_lod(store, 0, 64, 48) == 2
    assert choose_pyramid_lod(store, 0, 128, 96) == 1
    assert choose_pyramid_lod(store, 0, 512, 384) == 0
    payload = assets.load_asset(MatrixAssetRef("a.png", "p", metadata={"frame_id": 0}), width=128, height=96)
    loaded = QImage.fromData(payload)
    assert (loaded.width(), loaded.height()) == (128, 96)


def test_data_source_returns_only_cells_inside_the_viewport(_qt_application) -> None:
    view = _matrix(6, 3)
    item = view.item(4)
    pixmap = QPixmap(8, 8)
    pixmap.fill()
    item.setData(_PIXMAP_ROLE, pixmap)
    view._publish_now()

    result = view._matrix_data_source.load_viewport(MatrixViewportRequest(MatrixBounds(2, 2, 3, 2)))

    assert [(entry.x, entry.y) for entry in result.items] == [(2, 2), (3, 2)]
    assert result.items[0].asset is not None
    assert result.items[1].asset is None

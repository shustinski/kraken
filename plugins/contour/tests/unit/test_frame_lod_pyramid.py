from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QListWidgetItem

import contour.application.frame_lod as frame_lod_module
from contour.application.frame_lod import FixedGridFrameLayout, PyramidFrameStore
from contour.graphics.editor_view import PolygonEditorView
from contour.ui.frame_matrix_view import FrameMatrixGraphicsView


class FakePyramidStore(PyramidFrameStore):
    def __init__(self) -> None:
        super().__init__([])

    def has_lod(self) -> bool:
        return True

    def available_lods(self) -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)

    def frame_count(self) -> int:
        return 12

    def get_frame_size(self, frame_id: int, lod: int = 0) -> tuple[int, int]:
        scale = 2**int(lod)
        return (64 // scale, 48 // scale)

    def get_frame(self, frame_id: int, lod: int = 0) -> np.ndarray:
        width, height = self.get_frame_size(frame_id, lod)
        return np.full((height, width, 3), int(frame_id) % 255, dtype=np.uint8)

    def get_thumbnail(self, frame_id: int, lod: int = 0, max_size: int = 256) -> np.ndarray:
        return self.get_frame(frame_id, lod)


def test_fixed_grid_layout_maps_frame_ids_and_scene_positions() -> None:
    layout = FixedGridFrameLayout(frame_count=12, columns=4, frame_store=FakePyramidStore(), gap=4)

    assert layout.frame_id_to_row_col(6) == (1, 2)
    assert layout.row_col_to_frame_id(2, 3) == 11
    assert layout.frame_id_to_scene_rect(6, 1) == QRectF(72.0, 28.0, 32.0, 24.0)
    assert layout.scene_pos_to_frame_id(73.0, 29.0, 1) == 6
    assert layout.scene_pos_to_frame_id(107.0, 29.0, 1) is None


def test_editor_pyramid_mode_is_opt_in_and_selects_lod(_qt_application) -> None:
    view = PolygonEditorView()
    store = FakePyramidStore()

    view.set_pyramid_frame_store(store, frame_count=store.frame_count(), columns=4, enabled=True)

    assert view.pyramid_mode_enabled() is True
    assert view.choose_lod(1.0, 4) == 0
    assert view.choose_lod(0.20, 4) >= 1
    view.set_current_frame_id(7, center=False, emit_signal=False)
    assert view.current_frame_id() == 7


def test_frame_matrix_uses_only_three_most_zoomed_out_lods(_qt_application) -> None:
    view = FrameMatrixGraphicsView()
    store = FakePyramidStore()
    for index in range(store.frame_count()):
        item = QListWidgetItem("")
        item.setData(257, f"frame_{index}.png")
        item.setData(1258, index)
        view.addItem(item)

    view.setPyramidFrameStore(store)

    assert view.navigatorLods() == (2, 3, 4)


def test_frame_matrix_store_setup_does_not_decode_source_image(_qt_application, tmp_path, monkeypatch) -> None:
    path = tmp_path / "frame_000.png"
    path.write_bytes(b"placeholder")
    store = PyramidFrameStore.from_image_paths([path])
    view = FrameMatrixGraphicsView()

    def _fail_decode(_path):
        raise AssertionError("source image decoded on UI path")

    monkeypatch.setattr(frame_lod_module, "_load_source_image", _fail_decode)

    view.setPyramidFrameStore(store)

    assert view.navigatorLods() == (5, 6, 7)


def test_pyramid_store_forms_lods_from_image_paths(tmp_path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index:03d}.png"
        cv2.imwrite(str(path), np.full((16, 20, 3), index * 40, dtype=np.uint8))
        paths.append(path)

    store = PyramidFrameStore.from_image_paths(paths)

    assert store.has_lod() is True
    assert store.frame_count() == 2
    assert store.available_lods() == (0, 1)
    assert store.get_frame_size(0, 0) == (20, 16)
    assert store.get_frame(1, 0).shape == (16, 20, 3)
    assert store.get_frame_size(1, 1) == (10, 8)
    assert store.get_frame(1, 1).shape == (8, 10, 3)

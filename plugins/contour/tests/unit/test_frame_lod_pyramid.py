from __future__ import annotations

import numpy as np
import cv2
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QListWidgetItem

from contour.application.frame_lod import FixedGridFrameLayout, PyramidFrameStore, ZarrFrameStore
from contour.graphics.editor_view import PolygonEditorView
from contour.ui.frame_matrix_view import FrameMatrixGraphicsView


class FakePyramidStore(PyramidFrameStore):
    def __init__(self) -> None:
        super().__init__([])

    def has_zarr(self) -> bool:
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


def test_zarr_store_without_zarr_does_not_fallback_to_image_files(tmp_path) -> None:
    image_path = tmp_path / "frame_001.png"
    image_path.write_bytes(b"not-used")
    store = ZarrFrameStore.from_image_paths([image_path])

    assert store.has_zarr() is False
    assert store.available_lods() == ()
    assert store.frame_count() == 0
    try:
        store.get_frame(0, 0)
    except RuntimeError as exc:
        assert "Zarr LOD" in str(exc)
    else:
        raise AssertionError("ZarrFrameStore should not decode image files as a fallback")


def test_zarr_store_forms_pyramid_from_image_paths(tmp_path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index:03d}.png"
        cv2.imwrite(str(path), np.full((16, 20, 3), index * 40, dtype=np.uint8))
        paths.append(path)

    store = ZarrFrameStore.from_image_paths(paths)

    assert store.has_zarr() is True
    assert not (tmp_path / "frames.zarr").exists()
    assert store.frame_count() == 2
    assert store.available_lods() == (0,)
    assert store.get_frame_size(0, 0) == (20, 16)
    assert store.get_frame(1, 0).shape == (16, 20, 3)
    assert store.needs_lod_build() is True

    assert ZarrFrameStore._build_zarr_pyramid(paths, tmp_path / "frames.zarr") == tmp_path / "frames.zarr"
    store.refresh()

    assert store.available_lods()[0] == 0
    assert store.available_lods()[1] == 1
    assert store.needs_lod_build() is False
    import zarr

    root = zarr.open(str(tmp_path / "frames.zarr"), mode="r")
    assert "lod_0" not in set(root.keys())
    assert "lod_1" in set(root.keys())

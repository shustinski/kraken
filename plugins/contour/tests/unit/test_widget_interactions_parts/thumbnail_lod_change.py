"""test_widget_interactions: test thumbnail lod change."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QGraphicsView,
    QListWidgetItem,
)


class ThumbnailLodChangeMixin:
    def test_neighbor_overlay_navigation_loads_main_editor_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        self.widget._on_neighbor_frame_activated(paths[1])

        self.assertEqual(loaded, [paths[1]])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[1])

    def test_ctrl_arrow_navigates_selected_frame_when_frame_matrix_enabled(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget.neighbor_columns_spin.setValue(3)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(9)
        self.widget.thumbnail_grid.setCurrentRow(4)
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)

        self.assertEqual(loaded, [paths[5], paths[8]])
        self.assertEqual(self.widget.thumbnail_grid.currentRow(), 8)

    def test_ctrl_arrow_does_not_navigate_when_frame_matrix_disabled(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[1]
        self.widget.show_frame_matrix_checkbox.setChecked(False)
        self.widget.thumbnail_grid.addItem(QListWidgetItem(""))
        self.widget.thumbnail_grid.addItem(QListWidgetItem(""))
        self.widget.thumbnail_grid.setCurrentRow(1)
        loaded: list[str] = []
        self.widget.load_image = lambda path: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)

        self.assertEqual(loaded, [])
        self.assertEqual(self.widget.thumbnail_grid.currentRow(), 1)

    def test_thumbnail_request_size_follows_matrix_lod(self) -> None:
        self.widget.thumbnail_grid._matrix_zoom = 4.0

        self.assertEqual(self.widget._thumbnail_request_size(), (256, 192))

    def test_thumbnail_request_size_scales_to_max_matrix_lod(self) -> None:
        self.widget.thumbnail_grid._matrix_zoom = 32.0

        self.assertEqual(self.widget._thumbnail_request_size(), (512, 384))

    def test_frame_matrix_uses_editor_view_render_optimizations(self) -> None:
        matrix = self.widget.thumbnail_grid

        self.assertEqual(matrix.viewportUpdateMode(), QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.assertTrue(matrix.cacheMode() & QGraphicsView.CacheModeFlag.CacheBackground)
        self.assertTrue(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontSavePainterState)
        self.assertFalse(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)
        self.assertTrue(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)

        matrix._enter_zoom_render_mode()

        self.assertFalse(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
        self.assertTrue(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

        matrix._leave_zoom_render_mode()

        self.assertTrue(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
        self.assertFalse(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

    def test_thumbnail_lod_change_requeues_visible_frames_at_new_size(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        self.widget._thumbnail_loaded_generation[path] = self.widget._thumbnail_generation
        self.widget._thumbnail_loaded_sizes[path] = (64, 48)
        queued: list[tuple[int, str, tuple[int, int]]] = []

        def _capture_queue(generation: int, queued_path: str) -> None:
            queued.append((generation, queued_path, self.widget._thumbnail_request_size()))

        self.widget._queue_thumbnail_load = _capture_queue  # type: ignore[method-assign]
        self.widget.thumbnail_grid._matrix_zoom = 4.0

        self.widget._on_thumbnail_lod_changed()

        self.assertEqual(queued, [(self.widget._thumbnail_generation, path, (256, 192))])

    def test_graphics_matrix_keeps_high_lod_pixmap_scaled_to_cell(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        qimage = QImage(256, 192, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 256, 192, qimage)
        self.widget._flush_thumbnail_icon_batch()
        self.widget.thumbnail_grid._matrix_zoom = 4.0
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        pixmap_item = group[3]

        self.assertEqual(pixmap_item.pixmap().size(), QSize(256, 192))
        self.assertAlmostEqual(pixmap_item.scale(), 0.25)

    def test_graphics_matrix_cover_crops_square_thumbnail_to_fill_cell(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        qimage = QImage(256, 256, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 256, 192, qimage)
        self.widget._flush_thumbnail_icon_batch()
        self.widget.thumbnail_grid._matrix_zoom = 4.0
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        pixmap_item = group[3]

        self.assertEqual(pixmap_item.pixmap().size(), QSize(256, 192))
        self.assertAlmostEqual(pixmap_item.scale(), 0.25)

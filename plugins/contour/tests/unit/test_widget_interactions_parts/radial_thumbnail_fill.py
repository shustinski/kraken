"""test_widget_interactions: test radial thumbnail fill."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage


class RadialThumbnailFillMixin:
    def test_enabling_frame_matrix_thumbnails_restarts_loading_existing_items(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(12)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(12)
        resumed: list[bool] = []
        self.widget._resume_frame_matrix_thumbnail_loading = lambda: resumed.append(True)  # type: ignore[method-assign]

        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)

        self.assertEqual(resumed, [True])

    def test_visible_thumbnail_scheduler_prioritizes_viewport_and_current_frame(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(200)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._workspace._current_image_path = str(Path(paths[155]))
        self.widget.neighbor_columns_spin.setValue(10)
        self.widget._thumbnail_build_chunk_size = 200
        queued: list[str] = []
        self.widget._queue_thumbnail_load = lambda _generation, path: queued.append(path)  # type: ignore[method-assign]

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(200)
        self.widget._schedule_visible_thumbnail_loads()

        queued_indexes = {paths.index(path) for path in queued if path in paths}
        self.assertTrue(any(150 <= index <= 159 for index in queued_indexes))
        self.assertLess(len(queued_indexes), len(paths))
        self.widget._workspace._current_image_path = "sample.png"

    def test_radial_thumbnail_fill_keeps_visible_thumbnail_queue(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(20)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[10]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget.neighbor_columns_spin.setValue(5)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(20)
        requested_size = self.widget._thumbnail_request_size()
        generation = self.widget._thumbnail_generation
        self.widget._thumbnail_queued_paths = {paths[0]}
        self.widget._thumbnail_queued_sizes = {paths[0]: requested_size}

        self.widget._reseed_thumbnail_radial_fill()

        self.assertEqual(self.widget._thumbnail_generation, generation)
        self.assertEqual(self.widget._thumbnail_queued_paths, {paths[0]})
        self.assertNotIn(paths[0], self.widget._thumbnail_radial_paths)

    def test_thumbnail_result_during_frame_load_is_applied_after_load_finishes(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        before = item.icon().cacheKey()
        self.widget._loading_image_path = path

        qimage = QImage(4, 4, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 64, 48, qimage)
        self.widget._loading_image_path = None
        self.widget._flush_thumbnail_icon_batch()

        self.assertNotEqual(item.icon().cacheKey(), before)

    def test_loaded_thumbnail_refreshes_graphics_matrix_pixmap(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        before = group[3].pixmap().cacheKey()

        qimage = QImage(64, 48, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 64, 48, qimage)
        self.widget._flush_thumbnail_icon_batch()

        self.assertNotEqual(group[3].pixmap().cacheKey(), before)

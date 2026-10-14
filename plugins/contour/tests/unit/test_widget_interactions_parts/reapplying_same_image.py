"""test_widget_interactions: test reapplying same image."""
from __future__ import annotations
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QIcon, QPixmap
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    ImageProcessingState,
)


class ReapplyingSameImageMixin:
    def test_large_thumbnail_matrix_keeps_all_frames(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(10_000)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.neighbor_columns_spin.setValue(10)
        self.widget.neighbor_overlap_spin.setValue(0)
        self.widget._workspace._current_image_path = paths[5_005]
        self.widget._thumbnail_build_chunk_size = 20_000
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(10_000)

        self.assertEqual(self.widget.thumbnail_grid.count(), 10_000)
        self.assertEqual(self.widget._thumbnail_path_to_row[str(Path(paths[5_005]))], 5_005)
        self.assertGreaterEqual(self.widget.thumbnail_grid.minimumWidth(), 10 * 64)
        self.assertGreaterEqual(self.widget.thumbnail_grid.minimumHeight(), 1_000 * 48)
        self.widget._workspace._current_image_path = "sample.png"

    def test_thumbnail_matrix_rebuild_reuses_cached_icons(self) -> None:
        path = str(Path(r"d:\frames\frame_000.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 10
        requested_size = self.widget._thumbnail_request_size()
        pixmap = QPixmap(*requested_size)
        pixmap.fill(QColor("#123456"))
        self.widget._thumbnail_icon_cache[(path, requested_size)] = QIcon(pixmap)
        self.widget._thumbnail_loading_blocked = lambda: True  # type: ignore[method-assign]

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)

        self.assertFalse(self.widget.thumbnail_grid.item(0).icon().isNull())
        self.assertEqual(
            self.widget._thumbnail_loaded_generation[path],
            self.widget._thumbnail_generation,
        )
        self.assertEqual(self.widget._thumbnail_loaded_sizes[path], requested_size)

    def test_reapplying_same_image_paths_preserves_full_scene_cache(self) -> None:
        path = str(Path(r"d:\frames\frame_000.png"))
        state = ImageProcessingState(
            image_path=path,
            source_image=np.ones((24, 32, 3), dtype=np.uint8),
        )
        self.widget._workspace._image_paths = [path]
        self.widget._workspace._state_cache[path] = state
        self.widget._workspace._current_image_path = path
        self.widget._workspace._current_state = state
        self.widget._rebuild_image_list_items = lambda _paths: None  # type: ignore[method-assign]

        self.widget._apply_image_paths_to_workspace(
            [path],
            clear_extra_layers=False,
            select_path=path,
            fallback_to_first=True,
            clear_state_cache=True,
        )

        self.assertIs(self.widget._workspace._state_cache[path], state)
        self.assertIs(self.widget._workspace._state_cache[path].source_image, state.source_image)

    def test_main_and_neighbor_scene_loads_share_full_resolution_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "frame_000.png")
            cv2.imwrite(path, np.full((24, 32, 3), 73, dtype=np.uint8))

            with patch.object(
                processing_mixin_module,
                "load_image_color",
                wraps=processing_mixin_module.load_image_color,
            ) as decode:
                neighbor_source = self.widget._load_scene_source_image_cached(path)
                main_source = self.widget._load_scene_source_image_cached(path)

            self.assertIs(main_source, neighbor_source)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(self.widget._scene_source_image_cache_bytes, neighbor_source.nbytes)

    def test_image_path_for_cif_stem_uses_indexed_lookup(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:04d}.png")) for index in range(80)]
        self.widget._workspace._image_paths = paths
        self.widget._set_image_list_paths(paths)

        matched = self.widget._image_path_for_cif_stem("frame_0042")
        missing = self.widget._image_path_for_cif_stem("frame_9999")

        self.assertEqual(matched, paths[42])
        self.assertIsNone(missing)
        self.assertEqual(self.widget._image_path_by_stem_lower["frame_0000"], paths[0])

    def test_refresh_image_list_item_states_does_not_rebuild_asset_lists(self) -> None:
        rebuilds: list[int] = []
        self.widget._rebuild_asset_filter_lists = lambda: rebuilds.append(1)  # type: ignore[method-assign]
        self.widget._refresh_vector_rows_for_workspace = lambda: None  # type: ignore[method-assign]
        self.widget._update_thumbnail_grid_selection = lambda: None  # type: ignore[method-assign]
        self.widget._refresh_asset_filter_list_item_states = lambda: None  # type: ignore[method-assign]

        self.widget._refresh_image_list_item_states()

        self.assertEqual(rebuilds, [])

    def test_large_frame_matrix_count_matches_images_tab_count(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(1_200)]
        normalized = [str(Path(path)) for path in paths]
        self.widget._workspace._image_paths = normalized
        self.widget._set_image_list_paths(normalized)
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 2_000

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(len(normalized))

        self.assertEqual(self.widget.thumbnail_grid.count(), self.widget._image_list_proxy.rowCount())
        self.assertEqual(self.widget.thumbnail_grid.count(), len(normalized))

    def test_thumbnail_lod_change_does_not_cancel_large_matrix_build(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:04d}.png")) for index in range(350)]
        self.widget._workspace._image_paths = paths
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100

        self.widget._rebuild_thumbnail_grid()
        QTimer.singleShot(5, self.widget._on_thumbnail_lod_changed)
        self._wait_for_thumbnail_grid_count(len(paths))

        self.assertEqual(self.widget.thumbnail_grid.count(), len(paths))
        self.assertFalse(self.widget._thumbnail_rebuild_in_progress)

    def test_frame_matrix_rebuild_yields_before_adding_items(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(300)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 300
        self.widget._thumbnail_build_interval_ms = 0

        self.widget._rebuild_thumbnail_grid()

        self.assertEqual(self.widget.thumbnail_grid.count(), 0)
        self._wait_for_thumbnail_grid_count(300)

    def test_disabling_frame_matrix_clears_and_skips_matrix_work(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(20)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(20)
        self.assertEqual(self.widget.thumbnail_grid.count(), 20)

        self.widget.show_frame_matrix_checkbox.setChecked(False)
        self.widget._schedule_thumbnail_grid_rebuild(force=True)
        self.widget._update_thumbnail_grid_selection()

        self.assertEqual(self.widget.thumbnail_grid.count(), 0)
        self.assertEqual(self.widget._thumbnail_path_to_row, {})
        self.assertFalse(self.widget.thumbnail_matrix_panel.isVisible())

    def test_disabling_frame_matrix_thumbnails_keeps_items_without_queueing_loads(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(12)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        queued: list[str] = []
        self.widget._queue_thumbnail_load = lambda _generation, path: queued.append(path)  # type: ignore[method-assign]

        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(12)
        self.widget._schedule_visible_thumbnail_loads()

        self.assertEqual(self.widget.thumbnail_grid.count(), 12)
        self.assertEqual(queued, [])

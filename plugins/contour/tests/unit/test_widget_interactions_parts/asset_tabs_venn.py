"""test_widget_interactions: test asset tabs venn."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QGraphicsView,
    QListWidgetItem,
)


class AssetTabsVennMixin:
    def test_thumbnail_geometry_keeps_space_for_filtered_late_slots(self) -> None:
        for index in range(9):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f"frame_{index}.png")
            self.widget.thumbnail_grid.addItem(item)
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.thumbnail_grid.item(2).setHidden(True)
        self.widget._asset_filter_match_only = True
        self.widget.neighbor_columns_spin.setValue(4)

        self.widget._configure_thumbnail_grid_geometry()

        self.assertGreaterEqual(self.widget.thumbnail_grid.minimumHeight(), 3 * 48)

    def test_thumbnail_requeues_after_generation_bump_without_icon_apply(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        generation = self.widget._thumbnail_generation
        self.widget._thumbnail_loaded_generation[path] = generation
        queued: list[str] = []
        self.widget._queue_thumbnail_load = lambda gen, p: queued.append(p)  # type: ignore[method-assign]
        self.widget._thumbnail_generation = generation + 1
        self.widget._queue_thumbnail_load(self.widget._thumbnail_generation, path)
        self.assertEqual(queued, [path])

    def test_thumbnail_stale_background_result_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frame_001.png")
            cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
            self.widget.load_images([path])
            self._wait_for_thumbnail_grid_count(1)
            item = self.widget.thumbnail_grid.item(0)
            before = item.icon().cacheKey()

            self.widget._on_thumbnail_loaded(
                self.widget._thumbnail_generation - 1,
                path,
                64,
                48,
                np.full((4, 4, 3), 255, dtype=np.uint8),
            )

            self.assertEqual(item.icon().cacheKey(), before)

    def test_thumbnail_grid_uses_graphics_view_scroll_surface(self) -> None:
        self.assertTrue(hasattr(self.widget, "thumbnail_grid_scroll_area"))
        self.assertIsInstance(self.widget.thumbnail_grid_scroll_area, QGraphicsView)
        self.assertIs(self.widget.thumbnail_grid_scroll_area, self.widget.thumbnail_grid)

    def test_asset_tabs_and_venn_filter_split_image_vector_sets_without_reordering_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_a = os.path.join(directory, "frame_a.png")
            image_b = os.path.join(directory, "frame_b.png")
            for path in (image_a, image_b):
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
            vector_a = os.path.join(directory, "frame_a.cif")
            vector_lonely = os.path.join(directory, "lonely.cif")
            valid_cif = "\n".join(("DS 1 1 1;", "L NM;", "( S 8 8 );", "B 4 4 4 4;", "DF;", "E"))
            Path(vector_a).write_text(valid_cif, encoding="utf-8")
            Path(vector_lonely).write_text(valid_cif, encoding="utf-8")

            self.widget.load_images([image_a, image_b])
            self._wait_for_thumbnail_grid_count(2)
            original_order = self.widget._workspace.image_paths
            self.widget._workspace.set_cif_index({"frame_a": vector_a, "lonely": vector_lonely})
            self.widget._rebuild_vector_list()
            self.widget._sync_after_cif_index_changed()
            self._wait_for_thumbnail_grid_count(2)
            QTest.qWait(50)

            self.assertEqual(self.widget.image_vector_list.count(), 1)
            self.assertEqual(self.widget.image_only_list.count(), 1)
            self.assertEqual(self.widget.vector_only_list.count(), 1)
            self.assertEqual(self.widget.vector_only_list.item(0).text(), "lonely")
            self.assertEqual(self.widget.image_only_list.item(0).background().color().name().lower(), "#6b2c2c")

            self.assertEqual(self.widget._workspace.image_paths, original_order)
            self.assertFalse(self.widget.image_list.item(0).isHidden())
            self.assertFalse(self.widget.thumbnail_grid.item(0).isHidden())
            self.assertFalse(self.widget.image_list.item(1).isHidden())
            self.assertFalse(self.widget.thumbnail_grid.item(1).isHidden())
            self.assertFalse(hasattr(self.widget, "files_list_label"))
            self.assertFalse(hasattr(self.widget, "show_matched_frames_button"))

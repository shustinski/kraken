"""test_widget_interactions: test thumbnail grid uses."""
from __future__ import annotations
import os
import tempfile
import cv2
import numpy as np


class ThumbnailGridUsesMixin:
    def test_thumbnail_grid_uses_display_frames_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(5):
                path = os.path.join(directory, f"frame_{index}.png")
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.neighbor_columns_spin.setValue(4)
            self.widget.neighbor_overlap_spin.setValue(0)
            self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
            self.widget.load_images(paths)
            self._wait_for_thumbnail_grid_count(5)
            self.widget.neighbor_columns_spin.setValue(4)
            self.widget.neighbor_overlap_spin.setValue(0)
            self.widget._thumbnail_thread_pool.waitForDone(1000)
            self.widget._configure_thumbnail_grid_geometry()

            self.assertEqual(self.widget._thumbnail_columns(), 4)
            frame = 2 * self.widget.thumbnail_grid.frameWidth()
            self.assertEqual(self.widget.thumbnail_grid.width(), 4 * 64 + 1 + frame)

            first_row_y = [
                self.widget.thumbnail_grid.visualItemRect(self.widget.thumbnail_grid.item(index)).y()
                for index in range(4)
            ]
            self.assertEqual(len(set(first_row_y)), 1)
            fifth_rect = self.widget.thumbnail_grid.visualItemRect(self.widget.thumbnail_grid.item(4))
            self.assertGreater(fifth_rect.y(), first_row_y[0])

    def test_thumbnail_grid_applies_frame_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(5):
                path = os.path.join(directory, f"frame_{index}.png")
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.neighbor_columns_spin.setValue(4)
            self.widget.neighbor_overlap_spin.setValue(12)
            self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
            self.widget.load_images(paths)
            self._wait_for_thumbnail_grid_count(5)
            self.widget.neighbor_columns_spin.setValue(4)
            self.widget.neighbor_overlap_spin.setValue(12)
            self.widget._thumbnail_thread_pool.waitForDone(1000)
            self.widget._configure_thumbnail_grid_geometry()

            frame = 2 * self.widget.thumbnail_grid.frameWidth()
            overlap_x, overlap_y = self.widget._thumbnail_overlap_pixels_for_full_frame_overlap(
                12, self.widget._thumbnail_icon_size
            )
            self.assertEqual(self.widget.thumbnail_grid.width(), 4 * 64 - 3 * overlap_x + 1 + frame)
            self.assertEqual(
                self.widget.thumbnail_grid.visualItemRect(self.widget.thumbnail_grid.item(1)).x(), 64 - overlap_x
            )
            self.assertEqual(
                self.widget.thumbnail_grid.visualItemRect(self.widget.thumbnail_grid.item(4)).y(), 48 - overlap_y
            )

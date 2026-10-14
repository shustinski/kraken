"""test_widget_interactions: test inner hole stays."""
from __future__ import annotations
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtTest import QTest

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class InnerHoleStaysMixin:
    def test_inner_hole_stays_a_hole_after_switching_frames_and_back(self) -> None:
        self.widget._set_default_extraction_disabled()
        self.widget.vector_geom_min_outer_spin.setValue(0.0)
        self.widget.vector_geom_min_hole_spin.setValue(0.0)

        def wait_for_frame(path: str) -> None:
            for _ in range(300):
                self.widget._editor_display_thread_pool.waitForDone(10)
                self._app.processEvents()
                if self.widget._last_editor_display_path == path:
                    return
                QTest.qWait(10)
            self.assertEqual(self.widget._last_editor_display_path, path)

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        first_path = str(Path(directory.name) / "frame_with_hole.png")
        second_path = str(Path(directory.name) / "frame_plain.png")
        cv2.imwrite(first_path, np.zeros((80, 80), dtype=np.uint8))
        cv2.imwrite(second_path, np.full((80, 80), 9, dtype=np.uint8))
        outer = _rectangle_polygon(0, 0, 80, 80)
        outer.id = 1
        hole = _rectangle_polygon(20, 20, 40, 40)
        hole.id = 2
        hole.is_hole = True
        hole.parent_id = 1
        plain = _rectangle_polygon(4, 4, 12, 12)
        self.widget._workspace.replace_image_selection(
            [first_path, second_path],
            is_supported_image=lambda _path: True,
        )
        self.widget._workspace.apply_loaded_frame(
            first_path,
            source_image=np.zeros((80, 80), dtype=np.uint8),
            polygons=[outer.clone(), hole.clone()],
        )
        self.widget._workspace.apply_loaded_frame(
            second_path,
            source_image=np.full((80, 80), 9, dtype=np.uint8),
            polygons=[plain.clone()],
        )
        with patch.object(self.widget, "_try_extract_if_recognition_enabled"):
            self.widget.load_image(first_path, load_vectors=True)
        wait_for_frame(first_path)
        first_holes = [polygon for polygon in self.widget.get_polygons() if polygon.is_hole]
        self.assertEqual(len(first_holes), 1)
        self.assertEqual(first_holes[0].parent_id, 1)

        self.widget._sync_editor_polygons_to_current_workspace()
        with patch.object(self.widget, "_try_extract_if_recognition_enabled"):
            self.widget.load_image(second_path, load_vectors=True)
        wait_for_frame(second_path)
        self.assertFalse(any(polygon.is_hole for polygon in self.widget.get_polygons()))

        self.widget._sync_editor_polygons_to_current_workspace()
        with patch.object(self.widget, "_try_extract_if_recognition_enabled"):
            self.widget.load_image(first_path, load_vectors=True)
        wait_for_frame(first_path)

        restored = {polygon.id: polygon for polygon in self.widget.get_polygons()}
        self.assertTrue(restored[2].is_hole)
        self.assertEqual(restored[2].parent_id, 1)
        scene = self.widget.polygon_editor._editor_scene
        hole_color = scene._display_settings.hole_color.lower()
        self.assertEqual(
            scene._polygon_items[2].pen().color().name().lower(), hole_color, scene.polygons_needing_repair_map()
        )
        self.assertEqual(scene._polygon_items[2].brush().color().alpha(), 0)
        self.assertTrue(scene._cutout_polygons_for(1))
        self.assertFalse(scene._polygon_items[1].contains(QPointF(30.0, 30.0)))

        self.widget._sync_editor_polygons_to_current_workspace()
        with patch.object(self.widget, "_try_extract_if_recognition_enabled"):
            self.widget.load_image(second_path, load_vectors=True)
        wait_for_frame(second_path)
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = self.widget._workspace._state_cache[first_path]
        with patch.object(self.widget, "_try_extract_if_recognition_enabled"):
            self.widget.load_image(first_path, load_vectors=True)
        wait_for_frame(first_path)

        restored = {polygon.id: polygon for polygon in self.widget.get_polygons()}
        self.assertTrue(restored[2].is_hole)
        self.assertEqual(restored[2].parent_id, 1)
        self.assertEqual(
            scene._polygon_items[2].pen().color().name().lower(), hole_color, scene.polygons_needing_repair_map()
        )
        self.assertEqual(scene._polygon_items[2].brush().color().alpha(), 0)
        self.assertTrue(scene._cutout_polygons_for(1))
        self.assertFalse(scene._polygon_items[1].contains(QPointF(30.0, 30.0)))

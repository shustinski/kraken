"""test_widget_interactions: test x hold shows."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class XHoldShowsMixin:
    def test_x_hold_shows_source_image_and_combines_with_space_vector_hide(self) -> None:
        source = np.full((16, 16), 20, dtype=np.uint8)
        preprocessed = np.full((16, 16), 220, dtype=np.uint8)
        polygon = _rectangle_polygon(2, 2, 10, 10)
        state = ImageProcessingState(
            image_path="frame_1.png",
            source_image=source,
            preprocessed_image=preprocessed,
            pipeline_config=self.widget.get_pipeline(),
            polygons=[polygon],
        )
        self.widget._workspace._current_image_path = "frame_1.png"
        self.widget._workspace._current_state = state
        self.widget._workspace._state_cache = {"frame_1.png": state}
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self.widget.show()
        self._app.processEvents()

        def _wait_for_editor_pixel(expected: int, timeout_ms: int = 1500) -> None:
            attempts = max(1, timeout_ms // 10)
            last_value: int | None = None
            for _ in range(attempts):
                self.widget._editor_display_thread_pool.waitForDone(10)
                self._app.processEvents()
                pixmap = self.widget.polygon_editor._editor_scene._image_item.pixmap()
                if not pixmap.isNull():
                    last_value = pixmap.toImage().pixelColor(0, 0).red()
                    if last_value == expected:
                        return
                QTest.qWait(10)
            self.assertEqual(last_value, expected)

        self.widget._sync_current_state_views(preserve_view=False, sync_neighbors=False)
        _wait_for_editor_pixel(220)
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        self.widget.polygon_editor.setFocus()
        self._app.processEvents()
        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_X)
        _wait_for_editor_pixel(20)
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        _wait_for_editor_pixel(20)
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        _wait_for_editor_pixel(20)
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_X)
        _wait_for_editor_pixel(220)
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        _wait_for_editor_pixel(220)
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_X)
        _wait_for_editor_pixel(20)
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_X)
        _wait_for_editor_pixel(220)
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        _wait_for_editor_pixel(220)
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

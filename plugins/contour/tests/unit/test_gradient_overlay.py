from __future__ import annotations

import os
import sys
import unittest

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from contour.application.processing import ImageProcessingState
from contour.graphics_view import PolygonEditorScene, PolygonEditorView
from contour.widget import PolygonExtractionWidget


def _app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv[:1] or ["unit-test"])
    return instance


class GradientOverlaySceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_set_gradient_overlay_shows_pixmap(self) -> None:
        scene = PolygonEditorScene()
        image = np.zeros((12, 15, 3), dtype=np.uint8)
        image[:, :, 1] = 200

        scene.set_gradient_overlay(image, opacity=0.5)

        self.assertTrue(scene._gradient_overlay_item.isVisible())
        self.assertFalse(scene._gradient_overlay_item.pixmap().isNull())
        self.assertAlmostEqual(scene._gradient_overlay_item.opacity(), 0.5, places=3)

    def test_clear_gradient_overlay_hides_item(self) -> None:
        scene = PolygonEditorScene()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        scene.set_gradient_overlay(image, opacity=0.4)

        scene.clear_gradient_overlay()

        self.assertFalse(scene._gradient_overlay_item.isVisible())
        self.assertTrue(scene._gradient_overlay_item.pixmap().isNull())

    def test_set_gradient_overlay_none_clears(self) -> None:
        scene = PolygonEditorScene()
        scene.set_gradient_overlay(np.zeros((4, 4, 3), dtype=np.uint8), opacity=0.5)
        scene.set_gradient_overlay(None)
        self.assertFalse(scene._gradient_overlay_item.isVisible())

    def test_view_forwards_gradient_overlay_calls(self) -> None:
        view = PolygonEditorView()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[..., 2] = 180

        view.set_gradient_overlay(image, opacity=0.3)

        scene_item = view._editor_scene._gradient_overlay_item
        self.assertTrue(scene_item.isVisible())
        self.assertAlmostEqual(scene_item.opacity(), 0.3, places=3)

        view.set_gradient_overlay_opacity(0.8)
        self.assertAlmostEqual(scene_item.opacity(), 0.8, places=3)

        view.clear_gradient_overlay()
        self.assertFalse(scene_item.isVisible())


class GradientOverlayWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()
        source = np.zeros((40, 50), dtype=np.uint8)
        cv2.circle(source, (25, 20), 6, 230, thickness=-1)
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=source,
        )

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def test_toggle_gradient_overlay_updates_editor_layer(self) -> None:
        self.widget.gradient_overlay_checkbox.setChecked(True)
        self._app.processEvents()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        self.assertTrue(overlay_item.isVisible())
        self.assertFalse(overlay_item.pixmap().isNull())

        self.widget.gradient_overlay_checkbox.setChecked(False)
        self._app.processEvents()
        self.assertFalse(overlay_item.isVisible())

    def test_threshold_mode_honours_via_min_contrast(self) -> None:
        self.widget.extraction_profile_combo.setCurrentIndex(self.widget.extraction_profile_combo.findData("vias"))
        self._app.processEvents()
        self.widget.via_min_contrast_spin.setValue(10.0)
        self.widget.gradient_overlay_mode_combo.setCurrentIndex(1)
        self.widget.gradient_overlay_checkbox.setChecked(True)
        self._app.processEvents()

        overlay_low = self.widget._build_gradient_overlay_image(self.widget._workspace.current_state.source_image)

        self.widget.via_min_contrast_spin.setValue(250.0)
        overlay_high = self.widget._build_gradient_overlay_image(self.widget._workspace.current_state.source_image)

        low_active = int(overlay_low[..., 1].sum())
        high_active = int(overlay_high[..., 1].sum())
        self.assertGreater(low_active, high_active)

    def test_overlay_opacity_control_propagates_to_scene(self) -> None:
        self.widget.gradient_overlay_checkbox.setChecked(True)
        self.widget.gradient_overlay_opacity_spin.setValue(0.8)
        self._app.processEvents()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        self.assertAlmostEqual(overlay_item.opacity(), 0.8, places=3)


if __name__ == "__main__":
    unittest.main()

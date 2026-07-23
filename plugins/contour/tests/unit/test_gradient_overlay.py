from __future__ import annotations

import os
import sys
import unittest

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QApplication

from contour.application.processing import ImageProcessingState
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_view import PolygonEditorScene, PolygonEditorView
from contour.widget import PolygonExtractionWidget


def _app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv[:1] or ["unit-test"])
    return instance


def _rect_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [(left, top), (right, top), (right, bottom), (left, bottom)]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


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

    def test_main_image_visible_fraction_reports_viewport_coverage(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        view = PolygonEditorView()
        view.resize(200, 200)
        view.set_image(np.zeros((100, 100), dtype=np.uint8))
        view.fit_to_view()
        self.assertGreaterEqual(view.main_image_visible_fraction(), 0.5)
        view.scale(8.0, 8.0)
        self.assertLess(view.main_image_visible_fraction(), 0.5)
        self.assertTrue(view.should_auto_reposition_view(force=False))
        self.assertTrue(view.should_auto_reposition_view(force=True))

    def test_space_hold_suppresses_gradient_overlay_without_clearing_pixmap(self) -> None:
        scene = PolygonEditorScene()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 1] = 180
        scene.set_gradient_overlay(image, opacity=0.5)
        self.assertTrue(scene._gradient_overlay_item.isVisible())

        scene.set_gradient_overlay_visible(False)
        self.assertFalse(scene._gradient_overlay_item.isVisible())
        self.assertFalse(scene._gradient_overlay_item.pixmap().isNull())

        scene.set_gradient_overlay_visible(True)
        self.assertTrue(scene._gradient_overlay_item.isVisible())

    def test_extra_layers_preserve_zero_opacity_and_have_stable_z_order(self) -> None:
        scene = PolygonEditorScene()
        first = QPixmap(4, 4)
        first.fill(QColor("#ff0000"))
        second = QPixmap(4, 4)
        second.fill(QColor("#00ff00"))

        scene.set_extra_layers(
            [
                {"name": "first", "pixmap": first, "opacity": 0.0},
                {"name": "second", "pixmap": second, "opacity": 0.5},
            ]
        )

        self.assertEqual(len(scene._extra_layer_items), 2)
        self.assertAlmostEqual(scene._extra_layer_items[0].opacity(), 0.0, places=3)
        self.assertAlmostEqual(scene._extra_layer_items[1].opacity(), 0.5, places=3)
        self.assertLess(scene._extra_layer_items[0].zValue(), scene._extra_layer_items[1].zValue())

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

    def test_source_image_is_the_default_opaque_view(self) -> None:
        self.assertEqual(self.widget.gradient_overlay_mode_combo.currentData(), "source")
        self.widget._refresh_gradient_overlay()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        self.assertTrue(overlay_item.isVisible())
        self.assertFalse(overlay_item.pixmap().isNull())
        self.assertAlmostEqual(overlay_item.opacity(), 1.0, places=3)

    def test_threshold_mode_honours_via_min_contrast(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("via"))
        self._app.processEvents()
        self.widget.via_min_contrast_spin.setValue(10.0)
        self.widget.gradient_overlay_mode_combo.setCurrentIndex(
            self.widget.gradient_overlay_mode_combo.findData("threshold")
        )
        self._app.processEvents()

        overlay_low = self.widget._build_gradient_overlay_image(self.widget._workspace.current_state.source_image)

        self.widget.via_min_contrast_spin.setValue(250.0)
        overlay_high = self.widget._build_gradient_overlay_image(self.widget._workspace.current_state.source_image)

        low_active = int(overlay_low[..., 1].sum())
        high_active = int(overlay_high[..., 1].sum())
        self.assertGreater(low_active, high_active)

    def test_contact_debug_masks_are_available_in_display_combo(self) -> None:
        expected_modes = {
            "source",
            "heatmap",
            "threshold",
            "elevation",
            "mask",
            "candidate_mask",
            "via_mask",
            "tophat_mask",
            "dog_mask",
            "spot_response",
            "ring_response",
        }
        actual_modes = {
            str(self.widget.gradient_overlay_mode_combo.itemData(index))
            for index in range(self.widget.gradient_overlay_mode_combo.count())
        }
        self.assertTrue(expected_modes.issubset(actual_modes))

        candidate_mask = np.zeros((40, 50), dtype=np.uint8)
        candidate_mask[10:20, 15:25] = 255
        self.widget._workspace.current_state.debug_gradient_maps = {"candidate_mask": candidate_mask}
        index = self.widget.gradient_overlay_mode_combo.findData("candidate_mask")
        self.assertGreaterEqual(index, 0)
        self.widget.gradient_overlay_mode_combo.setCurrentIndex(index)

        overlay = self.widget._build_gradient_overlay_image(self.widget._workspace.current_state.source_image)

        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.shape, (40, 50, 3))
        self.assertGreater(int(overlay[12, 17].sum()), 0)
        self.assertEqual(int(overlay[0, 0].sum()), 0)

    def test_display_modes_are_always_fully_opaque(self) -> None:
        self.assertFalse(hasattr(self.widget, "gradient_overlay_checkbox"))
        self.assertFalse(hasattr(self.widget, "gradient_overlay_opacity_spin"))
        self.widget.gradient_overlay_mode_combo.setCurrentIndex(
            self.widget.gradient_overlay_mode_combo.findData("heatmap")
        )
        self._app.processEvents()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        self.assertAlmostEqual(overlay_item.opacity(), 1.0, places=3)

    def test_refresh_gradient_overlay_uses_preprocessed_image(self) -> None:
        preprocessed = np.full((40, 50), 128, dtype=np.uint8)
        self.widget._workspace._current_state.preprocessed_image = preprocessed
        captured: dict[str, object] = {}
        original = self.widget._build_gradient_overlay_image

        def _spy(image: np.ndarray) -> np.ndarray | None:
            captured["image"] = image
            return original(image)

        self.widget._build_gradient_overlay_image = _spy  # type: ignore[method-assign]
        self.widget.gradient_overlay_mode_combo.setCurrentIndex(
            self.widget.gradient_overlay_mode_combo.findData("heatmap")
        )
        self._app.processEvents()

        self.assertIs(captured.get("image"), preprocessed)

    def test_metal_overlay_uses_preprocessed_image(self) -> None:
        source = np.zeros((40, 50), dtype=np.uint8)
        preprocessed = np.full((40, 50), 200, dtype=np.uint8)
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[5:35, 5:45] = 255
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=source,
            preprocessed_image=preprocessed,
            debug_gradient_maps={"metal_binary_mask": mask},
        )
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("conductors")
        )
        self.widget.metal_show_mask_checkbox.setChecked(True)
        self.widget.metal_overlay_opacity_spin.setValue(1.0)
        self._app.processEvents()
        self.widget._apply_metal_visual_overlay()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        pixmap = overlay_item.pixmap()
        self.assertFalse(pixmap.isNull())
        image = pixmap.toImage()
        # Outside the mask tint, brightness should reflect preprocessed (200), not source (0).
        sample = image.pixelColor(2, 2)
        self.assertGreater(sample.red(), 50)
        self.widget._workspace._current_state.preprocessed_image = None
        self.widget._apply_metal_visual_overlay()
        sample_source = overlay_item.pixmap().toImage().pixelColor(2, 2)
        self.assertLess(sample_source.red(), 20)

    def test_metal_overlay_clips_mask_to_vector_polygons(self) -> None:
        source = np.zeros((40, 50), dtype=np.uint8)
        mask = np.ones((40, 50), dtype=np.uint8) * 255
        polygon = _rect_polygon(10, 10, 30, 30)
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=source,
            debug_gradient_maps={"metal_binary_mask": mask},
            polygons=[polygon],
        )
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("conductors")
        )
        self.widget.metal_show_mask_checkbox.setChecked(True)
        self.widget.metal_overlay_opacity_spin.setValue(1.0)
        self._app.processEvents()

        self.widget._apply_metal_visual_overlay()

        overlay_item = self.widget.polygon_editor._editor_scene._gradient_overlay_item
        image = overlay_item.pixmap().toImage()
        outside = image.pixelColor(2, 2)
        inside = image.pixelColor(20, 20)
        self.assertLess(outside.green(), 20)
        self.assertGreater(inside.green(), 80)


if __name__ == "__main__":
    unittest.main()

"""test_widget_interactions: test recognition mode hides."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF
from contour.application.processing import (
    BatchImageResult,
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class RecognitionModeHidesMixin:
    def test_recognition_mode_hides_loaded_vector_overlay(self) -> None:
        self.widget.auto_apply_checkbox.setChecked(False)
        polygon = _rectangle_polygon(4, 4, 20, 20)
        state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[polygon.clone()],
        )
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = state
        self.widget._workspace._state_cache = {"sample.png": state}
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([polygon.clone()])
        self._app.processEvents()
        items = list(self.widget.polygon_editor._editor_scene._polygon_items.values())
        self.assertTrue(items)
        self.assertTrue(all(item.isVisible() for item in items))

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_editor.get_polygons(), [])
        self.assertTrue(self.widget.polygon_editor.polygon_overlays_visible())

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self._app.processEvents()

        restored = self.widget.polygon_editor.get_polygons()
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].points, polygon.points)

    def test_conductor_display_filters_apply_only_during_recognition(self) -> None:
        self.widget.auto_apply_checkbox.setChecked(False)
        polygon = _rectangle_polygon(4, 4, 20, 20)
        state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            mask_image=np.full((32, 32), 255, dtype=np.uint8),
            polygons=[polygon.clone()],
            recognition_base_polygons=[polygon.clone()],
        )
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = state
        self.widget._workspace._state_cache = {"sample.png": state}
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.metal_show_conductors_checkbox.setChecked(False)
        self.widget.metal_show_mask_checkbox.setChecked(True)
        items = list(self.widget.polygon_editor._editor_scene._polygon_items.values())
        self.assertTrue(items)
        self.assertFalse(any(item.isVisible() for item in items))
        self.assertTrue(self.widget.metal_display_group.isEnabled())

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self._app.processEvents()

        items = list(self.widget.polygon_editor._editor_scene._polygon_items.values())
        self.assertTrue(items)
        self.assertTrue(all(item.isVisible() for item in items))
        self.assertFalse(self.widget.metal_display_group.isEnabled())
        self.assertFalse(self.widget._conductor_display_filters_enabled())

    def test_recognition_mode_shows_red_frame_around_editor_scene(self) -> None:
        self.assertNotIn("#DC2626", self.widget.editor_scene_frame.styleSheet())

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self._app.processEvents()
        self.assertIn("#DC2626", self.widget.editor_scene_frame.styleSheet())

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("via"))
        self._app.processEvents()
        self.assertIn("#DC2626", self.widget.editor_scene_frame.styleSheet())

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self._app.processEvents()
        self.assertNotIn("#DC2626", self.widget.editor_scene_frame.styleSheet())

    def test_leaving_recognition_reloads_cif_for_current_image(self) -> None:
        self.widget.auto_apply_checkbox.setChecked(False)
        cif_polygon = _rectangle_polygon(1, 1, 8, 8)
        cif_polygon.id = 2
        self.widget._workspace.merge_cif_paths({"sample": "sample.cif"})
        self.widget._load_cif_overlay_polygons = lambda _path: [cif_polygon.clone()]  # type: ignore[method-assign]
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self.widget.polygon_editor.set_polygons([_rectangle_polygon(4, 4, 20, 20)])
        self._app.processEvents()

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self._app.processEvents()

        restored = self.widget.polygon_editor.get_polygons()
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].points, cif_polygon.points)
        self.assertEqual(self.widget._workspace.current_state.polygons[0].points, cif_polygon.points)

    def test_preview_result_does_not_resync_neighbors_or_recenter_view(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self.widget.polygon_editor.resize(320, 240)
        self.widget.polygon_editor.set_image(np.zeros((128, 128), dtype=np.uint8))
        self.widget.polygon_editor.scale(2.0, 2.0)
        self.widget.polygon_editor.centerOn(QPointF(48.0, 64.0))
        self._app.processEvents()
        before_transform = self.widget.polygon_editor.transform()
        before_center = self.widget.polygon_editor.mapToScene(self.widget.polygon_editor.viewport().rect().center())
        neighbor_sync_calls: list[int] = []
        self.widget._request_neighbor_frame_sync = lambda *, delay_ms=0: neighbor_sync_calls.append(delay_ms)  # type: ignore[method-assign]
        self.widget._preview_running_request_id = 7

        self.widget._on_preview_processing_result(
            7,
            BatchImageResult(
                image_path="sample.png",
                source_image=np.zeros((128, 128), dtype=np.uint8),
                preprocessed_image=np.full((128, 128), 255, dtype=np.uint8),
                pipeline_config=self.widget.get_pipeline(),
                mask_image=np.full((128, 128), 255, dtype=np.uint8),
                polygons=[],
            ),
        )

        after_center = self.widget.polygon_editor.mapToScene(self.widget.polygon_editor.viewport().rect().center())
        self.assertEqual(neighbor_sync_calls, [])
        self.assertAlmostEqual(self.widget.polygon_editor.transform().m11(), before_transform.m11())
        self.assertAlmostEqual(after_center.x(), before_center.x(), delta=1.0)
        self.assertAlmostEqual(after_center.y(), before_center.y(), delta=1.0)

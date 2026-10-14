"""test_widget_interactions: test frame apply defers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QCheckBox,
    QSlider,
)
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class FrameApplyDefersMixin:
    def test_frame_apply_defers_vectors_until_pixmap_ready(self) -> None:
        path = str(Path("frame_defer.png"))
        polygon = _rectangle_polygon(2, 2, 12, 12)
        state = ImageProcessingState(
            image_path=path,
            source_image=np.full((24, 24), 40, dtype=np.uint8),
            polygons=[polygon.clone()],
        )
        self.widget._workspace._image_paths = [path]
        self.widget._workspace._current_image_path = path
        self.widget._workspace._current_state = state
        self.widget._editor_pixmap_cache.clear()
        queued: list[str] = []
        vector_applies: list[int] = []

        def _queue(image_path: str, display_image: object, **kwargs) -> None:
            del display_image, kwargs
            queued.append(str(Path(image_path)))

        self.widget._queue_editor_display_pixmap = _queue  # type: ignore[method-assign]
        self.widget._apply_editor_vectors_for_frame = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: vector_applies.append(1)
        )
        self.widget._request_neighbor_frame_sync = lambda *, delay_ms=0: None  # type: ignore[method-assign]
        self.widget._sync_extra_layers = lambda: None  # type: ignore[method-assign]
        self.widget._refresh_gradient_overlay = lambda: None  # type: ignore[method-assign]
        self.widget._update_vector_edit_status_label = lambda: None  # type: ignore[method-assign]

        self.widget._apply_frame_to_editor(clear_neighbors=False, sync_neighbors=False)

        self.assertEqual(queued, [path])
        self.assertEqual(vector_applies, [])
        pending = self.widget._pending_editor_frame_apply
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(str(Path(pending[0])), path)
        self.assertEqual(len(pending[1]), 1)

        self.widget._flush_pending_editor_frame_apply(path)
        self.assertEqual(vector_applies, [1])
        self.assertIsNone(self.widget._pending_editor_frame_apply)

    def test_extraction_change_processes_when_auto_apply_enabled(self) -> None:
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(True)
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("conductors")
        )
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()
        QTest.qWait(200)
        self._app.processEvents()

        self.assertEqual(process_calls[-1], False)

    def test_conductor_contrast_slider_and_hole_settings_feed_recognition_settings(self) -> None:
        self.assertIsInstance(self.widget.metal_min_contrast_slider, QSlider)
        self.assertEqual(self.widget.metal_min_contrast_slider.minimum(), 1)
        self.assertEqual(self.widget.metal_min_contrast_slider.value(), 50)
        self.assertEqual(self.widget.metal_auto_contrast_step_spin.value(), 10.0)
        self.assertEqual(self.widget.metal_auto_source_contrast_step_spin.value(), 4.0)
        self.assertEqual(self.widget.metal_auto_directional_gap_bridge_spin.value(), 3)
        self.assertEqual(self.widget.metal_auto_directional_gap_min_source_spin.value(), 45.0)
        self.widget.metal_min_contrast_slider.setValue(37)
        self.widget.metal_auto_contrast_step_spin.setValue(7.0)
        self.widget.metal_auto_source_contrast_step_spin.setValue(6.0)
        self.widget.metal_auto_directional_gap_bridge_spin.setValue(4)
        self.widget.metal_auto_directional_gap_min_source_spin.setValue(52.0)
        self.widget.metal_min_object_source_contrast_spin.setValue(16.0)
        self.widget.metal_min_object_rim_contrast_spin.setValue(42.0)
        self.widget.metal_min_object_rim_area_fraction_spin.setValue(0.002)
        self.widget.metal_min_hole_source_contrast_spin.setValue(12.5)
        self.widget.metal_min_hole_source_contrast_fraction_spin.setValue(0.2)

        settings = self.widget._current_contour_settings()

        self.assertEqual(settings.metal_min_contrast, 37.0)
        self.assertEqual(settings.metal_auto_contrast_step, 7.0)
        self.assertEqual(settings.metal_auto_source_contrast_step, 6.0)
        self.assertEqual(settings.metal_auto_directional_gap_bridge_px, 4)
        self.assertEqual(settings.metal_auto_directional_gap_min_source_intensity, 52.0)
        self.assertEqual(settings.metal_min_object_source_contrast, 16.0)
        self.assertEqual(settings.metal_min_object_rim_contrast, 42.0)
        self.assertEqual(settings.metal_min_object_rim_area_fraction, 0.002)
        self.assertEqual(settings.metal_min_hole_source_contrast, 12.5)
        self.assertEqual(settings.metal_min_hole_source_contrast_fraction, 0.2)

    def test_every_heuristic_expert_parameter_schedules_via_rerecognition(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("via"))
        self.widget.auto_apply_checkbox.setChecked(True)
        self.widget._on_extraction_settings_changed()
        self.widget._extraction_settings_debounce.stop()

        controls = self.widget._heuristic_expert_parameter_widgets
        self.assertGreater(len(controls), 20)
        for control in controls:
            self.widget._extraction_settings_debounce.stop()
            if isinstance(control, QCheckBox):
                control.setChecked(not control.isChecked())
            else:
                step = control.singleStep()
                new_value = control.value() + step
                if new_value > control.maximum():
                    new_value = control.value() - step
                control.setValue(new_value)
            self.assertTrue(
                self.widget._extraction_settings_debounce.isActive(),
                f"{control.objectName() or type(control).__name__} did not schedule recognition",
            )

        self.widget._extraction_settings_debounce.stop()
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]
        self.widget.heuristic_w_contrast_spin.setValue(self.widget.heuristic_w_contrast_spin.value() + 1.0)
        QTest.qWait(200)
        self._app.processEvents()

        self.assertEqual(process_calls, [False])

    def test_via_roundness_is_included_in_current_settings(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("via"))
        self.widget.via_roundness_spin.setValue(73.0)

        settings = self.widget._current_contour_settings()

        self.assertEqual(settings.extraction_profile, "vias")
        self.assertEqual(settings.object_type, "via")
        self.assertEqual(settings.output_mode, "box")
        self.assertEqual(settings.via_min_roundness, 73.0)

    def test_via_threshold_ui_exposes_only_white_and_black_ranges(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("via"))
        self.widget.via_white_range_min_spin.setValue(150)
        self.widget.via_white_range_max_spin.setValue(230)
        self.widget.via_black_range_checkbox.setChecked(True)
        self.widget.via_black_range_min_spin.setValue(5)
        self.widget.via_black_range_max_spin.setValue(40)

        settings = self.widget._current_contour_settings()

        self.assertFalse(hasattr(self.widget, "via_threshold_range_widget"))
        self.assertEqual(settings.via_white_range_min, 150)
        self.assertEqual(settings.via_white_range_max, 230)
        self.assertTrue(settings.via_black_range_enabled)
        self.assertEqual(settings.via_black_range_min, 5)
        self.assertEqual(settings.via_black_range_max, 40)

    def test_build_preview_request_reuses_cached_preprocessed_image_for_same_pipeline(self) -> None:
        pipeline_config = self.widget.get_pipeline()
        preprocessed = np.ones((32, 32), dtype=np.uint8)
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            preprocessed_image=preprocessed,
            pipeline_config=pipeline_config,
        )

        request = self.widget._build_preview_request()

        self.assertIsNotNone(request)
        self.assertIs(request.preprocessed_image, preprocessed)

    def test_build_preview_request_ignores_cached_preprocessed_image_for_changed_pipeline(self) -> None:
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            preprocessed_image=np.ones((32, 32), dtype=np.uint8),
            pipeline_config={"steps": [{"operation": "threshold"}]},
        )

        request = self.widget._build_preview_request()

        self.assertIsNotNone(request)
        self.assertIsNone(request.preprocessed_image)

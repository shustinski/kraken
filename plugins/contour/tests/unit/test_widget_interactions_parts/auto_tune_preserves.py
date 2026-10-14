"""test_widget_interactions: test auto tune preserves."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    BatchImageResult,
    ContourExtractionSettings,
)
from contour.domain import PolygonData

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class AutoTunePreservesMixin:
    def test_image_recognition_profile_finishes_after_result_is_applied(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("conductors")
        )
        request = self.widget._build_preview_request()
        assert request is not None
        polygon = _rectangle_polygon(4, 4, 20, 20)

        with (
            patch.object(processing_mixin_module, "image_recognition_profiling_enabled", return_value=True),
            patch.object(processing_mixin_module, "write_profile_report") as console_output,
        ):
            self.widget._start_image_recognition_profile(request)
            session = self.widget._image_recognition_profile
            session.preview_request_id = 7
            self.widget._preview_running_request_id = 7
            self.widget._on_preview_processing_result(
                7,
                BatchImageResult(
                    image_path="sample.png",
                    source_image=np.zeros((32, 32), dtype=np.uint8),
                    preprocessed_image=np.zeros((32, 32), dtype=np.uint8),
                    pipeline_config=self.widget.get_pipeline(),
                    mask_image=np.zeros((32, 32), dtype=np.uint8),
                    polygons=[polygon],
                ),
            )
            self._app.processEvents()

        self.assertIsNone(self.widget._image_recognition_profile)
        output = "\n".join(
            str(message)
            for call in console_output.call_args_list
            for message in call.args
        )
        self.assertIn("[contour image recognition profiling] started", output)
        self.assertIn("status=displayed", output)
        self.assertIn("polygons=1", output)
        self.assertIn("[contour image recognition profiling stats]", output)

    def test_via_search_range_and_output_size_have_separate_controls(self) -> None:
        self.assertFalse(hasattr(self.widget, "via_fixed_diameters_edit"))
        self.assertFalse(hasattr(self.widget, "via_size_mode_combo"))
        self.assertFalse(hasattr(self.widget, "via_diameter_size_mode_combo"))
        self.assertFalse(hasattr(self.widget, "bright_via_diameter_fixed_spin"))
        self.assertTrue(hasattr(self.widget, "bright_via_diameter_min_spin"))
        self.assertTrue(hasattr(self.widget, "bright_via_diameter_max_spin"))
        self.assertTrue(hasattr(self.widget, "via_output_diameter_spin"))

    def test_contacts_are_normalized_to_output_diameter_for_saving(self) -> None:
        via = PolygonData(
            id=1,
            points=[(10, 20), (22, 20), (22, 26), (10, 26)],
            category="via",
            shape_hint="box",
            bbox=(10, 20, 12, 6),
        )

        saved = self.widget._uniform_contact_polygons_for_save([via], diameter=8)

        self.assertEqual(saved[0].bbox[2:], (8, 8))
        self.assertEqual(saved[0].area, 64.0)
        self.assertEqual(via.bbox, (10, 20, 12, 6))

        odd_saved = self.widget._uniform_contact_polygons_for_save([via], diameter=9)
        self.assertEqual(odd_saved[0].bbox[2:], (9, 9))

    def test_output_diameter_redraws_recognized_contacts_without_processing(self) -> None:
        recognized = PolygonData(
            id=1,
            points=[(10, 20), (18, 20), (18, 28), (10, 28)],
            category="via",
            shape_hint="box",
            bbox=(10, 20, 8, 8),
            recognition_score=84.0,
        )
        manual = PolygonData(
            id=2,
            points=[(22, 20), (30, 20), (30, 28), (22, 28)],
            category="via",
            shape_hint="box",
            bbox=(22, 20, 8, 8),
        )
        state = self.widget._workspace.current_state
        assert state is not None
        state.polygons = [recognized, manual]
        self.widget._workspace._state_cache["sample.png"] = state
        self.widget.polygon_editor.set_polygons(state.polygons, emit_signal=False)
        self.widget.process_current_image = MagicMock()

        self.widget.via_output_diameter_spin.setValue(14)

        self.widget.process_current_image.assert_not_called()
        assert self.widget._workspace.current_state is not None
        resized = self.widget._workspace.current_state.polygons
        self.assertEqual(resized[0].bbox[2:], (14, 14))
        self.assertEqual(resized[1].bbox, manual.bbox)
        self.assertEqual(self.widget.get_polygons()[0].bbox[2:], (14, 14))

    def test_auto_tune_preserves_manual_contact_thresholds(self) -> None:
        from contour.application.use_cases import AutoTuneResult

        self.widget.via_white_range_checkbox.setChecked(True)
        self.widget.via_white_range_min_spin.setValue(151)
        self.widget.via_white_range_max_spin.setValue(239)
        self.widget.via_black_range_checkbox.setChecked(True)
        self.widget.via_black_range_min_spin.setValue(7)
        self.widget.via_black_range_max_spin.setValue(43)
        self.widget.bright_via_min_final_score_spin.setValue(67.0)
        tuned = ContourExtractionSettings(
            via_white_range_enabled=False,
            via_white_range_min=1,
            via_white_range_max=2,
            via_black_range_enabled=False,
            via_black_range_min=3,
            via_black_range_max=4,
            bright_via_min_final_score=5.0,
        )
        applied: list[ContourExtractionSettings] = []
        self.widget._set_extraction_settings = applied.append  # type: ignore[method-assign]
        self.widget.process_current_image = MagicMock()

        self.widget._apply_auto_tune_result(
            AutoTuneResult(
                pipeline_config={"steps": []},
                contour_settings=tuned,
                score=1.0,
                mask_score=1.0,
                roi_bbox=(0, 0, 10, 10),
                evaluations=1,
            )
        )

        self.assertEqual(len(applied), 1)
        protected = applied[0]
        self.assertTrue(protected.via_white_range_enabled)
        self.assertEqual((protected.via_white_range_min, protected.via_white_range_max), (151, 239))
        self.assertTrue(protected.via_black_range_enabled)
        self.assertEqual((protected.via_black_range_min, protected.via_black_range_max), (7, 43))
        self.assertEqual(protected.bright_via_min_final_score, 67.0)

    def test_pipeline_preview_request_is_built_in_no_extraction_mode(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))

        request = self.widget._build_preview_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.contour_settings.recognition_mode, "disabled")
        self.assertIsNotNone(request.passthrough_polygons)

    def test_editor_display_cache_key_tracks_pipeline_config(self) -> None:
        state = self.widget._workspace.current_state
        assert state is not None
        state.preprocessed_image = np.zeros((32, 32), dtype=np.uint8)
        state.pipeline_config = {"steps": [{"operation": "threshold", "parameters": {"threshold": 80}}]}

        first_key = self.widget._editor_display_cache_key("sample.png", state)
        state.pipeline_config = {"steps": [{"operation": "threshold", "parameters": {"threshold": 160}}]}
        second_key = self.widget._editor_display_cache_key("sample.png", state)

        self.assertNotEqual(first_key, second_key)

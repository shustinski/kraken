"""test_widget_interactions: test prepared filter result."""
from __future__ import annotations
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PyQt6.QtTest import QTest
from contour.application.processing import (
    BatchImageResult,
    ImageProcessingState,
)
from contour.domain import PolygonData
from contour.widget import PolygonExtractionWidget

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
)


class PreparedFilterResultMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
        )

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def _wait_for_thumbnail_grid_count(self, expected: int, timeout_ms: int = 3000) -> None:
        attempts = max(1, timeout_ms // 10)
        for _ in range(attempts):
            self._app.processEvents()
            if self.widget.thumbnail_grid.count() == expected:
                return
            QTest.qWait(10)
        self.assertEqual(self.widget.thumbnail_grid.count(), expected)

    def test_dialog_start_directory_uses_line_edit_path_parent_for_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "target"
            folder.mkdir()
            file_path = folder / "pipeline.json"
            file_path.write_text("{}", encoding="utf-8")

            self.widget.input_dir_edit.setText(str(file_path))

            self.assertEqual(
                self.widget._dialog_start_directory_from_line_edit(self.widget.input_dir_edit), str(folder)
            )

    def test_extraction_change_does_not_process_when_auto_apply_disabled(self) -> None:
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(False)
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("conductors")
        )
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()

        self.assertEqual(process_calls, [])

    def test_filter_change_queues_preprocessing_without_starting_recognition(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("disabled")
        )
        prepared_calls: list[tuple[str, object]] = []
        process_calls: list[bool] = []
        self.widget._abort_in_flight_interactive_processing = lambda **_kwargs: None  # type: ignore[method-assign]
        self.widget._queue_prepared_image_update = lambda *args: prepared_calls.append(args)  # type: ignore[method-assign]
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget._auto_apply_pipeline()

        self.assertEqual(len(prepared_calls), 1)
        self.assertEqual(prepared_calls[0][0], "sample.png")
        self.assertIs(prepared_calls[0][1], self.widget._workspace.current_state.source_image)
        self.assertEqual(process_calls, [])

    def test_recognition_setting_change_does_not_process_when_recognition_is_disabled(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("disabled")
        )
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget._auto_apply_recognition_settings()

        self.assertEqual(process_calls, [])

    def test_prepared_filter_result_is_displayed_when_recognition_is_disabled(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(
            self.widget.recognition_mode_combo.findData("disabled")
        )
        state = self.widget._workspace.current_state
        assert state is not None
        self.widget._workspace._state_cache = {"sample.png": state}
        self.widget._prepared_image_running_request_id = 7
        display_calls: list[dict[str, bool]] = []
        process_calls: list[bool] = []
        self.widget._refresh_current_display_image_only = lambda **kwargs: display_calls.append(kwargs)  # type: ignore[method-assign]
        self.widget._sync_current_state_views = MagicMock()  # type: ignore[method-assign]
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]
        filtered = np.full((32, 32), 173, dtype=np.uint8)
        pipeline = self.widget.get_pipeline()

        self.widget._on_prepared_image_result(7, "sample.png", filtered, pipeline)

        self.assertIs(state.preprocessed_image, filtered)
        self.assertEqual(state.pipeline_config, pipeline)
        self.assertEqual(display_calls, [{"preserve_view": True}])
        self.widget._sync_current_state_views.assert_not_called()
        self.assertEqual(process_calls, [])

    def test_pending_filter_waits_until_running_recognition_finishes(self) -> None:
        cancel = threading.Event()
        self.widget._preview_running_request_id = 4
        self.widget._preview_run_cancel = cancel
        pending_request = object()
        self.widget._prepared_image_pending_request = pending_request  # type: ignore[assignment]

        self.widget._start_pending_prepared_image_update()

        self.assertTrue(cancel.is_set())
        self.assertIs(self.widget._prepared_image_pending_request, pending_request)
        self.assertIsNone(self.widget._prepared_image_running_request_id)

    def test_filter_has_priority_when_recognition_worker_finishes(self) -> None:
        self.widget._preview_running_request_id = 4
        self.widget._prepared_image_pending_request = object()  # type: ignore[assignment]
        self.widget._preview_pending_request = object()  # type: ignore[assignment]
        starts: list[str] = []
        self.widget._start_pending_prepared_image_update = lambda: starts.append("filter")  # type: ignore[method-assign]
        self.widget._start_pending_preview_processing = lambda: starts.append("recognition")  # type: ignore[method-assign]

        self.widget._on_preview_processing_finished(4)

        self.assertEqual(starts, ["filter"])
        self.assertIsNone(self.widget._preview_running_request_id)

    def test_pending_recognition_starts_after_filter_worker_finishes(self) -> None:
        self.widget._prepared_image_running_request_id = 6
        self.widget._preview_pending_request = object()  # type: ignore[assignment]
        starts: list[str] = []
        self.widget._start_pending_preview_processing = lambda: starts.append("recognition")  # type: ignore[method-assign]

        self.widget._on_prepared_image_finished(6)

        self.assertEqual(starts, ["recognition"])
        self.assertIsNone(self.widget._prepared_image_running_request_id)

    def test_recognition_does_not_start_while_filter_worker_is_running(self) -> None:
        pending_request = object()
        self.widget._preview_pending_request = pending_request  # type: ignore[assignment]
        self.widget._prepared_image_running_request_id = 6

        self.widget._start_pending_preview_processing()

        self.assertIs(self.widget._preview_pending_request, pending_request)
        self.assertIsNone(self.widget._preview_running_request_id)

    def test_extraction_mode_defaults_to_no_extraction(self) -> None:
        self.assertEqual(self.widget.recognition_mode_combo.currentData(), "disabled")
        self.assertEqual(self.widget.recognition_mode_combo.currentText(), "Отключено")

    def test_process_selected_button_requests_processing_and_save(self) -> None:
        queued: list[dict[str, bool]] = []
        self.widget._queue_preview_processing = lambda **kwargs: queued.append(kwargs)  # type: ignore[method-assign]

        self.widget.process_current_button.click()

        self.assertEqual(queued, [{"debounced": False, "save_result": True}])

    def test_explicit_preview_result_is_saved_with_processed_polygons(self) -> None:
        polygon = _rectangle_polygon(4, 4, 20, 20)
        save_calls: list[list[PolygonData]] = []
        self.widget.save_current_result = lambda *args, **kwargs: save_calls.append(kwargs["polygons"])  # type: ignore[method-assign]
        self.widget._preview_running_request_id = 7
        self.widget._preview_running_save_result = True

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

        self.assertEqual(len(save_calls), 1)
        self.assertEqual(save_calls[0][0].points, polygon.points)

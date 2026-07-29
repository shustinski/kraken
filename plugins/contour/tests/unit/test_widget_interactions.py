from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSignalBlocker, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap, QRegion, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsPathItem,
    QGraphicsView,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
)

import contour.widget as widget_module
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    BatchImageResult,
    ContourDebugCandidate,
    DisplaySettings,
    ImageProcessingState,
)
from contour.application.services.workspace_session import WorkspaceLoadResult
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_items import EditablePolygonItem
from contour.graphics_view import (
    BrushMode,
    DeleteVertexMode,
    EditorTool,
    PolygonCreateMode,
    PolygonEditorScene,
    PolygonEditorView,
)
from contour.utils import draw_polygon_overlay
from contour.widget import PolygonExtractionWidget


class ViaCandidateOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_rejected_candidate_is_drawn_and_can_be_cleared(self) -> None:
        scene = PolygonEditorScene()
        candidate = ContourDebugCandidate(
            contour_index=1,
            bbox=(10, 12, 8, 8),
            accepted=False,
            reason="rejected:hard:circularity",
            score=24.0,
        )

        scene.set_debug_candidates([candidate])

        self.assertEqual(len(scene._debug_candidate_items), 1)
        self.assertEqual(
            scene._debug_candidate_items[0].pen().color(),
            QColor("#EF4444"),
        )
        scene.set_debug_candidates([])
        self.assertEqual(scene._debug_candidate_items, [])

    def test_via_color_changes_from_red_to_green_with_score(self) -> None:
        scene = PolygonEditorScene()
        low = _rectangle_polygon(2, 2, 10, 10)
        low.id = 1
        low.category = "via"
        low.recognition_score = 0.0
        high = _rectangle_polygon(14, 2, 22, 10)
        high.id = 2
        high.category = "via"
        high.recognition_score = 100.0

        scene.set_polygons([low, high])

        low_color = scene._polygon_items[1].pen().color()
        high_color = scene._polygon_items[2].pen().color()
        self.assertGreater(low_color.red(), low_color.green())
        self.assertGreater(high_color.green(), high_color.red())

    def test_show_rejected_checkbox_controls_candidate_overlay(self) -> None:
        editor = MagicMock()
        rejected = ContourDebugCandidate(
            contour_index=1,
            bbox=(10, 12, 8, 8),
            accepted=False,
            reason="rejected:hard:circularity",
        )
        show_rejected = QCheckBox()
        show_rejected.setChecked(True)
        owner = SimpleNamespace(
            polygon_editor=editor,
            debug_candidates_checkbox=QCheckBox(),
            bright_via_show_rejected_checkbox=show_rejected,
            _sync_polygons_to_editor=MagicMock(),
            _via_debug_inspection_enabled=lambda: True,
        )
        owner.debug_candidates_checkbox.setChecked(True)
        state = SimpleNamespace(debug_candidates=[rejected])

        processing_mixin_module.WidgetProcessingMixin._apply_editor_vectors_for_frame(
            owner,
            "sample.png",
            state,
            [],
            defer_heavy_overlays=False,
        )
        editor.set_debug_candidates.assert_called_with([rejected])

        show_rejected.setChecked(False)
        processing_mixin_module.WidgetProcessingMixin._apply_editor_vectors_for_frame(
            owner,
            "sample.png",
            state,
            [],
            defer_heavy_overlays=False,
        )
        editor.set_debug_candidates.assert_called_with([])


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


def _oversampled_rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    points = [
        (float(left), float(top)),
        (mid_x, float(top)),
        (float(right), float(top)),
        (float(right), mid_y),
        (float(right), float(bottom)),
        (mid_x, float(bottom)),
        (float(left), float(bottom)),
        (float(left), mid_y),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


def _net_outline_area(polygons: list[PolygonData]) -> float:
    """Subtract hole areas from roots (handles flat lists after CSG raster ops)."""

    outers = sum(p.area for p in polygons if not p.is_hole)
    holes = sum(p.area for p in polygons if p.is_hole)
    return outers - holes


def _all_points_within(polygons: list[PolygonData], left: float, top: float, right: float, bottom: float) -> bool:
    return all(left <= x <= right and top <= y <= bottom for polygon in polygons for x, y in polygon.points)


class PolygonExtractionWidgetLoadImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()

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

    def _install_workspace_stub(self, image_path: str = "sample.png") -> None:
        state = ImageProcessingState(
            image_path=image_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
        )

        def _load_image(path: str, *, load_source_image, load_cif_overlay) -> WorkspaceLoadResult:
            del load_source_image, load_cif_overlay
            self.widget._workspace._current_image_path = str(path)
            self.widget._workspace._current_state = state
            return WorkspaceLoadResult(
                image_path=str(path),
                state=state,
                prepared_image_required=True,
            )

        self.widget._workspace.load_image = _load_image  # type: ignore[method-assign]
        self.widget._workspace.resolve_cached_load = lambda path: _load_image(  # type: ignore[method-assign]
            str(path),
            load_source_image=None,
            load_cif_overlay=None,
        )

    def test_load_image_keeps_prepared_preview_flow_when_auto_apply_enabled(self) -> None:
        self._install_workspace_stub()
        process_calls: list[bool] = []
        prepared_calls: list[tuple[str, object]] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]
        self.widget._queue_prepared_image_update = lambda *args: prepared_calls.append(args)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(True)
        self.widget.load_image("sample.png")

        self.assertEqual(process_calls, [])
        self.assertEqual(len(prepared_calls), 1)

    def test_load_image_preserves_prepared_preview_flow_when_auto_apply_disabled(self) -> None:
        self._install_workspace_stub()
        process_calls: list[bool] = []
        prepared_calls: list[tuple[str, object]] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]
        self.widget._queue_prepared_image_update = lambda *args: prepared_calls.append(args)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(False)
        self.widget.load_image("sample.png")

        self.assertEqual(process_calls, [])
        self.assertEqual(len(prepared_calls), 1)


class PolygonExtractionWidgetExtractionAutoApplyTests(unittest.TestCase):
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
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()

        self.assertEqual(process_calls, [])

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

    def test_no_extraction_keeps_loaded_vector_overlay_visible(self) -> None:
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([_rectangle_polygon(4, 4, 20, 20)])
        self._app.processEvents()
        items = list(self.widget.polygon_editor._editor_scene._polygon_items.values())
        self.assertTrue(items)
        self.assertTrue(all(item.isVisible() for item in items))

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        self._app.processEvents()

        self.assertTrue(self.widget.polygon_editor.polygon_overlays_visible())
        self.assertTrue(all(item.isVisible() for item in items))

        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))
        self._app.processEvents()

        self.assertTrue(self.widget.polygon_editor.polygon_overlays_visible())
        self.assertTrue(all(item.isVisible() for item in items))

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

    def test_cached_frame_with_stale_pipeline_is_reprocessed_on_switch(self) -> None:
        state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            preprocessed_image=np.full((32, 32), 255, dtype=np.uint8),
            pipeline_config={"steps": [{"operation": "threshold", "parameters": {"threshold": 80}}]},
        )
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = state
        self.widget._workspace._state_cache = {"sample.png": state}
        self.widget.get_pipeline = lambda: {"steps": [{"operation": "threshold", "parameters": {"threshold": 160}}]}  # type: ignore[method-assign]
        prepared_calls: list[tuple[str, object]] = []
        self.widget._queue_prepared_image_update = lambda image_path, source_image: prepared_calls.append(
            (image_path, source_image)
        )  # type: ignore[method-assign]
        self.widget._request_neighbor_frame_sync = lambda *, delay_ms=0: None  # type: ignore[method-assign]

        self.widget._finish_frame_load_ui(
            WorkspaceLoadResult(image_path="sample.png", state=state, cache_hit=True),
            load_vectors=False,
        )

        self.assertIsNone(state.preprocessed_image)
        self.assertEqual(state.pipeline_config, None)
        self.assertEqual(prepared_calls, [("sample.png", state.source_image)])

    def test_extraction_change_processes_when_auto_apply_enabled(self) -> None:
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(True)
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()
        QTest.qWait(200)
        self._app.processEvents()

        self.assertEqual(process_calls[-1], False)

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

    def test_neighbor_frames_render_around_current_image(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(25)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[12]
        self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(5)
        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget._sync_neighbor_frames()

        neighbor_items = self.widget.polygon_editor._editor_scene._neighbor_frame_items
        self.assertEqual(len(neighbor_items), 8)
        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.path().isEmpty())
        self.assertTrue(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_lod_store_does_not_replace_main_editor_neighbor_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(9):
                path = Path(directory) / f"frame_{index:02d}.png"
                cv2.imwrite(str(path), np.full((12, 12, 3), index, dtype=np.uint8))
                paths.append(str(path))

            self.widget._workspace._image_paths = paths
            self.widget._workspace._current_image_path = paths[4]
            self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
            self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
            self.widget.neighbor_columns_spin.setValue(3)
            self.widget.neighbor_max_grid_spin.setValue(3)
            self.widget.show_neighbor_frames_checkbox.setChecked(True)

            self.widget._configure_pyramid_frame_store(paths)
            self.widget._sync_neighbor_frames()

            self.assertFalse(self.widget.polygon_editor.pyramid_mode_enabled())
            self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 8)

    def test_neighbor_frames_render_when_main_image_arrives_after_neighbor_request(self) -> None:
        scene = PolygonEditorScene()
        scene.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 0)

        scene.set_image(np.zeros((12, 12), dtype=np.uint8))

        self.assertEqual(len(scene._neighbor_frame_items), 1)

    def test_applying_new_frame_clears_stale_neighbor_layout_before_sync(self) -> None:
        scene = self.widget.polygon_editor._editor_scene
        self.widget.polygon_editor.set_image(np.zeros((12, 48), dtype=np.uint8))
        self.widget.polygon_editor.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "old_neighbor.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 1)
        self.widget._neighbor_sync_image_path = "old_frame.png"
        self.widget._neighbor_frame_specs = [(1, 0, "old_neighbor.png")]
        self.widget._neighbor_queued_paths = {"old_neighbor.png"}
        state = ImageProcessingState(
            image_path="new_frame.png",
            source_image=np.zeros((48, 12), dtype=np.uint8),
            polygons=[],
        )
        self.widget._workspace._current_image_path = "new_frame.png"
        self.widget._workspace._current_state = state
        self.widget._apply_display_image_to_editor = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
        self.widget._request_neighbor_frame_sync = lambda *args, **kwargs: None  # type: ignore[method-assign]

        self.widget._apply_frame_to_editor()

        self.assertIsNone(scene._pending_neighbor_frames)
        self.assertEqual(len(scene._neighbor_frame_items), 0)
        self.assertIsNone(self.widget._neighbor_sync_image_path)
        self.assertEqual(self.widget._neighbor_frame_specs, [])
        self.assertEqual(self.widget._neighbor_queued_paths, set())

    def test_main_image_update_does_not_clear_neighbors_when_pending_list_empty(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((12, 12), dtype=np.uint8))
        scene.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 1)

        scene._pending_neighbor_frames = []
        scene.set_image(np.zeros((12, 12), dtype=np.uint8))

        self.assertEqual(len(scene._neighbor_frame_items), 1)

    def test_apply_cached_neighbors_skips_clear_while_neighbor_loads_are_queued(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        scene = self.widget.polygon_editor._editor_scene
        scene._pending_neighbor_frames = None
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_queued_paths = {paths[5]}

        self.widget._apply_cached_neighbor_frames()

        self.assertIsNone(scene._pending_neighbor_frames)
        self.assertEqual(len(scene._neighbor_frame_items), 0)

    def test_neighbor_sync_drops_stale_queued_paths_before_queueing_current_neighbors(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(3)
        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget._neighbor_sync_image_path = "previous.png"
        self.widget._neighbor_queued_paths = {"previous_neighbor.png"}
        queued: list[str] = []

        def _queue(path: str) -> None:
            queued.append(path)
            self.widget._neighbor_queued_paths.add(path)

        self.widget._queue_neighbor_frame_load = _queue  # type: ignore[method-assign]

        self.widget._sync_neighbor_frames()

        self.assertNotIn("previous_neighbor.png", self.widget._neighbor_queued_paths)
        self.assertEqual(set(queued), set(path for path in paths if path != paths[4]))

    def test_neighbor_load_finished_applies_when_queue_drains(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_queued_paths = {paths[5]}
        applied: list[int] = []
        self.widget._schedule_neighbor_frame_apply = lambda *, delay_ms=0: applied.append(delay_ms)  # type: ignore[method-assign]

        self.widget._on_neighbor_frame_load_finished(0, paths[5])

        self.assertEqual(self.widget._neighbor_queued_paths, set())
        self.assertEqual(applied, [0])

    def test_neighbor_grid_size_controls_displayed_neighbor_ring_count(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(49)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[24]
        self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(7)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)

        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget._sync_neighbor_frames()
        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 8)

        self.widget.neighbor_max_grid_spin.setValue(5)
        self.widget._sync_neighbor_frames()
        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 24)

    def test_file_list_uses_stems_and_thumbnail_click_navigates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
            self.widget.load_images(paths)
            self._wait_for_thumbnail_grid_count(2)
            self.widget.load_image = lambda path, **_kwargs: setattr(self.widget, "_last_loaded_from_thumb", path)  # type: ignore[method-assign]

            self.assertEqual(self.widget.image_list.item(0).text(), "frame_001")
            self.assertEqual(self.widget.thumbnail_grid.count(), 2)
            self.assertEqual(self.widget.thumbnail_grid.item(0).text(), "")
            self.assertEqual(self.widget.thumbnail_grid.item(0).toolTip(), "frame_001")
            self.widget._on_thumbnail_item_clicked(self.widget.thumbnail_grid.item(1))

            self.assertEqual(self.widget._last_loaded_from_thumb, paths[1])

    def test_append_images_keeps_existing_rows_skips_duplicates_and_underscore_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "frame_001.png")
            second = os.path.join(directory, "frame_002.png")
            hidden = os.path.join(directory, "_frame_003.png")
            for path in (first, second, hidden):
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))

            self.widget.load_images([first])
            self.widget.append_images([first, hidden, second])

            self.assertEqual(self.widget.image_list.count(), 2)
            self.assertEqual([self.widget.image_list.item(i).text() for i in range(2)], ["frame_001", "frame_002"])
            self.assertEqual(self.widget._workspace.current_image_path, str(Path(second)))

    def test_select_input_directory_replaces_instead_of_appending(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = os.path.join(first_dir, "frame_001.png")
            second = os.path.join(second_dir, "frame_001.png")
            cv2.imwrite(first, np.zeros((8, 8), dtype=np.uint8))
            cv2.imwrite(second, np.ones((8, 8), dtype=np.uint8))
            scans: list[tuple[str, bool]] = []

            self.widget.load_images([first])
            self.widget._begin_async_directory_scan = lambda directory, *, append=False: scans.append(
                (str(Path(directory)), bool(append))
            )  # type: ignore[method-assign]

            with patch(
                "contour.widget_parts.navigation_mixin.QFileDialog.getExistingDirectory", return_value=second_dir
            ):
                self.widget._select_input_directory()

            self.assertEqual(scans, [(str(Path(second_dir)), False)])
            self.assertEqual(self.widget.input_dir_edit.text(), str(Path(second_dir)))
            self.assertEqual(self.widget._image_list_mode, "directory_scan")

    def test_scan_finish_restores_persisted_current_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget._pending_restore_current_image_path = str(Path(paths[1]))
            self.widget._on_input_directory_scan_finished(paths)

            self.assertEqual(self.widget._workspace.current_image_path, str(Path(paths[1])))
            self.assertEqual(self.widget.image_list.currentRow(), 1)

    def test_restore_persisted_session_selection_loads_images_vectors_and_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                image_paths.append(path)
            vector_path = os.path.join(directory, "frame_999.cif")
            Path(vector_path).write_text("", encoding="utf-8")

            self.widget._session_settings_store.save_image_paths(image_paths)
            self.widget._session_settings_store.save_vector_paths([vector_path])
            self.widget._session_settings_store.save_current_image_path(image_paths[1])

            self.widget._restore_persisted_session_selection()
            self._wait_for_thumbnail_grid_count(len(image_paths))

            self.assertEqual(self.widget._workspace.image_paths, tuple(str(Path(path)) for path in image_paths))
            self.assertEqual(self.widget._workspace.current_image_path, str(Path(image_paths[1])))
            self.assertEqual(self.widget.thumbnail_grid.count(), len(image_paths))
            self.assertIn("frame_999", self.widget._workspace.cif_paths_by_stem)

    def test_restored_frame_matrix_build_does_not_wait_for_cif_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                image_paths.append(path)

            self.widget._session_settings_store.save_image_paths(image_paths)
            self.widget._session_settings_store.save_current_image_path(image_paths[0])
            self.widget.cif_dir_edit.setText(directory)
            self.widget._indexed_cif_directory = None
            index_requests: list[str] = []
            self.widget._begin_async_cif_directory_index = lambda path: index_requests.append(  # type: ignore[method-assign]
                str(Path(path))
            )

            self.widget._restore_persisted_session_selection()
            self._wait_for_thumbnail_grid_count(len(image_paths))

            self.assertEqual(self.widget.thumbnail_grid.count(), len(image_paths))
            self.assertEqual(index_requests, [str(Path(directory))])

    def test_directory_session_restore_rescans_and_loads_current_vectors(self) -> None:
        from contour.infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame_001.png"
            vector_path = Path(directory) / "frame_001.cif"
            cv2.imwrite(str(image_path), np.zeros((16, 16), dtype=np.uint8))
            vector_path.write_text(
                "\n".join(
                    (
                        "DS 1 1 1;",
                        "L NM;",
                        "( R frame_001.png );",
                        "( S 16 16 );",
                        "B 6 6 8 8;",
                        "DF;",
                        "E",
                    )
                ),
                encoding="utf-8",
            )
            self.widget.input_dir_edit.setText(directory)
            self.widget.cif_dir_edit.setText(directory)
            self.widget._session_settings_store.save_image_list_mode(IMAGE_LIST_MODE_DIRECTORY)
            self.widget._session_settings_store.save_image_paths([image_path])
            self.widget._session_settings_store.save_vector_paths([])
            self.widget._session_settings_store.save_current_image_path(image_path)

            self.widget._restore_persisted_session_selection()

            for _ in range(500):
                self._app.processEvents()
                state = self.widget._workspace.current_state
                if (
                    state is not None
                    and state.image_path == str(image_path)
                    and len(state.polygons) == 1
                ):
                    break
                QTest.qWait(10)

            state = self.widget._workspace.current_state
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.image_path, str(image_path))
            self.assertEqual(len(state.polygons), 1)
            self.assertEqual(
                self.widget._workspace.resolve_cif_path(image_path),
                str(vector_path),
            )

    def test_refresh_preserves_explicit_image_subset(self) -> None:
        from contour.infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_a.png", "frame_b.png", "frame_c.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)
            self.widget._set_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)
            self.widget.append_images(paths[:2])
            self.widget.input_dir_edit.setText(directory)
            self.widget.refresh_image_list()
            self._app.processEvents()

            self.assertEqual(
                self.widget._workspace.image_paths,
                tuple(str(Path(path)) for path in paths[:2]),
            )

    def test_reset_project_clears_loaded_state_without_resetting_display_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frame_001.png")
            cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
            self.widget.load_images([path])
            self.widget.input_dir_edit.setText(directory)
            self.widget.cif_dir_edit.setText(directory)
            self.widget.line_width_spin.setValue(4.0)

            self.widget.reset_project()

            self.assertEqual(self.widget.image_list.count(), 0)
            self.assertEqual(self.widget.thumbnail_grid.count(), 0)
            self.assertEqual(self.widget.vector_list.count(), 0)
            self.assertEqual(self.widget._workspace.image_paths, ())
            self.assertIsNone(self.widget._workspace.current_image_path)
            self.assertEqual(self.widget.input_dir_edit.text(), "")
            self.assertEqual(self.widget.cif_dir_edit.text(), "")
            self.assertEqual(self.widget.line_width_spin.value(), 4.0)
            self.assertIsNone(self.widget._session_settings_store.load_current_image_path())
            self.assertEqual(self.widget._session_settings_store.load_image_paths(), [])
            self.assertEqual(self.widget._session_settings_store.load_vector_paths(), [])

    def test_image_row_navigation_recenters_editor_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.load_images(paths)
            with patch.object(self.widget, "_center_editor_on_current_main_image") as center_mock:
                proxy_index = self.widget._image_list_proxy.index(1, 0)
                self.widget.image_list.setCurrentIndex(proxy_index)
                for _ in range(100):
                    self._app.processEvents()
                    if call(force=True) in center_mock.call_args_list:
                        break
                    QTest.qWait(20)

            center_mock.assert_any_call(force=True)

    def test_rapid_image_row_navigation_loads_only_latest_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png", "frame_003.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(str(Path(path)))

            self.widget.load_images(paths)
            for _ in range(50):
                self._app.processEvents()
                if self.widget._image_list_proxy.rowCount() == len(paths):
                    break
                QTest.qWait(10)

            loaded_paths: list[str] = []
            self.widget.load_image = lambda path: loaded_paths.append(str(Path(path)))  # type: ignore[method-assign]
            self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(1, 0))
            self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(2, 0))
            QTest.qWait(350)
            self._app.processEvents()

            self.assertEqual(loaded_paths, [paths[2]])

    def test_completed_background_load_does_not_replace_newer_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(str(Path(path)))

            queued_loads = []
            self.widget._frame_load_thread_pool.start = queued_loads.append  # type: ignore[method-assign]
            self.widget.load_images(paths)
            for _ in range(50):
                self._app.processEvents()
                if self.widget._image_list_proxy.rowCount() == len(paths):
                    break
                QTest.qWait(10)
            for timer in list(self.widget._deferred_image_load_timers):
                timer.stop()
            self.widget._deferred_image_load_timers = []
            self.widget._loading_image_path = None
            self.widget.load_image(paths[0], load_vectors=False)
            self.assertEqual(len(queued_loads), 1)

            self.widget.load_image(paths[1], load_vectors=False)
            self.assertEqual(self.widget._frame_load_pending, (paths[1], False, False))

            queued_loads[0].run()
            self._app.processEvents()

            self.assertEqual(self.widget._workspace.current_image_path, paths[0])
            self.assertIsNone(self.widget._workspace.current_state)
            self.assertIn(paths[0], self.widget._workspace._state_cache)
            self.assertEqual(len(queued_loads), 2)
            # Finish the deliberately intercepted successor request before the
            # TemporaryDirectory removes its Windows image files.
            queued_loads[1].run()
            self._app.processEvents()
            self.widget._thumbnail_thread_pool.waitForDone(3000)
            self.assertEqual(queued_loads[1].image_path, paths[1])

    def test_cif_index_completion_queues_vectors_for_inflight_initial_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = str(Path(directory) / "frame_001.png")
            vector_path = str(Path(directory) / "frame_001.cif")
            cv2.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
            Path(vector_path).write_text("", encoding="utf-8")
            self.widget._workspace.replace_image_selection(
                [image_path],
                is_supported_image=lambda _path: True,
            )
            self.widget._workspace.set_cif_index({"frame_001": vector_path})
            self.widget._desired_image_path = image_path
            self.widget._loading_image_path = image_path
            self.widget._frame_load_running_path = image_path
            self.widget._frame_load_pending = None

            self.widget._sync_after_cif_index_changed()

            self.assertEqual(
                self.widget._frame_load_pending,
                (image_path, True, False),
            )

    def test_cached_first_frame_is_applied_when_switching_back_from_second_frame(self) -> None:
        paths = [str(Path("frame_001.png")), str(Path("frame_002.png"))]
        self.widget._workspace.replace_image_selection(paths, is_supported_image=lambda _path: True)
        for path, value in zip(paths, (1, 2), strict=True):
            self.widget._workspace.apply_loaded_frame(
                path,
                source_image=np.full((8, 8), value, dtype=np.uint8),
                polygons=[],
            )
        self.widget._desired_image_path = paths[1]
        self.widget._last_editor_display_cache_key = (paths[1], "source", "")

        with (
            patch.object(self.widget, "_apply_frame_to_editor") as apply_frame,
            patch.object(self.widget, "_try_extract_if_recognition_enabled"),
        ):
            self.widget.load_image(paths[0], load_vectors=False)

        self.assertEqual(self.widget._workspace.current_image_path, paths[0])
        self.assertIsNotNone(self.widget._workspace.current_state)
        self.assertEqual(self.widget._workspace.current_state.image_path, paths[0])
        apply_frame.assert_called_once()

    def test_checked_toolbar_tool_has_explicit_high_contrast_style(self) -> None:
        stylesheet = self.widget.styleSheet()

        self.assertIn("QToolButton:checked", stylesheet)
        self.assertIn("#16A34A", stylesheet)

    def test_toolbar_icons_use_the_complete_button_area(self) -> None:
        buttons = [
            *self.widget._tool_buttons.values(),
            self.widget.undo_button,
            self.widget.redo_button,
            self.widget.zoom_in_button,
            self.widget.zoom_out_button,
            self.widget.fit_button,
            self.widget.antialias_opened_cif_button,
        ]

        for button in buttons:
            self.assertEqual(button.iconSize().width(), button.maximumWidth())
            self.assertEqual(button.iconSize().height(), button.maximumHeight())
            self.assertIn("padding: 0", button.styleSheet())
            self.assertTrue(button.toolTip())
            pixmap = button.icon().pixmap(button.iconSize())
            visible_rect = QRegion(pixmap.mask()).boundingRect()
            self.assertGreaterEqual(visible_rect.width(), pixmap.width() * 0.9)
            self.assertGreaterEqual(visible_rect.height(), pixmap.height() * 0.9)

    def test_hotkey_tool_switch_updates_toolbar_checked_state(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._tool_buttons[EditorTool.BRUSH].isChecked())
        self.assertFalse(self.widget._tool_buttons[EditorTool.SELECT].isChecked())

        self.widget.polygon_editor.set_tool(EditorTool.SELECT)
        self._app.processEvents()

        self.assertTrue(self.widget._tool_buttons[EditorTool.SELECT].isChecked())
        self.assertFalse(self.widget._tool_buttons[EditorTool.BRUSH].isChecked())

    def test_delete_polygon_tool_is_not_exposed_in_toolbar(self) -> None:
        self.assertNotIn(EditorTool.DELETE_POLYGON, self.widget._tool_buttons)

    def test_antialias_tool_is_exposed_in_toolbar(self) -> None:
        self.assertIn(EditorTool.ANTIALIAS, self.widget._tool_buttons)

    def test_trace_pen_tool_is_exposed_in_toolbar(self) -> None:
        self.assertIn(EditorTool.TRACE_PEN, self.widget._tool_buttons)

    def test_antialias_settings_are_only_visible_for_antialias_tool(self) -> None:
        self.assertTrue(self.widget._antialias_toolbar_block.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.ANTIALIAS)
        self._app.processEvents()

        self.assertFalse(self.widget._antialias_toolbar_block.isHidden())
        self.assertFalse(self.widget.antialias_grade_label.isHidden())
        self.assertFalse(self.widget.antialias_grade_spin.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._antialias_toolbar_block.isHidden())

    def test_trace_pen_settings_are_only_visible_for_trace_pen_tool(self) -> None:
        self.assertTrue(self.widget._trace_toolbar_block.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.TRACE_PEN)
        self._app.processEvents()

        self.assertFalse(self.widget._trace_toolbar_block.isHidden())
        self.assertFalse(self.widget.trace_width_label.isHidden())
        self.assertFalse(self.widget.trace_width_spin.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._trace_toolbar_block.isHidden())

    def test_tool_settings_strip_stays_in_fixed_toolbar_position(self) -> None:
        strip_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self._app.processEvents()
        polygon_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()
        brush_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.assertEqual(strip_index, polygon_index)
        self.assertEqual(strip_index, brush_index)
        self.assertFalse(self.widget._brush_toolbar_block.isHidden())
        self.assertTrue(self.widget._polygon_toolbar_block.isHidden())

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

    def test_large_thumbnail_matrix_keeps_all_frames(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(10_000)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.neighbor_columns_spin.setValue(10)
        self.widget.neighbor_overlap_spin.setValue(0)
        self.widget._workspace._current_image_path = paths[5_005]
        self.widget._thumbnail_build_chunk_size = 20_000
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(10_000)

        self.assertEqual(self.widget.thumbnail_grid.count(), 10_000)
        self.assertEqual(self.widget._thumbnail_path_to_row[str(Path(paths[5_005]))], 5_005)
        self.assertGreaterEqual(self.widget.thumbnail_grid.minimumWidth(), 10 * 64)
        self.assertGreaterEqual(self.widget.thumbnail_grid.minimumHeight(), 1_000 * 48)
        self.widget._workspace._current_image_path = "sample.png"

    def test_thumbnail_matrix_rebuild_reuses_cached_icons(self) -> None:
        path = str(Path(r"d:\frames\frame_000.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 10
        requested_size = self.widget._thumbnail_request_size()
        pixmap = QPixmap(*requested_size)
        pixmap.fill(QColor("#123456"))
        self.widget._thumbnail_icon_cache[(path, requested_size)] = QIcon(pixmap)
        self.widget._thumbnail_loading_blocked = lambda: True  # type: ignore[method-assign]

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)

        self.assertFalse(self.widget.thumbnail_grid.item(0).icon().isNull())
        self.assertEqual(
            self.widget._thumbnail_loaded_generation[path],
            self.widget._thumbnail_generation,
        )
        self.assertEqual(self.widget._thumbnail_loaded_sizes[path], requested_size)

    def test_reapplying_same_image_paths_preserves_full_scene_cache(self) -> None:
        path = str(Path(r"d:\frames\frame_000.png"))
        state = ImageProcessingState(
            image_path=path,
            source_image=np.ones((24, 32, 3), dtype=np.uint8),
        )
        self.widget._workspace._image_paths = [path]
        self.widget._workspace._state_cache[path] = state
        self.widget._workspace._current_image_path = path
        self.widget._workspace._current_state = state
        self.widget._rebuild_image_list_items = lambda _paths: None  # type: ignore[method-assign]

        self.widget._apply_image_paths_to_workspace(
            [path],
            clear_extra_layers=False,
            select_path=path,
            fallback_to_first=True,
            clear_state_cache=True,
        )

        self.assertIs(self.widget._workspace._state_cache[path], state)
        self.assertIs(self.widget._workspace._state_cache[path].source_image, state.source_image)

    def test_main_and_neighbor_scene_loads_share_full_resolution_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "frame_000.png")
            cv2.imwrite(path, np.full((24, 32, 3), 73, dtype=np.uint8))

            with patch.object(
                processing_mixin_module,
                "load_image_color",
                wraps=processing_mixin_module.load_image_color,
            ) as decode:
                neighbor_source = self.widget._load_scene_source_image_cached(path)
                main_source = self.widget._load_scene_source_image_cached(path)

            self.assertIs(main_source, neighbor_source)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(self.widget._scene_source_image_cache_bytes, neighbor_source.nbytes)

    def test_large_frame_matrix_count_matches_images_tab_count(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(1_200)]
        normalized = [str(Path(path)) for path in paths]
        self.widget._workspace._image_paths = normalized
        self.widget._set_image_list_paths(normalized)
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 2_000

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(len(normalized))

        self.assertEqual(self.widget.thumbnail_grid.count(), self.widget._image_list_proxy.rowCount())
        self.assertEqual(self.widget.thumbnail_grid.count(), len(normalized))

    def test_thumbnail_lod_change_does_not_cancel_large_matrix_build(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:04d}.png")) for index in range(350)]
        self.widget._workspace._image_paths = paths
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100

        self.widget._rebuild_thumbnail_grid()
        QTimer.singleShot(5, self.widget._on_thumbnail_lod_changed)
        self._wait_for_thumbnail_grid_count(len(paths))

        self.assertEqual(self.widget.thumbnail_grid.count(), len(paths))
        self.assertFalse(self.widget._thumbnail_rebuild_in_progress)

    def test_frame_matrix_rebuild_yields_before_adding_items(self) -> None:
        paths = [rf"d:\frames\frame_{index:05d}.png" for index in range(300)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 300
        self.widget._thumbnail_build_interval_ms = 0

        self.widget._rebuild_thumbnail_grid()

        self.assertEqual(self.widget.thumbnail_grid.count(), 0)
        self._wait_for_thumbnail_grid_count(300)

    def test_disabling_frame_matrix_clears_and_skips_matrix_work(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(20)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(20)
        self.assertEqual(self.widget.thumbnail_grid.count(), 20)

        self.widget.show_frame_matrix_checkbox.setChecked(False)
        self.widget._schedule_thumbnail_grid_rebuild(force=True)
        self.widget._update_thumbnail_grid_selection()

        self.assertEqual(self.widget.thumbnail_grid.count(), 0)
        self.assertEqual(self.widget._thumbnail_path_to_row, {})
        self.assertFalse(self.widget.thumbnail_matrix_panel.isVisible())

    def test_disabling_frame_matrix_thumbnails_keeps_items_without_queueing_loads(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(12)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        queued: list[str] = []
        self.widget._queue_thumbnail_load = lambda _generation, path: queued.append(path)  # type: ignore[method-assign]

        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(12)
        self.widget._schedule_visible_thumbnail_loads()

        self.assertEqual(self.widget.thumbnail_grid.count(), 12)
        self.assertEqual(queued, [])

    def test_enabling_frame_matrix_thumbnails_restarts_loading_existing_items(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(12)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(12)
        resumed: list[bool] = []
        self.widget._resume_frame_matrix_thumbnail_loading = lambda: resumed.append(True)  # type: ignore[method-assign]

        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)

        self.assertEqual(resumed, [True])

    def test_visible_thumbnail_scheduler_prioritizes_viewport_and_current_frame(self) -> None:
        paths = [rf"d:\frames\frame_{index:03d}.png" for index in range(200)]
        self.widget._workspace._image_paths = [str(Path(path)) for path in paths]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._workspace._current_image_path = str(Path(paths[155]))
        self.widget.neighbor_columns_spin.setValue(10)
        self.widget._thumbnail_build_chunk_size = 200
        queued: list[str] = []
        self.widget._queue_thumbnail_load = lambda _generation, path: queued.append(path)  # type: ignore[method-assign]

        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(200)
        self.widget._schedule_visible_thumbnail_loads()

        queued_indexes = {paths.index(path) for path in queued if path in paths}
        self.assertTrue(any(150 <= index <= 159 for index in queued_indexes))
        self.assertLess(len(queued_indexes), len(paths))
        self.widget._workspace._current_image_path = "sample.png"

    def test_radial_thumbnail_fill_keeps_visible_thumbnail_queue(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(20)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[10]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget.neighbor_columns_spin.setValue(5)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(20)
        requested_size = self.widget._thumbnail_request_size()
        generation = self.widget._thumbnail_generation
        self.widget._thumbnail_queued_paths = {paths[0]}
        self.widget._thumbnail_queued_sizes = {paths[0]: requested_size}

        self.widget._reseed_thumbnail_radial_fill()

        self.assertEqual(self.widget._thumbnail_generation, generation)
        self.assertEqual(self.widget._thumbnail_queued_paths, {paths[0]})
        self.assertNotIn(paths[0], self.widget._thumbnail_radial_paths)

    def test_thumbnail_result_during_frame_load_is_applied_after_load_finishes(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        before = item.icon().cacheKey()
        self.widget._loading_image_path = path

        qimage = QImage(4, 4, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 64, 48, qimage)
        self.widget._loading_image_path = None
        self.widget._flush_thumbnail_icon_batch()

        self.assertNotEqual(item.icon().cacheKey(), before)

    def test_loaded_thumbnail_refreshes_graphics_matrix_pixmap(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        before = group[3].pixmap().cacheKey()

        qimage = QImage(64, 48, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 64, 48, qimage)
        self.widget._flush_thumbnail_icon_batch()

        self.assertNotEqual(group[3].pixmap().cacheKey(), before)

    def test_frame_matrix_navigation_loads_main_editor_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        loaded: list[tuple[str, bool]] = []
        self.widget.load_image = lambda path, **kwargs: loaded.append(  # type: ignore[method-assign]
            (str(Path(path)), bool(kwargs.get("preserve_editor_view_position")))
        )

        self.widget._on_frame_navigation_requested(2)

        self.assertEqual(loaded, [(paths[2], True)])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[2])

    def test_frame_matrix_navigation_does_not_center_editor_scene(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget.load_image = lambda _path, **_kwargs: None  # type: ignore[method-assign]

        with patch.object(self.widget.polygon_editor, "set_current_frame_id") as set_frame_mock:
            self.widget._on_frame_navigation_requested(2)

        set_frame_mock.assert_called_once_with(2, center=False, emit_signal=False)

    def test_frame_matrix_navigation_cancel_keeps_current_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[0]
        self.widget._set_image_list_paths(paths)
        selection = self.widget.image_list.selectionModel()
        if selection is not None:
            with QSignalBlocker(selection):
                self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(0, 0))
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(3)
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]
        self.widget._try_leave_current_frame = lambda: False  # type: ignore[method-assign]

        self.widget._on_frame_navigation_requested(2)

        self.assertEqual(loaded, [])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[0])
        self.assertEqual(
            str(Path(self.widget._image_list_path_from_proxy_index(self.widget.image_list.currentIndex()) or "")),
            paths[0],
        )

    def test_neighbor_overlay_navigation_loads_main_editor_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        self.widget._on_neighbor_frame_activated(paths[1])

        self.assertEqual(loaded, [paths[1]])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[1])

    def test_ctrl_arrow_navigates_selected_frame_when_frame_matrix_enabled(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget.neighbor_columns_spin.setValue(3)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(9)
        self.widget.thumbnail_grid.setCurrentRow(4)
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)

        self.assertEqual(loaded, [paths[5], paths[8]])
        self.assertEqual(self.widget.thumbnail_grid.currentRow(), 8)

    def test_ctrl_arrow_does_not_navigate_when_frame_matrix_disabled(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[1]
        self.widget.show_frame_matrix_checkbox.setChecked(False)
        self.widget.thumbnail_grid.addItem(QListWidgetItem(""))
        self.widget.thumbnail_grid.addItem(QListWidgetItem(""))
        self.widget.thumbnail_grid.setCurrentRow(1)
        loaded: list[str] = []
        self.widget.load_image = lambda path: loaded.append(str(Path(path)))  # type: ignore[method-assign]

        QTest.keyClick(self.widget.polygon_editor.viewport(), Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)

        self.assertEqual(loaded, [])
        self.assertEqual(self.widget.thumbnail_grid.currentRow(), 1)

    def test_thumbnail_request_size_follows_matrix_lod(self) -> None:
        self.widget.thumbnail_grid._matrix_zoom = 4.0

        self.assertEqual(self.widget._thumbnail_request_size(), (256, 192))

    def test_thumbnail_request_size_scales_to_max_matrix_lod(self) -> None:
        self.widget.thumbnail_grid._matrix_zoom = 32.0

        self.assertEqual(self.widget._thumbnail_request_size(), (512, 384))

    def test_frame_matrix_uses_editor_view_render_optimizations(self) -> None:
        matrix = self.widget.thumbnail_grid

        self.assertEqual(matrix.viewportUpdateMode(), QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.assertTrue(matrix.cacheMode() & QGraphicsView.CacheModeFlag.CacheBackground)
        self.assertTrue(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontSavePainterState)
        self.assertFalse(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)
        self.assertTrue(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)

        matrix._enter_zoom_render_mode()

        self.assertFalse(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
        self.assertTrue(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

        matrix._leave_zoom_render_mode()

        self.assertTrue(matrix.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
        self.assertFalse(matrix.optimizationFlags() & QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

    def test_thumbnail_lod_change_requeues_visible_frames_at_new_size(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        self.widget._thumbnail_loaded_generation[path] = self.widget._thumbnail_generation
        self.widget._thumbnail_loaded_sizes[path] = (64, 48)
        queued: list[tuple[int, str, tuple[int, int]]] = []

        def _capture_queue(generation: int, queued_path: str) -> None:
            queued.append((generation, queued_path, self.widget._thumbnail_request_size()))

        self.widget._queue_thumbnail_load = _capture_queue  # type: ignore[method-assign]
        self.widget.thumbnail_grid._matrix_zoom = 4.0

        self.widget._on_thumbnail_lod_changed()

        self.assertEqual(queued, [(self.widget._thumbnail_generation, path, (256, 192))])

    def test_graphics_matrix_keeps_high_lod_pixmap_scaled_to_cell(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        qimage = QImage(256, 192, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 256, 192, qimage)
        self.widget._flush_thumbnail_icon_batch()
        self.widget.thumbnail_grid._matrix_zoom = 4.0
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        pixmap_item = group[3]

        self.assertEqual(pixmap_item.pixmap().size(), QSize(256, 192))
        self.assertAlmostEqual(pixmap_item.scale(), 0.25)

    def test_graphics_matrix_cover_crops_square_thumbnail_to_fill_cell(self) -> None:
        path = str(Path(r"d:\frames\frame_001.png"))
        self.widget._workspace._image_paths = [path]
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(True)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(1)
        item = self.widget.thumbnail_grid.item(0)
        qimage = QImage(256, 256, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation, path, 256, 192, qimage)
        self.widget._flush_thumbnail_icon_batch()
        self.widget.thumbnail_grid._matrix_zoom = 4.0
        self.widget.thumbnail_grid.refreshItems()
        group = self.widget.thumbnail_grid._item_groups[id(item)]
        pixmap_item = group[3]

        self.assertEqual(pixmap_item.pixmap().size(), QSize(256, 192))
        self.assertAlmostEqual(pixmap_item.scale(), 0.25)

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
            Path(vector_a).write_text("placeholder", encoding="utf-8")
            Path(vector_lonely).write_text("placeholder", encoding="utf-8")

            self.widget.load_images([image_a, image_b])
            self._wait_for_thumbnail_grid_count(2)
            original_order = self.widget._workspace.image_paths
            self.widget._workspace.set_cif_index({"frame_a": vector_a, "lonely": vector_lonely})
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

    def test_work_simulation_restores_loaded_vectors_without_saving(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(2, 2, 12, 12)
        second_polygon = _rectangle_polygon(4, 4, 18, 18)
        states = {
            first_path: ImageProcessingState(
                image_path=first_path,
                source_image=np.zeros((24, 24), dtype=np.uint8),
                polygons=[first_polygon.clone()],
                reference_polygons=[first_polygon.clone()],
                loaded_cif_path="frame_1.cif",
                polygons_dirty=False,
            ),
            second_path: ImageProcessingState(
                image_path=second_path,
                source_image=np.zeros((24, 24), dtype=np.uint8),
                polygons=[second_polygon.clone()],
                reference_polygons=[second_polygon.clone()],
                loaded_cif_path="frame_2.cif",
                polygons_dirty=False,
            ),
        }
        self.widget._workspace.replace_image_selection([first_path, second_path], is_supported_image=lambda _path: True)
        self.widget._workspace.set_cif_index({"frame_1": "frame_1.cif", "frame_2": "frame_2.cif"})
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(Path(path).stem)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)

        def _fake_load_image(path: str) -> None:
            self.widget._workspace._current_image_path = path
            self.widget._workspace._current_state = states[path]
            self.widget.polygon_editor.set_image(states[path].source_image)
            self.widget.polygon_editor.set_polygons([polygon.clone() for polygon in states[path].polygons])

        save_calls: list[str] = []
        self.widget.load_image = _fake_load_image  # type: ignore[method-assign]
        self.widget.save_current_result = lambda *args, **kwargs: save_calls.append("save")  # type: ignore[method-assign]
        self.widget.export_current_frame_to_dataset = lambda *args, **kwargs: save_calls.append("export")  # type: ignore[method-assign]
        self.widget._work_simulation_interval_ms = 1

        self.widget._start_work_simulation()
        for _ in range(100):
            self._app.processEvents()
            if not self.widget._work_simulation_timer.isActive() and not self.widget._work_simulation_paths:
                break
            QTest.qWait(10)

        self.assertFalse(self.widget._work_simulation_timer.isActive())
        self.assertEqual(save_calls, [])
        self.assertEqual(self.widget._workspace.current_image_path, second_path)
        self.assertEqual(self.widget.get_polygons()[0].points, second_polygon.points)
        self.assertFalse(states[first_path].polygons_dirty)
        self.assertFalse(states[second_path].polygons_dirty)

    def test_work_simulation_toggle_stops_running_simulation(self) -> None:
        path = "frame_1.png"
        polygon = _rectangle_polygon(2, 2, 12, 12)
        state = ImageProcessingState(
            image_path=path,
            source_image=np.zeros((24, 24), dtype=np.uint8),
            polygons=[polygon.clone()],
            reference_polygons=[polygon.clone()],
            loaded_cif_path="frame_1.cif",
            polygons_dirty=False,
        )
        self.widget._workspace.replace_image_selection([path], is_supported_image=lambda _path: True)
        self.widget._workspace.set_cif_index({"frame_1": "frame_1.cif"})
        item = QListWidgetItem(Path(path).stem)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.widget.image_list.addItem(item)
        self.widget.load_image = lambda _path: (
            setattr(self.widget._workspace, "_current_image_path", path),
            setattr(self.widget._workspace, "_current_state", state),
            self.widget.polygon_editor.set_polygons([polygon.clone()]),
        )  # type: ignore[method-assign]

        self.widget._start_work_simulation()
        self.assertTrue(self.widget._work_simulation_running)

        self.widget._toggle_work_simulation()

        self.assertFalse(self.widget._work_simulation_running)
        self.assertFalse(self.widget._work_simulation_timer.isActive())

    def test_additional_layer_plus_is_disabled_without_base_and_enabled_with_base(self) -> None:
        self.assertFalse(self.widget.add_extra_layers_button.isEnabled())
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "frame_001.png")
            cv2.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
            self.widget.load_images([image_path])
            self.assertTrue(self.widget.add_extra_layers_button.isEnabled())

    def test_additional_layer_loading_is_blocked_without_base_layer(self) -> None:
        with patch.object(widget_module.QMessageBox, "information") as info_mock:
            self.widget._load_extra_layers()
        info_mock.assert_called_once()
        self.assertIn("Сначала загрузите базовый слой", str(info_mock.call_args))

    def test_extra_layer_row_controls_have_tooltips_and_compact_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_path = os.path.join(directory, "base_1.png")
            cv2.imwrite(base_path, np.zeros((8, 8), dtype=np.uint8))
            self.widget.load_images([base_path])
            layer_dir = os.path.join(directory, "layer")
            os.makedirs(layer_dir, exist_ok=True)
            cv2.imwrite(os.path.join(layer_dir, "overlay_1.png"), np.zeros((8, 8), dtype=np.uint8))
            layer = self.widget._extra_layer_from_directory(layer_dir)
            self.assertIsNotNone(layer)
            layer["opacity"] = 0
            self.widget._extra_layers.append(layer)
            self.widget._refresh_extra_layers_list()

            row_item = self.widget.extra_layers_list.item(0)
            row_widget = self.widget.extra_layers_list.itemWidget(row_item)
            checkbox = row_widget.findChild(QCheckBox)
            spinboxes = row_widget.findChildren(QSpinBox)
            buttons = row_widget.findChildren(QPushButton)
            self.assertIsNotNone(checkbox)
            self.assertEqual(checkbox.text(), "")
            self.assertEqual(checkbox.toolTip(), "Показать/скрыть слой")
            self.assertEqual(len(spinboxes), 3)
            self.assertEqual(spinboxes[0].toolTip(), "Смещение слоя по X")
            self.assertEqual(spinboxes[1].toolTip(), "Смещение слоя по Y")
            self.assertEqual(spinboxes[2].toolTip(), "Прозрачность слоя")

            self.assertEqual(spinboxes[2].value(), 0)
            self.assertLessEqual(spinboxes[0].width(), 48)
            self.assertLessEqual(spinboxes[1].width(), 48)
            self.assertLessEqual(spinboxes[2].width(), 46)
            self.assertEqual(len(buttons), 1)
            self.assertEqual(buttons[0].text(), "X")
            self.assertEqual(buttons[0].width(), 28)

    def test_reorder_extra_layers_updates_render_order_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_paths = []
            for index in (1, 2):
                path = os.path.join(directory, f"base_{index}.png")
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                base_paths.append(path)
            self.widget.load_images(base_paths)

            first_dir = os.path.join(directory, "layer_a")
            second_dir = os.path.join(directory, "layer_b")
            os.makedirs(first_dir, exist_ok=True)
            os.makedirs(second_dir, exist_ok=True)
            cv2.imwrite(os.path.join(first_dir, "a_1.png"), np.zeros((8, 8), dtype=np.uint8))
            cv2.imwrite(os.path.join(second_dir, "b_1.png"), np.zeros((8, 8), dtype=np.uint8))
            first = self.widget._extra_layer_from_directory(first_dir)
            second = self.widget._extra_layer_from_directory(second_dir)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.widget._extra_layers = [first, second]
            self.widget._refresh_extra_layers_list()

            moved = self.widget.extra_layers_list.takeItem(0)
            self.widget.extra_layers_list.insertItem(1, moved)
            self.widget._on_extra_layers_rows_moved()
            self.assertEqual(self.widget._extra_layers[0]["name"], second["name"])

    def test_manual_tool_postprocess_settings_are_exposed_in_help_menu(self) -> None:
        menu = QMenu()
        self.widget.attach_help_menu(menu)

        action = next(
            (action for action in menu.actions() if action.objectName() == "manualToolPostprocessAction"), None
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.text(), "Постобработка ручных инструментов")

    def test_view_sync_does_not_postprocess_untouched_vectors_or_mark_dirty(self) -> None:
        tiny = _rectangle_polygon(4, 4, 5, 5)
        state = ImageProcessingState(
            image_path="frame_1.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[tiny.clone()],
            reference_polygons=[tiny.clone()],
        )
        self.widget._workspace._current_image_path = "frame_1.png"
        self.widget._workspace._current_state = state
        self.widget._workspace._state_cache = {"frame_1.png": state}

        self.widget._sync_current_state_views()

        self.assertFalse(self.widget._workspace.current_image_has_changes())
        self.assertEqual(len(self.widget.polygon_editor.get_polygons()), 1)

    def test_recognition_view_sync_preserves_editor_scroll(self) -> None:
        state = ImageProcessingState(
            image_path="frame_1.png",
            source_image=np.zeros((120, 160), dtype=np.uint8),
            preprocessed_image=np.full((120, 160), 180, dtype=np.uint8),
            pipeline_config={"steps": []},
            polygons=[],
        )
        self.widget._workspace._current_image_path = "frame_1.png"
        self.widget._workspace._current_state = state
        self.widget.polygon_editor.set_image(state.source_image, preserve_view=True)
        self.widget.polygon_editor.scale(3.0, 3.0)
        self.widget.polygon_editor.horizontalScrollBar().setValue(42)
        self.widget.polygon_editor.verticalScrollBar().setValue(17)

        with patch.object(self.widget, "_center_editor_on_current_main_image") as center_mock:
            self.widget._sync_current_state_views(preserve_view=True, sync_neighbors=False)
            self._app.processEvents()

        center_mock.assert_not_called()

    def test_center_editor_on_main_image_skips_when_frame_mostly_visible(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((80, 80), dtype=np.uint8))
        self.widget.polygon_editor.fit_to_view()
        with patch.object(self.widget.polygon_editor, "center_main_image") as center_mock:
            self.widget._center_editor_on_current_main_image(force=False)
        center_mock.assert_not_called()

        self.widget.neighbor_max_grid_spin.setValue(7)
        self.widget.polygon_editor.resetTransform()
        self.widget.polygon_editor.scale(1.0, 1.0)

        self.assertEqual(self.widget._neighbor_grid_size(), 7)

        self.widget.polygon_editor.resetTransform()
        self.widget.polygon_editor.scale(0.2, 0.2)
        self.widget.neighbor_max_grid_spin.setValue(5)

        self.assertEqual(self.widget._neighbor_grid_size(), 5)

    def test_frame_matrix_load_suppresses_deferred_editor_centering(self) -> None:
        path = str(Path(r"d:\frames\frame_002.png"))
        self.widget._workspace._current_image_path = path
        self.widget._preserve_editor_view_position_path = path

        with patch.object(self.widget.polygon_editor, "center_main_image") as center_mock:
            self.widget._center_editor_on_current_main_image(force=True)

        center_mock.assert_not_called()

    def test_neighbor_frame_border_is_hidden_when_neighbors_are_disabled(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((24, 24), dtype=np.uint8))

        self.widget.show_neighbor_frames_checkbox.setChecked(False)
        self.widget._sync_neighbor_frames()

        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_neighbor_request_clears_existing_neighbors_when_disabled(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((24, 24), dtype=np.uint8))
        self.widget.polygon_editor.set_neighbor_frames(
            [(1, 0, np.zeros((24, 24), dtype=np.uint8), "right.png")],
            0.5,
            show_main_frame=True,
        )
        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 1)

        self.widget.show_neighbor_frames_checkbox.setChecked(False)
        self.widget._request_neighbor_frame_sync()

        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 0)
        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_stale_neighbor_apply_does_not_redraw_when_disabled(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((24, 24), dtype=np.uint8))
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_image_cache[str(Path(paths[5]))] = np.zeros((24, 24), dtype=np.uint8)
        self.widget.show_neighbor_frames_checkbox.setChecked(False)

        self.widget._apply_cached_neighbor_frames()

        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 0)
        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_neighbor_frame_overlap_moves_tiles_closer(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        frames = [
            (-1, 0, np.zeros((12, 12), dtype=np.uint8), "left.png"),
            (1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png"),
        ]

        self.widget.polygon_editor.set_neighbor_frames(frames, 0.5, overlap_pixels=3, show_main_frame=True)

        positions = sorted(
            round(item.pos().x()) for item in self.widget.polygon_editor._editor_scene._neighbor_frame_items
        )
        self.assertEqual(positions, [-9, 9])

    def test_display_settings_are_saved_when_changed(self) -> None:
        saved_payloads: list[dict[str, object]] = []

        class _Store:
            def load(self) -> dict[str, object]:
                return {}

            def save(self, payload: dict[str, object]) -> None:
                saved_payloads.append(dict(payload))

        self.widget._display_settings_store = _Store()  # type: ignore[assignment]
        self.widget.neighbor_overlap_spin.setValue(5)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget.show_neighbor_vectors_checkbox.setChecked(True)

        self.assertTrue(saved_payloads)
        self.assertEqual(saved_payloads[-1]["neighbor_overlap_pixels"], 5)
        self.assertTrue(saved_payloads[-1]["show_neighbor_frames"])
        self.assertTrue(saved_payloads[-1]["show_neighbor_vectors"])


class PolygonEditorViewMiddleClickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.view = PolygonEditorView()
        self.view.resize(320, 320)
        self.view.set_image(np.zeros((100, 100), dtype=np.uint8))
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.SELECT)
        self.view.show()
        self._app.processEvents()

    def test_neighbor_frame_vectors_are_rendered_over_neighbor_image(self) -> None:
        neighbor = QImage(25, 25, QImage.Format.Format_RGB32)
        neighbor.fill(0)
        polygon = _rectangle_polygon(10, 10, 40, 40)

        self.view.set_neighbor_frames([(1, 0, neighbor, "neighbor.jpg", [polygon], (100, 100))], 0.5)

        scene = self.view._editor_scene
        vector_items = [item for item in scene._neighbor_frame_items if isinstance(item, QGraphicsPathItem)]
        self.assertEqual(len(vector_items), 1)
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(20, 20)), None)
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(120, 20)), "neighbor.jpg")

    def test_neighbor_frame_path_at_overlap_prefers_neighbor_without_main_vectors(self) -> None:
        neighbor = QImage(100, 100, QImage.Format.Format_RGB32)
        neighbor.fill(0)

        self.view.set_polygons([])
        self.view.set_neighbor_frames([(1, 0, neighbor, "neighbor.jpg", [], (100, 100))], 0.5, overlap_pixels=20)

        scene = self.view._editor_scene
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(90, 50)), "neighbor.jpg")
        self.assertIsNone(scene.neighbor_frame_path_at(QPointF(40, 50)))

    def test_neighbor_frame_navigation_requires_double_click(self) -> None:
        neighbor = QImage(25, 25, QImage.Format.Format_RGB32)
        neighbor.fill(0)
        self.view.set_neighbor_frames(
            [(1, 0, neighbor, "neighbor.jpg", [], (100, 100))],
            0.5,
        )
        activated: list[str] = []
        self.view.neighborFrameActivated.connect(activated.append)
        position = self.view.mapFromScene(QPointF(120.0, 20.0))

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            position,
        )
        self._app.processEvents()
        self.assertEqual(activated, [])

        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            position,
        )
        self._app.processEvents()
        self.assertEqual(activated, ["neighbor.jpg"])

    def test_contact_tool_click_on_existing_via_requests_diagnostics(self) -> None:
        via = _rectangle_polygon(40, 40, 60, 60)
        via.category = "via"
        via.shape_hint = "box"
        requested: list[PolygonData] = []
        self.view.set_polygons([via])
        self.view.set_contact_recognition_mode(True)
        self.view.set_via_debug_inspection_enabled(True)
        self.view.set_tool(EditorTool.ADD_VIA)
        self.view.viaDebugRequested.connect(requested.append)

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(50.0, 50.0)),
        )
        self._app.processEvents()

        self.assertEqual([polygon.id for polygon in requested], [1])
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_copy_and_paste_emit_profile_boundaries_and_counts(self) -> None:
        self.view._editor_scene.select_polygon(1)
        copy_started: list[None] = []
        copy_finished: list[int] = []
        paste_started: list[int] = []
        paste_finished: list[int] = []
        self.view.contactCopyStarted.connect(lambda: copy_started.append(None))
        self.view.contactCopyFinished.connect(copy_finished.append)
        self.view.contactPasteStarted.connect(paste_started.append)
        self.view.contactPasteFinished.connect(paste_finished.append)

        self.view.copy_selected()
        self.view.start_paste_mode()
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(15.0, 15.0)),
        )
        self._app.processEvents()

        self.assertEqual(len(copy_started), 1)
        self.assertEqual(copy_finished, [1])
        self.assertEqual(paste_started, [1])
        self.assertEqual(paste_finished, [1])
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_paste_preview_reuses_items_while_pointer_moves(self) -> None:
        self.view._editor_scene.select_polygon(1)
        self.view.copy_selected()
        self.view.start_paste_mode()
        initial_items = list(self.view._paste_preview_items)

        self.view._update_paste_preview(QPointF(30.0, 30.0))
        self.view._update_paste_preview(QPointF(60.0, 70.0))

        self.assertEqual(len(initial_items), 1)
        self.assertEqual(self.view._paste_preview_items, initial_items)
        self.assertEqual(
            initial_items[0].pos(),
            QPointF(
                60.0 - self.view._clipboard_anchor.x(),
                70.0 - self.view._clipboard_anchor.y(),
            ),
        )

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._app.processEvents()

    def test_middle_button_temporarily_hides_vectors_while_panning(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        origin = self.view.mapFromScene(QPointF(50.0, 50.0))
        h_before = self.view.horizontalScrollBar().value()

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            origin,
        )
        self._app.processEvents()
        self.assertFalse(self.view._editor_scene.polygon_overlays_visible())

        QTest.mouseMove(self.view.viewport(), origin + QPoint(30, -12), delay=10)
        self._app.processEvents()

        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            origin + QPoint(30, -12),
        )
        self._app.processEvents()

        self.assertTrue(self.view._editor_scene.polygon_overlays_visible())
        self.assertEqual(len(self.view.get_polygons()), 1)
        self.assertEqual(self.view.current_tool, EditorTool.ADD_POLYGON)
        self.assertLessEqual(self.view.horizontalScrollBar().value(), h_before - 25)

    def test_f_key_fits_main_image_to_view(self) -> None:
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        zoom_before = self.view.zoom_factor()

        QTest.keyClick(self.view, Qt.Key.Key_F)
        self._app.processEvents()

        self.assertLess(self.view.zoom_factor(), zoom_before)
        self.assertGreater(self.view.main_image_visible_fraction(), 0.99)

    def test_space_hold_hides_vectors_without_mutating_polygon_data(self) -> None:
        QTest.mouseClick(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(40, 40)
        )
        before = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        QTest.keyPress(self.view, Qt.Key.Key_Space)

        self._app.processEvents()
        self.assertFalse(self.view._editor_scene.polygon_overlays_visible())
        after_press = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        self.assertIsNotNone(self.view._editor_scene.selected_polygon_id())
        self.assertEqual(before, after_press)

        QTest.keyRelease(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(self.view._editor_scene.polygon_overlays_visible())
        after_release = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        self.assertEqual(after_release, before)

    def test_space_hold_hides_recognition_mask_overlay(self) -> None:
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[..., 1] = 200
        self.view.set_gradient_overlay(overlay, opacity=0.6)
        overlay_item = self.view._editor_scene._gradient_overlay_item
        self.assertTrue(overlay_item.isVisible())

        QTest.keyPress(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertFalse(overlay_item.isVisible())
        self.assertFalse(overlay_item.pixmap().isNull())

        QTest.keyRelease(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(overlay_item.isVisible())

    def test_trace_pen_commits_polygonal_chain_on_enter(self) -> None:
        self.view.set_tool(EditorTool.TRACE_PEN)
        recorded: list[tuple[list[tuple[float, float]], float, bool]] = []

        def _record_trace(points: list[tuple[float, float]], width: float, erase: bool = False) -> bool:
            recorded.append((points, width, erase))
            self.view._editor_scene.cancel_pending_polygon()
            return True

        self.view._editor_scene.add_trace_stroke = _record_trace  # type: ignore[method-assign]

        for point in (QPointF(10.0, 10.0), QPointF(80.0, 10.0), QPointF(80.0, 70.0)):
            QTest.mouseClick(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                self.view.mapFromScene(point),
            )
        self._app.processEvents()

        self.assertEqual(len(self.view._editor_scene.pending_points_snapshot()), 3)

        QTest.keyClick(self.view, Qt.Key.Key_Return)
        self._app.processEvents()

        self.assertEqual(len(recorded), 1)
        points, width, erase = recorded[0]
        self.assertEqual(points, [(10.0, 10.0), (80.0, 10.0), (80.0, 70.0)])
        self.assertEqual(width, self.view._trace_width)
        self.assertFalse(erase)

    def test_ctrl_wheel_keeps_scene_point_under_cursor_stable(self) -> None:
        self.view.fit_to_view()
        self._app.processEvents()
        pos = QPoint(90, 80)
        view_pos = self.view._viewport_to_view_point(pos)
        scene_before = self.view.mapToScene(view_pos)
        event = QWheelEvent(
            QPointF(pos),
            QPointF(pos),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        for _ in range(5):
            self._app.sendEvent(self.view.viewport(), event)
            self._app.processEvents()
        QTest.qWait(120)
        self._app.processEvents()
        scene_after = self.view.mapToScene(view_pos)
        self.assertAlmostEqual(scene_after.x(), scene_before.x(), delta=0.5)
        self.assertAlmostEqual(scene_after.y(), scene_before.y(), delta=0.5)

    def test_ruler_tool_reports_measurement_without_changing_polygons(self) -> None:
        measurements: list[str] = []
        self.view.rulerMeasurementChanged.connect(measurements.append)
        self.view.set_tool(EditorTool.RULER)

        start_pos = self.view.mapFromScene(QPointF(20.0, 20.0))
        end_pos = self.view.mapFromScene(QPointF(80.0, 60.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start_pos,
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end_pos,
        )
        self._app.processEvents()

        self.assertTrue(measurements)
        self.assertIn("L=", measurements[-1])
        self.assertFalse(self.view._editor_scene._measurement_item.path().isEmpty())
        self.assertTrue(self.view._editor_scene._measurement_label_item.isVisible())
        self.assertIn("L=", self.view._editor_scene._measurement_label_item.text())
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_ruler_shift_snaps_measurement_to_45_degree_step(self) -> None:
        self.view.set_tool(EditorTool.RULER)
        start = QPointF(20.0, 20.0)
        target = QPointF(80.0, 50.0)

        snapped = self.view._ruler_target(start, target, Qt.KeyboardModifier.ShiftModifier)

        dx = snapped.x() - start.x()
        dy = snapped.y() - start.y()
        self.assertAlmostEqual(abs(dx), abs(dy), delta=1e-6)

    def test_right_button_brush_erases_existing_polygon_area(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        initial_area = _net_outline_area(self.view.get_polygons())
        start_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        end_pos = self.view.mapFromScene(QPointF(50.0, 80.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            start_pos,
        )
        QTest.mouseMove(self.view.viewport(), end_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            end_pos,
        )
        self._app.processEvents()

        final_area = _net_outline_area(self.view.get_polygons())
        self.assertLess(final_area, initial_area)

    def test_right_button_rectangle_polygon_erases_existing_polygon_area(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)
        initial_area = _net_outline_area(self.view.get_polygons())
        start_pos = self.view.mapFromScene(QPointF(42.0, 20.0))
        end_pos = self.view.mapFromScene(QPointF(58.0, 80.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            start_pos,
        )
        QTest.mouseMove(self.view.viewport(), end_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            end_pos,
        )
        self._app.processEvents()

        final_area = _net_outline_area(self.view.get_polygons())
        self.assertLess(final_area, initial_area)

    def test_brush_drag_skips_conductor_hover_sync_on_mouse_move(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        calls: list[QPointF] = []
        original_sync = self.view._editor_scene.sync_conductor_hover_highlight
        self.view._editor_scene.sync_conductor_hover_highlight = lambda pos: calls.append(pos)  # type: ignore[method-assign]
        try:
            start_pos = self.view.mapFromScene(QPointF(25.0, 25.0))
            move_pos = self.view.mapFromScene(QPointF(75.0, 75.0))
            QTest.mousePress(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                start_pos,
            )
            QTest.mouseMove(self.view.viewport(), move_pos)
            QTest.mouseRelease(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                move_pos,
            )
            self._app.processEvents()
        finally:
            self.view._editor_scene.sync_conductor_hover_highlight = original_sync  # type: ignore[method-assign]
        self.assertEqual(calls, [])

    def test_antialias_tool_hover_shows_vertices_and_click_simplifies_polygon(self) -> None:
        self.view.set_polygons([_oversampled_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.ANTIALIAS)
        self.view.set_antialias_grade(1)
        hover_pos = self.view.mapFromScene(QPointF(50.0, 50.0))

        QTest.mouseMove(self.view.viewport(), hover_pos)
        self._app.processEvents()

        item = self.view._editor_scene._polygon_items[1]
        self.assertGreater(len(item._handles), 0)
        before_count = len(self.view.get_polygons()[0].points)

        QTest.mouseClick(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, hover_pos)
        self._app.processEvents()

        after = self.view.get_polygons()[0]
        self.assertLess(len(after.points), before_count)
        self.assertEqual(self.view.current_tool, EditorTool.ANTIALIAS)

    def test_antialias_tool_drag_simplifies_polygons_in_area(self) -> None:
        first = _oversampled_rectangle_polygon(20, 20, 40, 40)
        second = _oversampled_rectangle_polygon(60, 60, 80, 80)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view.set_tool(EditorTool.ANTIALIAS)
        self.view.set_antialias_grade(1)
        before_counts = {polygon.id: len(polygon.points) for polygon in self.view.get_polygons()}

        start_pos = self.view.mapFromScene(QPointF(15.0, 15.0))
        end_pos = self.view.mapFromScene(QPointF(85.0, 85.0))
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_pos)
        QTest.mouseMove(self.view.viewport(), end_pos)
        QTest.mouseRelease(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end_pos)
        self._app.processEvents()

        after_counts = {polygon.id: len(polygon.points) for polygon in self.view.get_polygons()}
        self.assertLess(after_counts[1], before_counts[1])
        self.assertLess(after_counts[2], before_counts[2])
        self.assertEqual(self.view.current_tool, EditorTool.ANTIALIAS)

    def test_noop_brush_erase_does_not_create_undo_action(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        undo_before = self.view.undo_stack.count()

        changed = self.view._editor_scene.add_brush_stroke([(150.0, 150.0), (180.0, 180.0)], thickness=12.0, erase=True)

        self.assertFalse(changed)
        self.assertEqual(self.view.undo_stack.count(), undo_before)

    def test_brush_records_movement_of_at_least_one_pixel_as_segment(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)

        self.view._editor_scene.append_brush_vertex(QPointF(40.0, 40.0), 12.0)
        self.view._editor_scene.append_brush_vertex(QPointF(41.2, 40.0), 12.0)

        points = self.view._editor_scene.pending_points_snapshot()
        self.assertGreaterEqual(len(points), 2)

    def test_brush_drops_vertices_closer_than_one_pixel(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)
        self.view._append_brush_point(QPointF(40.0, 40.0))
        self.view._append_brush_point(QPointF(40.8, 40.0))

        points = self.view._editor_scene.pending_points_snapshot()
        self.assertEqual(len(points), 1)

    def test_brush_vertex_spacing_rule_is_one_image_pixel(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        self._app.processEvents()
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)
        self.view._append_brush_point(QPointF(40.0, 40.0))
        self.view._append_brush_point(QPointF(40.8, 40.0))
        points_after_small = self.view._editor_scene.pending_points_snapshot()
        self.assertEqual(len(points_after_small), 1)

        self.view._append_brush_point(QPointF(41.2, 40.0))
        points_after_large = self.view._editor_scene.pending_points_snapshot()
        self.assertGreaterEqual(len(points_after_large), 2)

    def test_middle_pan_during_brush_does_not_shift_brush_position(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.BRUSH)
        self.view.set_brush_thickness(12.0)
        start = self.view.mapFromScene(QPointF(30.0, 30.0))

        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.view.viewport(), start + QPoint(90, 0), delay=10)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(90, 0),
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(90, 0),
        )
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        # Stroke remains around the original press location; pan does not drag the brush.
        left, top, width, height = polygons[0].bbox
        self.assertLessEqual(left, 31)
        self.assertLessEqual(top, 31)
        self.assertGreaterEqual(left + width, 29)
        self.assertGreaterEqual(top + height, 29)

    def test_middle_pan_during_brush_at_image_edge_keeps_brush_anchor(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.BRUSH)
        self.view.set_brush_thickness(12.0)
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        self._app.processEvents()

        edge_scene = QPointF(95.0, 50.0)
        start = self.view.mapFromScene(edge_scene)

        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start)
        # Push pan strongly so viewport hits scroll limits at image boundary.
        for _ in range(3):
            QTest.mouseMove(self.view.viewport(), start + QPoint(-600, 0), delay=8)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(-600, 0),
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(-600, 0),
        )
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        left, top, width, height = polygons[0].bbox
        center_x = left + width / 2.0
        center_y = top + height / 2.0
        self.assertAlmostEqual(center_x, edge_scene.x(), delta=2.0)
        self.assertAlmostEqual(center_y, edge_scene.y(), delta=2.0)

    def test_closed_brush_contour_preserves_empty_center(self) -> None:
        self.view.set_polygons([])
        points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]

        changed = self.view._editor_scene.add_brush_stroke(points, thickness=10.0)
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertTrue(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(50.0, 50.0)))

    def test_closed_brush_contour_fills_center_below_manual_min_hole_area(self) -> None:
        self.view.set_polygons([])
        self.view._editor_scene.set_vector_geometry_settings(VectorGeometrySettings(min_hole_area_to_remove_px2=150.0))
        points = [
            (30.0, 30.0),
            (50.0, 30.0),
            (50.0, 50.0),
            (30.0, 50.0),
            (30.0, 30.0),
        ]

        changed = self.view._editor_scene.add_brush_stroke(points, thickness=10.0)
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertFalse(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertTrue(outer_item.contains(QPointF(40.0, 40.0)))

    def test_brush_erase_fills_hole_below_manual_min_hole_area(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=11.0)
        )

        changed = self.view._editor_scene.add_brush_stroke(
            [(44.0, 44.0), (46.0, 44.0), (46.0, 46.0), (44.0, 46.0), (44.0, 44.0)],
            thickness=1.0,
            erase=True,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        self.assertFalse(any(polygon.is_hole for polygon in self.view.get_polygons()))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertTrue(outer_item.contains(QPointF(45.0, 45.0)))

    def test_brush_erase_keeps_hole_above_manual_min_hole_area(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=11.0)
        )

        changed = self.view._editor_scene.add_brush_stroke(
            [(40.0, 40.0), (50.0, 40.0), (50.0, 50.0), (40.0, 50.0), (40.0, 40.0)],
            thickness=1.0,
            erase=True,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        self.assertTrue(any(polygon.is_hole for polygon in self.view.get_polygons()))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(45.0, 45.0)))

    def test_small_rectangle_fully_inside_existing_contour_is_not_drawn(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(VectorGeometrySettings(min_hole_area_to_remove_px2=11.0))

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(46.0, 46.0))
        self._app.processEvents()

        self.assertFalse(added)
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_small_rectangle_fully_inside_existing_cutout_is_not_drawn(self) -> None:
        self.view.set_polygons([])
        self.view._editor_scene.set_vector_geometry_settings(VectorGeometrySettings(min_hole_area_to_remove_px2=11.0))
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        before = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(46.0, 46.0))
        self._app.processEvents()

        self.assertFalse(added)
        after = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]
        self.assertEqual(after, before)

    def test_can_draw_polygon_inside_existing_cutout(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(56.0, 56.0))
        self._app.processEvents()

        self.assertTrue(added)
        polygons = self.view.get_polygons()
        self.assertEqual(sum(1 for polygon in polygons if polygon.is_hole), 1)
        self.assertGreaterEqual(sum(1 for polygon in polygons if not polygon.is_hole), 2)
        self.assertTrue(
            any(not polygon.is_hole and polygon.bbox[0] >= 44 and polygon.bbox[1] >= 44 for polygon in polygons)
        )

    def test_brush_crossing_inner_contour_updates_hole_geometry(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        before_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]

        changed = self.view._editor_scene.add_brush_stroke(
            [(28.0, 50.0), (72.0, 50.0)],
            thickness=12.0,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        after_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertNotEqual(
            [(polygon.bbox, polygon.points) for polygon in before_holes],
            [(polygon.bbox, polygon.points) for polygon in after_holes],
        )

    def test_brush_editing_outer_ring_keeps_inner_object_if_not_touched(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(56.0, 56.0))
        before_polygons = self.view.get_polygons()

        changed = self.view._editor_scene.add_brush_stroke(
            [(20.0, 28.0), (80.0, 28.0)],
            thickness=8.0,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        after_polygons = self.view.get_polygons()
        self.assertGreaterEqual(sum(1 for polygon in after_polygons if not polygon.is_hole), 2)
        self.assertTrue(
            any(not polygon.is_hole and polygon.bbox[0] >= 44 and polygon.bbox[1] >= 44 for polygon in after_polygons)
        )
        self.assertGreaterEqual(len(after_polygons), len(before_polygons))

    def test_delete_vertices_area_affects_unselected_polygons_too(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)

        deleted = self.view._editor_scene.delete_vertices_in_rect(QRectF(QPointF(56.0, 6.0), QPointF(66.0, 16.0)))

        self.assertEqual(deleted, 1)
        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 3)
        self.assertEqual(len(polygons[1].points), 4)

    def test_delete_vertices_area_preview_highlights_all_touched_polygons(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])

        self.view._editor_scene.preview_delete_vertices_in_rect(QPointF(5.0, 5.0), QPointF(65.0, 15.0))

        self.assertEqual(self.view._editor_scene._delete_area_highlight_ids, {1, 2})
        self.view._editor_scene.clear_preview_rect()
        self.assertEqual(self.view._editor_scene._delete_area_highlight_ids, set())

    def test_add_vertex_click_on_unselected_polygon_selects_and_edits_it(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.ADD_VERTEX)

        select_pos = self.view.mapFromScene(QPointF(75.0, 25.0))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            select_pos,
        )
        self._app.processEvents()

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 4)
        self.assertEqual(len(polygons[1].points), 4)
        self.assertEqual(self.view._editor_scene.selected_polygon_id(), 2)

        add_pos = self.view.mapFromScene(QPointF(75.0, 10.0))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            add_pos,
        )
        self._app.processEvents()

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 5)
        self.assertEqual(len(polygons[1].points), 4)
        self.assertEqual(self.view._editor_scene.selected_polygon_id(), 2)

    def test_move_vertex_click_inside_selected_polygon_moves_nearest_vertex(self) -> None:
        poly = _rectangle_polygon(20, 20, 80, 80)
        self.view.set_polygons([poly])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        before = self.view.get_polygons()[0].points

        press_pos = self.view.mapFromScene(QPointF(50.0, 50.0))
        release_pos = self.view.mapFromScene(QPointF(62.0, 58.0))
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        after = self.view.get_polygons()[0].points
        self.assertNotEqual(before, after)

    def test_polygon_tool_right_click_deletes_existing_polygon(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        before_undo_count = self.view.undo_stack.count()

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(50.0, 50.0)),
        )
        self._app.processEvents()

        self.assertEqual(self.view.get_polygons(), [])
        self.assertGreater(self.view.undo_stack.count(), before_undo_count)

    def test_polygon_tool_right_click_while_drawing_keeps_erase_drawing_behavior(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(5.0, 5.0)),
        )
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(10.0, 5.0)),
        )
        self._app.processEvents()

        self.assertEqual(len(self.view.get_polygons()), 1)
        self.assertEqual(len(self.view._editor_scene.pending_points_snapshot()), 2)

    def test_finished_point_polygon_is_cropped_to_image_bounds(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        for point in (
            QPointF(-20.0, -20.0),
            QPointF(60.0, -20.0),
            QPointF(60.0, 60.0),
            QPointF(-20.0, 60.0),
        ):
            self.view._editor_scene.append_pending_point(point)

        self.view._editor_scene.finish_pending_polygon()
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        self.assertTrue(_all_points_within(polygons, 0.0, 0.0, 100.0, 100.0))

    def test_finished_rectangle_polygon_is_cropped_to_image_bounds(self) -> None:
        self.view.set_polygons([])

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(-20.0, -20.0), QPointF(60.0, 60.0))
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(added)
        self.assertTrue(polygons)
        self.assertTrue(_all_points_within(polygons, 0.0, 0.0, 100.0, 100.0))

    def test_delete_key_still_deletes_selected_polygon(self) -> None:
        self.view._editor_scene.select_polygon(1)

        QTest.keyClick(self.view.viewport(), Qt.Key.Key_Delete)
        self._app.processEvents()

        self.assertEqual(self.view.get_polygons(), [])

    def test_contact_delete_key_emits_separate_profiling_boundaries(self) -> None:
        via = _rectangle_polygon(20, 20, 40, 40)
        via.category = "via"
        via.shape_hint = "box"
        self.view.set_polygons([via])
        self.view._editor_scene.select_polygon(1)
        started: list[int] = []
        finished: list[int] = []
        self.view.contactDeletionStarted.connect(started.append)
        self.view.contactDeletionFinished.connect(finished.append)

        QTest.keyClick(self.view.viewport(), Qt.Key.Key_Delete)
        self._app.processEvents()

        self.assertEqual(started, [1])
        self.assertEqual(finished, [1])
        self.assertEqual(self.view.get_polygons(), [])

    def test_contact_undo_and_redo_emit_profiling_boundaries(self) -> None:
        via = _rectangle_polygon(20, 20, 40, 40)
        via.category = "via"
        via.shape_hint = "box"
        self.view.set_polygons([via])
        self.view._editor_scene.select_polygon(1)
        self.assertTrue(self.view._editor_scene.delete_polygon())
        undo_started: list[bool] = []
        undo_finished: list[tuple[bool, int]] = []
        redo_started: list[bool] = []
        redo_finished: list[tuple[bool, int]] = []
        self.view.contactUndoStarted.connect(lambda: undo_started.append(True))
        self.view.contactUndoFinished.connect(
            lambda applied, count: undo_finished.append((applied, count))
        )
        self.view.contactRedoStarted.connect(lambda: redo_started.append(True))
        self.view.contactRedoFinished.connect(
            lambda applied, count: redo_finished.append((applied, count))
        )

        self.view.undo()
        self.view.redo()

        self.assertEqual(undo_started, [True])
        self.assertEqual(undo_finished, [(True, 1)])
        self.assertEqual(redo_started, [True])
        self.assertEqual(redo_finished, [(True, 1)])

    def test_area_selection_emits_selected_contact_count(self) -> None:
        first = _rectangle_polygon(20, 20, 35, 35)
        first.category = "via"
        first.shape_hint = "box"
        second = _rectangle_polygon(55, 55, 70, 70)
        second.id = 2
        second.category = "via"
        second.shape_hint = "box"
        self.view.set_polygons([first, second])
        started: list[bool] = []
        finished: list[int] = []
        self.view.contactMultiSelectionStarted.connect(lambda: started.append(True))
        self.view.contactMultiSelectionFinished.connect(finished.append)
        press_pos = self.view.mapFromScene(QPointF(5.0, 5.0))
        release_pos = self.view.mapFromScene(QPointF(85.0, 85.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        self.assertEqual(started, [True])
        self.assertEqual(finished, [2])

    def test_move_vertex_keeps_closed_duplicate_endpoint_together(self) -> None:
        points = [
            (20.0, 20.0),
            (80.0, 20.0),
            (80.0, 80.0),
            (20.0, 80.0),
            (20.0, 20.0),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        poly = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)
        self.view.set_polygons([poly])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)

        press_pos = self.view.mapFromScene(QPointF(20.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(30.0, 30.0))
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        after = self.view.get_polygons()[0].points
        self.assertAlmostEqual(after[0][0], 30.0, places=1)
        self.assertAlmostEqual(after[0][1], 30.0, places=1)
        self.assertEqual(after[-1], after[0])

    def test_move_vertex_uses_local_cleanup_without_global_merge(self) -> None:
        first = _rectangle_polygon(20, 20, 50, 50)
        second = _rectangle_polygon(55, 20, 85, 50)
        second.id = 2
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0)
        )
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        before_undo_count = self.view.undo_stack.count()

        press_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(65.0, 20.0))
        with patch("contour.application.vector_geometry_postprocess.merge_overlapping_root_families") as merge_mock:
            QTest.mousePress(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                press_pos,
            )
            QTest.mouseMove(self.view.viewport(), release_pos)
            QTest.mouseRelease(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                release_pos,
            )
        self._app.processEvents()

        roots = [polygon for polygon in self.view.get_polygons() if polygon.parent_id is None and not polygon.is_hole]
        self.assertEqual(len(roots), 2)
        self.assertEqual(self.view.undo_stack.count(), before_undo_count + 1)
        merge_mock.assert_not_called()

    def test_repeated_outer_edits_do_not_expand_untouched_inner_contour(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        initial_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(initial_holes), 1)

        for y_coord in (28.0, 26.0, 24.0):
            changed = self.view._editor_scene.add_brush_stroke(
                [(20.0, y_coord), (80.0, y_coord)],
                thickness=4.0,
            )
            self.assertTrue(changed)

        final_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(final_holes), 1)
        self.assertEqual(final_holes[0].bbox, initial_holes[0].bbox)
        self.assertEqual(final_holes[0].points, initial_holes[0].points)

    def test_repeated_inner_edge_edits_do_not_expand_hole_bbox(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (90.0, 30.0),
            (90.0, 90.0),
            (30.0, 90.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        initial_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(initial_holes), 1)
        initial_bbox = initial_holes[0].bbox

        for _index in range(4):
            changed = self.view._editor_scene.add_brush_stroke(
                [(20.0, 34.0), (100.0, 34.0)],
                thickness=2.0,
            )
            self.assertTrue(changed)

        final_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(final_holes), 1)
        self.assertGreaterEqual(final_holes[0].bbox[0], initial_bbox[0])
        self.assertGreaterEqual(final_holes[0].bbox[1], initial_bbox[1])
        self.assertLessEqual(final_holes[0].bbox[2], initial_bbox[2])
        self.assertLessEqual(final_holes[0].bbox[3], initial_bbox[3])

    def test_cutting_shape_with_hole_keeps_center_empty(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)

        changed = self.view._editor_scene.add_rectangle_polygon(
            QPointF(22.0, 38.0),
            QPointF(30.0, 62.0),
            erase=True,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertTrue(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(50.0, 50.0)))

    def test_box_and_via_items_display_as_ellipses(self) -> None:
        polygon = _rectangle_polygon(20, 20, 80, 60)
        polygon.category = "via"
        polygon.shape_hint = "box"

        item = EditablePolygonItem(polygon, DisplaySettings(show_vertices=True))

        self.assertGreater(item.path().elementCount(), len(polygon.points) + 1)
        self.assertEqual(len(item._handles), 0)

    def test_overlay_preview_draws_box_and_via_as_ellipse(self) -> None:
        polygon = _rectangle_polygon(10, 10, 30, 30)
        polygon.category = "via"
        polygon.shape_hint = "box"
        image = np.zeros((48, 48, 3), dtype=np.uint8)

        overlay = draw_polygon_overlay(
            image,
            [polygon],
            DisplaySettings(external_color="#00FF00", fill_opacity=1.0, show_vertices=False),
        )

        self.assertEqual(int(overlay[20, 20, 1]), 255)
        self.assertEqual(int(overlay[10, 10, 1]), 0)


class PolygonExtractionWidgetColorPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()
        color_image = np.zeros((32, 32, 3), dtype=np.uint8)
        color_image[10, 12] = (48, 32, 16)
        self.widget._workspace._current_image_path = "sample.png"
        self.widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=color_image,
        )
        self.widget._pipeline.steps = []
        self.widget._pipeline.steps.append(self.widget._pipeline.create_step("color_binarize"))
        self.widget._populate_pipeline_list()
        self.widget.pipeline_list.setCurrentRow(0)

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def test_clicking_image_adds_color_to_color_binarize_step(self) -> None:
        self.widget._set_color_pick_active(0)

        self.widget._on_editor_image_clicked(12.0, 10.0)

        entries = self.widget._pipeline.steps[0].parameters.get("selected_colors", [])
        self.assertEqual(entries, [{"rgb": [16, 32, 48], "enabled": True}])
        self.assertEqual(self.widget._color_pick_pipeline_row, 0)


class PolygonExtractionWidgetAutosaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def test_switching_frames_autosaves_loaded_cif_when_polygons_changed(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(4, 4, 20, 20)
        changed_polygon = _rectangle_polygon(4, 4, 24, 20)
        first_state = ImageProcessingState(
            image_path=first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed_polygon.clone()],
            loaded_cif_path="frame_1.cif",
            reference_polygons=[first_polygon.clone()],
        )
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            reference_polygons=[],
        )
        self.widget._workspace._state_cache = {
            first_path: first_state,
            second_path: second_state,
        }
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = first_state
        self.widget._viewed_image_paths.update({str(Path(first_path)), str(Path(second_path))})
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([changed_polygon.clone()])
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)
        self.widget._refresh_image_list_item_states()
        first_item = self.widget.image_list.item(0)
        second_item = self.widget.image_list.item(1)

        saved_calls: list[tuple[str, str, tuple[int, int], int]] = []
        original_save_polygons_vector = processing_mixin_module.save_polygons_vector
        original_load_image = self.widget.load_image
        try:
            self.widget.autosave_on_frame_transition_checkbox.setChecked(True)
            processing_mixin_module.save_polygons_vector = lambda path, image_path, polygons, image_size: (
                saved_calls.append((str(path), image_path, image_size, len(polygons)))
            )
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]

            self.widget._on_image_item_changed(second_item, first_item)
        finally:
            processing_mixin_module.save_polygons_vector = original_save_polygons_vector
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(saved_calls, [("frame_1.cif", first_path, (32, 32), 1)])
        self.assertEqual(first_item.background().color().name().lower(), "#1e4a35")
        self.assertEqual(second_item.background().color().name().lower(), "#3d4f66")

    def test_switching_frames_without_edits_does_not_prompt_or_save(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        polygon = _rectangle_polygon(4, 4, 20, 20)
        first_state = ImageProcessingState(
            image_path=first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[polygon.clone()],
            loaded_cif_path="frame_1.cif",
            reference_polygons=[polygon.clone()],
        )
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            reference_polygons=[],
        )
        self.widget._workspace._state_cache = {first_path: first_state, second_path: second_state}
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = first_state
        self.widget._viewed_image_paths.add(str(Path(first_path)))
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([polygon.clone()])
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(Path(path).stem)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)
        first_item = self.widget.image_list.item(0)
        second_item = self.widget.image_list.item(1)

        saved_calls: list[str] = []
        original_save_polygons_vector = processing_mixin_module.save_polygons_vector
        original_load_image = self.widget.load_image
        try:
            processing_mixin_module.save_polygons_vector = lambda *args, **kwargs: saved_calls.append("save")
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]
            with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            processing_mixin_module.save_polygons_vector = original_save_polygons_vector
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(saved_calls, [])
        self.widget._refresh_image_list_item_states()
        self.assertEqual(first_item.background().color().name().lower(), "#3d4f66")

    def test_fast_switch_before_editor_apply_does_not_mark_loaded_frame_dirty(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(4, 4, 20, 20)
        second_polygon = _rectangle_polygon(6, 6, 22, 22)
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[second_polygon.clone()],
            loaded_cif_path="frame_2.cif",
            reference_polygons=[second_polygon.clone()],
            polygons_dirty=False,
        )
        self.widget._workspace._state_cache = {second_path: second_state}
        self.widget._workspace._current_image_path = second_path
        self.widget._workspace._current_state = second_state
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([first_polygon.clone()])
        self.widget._editor_polygons_signature = self.widget._polygons_editor_signature(
            first_path,
            [first_polygon.clone()],
        )

        with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
            allowed = self.widget._try_leave_current_frame()

        self.assertTrue(allowed)
        self.assertFalse(self.widget._workspace.current_image_has_changes())
        self.assertEqual(second_state.polygons[0].points, second_polygon.points)

    def test_switching_frames_does_not_save_when_autosave_disabled_even_if_dialog_discards(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(4, 4, 20, 20)
        changed_polygon = _rectangle_polygon(4, 4, 24, 20)
        first_state = ImageProcessingState(
            image_path=first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed_polygon.clone()],
            loaded_cif_path="frame_1.cif",
            reference_polygons=[first_polygon.clone()],
        )
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            reference_polygons=[],
        )
        self.widget._workspace._state_cache = {
            first_path: first_state,
            second_path: second_state,
        }
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = first_state
        self.widget._viewed_image_paths.update({str(Path(first_path)), str(Path(second_path))})
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([changed_polygon.clone()])
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)
        self.widget._refresh_image_list_item_states()
        first_item = self.widget.image_list.item(0)
        second_item = self.widget.image_list.item(1)

        saved_calls: list[tuple[str, str, tuple[int, int], int]] = []
        original_save_polygons_vector = processing_mixin_module.save_polygons_vector
        original_load_image = self.widget.load_image
        try:
            self.widget.autosave_on_frame_transition_checkbox.setChecked(False)
            processing_mixin_module.save_polygons_vector = lambda path, image_path, polygons, image_size: (
                saved_calls.append((str(path), image_path, image_size, len(polygons)))
            )
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]

            with patch.object(
                widget_module.QMessageBox, "exec", return_value=widget_module.QMessageBox.StandardButton.Discard
            ):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            processing_mixin_module.save_polygons_vector = original_save_polygons_vector
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(saved_calls, [])

    def test_extraction_mode_switch_does_not_prompt_save_or_mark_viewed(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        changed_polygon = _rectangle_polygon(4, 4, 24, 20)
        first_state = ImageProcessingState(
            image_path=first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed_polygon.clone()],
            reference_polygons=[],
        )
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            reference_polygons=[],
        )
        self.widget._workspace._state_cache = {first_path: first_state, second_path: second_state}
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = first_state
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)
        first_item = self.widget.image_list.item(0)
        second_item = self.widget.image_list.item(1)
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("conductors"))

        saved_calls: list[str] = []
        original_save_polygons_vector = processing_mixin_module.save_polygons_vector
        original_load_image = self.widget.load_image
        try:
            processing_mixin_module.save_polygons_vector = lambda *args, **kwargs: saved_calls.append("save")
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]
            with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            processing_mixin_module.save_polygons_vector = original_save_polygons_vector
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(saved_calls, [])
        self.assertNotIn(first_path, self.widget._viewed_image_paths)

    def test_dataset_mode_exports_changed_frame_when_switching_frames(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(4, 4, 20, 20)
        changed_polygon = _rectangle_polygon(4, 4, 24, 20)
        first_state = ImageProcessingState(
            image_path=first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed_polygon.clone()],
            reference_polygons=[first_polygon.clone()],
        )
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            reference_polygons=[],
        )
        self.widget._workspace._state_cache = {
            first_path: first_state,
            second_path: second_state,
        }
        self.widget._workspace._current_image_path = first_path
        self.widget._workspace._current_state = first_state
        self.widget._viewed_image_paths.update({str(Path(first_path)), str(Path(second_path))})
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([changed_polygon.clone()])
        self.widget.dataset_dir_edit.setText("dataset")
        self.widget.dataset_mode_checkbox.setChecked(True)
        self.widget.autosave_on_frame_transition_checkbox.setChecked(True)
        self.widget.image_list.clear()
        for path in (first_path, second_path):
            item = QListWidgetItem(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.widget.image_list.addItem(item)
        first_item = self.widget.image_list.item(0)
        second_item = self.widget.image_list.item(1)

        from contour.application.services import dataset_exporter as dataset_exporter_module

        exported_calls: list[tuple[str, str, int]] = []
        original_export_dataset_frame = dataset_exporter_module.export_dataset_frame
        original_load_image = self.widget.load_image
        try:
            dataset_exporter_module.export_dataset_frame = (
                lambda dataset_directory, image_path, polygons, source_image: (
                    exported_calls.append((str(dataset_directory), image_path, len(polygons)))
                    or {"image": "dataset/images/frame_1.png", "cif": "dataset/cif/frame_1.cif"}
                )
            )
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]

            self.widget._on_image_item_changed(second_item, first_item)
        finally:
            dataset_exporter_module.export_dataset_frame = original_export_dataset_frame
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(exported_calls, [("dataset", first_path, 1)])


class PolygonExtractionWidgetBrushModeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def test_brush_mode_combo_exposes_all_brush_modes(self) -> None:
        modes = [
            str(self.widget.brush_mode_combo.itemData(index)) for index in range(self.widget.brush_mode_combo.count())
        ]
        self.assertEqual(self.widget.brush_mode_combo.count(), 2)
        self.assertEqual(modes, ["freeform", "angled"])

    def test_trace_pen_width_control_updates_editor(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.TRACE_PEN)
        self.widget.trace_width_spin.setValue(24)
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_editor._trace_width, 24.0)

    def test_contact_tool_uses_plain_width_and_height_labels(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.ADD_VIA)
        self._app.processEvents()

        self.assertEqual(self.widget.via_width_label.text(), "Ширина")
        self.assertEqual(self.widget.via_height_label.text(), "Высота")

        self.widget.via_width_spin.setValue(13)
        self.widget.via_height_spin.setValue(21)
        self._app.processEvents()
        self.assertEqual(self.widget.polygon_editor._via_width, 13.0)
        self.assertEqual(self.widget.polygon_editor._via_height, 21.0)

    def test_polygon_mode_indicator_is_hidden_but_mode_switch_stays_operational(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.widget.polygon_mode_combo.setCurrentIndex(
            self.widget.polygon_mode_combo.findData(PolygonCreateMode.RECTANGLE)
        )
        self._app.processEvents()

        self.assertFalse(self.widget.polygon_draw_mode_indicator.isVisible())
        self.assertEqual(self.widget.polygon_editor.effective_polygon_create_mode(), PolygonCreateMode.RECTANGLE)

    def test_space_hold_hides_vectors_after_selecting_drawing_tool_button(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((100, 100), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.widget.show()
        self._app.processEvents()

        QTest.mouseClick(self.widget._tool_buttons[EditorTool.BRUSH], Qt.MouseButton.LeftButton)
        self._app.processEvents()

        self.assertEqual(QApplication.focusWidget(), self.widget.polygon_editor)
        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

    def test_tool_mode_cycle_matches_shift_click_targets(self) -> None:
        self.widget._cycle_editor_tool_mode(EditorTool.ADD_POLYGON)
        self.widget._cycle_editor_tool_mode(EditorTool.BRUSH)
        self.widget._cycle_editor_tool_mode(EditorTool.DELETE_VERTEX)
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_mode_combo.currentData(), PolygonCreateMode.RECTANGLE)
        self.assertEqual(self.widget.brush_mode_combo.currentData(), BrushMode.ANGLED)
        self.assertEqual(self.widget.delete_vertex_mode_combo.currentData(), DeleteVertexMode.AREA)

    def test_shift_key_cycles_active_tool_mode_and_updates_combo(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        QTest.keyClick(self.widget.polygon_editor, Qt.Key.Key_Shift)
        QTest.keyClick(self.widget.polygon_editor, Qt.Key.Key_Shift)
        self._app.processEvents()

        self.assertEqual(self.widget.brush_mode_combo.currentData(), BrushMode.FREEFORM)

    def test_toolbar_hides_incompatible_tools_and_falls_back_to_select(self) -> None:
        via = _rectangle_polygon(10, 10, 20, 20)
        via.category = "via"
        via.shape_hint = "box"
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.widget.polygon_editor.set_polygons([via])
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_editor.current_tool, EditorTool.SELECT)
        self.assertFalse(self.widget._tool_buttons[EditorTool.ADD_VIA].isHidden())
        for tool in (
            EditorTool.ADD_POLYGON,
            EditorTool.BRUSH,
            EditorTool.TRACE_PEN,
            EditorTool.ADD_VERTEX,
            EditorTool.DELETE_VERTEX,
            EditorTool.MOVE_VERTEX,
            EditorTool.ANTIALIAS,
        ):
            self.assertTrue(self.widget._tool_buttons[tool].isHidden())
        self.widget.polygon_editor.set_tool(EditorTool.ADD_VIA)
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.assertEqual(self.widget.polygon_editor.current_tool, EditorTool.ADD_VIA)

        self.widget.polygon_editor.set_polygons([_rectangle_polygon(0, 0, 30, 30)])
        self._app.processEvents()
        self.assertTrue(self.widget._tool_buttons[EditorTool.ADD_VIA].isHidden())
        self.assertFalse(self.widget._tool_buttons[EditorTool.ADD_POLYGON].isHidden())


class PolygonEditorSceneBrushPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_brush_preview_keeps_configured_width_across_mouse_updates(self) -> None:
        scene = PolygonEditorScene()
        scene.start_pending_polygon(for_brush=True)
        scene.set_pending_path_width(40.0, cosmetic=False)
        scene.append_brush_vertex(QPointF(10.0, 10.0), 40.0)
        scene.update_pending_cursor(QPointF(110.0, 10.0))
        first_height = scene._pending_path_item.path().boundingRect().height()
        scene.update_pending_cursor(QPointF(120.0, 10.0))
        second_height = scene._pending_path_item.path().boundingRect().height()

        self.assertGreater(first_height, 35.0)
        self.assertGreater(second_height, 35.0)

    def test_trace_pen_adds_fixed_width_conductor_polygon(self) -> None:
        scene = PolygonEditorScene()

        changed = scene.add_trace_stroke([(10.0, 20.0), (110.0, 20.0)], width=16.0, erase=False)
        polygons = scene.get_polygons()

        self.assertTrue(changed)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "conductor")
        self.assertEqual(polygons[0].shape_hint, "trace_pen")
        self.assertGreater(polygons[0].area, 1400.0)

    def test_trace_pen_clips_new_polygon_to_image_bounds(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((40, 80), dtype=np.uint8))

        changed = scene.add_trace_stroke([(-20.0, 20.0), (100.0, 20.0)], width=16.0, erase=False)
        polygons = scene.get_polygons()

        self.assertTrue(changed)
        self.assertEqual(len(polygons), 1)
        for x_coord, y_coord in polygons[0].points:
            self.assertGreaterEqual(x_coord, 0)
            self.assertLessEqual(x_coord, 80)
            self.assertGreaterEqual(y_coord, 0)
            self.assertLessEqual(y_coord, 40)

    def test_via_placement_rejects_overlapping_existing_via(self) -> None:
        scene = PolygonEditorScene()

        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))
        self.assertFalse(scene.add_via_at(QPointF(24.0, 20.0), 10.0, 10.0))
        self.assertTrue(scene.add_via_at(QPointF(40.0, 20.0), 10.0, 10.0))

        self.assertEqual(len(scene.get_polygons()), 2)

    def test_via_placement_rejects_rect_outside_image_bounds(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((40, 40), dtype=np.uint8))

        self.assertFalse(scene.add_via_at(QPointF(2.0, 20.0), 10.0, 10.0))
        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))

        self.assertEqual(len(scene.get_polygons()), 1)

    def test_delete_via_at_deletes_only_vias(self) -> None:
        scene = PolygonEditorScene()
        conductor = _rectangle_polygon(0, 0, 20, 20)
        via = _rectangle_polygon(35, 35, 45, 45)
        via.id = 2
        via.category = "via"
        via.shape_hint = "box"
        scene.set_polygons([conductor, via])

        self.assertFalse(scene.delete_via_at(QPointF(10.0, 10.0)))
        self.assertEqual(len(scene.get_polygons()), 2)
        self.assertTrue(scene.delete_via_at(QPointF(40.0, 40.0)))
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_polygon_and_via_creation_are_mutually_exclusive(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))

        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))
        self.assertFalse(scene.add_rectangle_polygon(QPointF(40.0, 40.0), QPointF(60.0, 60.0)))

        scene.set_polygons([_rectangle_polygon(10, 10, 30, 30)])
        self.assertFalse(scene.add_via_at(QPointF(60.0, 60.0), 10.0, 10.0))

    def test_paste_cannot_create_mixed_content(self) -> None:
        scene = PolygonEditorScene()
        via = _rectangle_polygon(10, 10, 20, 20)
        via.category = "via"
        via.shape_hint = "box"
        scene.set_polygons([via])

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(30, 30, 50, 50)],
            QPointF(40.0, 40.0),
            QPointF(70.0, 70.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_paste_merges_intersecting_conductors_and_is_undoable(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        existing = _rectangle_polygon(10, 10, 40, 40)
        scene.set_polygons([existing])

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(30, 10, 60, 40)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        polygons = scene.get_polygons()
        self.assertEqual(len(polygons), 1)
        self.assertEqual(min(x for x, _y in polygons[0].points), 10)
        self.assertEqual(max(x for x, _y in polygons[0].points), 60)
        self.assertAlmostEqual(polygons[0].area, 1500.0)

        scene.undo_stack.undo()
        restored = scene.get_polygons()
        self.assertEqual(len(restored), 1)
        self.assertEqual(max(x for x, _y in restored[0].points), 40)

        scene.undo_stack.redo()
        self.assertEqual(max(x for x, _y in scene.get_polygons()[0].points), 60)

    def test_nonintersecting_paste_preserves_points_without_shapely_rebuild(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        source = _rectangle_polygon(10, 10, 30, 30)

        with patch(
            "contour.graphics.editor_scene.shapely_to_polygon_data_list"
        ) as rebuild:
            pasted = scene.add_cloned_polygons_at(
                [source],
                QPointF(0.0, 0.0),
                QPointF(40.0, 0.0),
            )

        self.assertEqual(len(pasted), 1)
        self.assertEqual(scene.get_polygons()[0].points, [(50, 10), (70, 10), (70, 30), (50, 30)])
        rebuild.assert_not_called()

    def test_paste_clips_conductor_to_three_pixel_image_inset(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(-10, -20, 110, 120)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        polygon = scene.get_polygons()[0]
        self.assertEqual(min(x for x, _y in polygon.points), 3)
        self.assertEqual(max(x for x, _y in polygon.points), 97)
        self.assertEqual(min(y for _x, y in polygon.points), 3)
        self.assertEqual(max(y for _x, y in polygon.points), 97)

    def test_paste_rejects_conductor_fully_outside_image_inset(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        undo_count = scene.undo_stack.count()

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(-30, -30, -10, -10)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(scene.get_polygons(), [])
        self.assertEqual(scene.undo_stack.count(), undo_count)

    def test_paste_skips_contacts_closer_than_minimum_distance(self) -> None:
        scene = PolygonEditorScene()
        existing = _rectangle_polygon(0, 0, 10, 10)
        existing.category = "via"
        existing.shape_hint = "box"
        too_close = _rectangle_polygon(10, 0, 20, 10)
        too_close.category = "via"
        too_close.shape_hint = "box"
        far_enough = _rectangle_polygon(20, 0, 30, 10)
        far_enough.id = 2
        far_enough.category = "via"
        far_enough.shape_hint = "box"
        scene.set_polygons([existing])
        scene.set_minimum_contact_distance(20.0)

        pasted = scene.add_cloned_polygons_at(
            [too_close, far_enough],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        centers = sorted(
            (min(x for x, _y in polygon.points) + max(x for x, _y in polygon.points)) / 2.0
            for polygon in scene.get_polygons()
        )
        self.assertEqual(centers, [5.0, 25.0])
        scene.undo_stack.undo()
        self.assertEqual([polygon.id for polygon in scene.get_polygons()], [1])

    def test_paste_filters_contacts_against_contacts_accepted_in_same_batch(self) -> None:
        scene = PolygonEditorScene()
        first = _rectangle_polygon(0, 0, 10, 10)
        first.category = "via"
        first.shape_hint = "box"
        second = _rectangle_polygon(12, 0, 22, 10)
        second.id = 2
        second.category = "via"
        second.shape_hint = "box"
        scene.set_minimum_contact_distance(20.0)

        pasted = scene.add_cloned_polygons_at(
            [first, second],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_fully_filtered_contact_paste_does_not_create_undo_entry(self) -> None:
        scene = PolygonEditorScene()
        existing = _rectangle_polygon(0, 0, 10, 10)
        existing.category = "via"
        existing.shape_hint = "box"
        duplicate = existing.clone()
        scene.set_polygons([existing])
        scene.set_minimum_contact_distance(20.0)
        undo_count = scene.undo_stack.count()
        changed: list[None] = []
        scene.polygonsChanged.connect(lambda: changed.append(None))

        pasted = scene.add_cloned_polygons_at(
            [duplicate],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(scene.undo_stack.count(), undo_count)
        self.assertEqual(changed, [])
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_paste_refreshes_and_emits_once_per_undo_operation(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(10):
            contact = _rectangle_polygon(index * 20, 10, index * 20 + 10, 20)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        changed: list[None] = []
        scene.polygonsChanged.connect(lambda: changed.append(None))

        with patch.object(scene, "_refresh_all_items", wraps=scene._refresh_all_items) as refresh:
            pasted = scene.add_cloned_polygons_at(
                contacts,
                QPointF(95.0, 15.0),
                QPointF(95.0, 45.0),
            )
            self.assertEqual(len(pasted), len(contacts))
            # Adding and applying the final selection share one scene refresh.
            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(len(changed), 1)

            scene.undo_stack.undo()
            self.assertEqual(refresh.call_count, 2)
            self.assertEqual(len(changed), 2)
            self.assertEqual(scene.get_polygons(), [])

            scene.undo_stack.redo()
            self.assertEqual(refresh.call_count, 3)
            self.assertEqual(len(changed), 3)
            self.assertEqual(len(scene.get_polygons()), len(contacts))

    def test_contact_recognition_mode_protects_only_recognized_vias(self) -> None:
        scene = PolygonEditorScene()
        manual = _rectangle_polygon(10, 10, 20, 20)
        manual.category = "via"
        manual.shape_hint = "box"
        automatic = _rectangle_polygon(30, 30, 40, 40)
        automatic.id = 2
        automatic.category = "via"
        automatic.shape_hint = "box"
        automatic.recognition_score = 82.0
        scene.set_polygons([manual, automatic])
        scene.set_protect_recognized_vias(True)

        self.assertFalse(scene.delete_via_at(QPointF(35.0, 35.0)))
        self.assertTrue(scene.delete_via_at(QPointF(15.0, 15.0)))
        self.assertEqual([polygon.id for polygon in scene.get_polygons()], [2])

    def test_manual_via_uses_configured_color_and_recognized_via_keeps_score_color(self) -> None:
        scene = PolygonEditorScene()
        scene.set_display_settings(
            DisplaySettings(
                external_color="#7C3AED",
                via_selection_color="#FACC15",
            )
        )
        manual = _rectangle_polygon(10, 10, 20, 20)
        manual.category = "via"
        manual.shape_hint = "box"
        automatic = _rectangle_polygon(30, 30, 40, 40)
        automatic.id = 2
        automatic.category = "via"
        automatic.shape_hint = "box"
        automatic.recognition_score = 100.0
        scene.set_polygons([manual, automatic])

        self.assertEqual(scene._polygon_items[1].pen().color().name().lower(), "#7c3aed")
        self.assertNotEqual(scene._polygon_items[2].pen().color().name().lower(), "#7c3aed")
        scene.select_polygon(1)
        self.assertEqual(scene._polygon_items[1].pen().color().name().lower(), "#facc15")

    def test_via_display_mode_switches_between_circle_and_rectangle_paths(self) -> None:
        polygon = PolygonData(
            id=1,
            points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            category="via",
            shape_hint="box",
        )

        circle_item = EditablePolygonItem(polygon, DisplaySettings(via_display_mode="circle"))
        rectangle_item = EditablePolygonItem(polygon, DisplaySettings(via_display_mode="rectangle"))

        self.assertFalse(circle_item.path().contains(QPointF(1.0, 1.0)))
        self.assertTrue(rectangle_item.path().contains(QPointF(1.0, 1.0)))

    def test_via_cursor_turns_red_when_placement_overlaps_existing_via(self) -> None:
        scene = PolygonEditorScene()
        scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0)

        scene.set_via_cursor(QPointF(24.0, 20.0), 10.0, 10.0, True)
        blocked_color = scene._via_cursor_item.pen().color().name().lower()
        scene.set_via_cursor(QPointF(40.0, 20.0), 10.0, 10.0, True)
        allowed_color = scene._via_cursor_item.pen().color().name().lower()

        self.assertEqual(blocked_color, "#ef4444")
        self.assertEqual(allowed_color, "#a78bfa")


if __name__ == "__main__":
    unittest.main()

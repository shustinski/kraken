from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox, QListWidgetItem, QMenu, QScrollArea, QSpinBox

import contour.widget as widget_module
from contour.application.processing import DisplaySettings, ImageProcessingState
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


def _net_outline_area(polygons: list[PolygonData]) -> float:
    """Subtract hole areas from roots (handles flat lists after CSG raster ops)."""

    outers = sum(p.area for p in polygons if not p.is_hole)
    holes = sum(p.area for p in polygons if p.is_hole)
    return outers - holes


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

    def test_extraction_change_does_not_process_when_auto_apply_disabled(self) -> None:
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(False)
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()

        self.assertEqual(process_calls, [])

    def test_extraction_mode_defaults_to_no_extraction(self) -> None:
        self.assertEqual(self.widget.recognition_mode_combo.currentData(), "disabled")
        self.assertEqual(self.widget.recognition_mode_combo.currentText(), "Без извлечения")

    def test_extraction_change_processes_when_auto_apply_enabled(self) -> None:
        process_calls: list[bool] = []
        self.widget.process_current_image = lambda *_args, debounced=False: process_calls.append(debounced)  # type: ignore[method-assign]

        self.widget.auto_apply_checkbox.setChecked(True)
        self.widget.min_area_spin.setValue(self.widget.min_area_spin.value() + 5.0)
        self._app.processEvents()
        QTest.qWait(200)
        self._app.processEvents()

        self.assertEqual(process_calls[-1], False)

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

    def test_neighbor_frames_render_around_current_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(25):
                path = os.path.join(directory, f"frame_{index:02d}.png")
                image = np.full((12, 12), index, dtype=np.uint8)
                cv2.imwrite(path, image)
                paths.append(path)

            self.widget.load_images(paths)
            self.widget.neighbor_columns_spin.setValue(5)
            self.widget.neighbor_max_grid_spin.setValue(3)
            self.widget.show_neighbor_frames_checkbox.setChecked(True)
            self.widget.image_list.setCurrentRow(12)
            self._app.processEvents()

            neighbor_items = self.widget.polygon_editor._editor_scene._neighbor_frame_items
            self.assertEqual(len(neighbor_items), 8)
            self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.path().isEmpty())
            self.assertTrue(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_file_list_uses_stems_and_thumbnail_click_navigates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.load_images(paths)
            self.widget.load_image = lambda path: setattr(self.widget, "_last_loaded_from_thumb", path)  # type: ignore[method-assign]

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

    def test_image_row_navigation_recenters_editor_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.load_images(paths)
            with patch.object(self.widget.polygon_editor, "center_main_image") as center_mock:
                self.widget.image_list.setCurrentRow(1)
                self._app.processEvents()

            center_mock.assert_called()

    def test_checked_toolbar_tool_has_explicit_high_contrast_style(self) -> None:
        stylesheet = self.widget.styleSheet()

        self.assertIn("QToolButton:checked", stylesheet)
        self.assertIn("#2563EB", stylesheet)

    def test_thumbnail_grid_uses_display_frames_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(5):
                path = os.path.join(directory, f"frame_{index}.png")
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.neighbor_columns_spin.setValue(4)
            self.widget.load_images(paths)

            self.assertEqual(self.widget._thumbnail_columns(), 4)
            self.assertGreaterEqual(self.widget.thumbnail_grid.minimumWidth(), 4 * 64)

    def test_thumbnail_stale_background_result_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frame_001.png")
            cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
            self.widget.load_images([path])
            item = self.widget.thumbnail_grid.item(0)
            before = item.icon().cacheKey()

            self.widget._on_thumbnail_loaded(self.widget._thumbnail_generation - 1, path, np.full((4, 4, 3), 255, dtype=np.uint8))

            self.assertEqual(item.icon().cacheKey(), before)

    def test_thumbnail_grid_is_inside_scroll_area(self) -> None:
        self.assertTrue(hasattr(self.widget, "thumbnail_grid_scroll_area"))
        self.assertIsInstance(self.widget.thumbnail_grid_scroll_area, QScrollArea)
        self.assertIs(self.widget.thumbnail_grid_scroll_area.widget(), self.widget.thumbnail_grid)

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
            self.widget._extra_layers.append(layer)
            self.widget._refresh_extra_layers_list()

            row_item = self.widget.extra_layers_list.item(0)
            row_widget = self.widget.extra_layers_list.itemWidget(row_item)
            checkbox = row_widget.findChild(QCheckBox)
            spinboxes = row_widget.findChildren(QSpinBox)
            self.assertIsNotNone(checkbox)
            self.assertEqual(checkbox.text(), "")
            self.assertEqual(checkbox.toolTip(), "Показать/скрыть слой")
            self.assertEqual(len(spinboxes), 3)
            self.assertEqual(spinboxes[0].toolTip(), "Смещение слоя по X")
            self.assertEqual(spinboxes[1].toolTip(), "Смещение слоя по Y")
            self.assertEqual(spinboxes[2].toolTip(), "Прозрачность слоя")

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

        action = next((action for action in menu.actions() if action.objectName() == "manualToolPostprocessAction"), None)

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

    def test_neighbor_grid_expands_when_zoomed_out(self) -> None:
        self.widget.neighbor_max_grid_spin.setValue(7)
        self.widget.polygon_editor.resetTransform()
        self.widget.polygon_editor.scale(0.2, 0.2)

        self.assertEqual(self.widget._neighbor_grid_size_for_zoom(), 7)

    def test_neighbor_frame_border_is_hidden_when_neighbors_are_disabled(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((24, 24), dtype=np.uint8))

        self.widget.show_neighbor_frames_checkbox.setChecked(False)
        self.widget._sync_neighbor_frames()

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

        self.assertTrue(saved_payloads)
        self.assertEqual(saved_payloads[-1]["neighbor_overlap_pixels"], 5)
        self.assertTrue(saved_payloads[-1]["show_neighbor_frames"])


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

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._app.processEvents()

    def test_middle_button_pans_without_hiding_polygon_overlays_or_changing_polygons(self) -> None:
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
        self.assertTrue(self.view._editor_scene.polygon_overlays_visible())

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

    def test_space_hold_hides_vectors_without_mutating_polygon_data(self) -> None:
        QTest.mouseClick(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(40, 40))
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

    def test_ctrl_wheel_keeps_scene_point_under_cursor_stable(self) -> None:
        self.view.fit_to_view()
        self._app.processEvents()
        pos = QPoint(90, 80)
        scene_before = self.view.mapToScene(pos)
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
        self._app.sendEvent(self.view.viewport(), event)
        self._app.processEvents()
        scene_after = self.view.mapToScene(pos)
        self.assertAlmostEqual(scene_after.x(), scene_before.x(), delta=2.0)
        self.assertAlmostEqual(scene_after.y(), scene_before.y(), delta=2.0)

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

        click_pos = self.view.mapFromScene(QPointF(75.0, 10.0))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            click_pos,
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

    def test_move_vertex_merges_when_it_overlaps_another_polygon(self) -> None:
        first = _rectangle_polygon(20, 20, 50, 50)
        second = _rectangle_polygon(55, 20, 85, 50)
        second.id = 2
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0)
        )
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)

        press_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(65.0, 20.0))
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
        self.assertEqual(len(roots), 1)

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
        original_save_polygons_cif = widget_module.save_polygons_cif
        original_load_image = self.widget.load_image
        try:
            self.widget.autosave_on_frame_transition_checkbox.setChecked(True)
            widget_module.save_polygons_cif = lambda path, image_path, polygons, image_size, layer_name="NM": (
                saved_calls.append((str(path), image_path, image_size, len(polygons)))
            )
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]

            self.widget._on_image_item_changed(second_item, first_item)
        finally:
            widget_module.save_polygons_cif = original_save_polygons_cif
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
        original_save_polygons_cif = widget_module.save_polygons_cif
        original_load_image = self.widget.load_image
        try:
            widget_module.save_polygons_cif = lambda *args, **kwargs: saved_calls.append("save")
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]
            with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            widget_module.save_polygons_cif = original_save_polygons_cif
            self.widget.load_image = original_load_image  # type: ignore[method-assign]

        self.assertEqual(saved_calls, [])
        self.widget._refresh_image_list_item_states()
        self.assertEqual(first_item.background().color().name().lower(), "#3d4f66")

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
        original_save_polygons_cif = widget_module.save_polygons_cif
        original_load_image = self.widget.load_image
        try:
            self.widget.autosave_on_frame_transition_checkbox.setChecked(False)
            widget_module.save_polygons_cif = lambda path, image_path, polygons, image_size, layer_name="NM": (
                saved_calls.append((str(path), image_path, image_size, len(polygons)))
            )
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]

            with patch.object(widget_module.QMessageBox, "exec", return_value=widget_module.QMessageBox.StandardButton.Discard):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            widget_module.save_polygons_cif = original_save_polygons_cif
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
        original_save_polygons_cif = widget_module.save_polygons_cif
        original_load_image = self.widget.load_image
        try:
            widget_module.save_polygons_cif = lambda *args, **kwargs: saved_calls.append("save")
            self.widget.load_image = lambda path: None  # type: ignore[method-assign]
            with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
                self.widget._on_image_item_changed(second_item, first_item)
        finally:
            widget_module.save_polygons_cif = original_save_polygons_cif
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
        modes = [str(self.widget.brush_mode_combo.itemData(index)) for index in range(self.widget.brush_mode_combo.count())]
        self.assertEqual(self.widget.brush_mode_combo.count(), 2)
        self.assertEqual(modes, ["freeform", "angled"])

    def test_polygon_mode_indicator_is_hidden_but_mode_switch_stays_operational(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.widget.polygon_mode_combo.setCurrentIndex(self.widget.polygon_mode_combo.findData(PolygonCreateMode.RECTANGLE))
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


if __name__ == "__main__":
    unittest.main()

"""test_widget_interactions: test extra layer row."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QMenu,
    QPushButton,
    QSpinBox,
)
import contour.widget as widget_module
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class ExtraLayerRowMixin:
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
        self.widget._set_default_extraction_disabled()
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
        self.widget._editor_display_thread_pool.waitForDone(1000)
        self._app.processEvents()
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

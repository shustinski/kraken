"""test_widget_interactions: test switching frames autosaves."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
import contour.widget as widget_module
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class SwitchingFramesAutosavesMixin:
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

"""test_widget_interactions: test switching frames does."""
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


class SwitchingFramesDoesMixin:
    def test_switching_frames_after_loading_cif_overlay_does_not_prompt(self) -> None:
        first_path = str(Path("frame_1.png"))
        second_path = str(Path("frame_2.png"))
        overlay = _rectangle_polygon(4, 4, 20, 20)
        self.widget._workspace.replace_image_selection(
            [first_path, second_path],
            is_supported_image=lambda _path: True,
        )
        self.widget._workspace.apply_loaded_frame(
            first_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[overlay.clone()],
            make_current=True,
        )
        self.widget._workspace.apply_loaded_frame(
            second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[],
            make_current=False,
        )
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([overlay.clone()])
        self.widget._editor_polygons_signature = self.widget._polygons_editor_signature(
            first_path,
            [overlay.clone()],
        )
        self.widget.autosave_on_frame_transition_checkbox.setChecked(False)

        with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
            allowed = self.widget._try_leave_current_frame()

        self.assertTrue(allowed)
        self.assertFalse(self.widget._workspace.image_has_changes(first_path))
        self.assertFalse(self.widget._workspace.current_image_has_changes())

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

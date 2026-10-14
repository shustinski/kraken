"""test_widget_interactions: test extraction mode switch."""
from __future__ import annotations
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


class ExtractionModeSwitchMixin:
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

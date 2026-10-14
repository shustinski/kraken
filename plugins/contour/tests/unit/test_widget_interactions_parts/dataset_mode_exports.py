"""test_widget_interactions: test dataset mode exports."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class DatasetModeExportsMixin:
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

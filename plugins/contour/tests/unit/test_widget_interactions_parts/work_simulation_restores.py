"""test_widget_interactions: test work simulation restores."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class WorkSimulationRestoresMixin:
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

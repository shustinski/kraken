"""test_widget_interactions: polygon extraction widget load."""
from __future__ import annotations
import unittest
import numpy as np
from PyQt6.QtTest import QTest
from contour.application.processing import (
    ImageProcessingState,
)
from contour.application.services.workspace_session import WorkspaceLoadResult
from contour.widget import PolygonExtractionWidget

from test_widget_interactions_parts.oversampled_rectangle_polygon import _app

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

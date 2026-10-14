"""test_widget_interactions: test cached frame switch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from contour.application.processing import (
    ImageProcessingState,
)
from contour.application.services.workspace_session import WorkspaceLoadResult

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class CachedFrameSwitchMixin:
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

    def test_cached_frame_switch_holds_previous_frame_until_vectors_reload(self) -> None:
        self.widget.auto_apply_checkbox.setChecked(False)
        self.widget.recognition_mode_combo.setCurrentIndex(self.widget.recognition_mode_combo.findData("disabled"))
        path_a = str(Path("frame_a.png"))
        path_b = str(Path("frame_b.png"))
        polygon_a = _rectangle_polygon(1, 1, 8, 8)
        state_a = ImageProcessingState(
            image_path=path_a,
            source_image=np.full((32, 32), 10, dtype=np.uint8),
            polygons=[polygon_a.clone()],
        )
        state_b = ImageProcessingState(
            image_path=path_b,
            source_image=np.full((32, 32), 200, dtype=np.uint8),
            polygons=[],
        )
        self.widget._workspace._image_paths = [path_a, path_b]
        self.widget._workspace._current_image_path = path_a
        self.widget._workspace._current_state = state_a
        self.widget._workspace._state_cache = {path_a: state_a, path_b: state_b}
        cache_key_a = self.widget._editor_display_cache_key(path_a, state_a)
        seed = np.full((32, 32, 3), 10, dtype=np.uint8)
        qimage = QImage(seed.data, 32, 32, 96, QImage.Format.Format_RGB888).copy()
        self.widget._editor_pixmap_cache[cache_key_a] = QPixmap.fromImage(qimage)
        self.widget._last_editor_display_cache_key = ("stale", "source", "")
        self.widget._last_editor_display_path = path_a
        self.widget._abort_in_flight_interactive_processing = lambda **_kwargs: None  # type: ignore[method-assign]
        self.widget._start_frame_switch_profile = lambda _path: None  # type: ignore[method-assign]
        self.widget._frame_switch_profile_for_path = lambda _path: None  # type: ignore[method-assign]
        self.widget._workspace.set_cif_index({"frame_b": "frame_b.cif"})
        # set_cif_index clears overlays; restore the prepared source caches used by this test.
        state_a.polygons = [polygon_a.clone()]
        state_a.loaded_cif_path = None
        state_b.polygons = []
        state_b.loaded_cif_path = None
        self.widget._workspace._state_cache = {path_a: state_a, path_b: state_b}
        self.widget._workspace._current_state = state_a
        reloads: list[str] = []
        self.widget._begin_frame_vectors_reload = lambda image_path: reloads.append(str(Path(image_path)))  # type: ignore[method-assign]
        self.widget._request_neighbor_frame_sync = lambda *, delay_ms=0: None  # type: ignore[method-assign]
        self.widget._queue_prepared_image_update = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        self.widget._try_extract_if_recognition_enabled = lambda: None  # type: ignore[method-assign]
        self.widget._apply_frame_to_editor()
        self._app.processEvents()

        self.widget._begin_frame_load(path_b, load_vectors=True)
        self._app.processEvents()

        self.assertIs(self.widget._workspace.current_state, state_b)
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), path_b)
        self.assertEqual(reloads, [path_b])
        # Previous complete frame stays visible until vectors arrive with the new image.
        pixmap = self.widget.polygon_editor._editor_scene._image_item.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertEqual(pixmap.toImage().pixelColor(0, 0).red(), 10)
        self.assertEqual(len(self.widget.polygon_editor.get_polygons()), 1)

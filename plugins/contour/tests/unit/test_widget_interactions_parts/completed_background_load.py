"""test_widget_interactions: test completed background load."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt6.QtTest import QTest


class CompletedBackgroundLoadMixin:
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

"""test_widget_interactions: test directory session restore."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from unittest.mock import call, patch
import cv2
import numpy as np
from PyQt6.QtCore import QSignalBlocker, QTimer
from PyQt6.QtTest import QTest


class DirectorySessionRestoreMixin:
    def test_directory_session_restore_rescans_and_loads_current_vectors(self) -> None:
        from contour.infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame_001.png"
            vector_path = Path(directory) / "frame_001.cif"
            cv2.imwrite(str(image_path), np.zeros((16, 16), dtype=np.uint8))
            vector_path.write_text(
                "\n".join(
                    (
                        "DS 1 1 1;",
                        "L NM;",
                        "( R frame_001.png );",
                        "( S 16 16 );",
                        "B 6 6 8 8;",
                        "DF;",
                        "E",
                    )
                ),
                encoding="utf-8",
            )
            self.widget.input_dir_edit.setText(directory)
            self.widget.cif_dir_edit.setText(directory)
            self.widget._session_settings_store.save_image_list_mode(IMAGE_LIST_MODE_DIRECTORY)
            self.widget._session_settings_store.save_image_paths([image_path])
            self.widget._session_settings_store.save_vector_paths([])
            self.widget._session_settings_store.save_current_image_path(image_path)

            self.widget._restore_persisted_session_selection()

            for _ in range(500):
                self._app.processEvents()
                state = self.widget._workspace.current_state
                if (
                    state is not None
                    and state.image_path == str(image_path)
                    and len(state.polygons) == 1
                ):
                    break
                QTest.qWait(10)

            state = self.widget._workspace.current_state
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.image_path, str(image_path))
            self.assertEqual(len(state.polygons), 1)
            self.assertEqual(
                self.widget._workspace.resolve_cif_path(image_path),
                str(vector_path),
            )

    def test_refresh_preserves_explicit_image_subset(self) -> None:
        from contour.infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_a.png", "frame_b.png", "frame_c.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)
            self.widget._set_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)
            self.widget.append_images(paths[:2])
            self.widget.input_dir_edit.setText(directory)
            self.widget.refresh_image_list()
            self._app.processEvents()

            self.assertEqual(
                self.widget._workspace.image_paths,
                tuple(str(Path(path)) for path in paths[:2]),
            )

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
            self.assertEqual(self.widget._session_settings_store.load_image_paths(), [])
            self.assertEqual(self.widget._session_settings_store.load_vector_paths(), [])

    def test_image_row_navigation_recenters_editor_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.load_images(paths)
            with patch.object(self.widget, "_center_editor_on_current_main_image") as center_mock:
                proxy_index = self.widget._image_list_proxy.index(1, 0)
                self.widget.image_list.setCurrentIndex(proxy_index)
                for _ in range(100):
                    self._app.processEvents()
                    if call(force=True) in center_mock.call_args_list:
                        break
                    QTest.qWait(20)

            center_mock.assert_any_call(force=True)

    def test_rapid_image_row_navigation_loads_only_latest_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png", "frame_003.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(str(Path(path)))

            self.widget.load_images(paths)
            for _ in range(50):
                self._app.processEvents()
                if self.widget._image_list_proxy.rowCount() == len(paths):
                    break
                QTest.qWait(10)

            loaded_paths: list[str] = []
            self.widget.load_image = lambda path: loaded_paths.append(str(Path(path)))  # type: ignore[method-assign]
            self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(1, 0))
            self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(2, 0))
            QTest.qWait(350)
            self._app.processEvents()

            self.assertEqual(loaded_paths, [paths[2]])

    def test_click_does_not_repeat_pending_current_change_request(self) -> None:
        path = str(Path("frame_001.png"))
        self.widget._image_list_model.set_paths([path])
        self._app.processEvents()
        index = self.widget._image_list_proxy.index(0, 0)
        selection = self.widget.image_list.selectionModel()
        self.assertIsNotNone(selection)
        with QSignalBlocker(selection):
            self.widget.image_list.setCurrentIndex(index)
        timer = QTimer(self.widget)
        timer.setSingleShot(True)
        timer.start(1000)
        self.widget._deferred_image_load_timers = [timer]
        self.widget._desired_image_path = path

        with patch.object(self.widget, "_on_image_list_current_changed") as current_changed:
            self.widget._on_image_list_clicked(index)

        current_changed.assert_not_called()
        timer.stop()
        self.widget._deferred_image_load_timers = []

    def test_frame_switch_interactive_does_not_wait_for_background_geometry_validation(self) -> None:
        path = str(Path("frame_001.png"))
        self.widget._workspace._current_image_path = path
        self.widget._loading_image_path = None
        self.widget._frame_load_running_path = None
        self.widget._pending_editor_frame_apply = None
        self.widget._thumbnail_rebuild_in_progress = False
        frame_chrome_timer = getattr(self.widget, "_frame_chrome_update_timer", None)
        if frame_chrome_timer is not None:
            frame_chrome_timer.stop()
        self.widget._neighbor_sync_timer.stop()
        self.widget._editor_display_thread_pool.waitForDone(1000)

        class UnexpectedFramePoolAccess:
            def activeThreadCount(self) -> int:
                raise AssertionError("background geometry validation must not block interactivity")

        frame_pool = self.widget._frame_load_thread_pool
        self.widget._frame_load_thread_pool = UnexpectedFramePoolAccess()
        try:
            self.assertTrue(self.widget._frame_switch_profile_is_interactive(path))
        finally:
            self.widget._frame_load_thread_pool = frame_pool

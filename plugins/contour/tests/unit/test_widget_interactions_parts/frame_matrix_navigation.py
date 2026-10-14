"""test_widget_interactions: test frame matrix navigation."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtTest import QTest


class FrameMatrixNavigationMixin:
    def test_frame_matrix_navigation_loads_main_editor_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        loaded: list[tuple[str, bool]] = []
        self.widget.load_image = lambda path, **kwargs: loaded.append(  # type: ignore[method-assign]
            (str(Path(path)), bool(kwargs.get("preserve_editor_view_position")))
        )

        self.widget._on_frame_navigation_requested(2)

        self.assertEqual(loaded, [(paths[2], True)])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[2])

    def test_frame_matrix_navigation_uses_image_list_signal_and_starts_loading(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget._set_image_list_paths(paths)
        loaded: list[tuple[str, bool]] = []
        self.widget.load_image = lambda path, **kwargs: loaded.append(  # type: ignore[method-assign]
            (str(Path(path)), bool(kwargs.get("preserve_editor_view_position")))
        )

        self.widget.thumbnail_grid.frameNavigationRequested.emit(2)
        QTest.qWait(300)

        self.assertEqual(
            str(Path(self.widget._image_list_path_from_proxy_index(self.widget.image_list.currentIndex()) or "")),
            paths[2],
        )
        self.assertEqual(loaded, [(paths[2], False)])

    def test_frame_matrix_navigation_does_not_center_editor_scene(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget.load_image = lambda _path, **_kwargs: None  # type: ignore[method-assign]

        with patch.object(self.widget.polygon_editor, "set_current_frame_id") as set_frame_mock:
            self.widget._on_frame_navigation_requested(2)

        set_frame_mock.assert_called_once_with(2, center=False, emit_signal=False)

    def test_frame_matrix_navigation_cancel_keeps_current_frame(self) -> None:
        paths = [str(Path(rf"d:\frames\frame_{index:03d}.png")) for index in range(3)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[0]
        self.widget._set_image_list_paths(paths)
        selection = self.widget.image_list.selectionModel()
        if selection is not None:
            with QSignalBlocker(selection):
                self.widget.image_list.setCurrentIndex(self.widget._image_list_proxy.index(0, 0))
        self.widget.show_frame_matrix_checkbox.setChecked(True)
        self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
        self.widget._thumbnail_build_chunk_size = 100
        self.widget._rebuild_thumbnail_grid()
        self._wait_for_thumbnail_grid_count(3)
        loaded: list[str] = []
        self.widget.load_image = lambda path, **_kwargs: loaded.append(str(Path(path)))  # type: ignore[method-assign]
        self.widget._try_leave_current_frame = lambda: False  # type: ignore[method-assign]

        self.widget._on_frame_navigation_requested(2)

        self.assertEqual(loaded, [])
        self.assertEqual(str(Path(self.widget._workspace.current_image_path or "")), paths[0])
        self.assertEqual(
            str(Path(self.widget._image_list_path_from_proxy_index(self.widget.image_list.currentIndex()) or "")),
            paths[0],
        )

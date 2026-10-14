"""test_widget_interactions: test ingest dropped paths."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt6.QtTest import QTest


class IngestDroppedPathsMixin:
    def test_file_list_uses_stems_and_thumbnail_click_navigates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget.show_frame_matrix_thumbnails_checkbox.setChecked(False)
            self.widget.load_images(paths)
            self._wait_for_thumbnail_grid_count(2)
            self.widget.load_image = lambda path, **_kwargs: setattr(self.widget, "_last_loaded_from_thumb", path)  # type: ignore[method-assign]

            self.assertEqual(self.widget.image_list.item(0).text(), "frame_001")
            self.assertEqual(self.widget.thumbnail_grid.count(), 2)
            self.assertEqual(self.widget.thumbnail_grid.item(0).text(), "")
            self.assertEqual(self.widget.thumbnail_grid.item(0).toolTip(), "frame_001")
            self.widget._on_thumbnail_item_clicked(self.widget.thumbnail_grid.item(1))
            QTest.qWait(300)

            self.assertEqual(self.widget._last_loaded_from_thumb, paths[1])

    def test_append_images_keeps_existing_rows_skips_duplicates_and_underscore_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "frame_001.png")
            second = os.path.join(directory, "frame_002.png")
            hidden = os.path.join(directory, "_frame_003.png")
            for path in (first, second, hidden):
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))

            self.widget.load_images([first])
            self.widget.append_images([first, hidden, second])

            self.assertEqual(self.widget.image_list.count(), 2)
            self.assertEqual([self.widget.image_list.item(i).text() for i in range(2)], ["frame_001", "frame_002"])
            self.assertEqual(self.widget._workspace.current_image_path, str(Path(second)))

    def test_remove_images_drops_frame_from_project_but_keeps_file_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "frame_001.png")
            second = os.path.join(directory, "frame_002.png")
            for path in (first, second):
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))

            self.widget.load_images([first, second])
            self.widget.remove_images([first])

            self.assertTrue(os.path.isfile(first))
            self.assertEqual(self.widget.image_list.count(), 1)
            self.assertEqual(self.widget.image_list.item(0).text(), "frame_002")
            self.assertEqual(self.widget._workspace.current_image_path, str(Path(second)))

    def test_clear_frame_vectors_marks_overlay_cleared_without_deleting_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "frame_001.png")
            cif_path = os.path.join(directory, "frame_001.cif")
            cv2.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
            Path(cif_path).write_text("CIF", encoding="utf-8")

            self.widget.load_images([image_path])
            self.widget._workspace.merge_cif_paths({"frame_001": cif_path})
            self.widget.clear_frame_vectors(image_path)

            self.assertTrue(os.path.isfile(cif_path))
            self.assertTrue(self.widget._workspace.vectors_are_cleared(str(Path(image_path))))
            self.assertEqual(self.widget._load_cif_overlay_polygons(image_path), [])

    def test_ingest_dropped_paths_appends_images_and_vectors_without_replacing(self) -> None:
        directory = tempfile.TemporaryDirectory()
        # tearDown closes the widget and its workers before removing their inputs.
        self.addCleanup(directory.cleanup)
        first = os.path.join(directory.name, "frame_001.png")
        second = os.path.join(directory.name, "frame_002.png")
        cif_path = os.path.join(directory.name, "frame_002.cif")
        cv2.imwrite(first, np.zeros((8, 8), dtype=np.uint8))
        cv2.imwrite(second, np.zeros((8, 8), dtype=np.uint8))
        Path(cif_path).write_text(
            "\n".join(("DS 1 1 1;", "L NM;", "( S 8 8 );", "B 4 4 4 4;", "DF;", "E")),
            encoding="utf-8",
        )

        self.widget.load_images([first])
        self.widget._ingest_dropped_paths([second, cif_path])

        self.assertEqual(self.widget.image_list.count(), 2)
        self.assertEqual(
            [self.widget.image_list.item(i).text() for i in range(2)],
            ["frame_001", "frame_002"],
        )
        self.assertIn("frame_002", self.widget._workspace.cif_paths_by_stem)

    def test_select_input_directory_replaces_instead_of_appending(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = os.path.join(first_dir, "frame_001.png")
            second = os.path.join(second_dir, "frame_001.png")
            cv2.imwrite(first, np.zeros((8, 8), dtype=np.uint8))
            cv2.imwrite(second, np.ones((8, 8), dtype=np.uint8))
            scans: list[tuple[str, bool]] = []

            self.widget.load_images([first])
            self.widget._begin_async_directory_scan = lambda directory, *, append=False: scans.append(
                (str(Path(directory)), bool(append))
            )  # type: ignore[method-assign]

            with patch(
                "contour.widget_parts.navigation_mixin.QFileDialog.getExistingDirectory", return_value=second_dir
            ):
                self.widget._select_input_directory()

            self.assertEqual(scans, [(str(Path(second_dir)), False)])
            self.assertEqual(self.widget.input_dir_edit.text(), str(Path(second_dir)))
            self.assertEqual(self.widget._image_list_mode, "directory_scan")

    def test_scan_finish_restores_persisted_current_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                paths.append(path)

            self.widget._pending_restore_current_image_path = str(Path(paths[1]))
            self.widget._on_input_directory_scan_finished(paths)

            self.assertEqual(self.widget._workspace.current_image_path, str(Path(paths[1])))
            self.assertEqual(self.widget.image_list.currentRow(), 1)

    def test_restore_persisted_session_selection_loads_images_vectors_and_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                image_paths.append(path)
            vector_path = os.path.join(directory, "frame_999.cif")
            Path(vector_path).write_text("", encoding="utf-8")

            self.widget._session_settings_store.save_image_paths(image_paths)
            self.widget._session_settings_store.save_vector_paths([vector_path])
            self.widget._session_settings_store.save_current_image_path(image_paths[1])

            self.widget._restore_persisted_session_selection()
            self._wait_for_thumbnail_grid_count(len(image_paths))

            self.assertEqual(self.widget._workspace.image_paths, tuple(str(Path(path)) for path in image_paths))
            self.assertEqual(self.widget._workspace.current_image_path, str(Path(image_paths[1])))
            self.assertEqual(self.widget.thumbnail_grid.count(), len(image_paths))
            self.assertIn("frame_999", self.widget._workspace.cif_paths_by_stem)

    def test_restored_frame_matrix_build_does_not_wait_for_cif_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for name in ("frame_001.png", "frame_002.png"):
                path = os.path.join(directory, name)
                cv2.imwrite(path, np.zeros((8, 8), dtype=np.uint8))
                image_paths.append(path)

            self.widget._session_settings_store.save_image_paths(image_paths)
            self.widget._session_settings_store.save_current_image_path(image_paths[0])
            self.widget.cif_dir_edit.setText(directory)
            self.widget._indexed_cif_directory = None
            index_requests: list[str] = []
            self.widget._begin_async_cif_directory_index = lambda path: index_requests.append(  # type: ignore[method-assign]
                str(Path(path))
            )

            self.widget._restore_persisted_session_selection()
            self._wait_for_thumbnail_grid_count(len(image_paths))

            self.assertEqual(self.widget.thumbnail_grid.count(), len(image_paths))
            self.assertEqual(index_requests, [str(Path(directory))])

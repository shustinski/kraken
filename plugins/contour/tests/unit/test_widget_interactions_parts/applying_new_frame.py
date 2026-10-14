"""test_widget_interactions: test applying new frame."""
from __future__ import annotations
import tempfile
from pathlib import Path
import cv2
import numpy as np
from contour.application.processing import (
    ImageProcessingState,
)
from contour.graphics_view import (
    PolygonEditorScene,
)


class ApplyingNewFrameMixin:
    def test_neighbor_frames_render_around_current_image(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(25)]
        self.widget._set_image_list_paths(paths)
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[12]
        self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(5)
        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget._sync_neighbor_frames()

        neighbor_items = self.widget.polygon_editor._editor_scene._neighbor_frame_items
        self.assertEqual(len(neighbor_items), 8)
        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.path().isEmpty())
        self.assertTrue(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_lod_store_does_not_replace_main_editor_neighbor_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index in range(9):
                path = Path(directory) / f"frame_{index:02d}.png"
                cv2.imwrite(str(path), np.full((12, 12, 3), index, dtype=np.uint8))
                paths.append(str(path))

            self.widget._workspace._image_paths = paths
            self.widget._workspace._current_image_path = paths[4]
            self.widget._set_image_list_paths(paths)
            self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
            self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
            self.widget.neighbor_columns_spin.setValue(3)
            self.widget.neighbor_max_grid_spin.setValue(3)
            self.widget.show_neighbor_frames_checkbox.setChecked(True)

            self.widget._configure_pyramid_frame_store(paths)
            self.widget._sync_neighbor_frames()

            self.assertFalse(self.widget.polygon_editor.pyramid_mode_enabled())
            self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 8)

    def test_neighbor_frames_render_when_main_image_arrives_after_neighbor_request(self) -> None:
        scene = PolygonEditorScene()
        scene.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 0)

        scene.set_image(np.zeros((12, 12), dtype=np.uint8))

        self.assertEqual(len(scene._neighbor_frame_items), 1)

    def test_applying_new_frame_clears_stale_neighbor_layout_before_sync(self) -> None:
        scene = self.widget.polygon_editor._editor_scene
        self.widget.polygon_editor.set_image(np.zeros((12, 48), dtype=np.uint8))
        self.widget.polygon_editor.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "old_neighbor.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 1)
        self.widget._neighbor_sync_image_path = "old_frame.png"
        self.widget._neighbor_frame_specs = [(1, 0, "old_neighbor.png")]
        self.widget._neighbor_queued_paths = {"old_neighbor.png"}
        state = ImageProcessingState(
            image_path="new_frame.png",
            source_image=np.zeros((48, 12), dtype=np.uint8),
            polygons=[],
        )
        self.widget._workspace._current_image_path = "new_frame.png"
        self.widget._workspace._current_state = state
        self.widget._apply_display_image_to_editor = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
        self.widget._request_neighbor_frame_sync = lambda *args, **kwargs: None  # type: ignore[method-assign]

        self.widget._apply_frame_to_editor()

        self.assertIsNone(scene._pending_neighbor_frames)
        self.assertEqual(len(scene._neighbor_frame_items), 0)
        self.assertIsNone(self.widget._neighbor_sync_image_path)
        self.assertEqual(self.widget._neighbor_frame_specs, [])
        self.assertEqual(self.widget._neighbor_queued_paths, set())

    def test_main_image_update_does_not_clear_neighbors_when_pending_list_empty(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((12, 12), dtype=np.uint8))
        scene.set_neighbor_frames(
            [(1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png")],
            0.5,
            overlap_pixels=0,
            show_main_frame=True,
        )
        self.assertEqual(len(scene._neighbor_frame_items), 1)

        scene._pending_neighbor_frames = []
        scene.set_image(np.zeros((12, 12), dtype=np.uint8))

        self.assertEqual(len(scene._neighbor_frame_items), 1)

    def test_apply_cached_neighbors_skips_clear_while_neighbor_loads_are_queued(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        scene = self.widget.polygon_editor._editor_scene
        scene._pending_neighbor_frames = None
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_queued_paths = {paths[5]}

        self.widget._apply_cached_neighbor_frames()

        self.assertIsNone(scene._pending_neighbor_frames)
        self.assertEqual(len(scene._neighbor_frame_items), 0)

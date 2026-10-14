"""test_widget_interactions: test neighbor sync drops."""
from __future__ import annotations
import numpy as np


class NeighborSyncDropsMixin:
    def test_neighbor_sync_drops_stale_queued_paths_before_queueing_current_neighbors(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._set_image_list_paths(paths)
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(3)
        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget._neighbor_sync_image_path = "previous.png"
        self.widget._neighbor_queued_paths = {"previous_neighbor.png"}
        queued: list[str] = []

        def _queue(path: str) -> None:
            queued.append(path)
            self.widget._neighbor_queued_paths.add(path)

        self.widget._queue_neighbor_frame_load = _queue  # type: ignore[method-assign]

        self.widget._sync_neighbor_frames()

        self.assertNotIn("previous_neighbor.png", self.widget._neighbor_queued_paths)
        self.assertEqual(set(queued), set(path for path in paths if path != paths[4]))

    def test_neighbor_load_finished_applies_when_queue_drains(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_queued_paths = {paths[5]}
        applied: list[int] = []
        self.widget._schedule_neighbor_frame_apply = lambda *, delay_ms=0: applied.append(delay_ms)  # type: ignore[method-assign]

        self.widget._on_neighbor_frame_load_finished(0, paths[5])

        self.assertEqual(self.widget._neighbor_queued_paths, set())
        self.assertEqual(applied, [0])

    def test_neighbor_grid_size_controls_displayed_neighbor_ring_count(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(49)]
        self.widget._set_image_list_paths(paths)
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[24]
        self.widget._neighbor_frame_image = lambda _path: np.zeros((12, 12), dtype=np.uint8)  # type: ignore[method-assign]
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        self.widget.neighbor_columns_spin.setValue(7)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)

        self.widget.neighbor_max_grid_spin.setValue(3)
        self.widget._sync_neighbor_frames()
        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 8)

        self.widget.neighbor_max_grid_spin.setValue(5)
        self.widget._sync_neighbor_frames()
        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 24)

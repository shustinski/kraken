"""test_widget_interactions: test display settings are."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class DisplaySettingsAreMixin:
    def test_stale_neighbor_apply_does_not_redraw_when_disabled(self) -> None:
        paths = [f"frame_{index:02d}.png" for index in range(9)]
        self.widget._workspace._image_paths = paths
        self.widget._workspace._current_image_path = paths[4]
        self.widget.polygon_editor.set_image(np.zeros((24, 24), dtype=np.uint8))
        self.widget._neighbor_frame_specs = [(1, 0, paths[5])]
        self.widget._neighbor_sync_image_path = paths[4]
        self.widget._neighbor_image_cache[str(Path(paths[5]))] = np.zeros((24, 24), dtype=np.uint8)
        self.widget.show_neighbor_frames_checkbox.setChecked(False)

        self.widget._apply_cached_neighbor_frames()

        self.assertEqual(len(self.widget.polygon_editor._editor_scene._neighbor_frame_items), 0)
        self.assertFalse(self.widget.polygon_editor._editor_scene._main_frame_item.isVisible())

    def test_neighbor_frame_overlap_moves_tiles_closer(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((12, 12), dtype=np.uint8))
        frames = [
            (-1, 0, np.zeros((12, 12), dtype=np.uint8), "left.png"),
            (1, 0, np.zeros((12, 12), dtype=np.uint8), "right.png"),
        ]

        self.widget.polygon_editor.set_neighbor_frames(frames, 0.5, overlap_pixels=3, show_main_frame=True)

        positions = sorted(
            round(item.pos().x()) for item in self.widget.polygon_editor._editor_scene._neighbor_frame_items
        )
        self.assertEqual(positions, [-9, 9])

    def test_display_settings_are_saved_when_changed(self) -> None:
        saved_payloads: list[dict[str, object]] = []

        class _Store:
            def load(self) -> dict[str, object]:
                return {}

            def save(self, payload: dict[str, object]) -> None:
                saved_payloads.append(dict(payload))

        self.widget._display_settings_store = _Store()  # type: ignore[assignment]
        self.widget.neighbor_overlap_spin.setValue(5)
        self.widget.show_neighbor_frames_checkbox.setChecked(True)
        self.widget.show_neighbor_vectors_checkbox.setChecked(True)

        self.assertTrue(saved_payloads)
        self.assertEqual(saved_payloads[-1]["neighbor_overlap_pixels"], 5)
        self.assertTrue(saved_payloads[-1]["show_neighbor_frames"])
        self.assertTrue(saved_payloads[-1]["show_neighbor_vectors"])

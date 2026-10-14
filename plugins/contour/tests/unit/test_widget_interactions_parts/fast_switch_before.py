"""test_widget_interactions: test fast switch before."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import contour.widget as widget_module
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class FastSwitchBeforeMixin:
    def test_fast_switch_before_editor_apply_does_not_mark_loaded_frame_dirty(self) -> None:
        first_path = "frame_1.png"
        second_path = "frame_2.png"
        first_polygon = _rectangle_polygon(4, 4, 20, 20)
        second_polygon = _rectangle_polygon(6, 6, 22, 22)
        second_state = ImageProcessingState(
            image_path=second_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[second_polygon.clone()],
            loaded_cif_path="frame_2.cif",
            reference_polygons=[second_polygon.clone()],
            polygons_dirty=False,
        )
        self.widget._workspace._state_cache = {second_path: second_state}
        self.widget._workspace._current_image_path = second_path
        self.widget._workspace._current_state = second_state
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([first_polygon.clone()])
        self.widget._editor_polygons_signature = self.widget._polygons_editor_signature(
            first_path,
            [first_polygon.clone()],
        )

        with patch.object(widget_module.QMessageBox, "exec", side_effect=AssertionError("unexpected prompt")):
            allowed = self.widget._try_leave_current_frame()

        self.assertTrue(allowed)
        self.assertFalse(self.widget._workspace.current_image_has_changes())
        self.assertEqual(second_state.polygons[0].points, second_polygon.points)

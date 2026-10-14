"""test_widget_interactions: via candidate overlay tests."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
)
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    ContourDebugCandidate,
)
from contour.graphics_view import (
    PolygonEditorScene,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
)

class ViaCandidateOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_rejected_candidate_is_drawn_and_can_be_cleared(self) -> None:
        scene = PolygonEditorScene()
        candidate = ContourDebugCandidate(
            contour_index=1,
            bbox=(10, 12, 8, 8),
            accepted=False,
            reason="rejected:hard:circularity",
            score=24.0,
        )

        scene.set_debug_candidates([candidate])

        self.assertEqual(len(scene._debug_candidate_items), 1)
        self.assertEqual(
            scene._debug_candidate_items[0].pen().color(),
            QColor("#EF4444"),
        )
        original_item = scene._debug_candidate_items[0]
        scene.set_debug_candidates([])
        self.assertEqual(scene._debug_candidate_items, [])
        self.assertEqual(scene._recycled_debug_candidate_items, [original_item])
        self.assertTrue(scene._recycled_debug_cleanup_timer.isActive())

        scene.set_debug_candidates([candidate])

        self.assertIs(scene._debug_candidate_items[0], original_item)
        self.assertEqual(scene._recycled_debug_candidate_items, [])

    def test_via_color_changes_from_red_to_green_with_score(self) -> None:
        scene = PolygonEditorScene()
        low = _rectangle_polygon(2, 2, 10, 10)
        low.id = 1
        low.category = "via"
        low.recognition_score = 0.0
        high = _rectangle_polygon(14, 2, 22, 10)
        high.id = 2
        high.category = "via"
        high.recognition_score = 100.0

        scene.set_polygons([low, high])

        low_color = scene._polygon_items[1].pen().color()
        high_color = scene._polygon_items[2].pen().color()
        self.assertGreater(low_color.red(), low_color.green())
        self.assertGreater(high_color.green(), high_color.red())

    def test_show_rejected_checkbox_controls_candidate_overlay(self) -> None:
        editor = MagicMock()
        rejected = ContourDebugCandidate(
            contour_index=1,
            bbox=(10, 12, 8, 8),
            accepted=False,
            reason="rejected:hard:circularity",
        )
        show_rejected = QCheckBox()
        show_rejected.setChecked(True)
        owner = SimpleNamespace(
            polygon_editor=editor,
            debug_candidates_checkbox=QCheckBox(),
            bright_via_show_rejected_checkbox=show_rejected,
            _sync_polygons_to_editor=MagicMock(),
            _polygons_visible_in_editor=lambda _state, polygons: list(polygons),
            _apply_conductor_display_visibility=MagicMock(),
            _via_debug_inspection_enabled=lambda: True,
        )
        owner.debug_candidates_checkbox.setChecked(True)
        owner._apply_deferred_editor_vector_overlays = lambda state: (
            processing_mixin_module.WidgetProcessingMixin._apply_deferred_editor_vector_overlays(owner, state)
        )
        state = SimpleNamespace(debug_candidates=[rejected])

        processing_mixin_module.WidgetProcessingMixin._apply_editor_vectors_for_frame(
            owner,
            "sample.png",
            state,
            [],
            defer_heavy_overlays=False,
        )
        editor.set_debug_candidates.assert_called_with([rejected])

        show_rejected.setChecked(False)
        processing_mixin_module.WidgetProcessingMixin._apply_editor_vectors_for_frame(
            owner,
            "sample.png",
            state,
            [],
            defer_heavy_overlays=False,
        )
        editor.set_debug_candidates.assert_called_with([])

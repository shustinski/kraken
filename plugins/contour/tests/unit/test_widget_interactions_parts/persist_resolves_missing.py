"""test_widget_interactions: test persist resolves missing."""
from __future__ import annotations
import numpy as np
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    ImageProcessingState,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class PersistResolvesMissingMixin:
    def test_persist_resolves_missing_loaded_cif_path_from_index(self) -> None:
        image_path = "0555.png"
        cif_path = "0555.cif"
        original = _rectangle_polygon(4, 4, 20, 20)
        changed = _rectangle_polygon(4, 4, 24, 20)
        state = ImageProcessingState(
            image_path=image_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed.clone()],
            loaded_cif_path=None,
            reference_polygons=[original.clone()],
        )
        self.widget._workspace._state_cache = {image_path: state}
        self.widget._workspace._current_image_path = image_path
        self.widget._workspace._current_state = state
        self.widget._workspace._cif_paths_by_stem = {"0555": cif_path}
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([changed.clone()])
        self.widget._editor_polygons_signature = self.widget._polygons_editor_signature(
            image_path,
            [changed.clone()],
        )
        self.widget.dataset_mode_checkbox.setChecked(False)

        saved_calls: list[tuple[str, str, int]] = []
        original_save = processing_mixin_module.save_polygons_vector
        try:
            processing_mixin_module.save_polygons_vector = lambda path, image_path, polygons, image_size: (
                saved_calls.append((str(path), image_path, len(polygons)))
            )
            ok = self.widget._persist_current_overlay_changes()
        finally:
            processing_mixin_module.save_polygons_vector = original_save

        self.assertTrue(ok)
        self.assertEqual(saved_calls, [(cif_path, image_path, 1)])
        self.assertEqual(state.loaded_cif_path, cif_path)

    def test_persist_failure_without_target_sets_visible_error(self) -> None:
        image_path = "0555.png"
        original = _rectangle_polygon(4, 4, 20, 20)
        changed = _rectangle_polygon(4, 4, 24, 20)
        state = ImageProcessingState(
            image_path=image_path,
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[changed.clone()],
            loaded_cif_path=None,
            reference_polygons=[original.clone()],
        )
        self.widget._workspace._state_cache = {image_path: state}
        self.widget._workspace._current_image_path = image_path
        self.widget._workspace._current_state = state
        self.widget._workspace._cif_paths_by_stem = {}
        self.widget.polygon_editor.set_image(np.zeros((32, 32), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([changed.clone()])
        self.widget._editor_polygons_signature = self.widget._polygons_editor_signature(
            image_path,
            [changed.clone()],
        )
        self.widget.dataset_mode_checkbox.setChecked(False)

        ok = self.widget._persist_current_overlay_changes()

        self.assertFalse(ok)
        self.assertTrue(self.widget._last_vector_persist_error)

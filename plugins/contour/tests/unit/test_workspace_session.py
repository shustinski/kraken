from __future__ import annotations

from pathlib import Path
import unittest

from contour.application.services import WorkspaceSession
from contour.domain import PolygonData, compute_polygon_metrics


def _triangle_polygon() -> PolygonData:
    points = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


class WorkspaceSessionTests(unittest.TestCase):
    def test_load_image_reuses_cached_state_without_reloading_source(self) -> None:
        session = WorkspaceSession()
        loader_calls: list[str] = []

        def load_source_image(path: str) -> str:
            loader_calls.append(path)
            return f"source:{path}"

        session.replace_image_selection(["image.png"], is_supported_image=lambda _path: True)
        first = session.load_image(
            "image.png",
            load_source_image=load_source_image,
            load_cif_overlay=lambda _path: [],
        )
        self.assertFalse(first.cache_hit)
        self.assertTrue(first.prepared_image_required)
        self.assertEqual(loader_calls, ["image.png"])

        session.store_preprocessed_image("image.png", "prepared:image.png")
        session.clear_current_selection()

        second = session.load_image(
            "image.png",
            load_source_image=lambda _path: self.fail("cache miss triggered source reload"),
            load_cif_overlay=lambda _path: self.fail("cache miss triggered cif reload"),
        )
        self.assertTrue(second.cache_hit)
        self.assertFalse(second.prepared_image_required)
        self.assertEqual(second.state.preprocessed_image, "prepared:image.png")

    def test_updating_cif_index_invalidates_cached_image_state(self) -> None:
        session = WorkspaceSession()
        loader_calls: list[str] = []

        def load_source_image(path: str) -> str:
            loader_calls.append(path)
            return f"source:{path}"

        session.load_image(
            "sample.png",
            load_source_image=load_source_image,
            load_cif_overlay=lambda _path: [],
        )
        session.set_cif_index({"sample": "sample.cif"})
        session.clear_current_selection()

        reloaded = session.load_image(
            "sample.png",
            load_source_image=load_source_image,
            load_cif_overlay=lambda _path: [
                PolygonData(
                    id=1,
                    points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                )
            ],
        )

        self.assertFalse(reloaded.cache_hit)
        self.assertEqual(loader_calls, ["sample.png", "sample.png"])
        self.assertEqual(len(reloaded.state.polygons), 1)
        self.assertEqual(reloaded.state.polygons[0].id, 1)

    def test_image_has_changes_compares_against_reference_polygons(self) -> None:
        session = WorkspaceSession()
        polygon = _triangle_polygon()
        session.load_image(
            "sample.png",
            load_source_image=lambda _path: "source",
            load_cif_overlay=lambda _path: [polygon.clone()],
        )
        session.current_state.reference_polygons = [polygon.clone()]

        self.assertFalse(session.current_image_has_changes())

        changed_polygon = polygon.clone()
        changed_polygon.points[1] = (6.0, 0.0)
        changed_polygon.area, changed_polygon.perimeter, changed_polygon.bbox = compute_polygon_metrics(
            changed_polygon.points
        )
        session.update_current_polygons([changed_polygon])

        self.assertTrue(session.current_image_has_changes())
        self.assertTrue(session.image_has_changes("sample.png"))

        session.update_current_polygons([polygon.clone()])

        self.assertFalse(session.current_image_has_changes())

    def test_image_has_changes_handles_mixed_parent_ids(self) -> None:
        session = WorkspaceSession()
        parent = _triangle_polygon()
        hole = PolygonData(
            id=2,
            points=[(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)],
            is_hole=True,
            parent_id=1,
        )
        session.load_image(
            "sample.png",
            load_source_image=lambda _path: "source",
            load_cif_overlay=lambda _path: [hole.clone(), parent.clone()],
        )
        session.current_state.reference_polygons = [parent.clone(), hole.clone()]

        self.assertFalse(session.current_image_has_changes())

    def test_open_frame_is_clean_until_polygons_change(self) -> None:
        session = WorkspaceSession()
        polygon = _triangle_polygon()
        session.load_image(
            "sample.png",
            load_source_image=lambda _path: "source",
            load_cif_overlay=lambda _path: [polygon.clone()],
        )
        session.current_state.reference_polygons = [polygon.clone()]

        self.assertFalse(session.current_image_has_changes())

    def test_edit_polygon_marks_dirty_and_save_sync_marks_saved(self) -> None:
        session = WorkspaceSession()
        polygon = _triangle_polygon()
        session.load_image(
            "sample.png",
            load_source_image=lambda _path: "source",
            load_cif_overlay=lambda _path: [polygon.clone()],
        )
        session.current_state.reference_polygons = [polygon.clone()]
        changed = polygon.clone()
        changed.points[1] = (7.0, 0.0)
        changed.area, changed.perimeter, changed.bbox = compute_polygon_metrics(changed.points)

        session.update_current_polygons([changed])

        self.assertTrue(session.current_image_has_changes())
        self.assertTrue(session.sync_polygon_reference_to_current("sample.png"))
        self.assertFalse(session.current_image_has_changes())

    def test_merge_cif_paths_overrides_conflicting_stems(self) -> None:
        session = WorkspaceSession()
        session.set_cif_index({"a": "/old/a.cif"})
        session.merge_cif_paths({"a": "/new/a.cif", "b": "/q/b.cif"})
        self.assertEqual(session.cif_paths_by_stem["a"], "/new/a.cif")
        self.assertEqual(session.cif_paths_by_stem["b"], "/q/b.cif")

    def test_invalidate_image_states_clears_cache_for_paths(self) -> None:
        session = WorkspaceSession()

        def load_source_image(path: str) -> str:
            return f"src:{path}"

        session.replace_image_selection(["a.png", "b.png"], is_supported_image=lambda _p: True)
        session.load_image("a.png", load_source_image=load_source_image, load_cif_overlay=lambda _path: [])
        session.load_image("b.png", load_source_image=load_source_image, load_cif_overlay=lambda _path: [])
        key_a = str(Path("a.png"))
        key_b = str(Path("b.png"))
        self.assertIn(key_a, session._state_cache)

        session.invalidate_image_states(["a.png"])
        self.assertNotIn(key_a, session._state_cache)
        self.assertIn(key_b, session._state_cache)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from polygon_widget.application.services import WorkspaceSession
from polygon_widget.domain import PolygonData


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


if __name__ == "__main__":
    unittest.main()

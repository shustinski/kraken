"""A recognition result already produced is shown when the file list changes."""

from __future__ import annotations

import unittest

from contour.application.processing_parts.display_settings import BatchImageResult
from contour.application.recognition_display import (
    RecognitionResultCatalog,
    install_produced_recognition_for_switch,
    should_repeat_recognition,
)
from contour.application.services.workspace_session import WorkspaceSession
from contour.domain.polygon import PolygonData


def _polygon() -> PolygonData:
    return PolygonData(id=7, points=[(0, 0), (4, 0), (0, 3)])


class RecognitionFileSwitchTests(unittest.TestCase):
    def test_switching_to_a_recognized_file_shows_its_polygons(self) -> None:
        session = WorkspaceSession()
        session.apply_loaded_frame("a.png", source_image=object(), polygons=[])
        session.apply_loaded_frame("b.png", source_image=object(), polygons=[], make_current=False)

        catalog = RecognitionResultCatalog()
        catalog.remember("b.png", [_polygon()])
        self.assertFalse(should_repeat_recognition(produced=True))
        self.assertTrue(should_repeat_recognition(produced=catalog.has_result("a.png")))

        self.assertTrue(install_produced_recognition_for_switch(session, catalog, "b.png"))
        loaded = session.resolve_cached_load("b.png")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.state.recognition_produced)
        self.assertEqual([7], [polygon.id for polygon in loaded.state.polygons])
        self.assertFalse(should_repeat_recognition(produced=loaded.state.recognition_produced))

    def test_result_for_a_file_that_is_not_current_is_kept_for_the_switch(self) -> None:
        session = WorkspaceSession()
        session.apply_loaded_frame("a.png", source_image=object(), polygons=[])
        polygon = _polygon()
        applied = session.apply_processing_result(
            BatchImageResult(
                image_path="b.png",
                source_image=object(),
                preprocessed_image=object(),
                pipeline_config={"mode": "conductors"},
                mask_image=None,
                polygons=[polygon],
            )
        )
        self.assertFalse(applied)
        self.assertEqual("a.png", session.current_image_path)

        loaded = session.resolve_cached_load("b.png")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.state.recognition_produced)
        self.assertEqual([7], [polygon.id for polygon in loaded.state.polygons])
        self.assertEqual([7], [polygon.id for polygon in loaded.state.recognition_base_polygons])


if __name__ == "__main__":
    unittest.main()

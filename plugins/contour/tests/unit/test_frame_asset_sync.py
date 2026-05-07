"""Unit coverage for basename matching and sidebar status classification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from contour.application.frame_asset_sync import (
    VectorSideListStatus,
    ImageSideListPaintStatus,
    build_image_cif_matching_report,
    background_hex_vector_status,
    background_hex_image_paint_status,
    classify_image_side_paint_status,
    classify_vector_side_status,
    index_cif_file_paths,
)


class FrameAssetSyncTests(unittest.TestCase):
    def test_subset_image_selection_stems_detect_gaps_vs_cifs(self) -> None:
        report = build_image_cif_matching_report(
            ["D:/proj/a.PNG", Path("b.jpg")],
            {"a": "D:/vectors/a.cif", "lonely": "/x/y/lonely.cif"},
        )
        self.assertEqual(report.stems_with_image_but_no_cif, frozenset({"b"}))
        self.assertEqual(report.stems_with_cif_but_no_image, frozenset({"lonely"}))

    def test_index_selected_cif_file_paths_requires_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "one.cif"
            kept_cv = root / "two.cv"
            kept.write_text("placeholder", encoding="utf-8")
            kept_cv.write_text("placeholder", encoding="utf-8")
            missing = root / "gone.cif"
            indexed = index_cif_file_paths([str(kept), str(kept_cv), str(missing)])
        self.assertEqual(indexed, {"one": str(kept.resolve()), "two": str(kept_cv.resolve())})

    def test_vector_status_prioritizes_missing_image_then_error_then_dirty(self) -> None:
        status_no_image = classify_vector_side_status(
            has_matching_image=False,
            cif_load_failed=True,
            image_never_viewed=False,
            polygons_dirty=False,
            persist_highlight=False,
        )
        self.assertEqual(status_no_image, VectorSideListStatus.NO_MATCHING_IMAGE)

        status_error = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=True,
            image_never_viewed=False,
            polygons_dirty=False,
            persist_highlight=True,
        )
        self.assertEqual(status_error, VectorSideListStatus.LOAD_ERROR)

        status_dirty = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=False,
            image_never_viewed=False,
            polygons_dirty=True,
            persist_highlight=True,
        )
        self.assertEqual(status_dirty, VectorSideListStatus.MODIFIED)

    def test_vector_status_saved_vs_viewed_when_clean(self) -> None:
        saved = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=False,
            image_never_viewed=False,
            polygons_dirty=False,
            persist_highlight=True,
        )
        self.assertEqual(saved, VectorSideListStatus.SAVED)
        unseen = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=False,
            image_never_viewed=True,
            polygons_dirty=False,
            persist_highlight=False,
        )
        self.assertEqual(unseen, VectorSideListStatus.UNSEEN)
        viewed = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=False,
            image_never_viewed=False,
            polygons_dirty=False,
            persist_highlight=False,
        )
        self.assertEqual(viewed, VectorSideListStatus.VIEWED)

    def test_vector_unseen_takes_priority_over_persist_highlight(self) -> None:
        """Never-opened frames should not show saved (green) from a stale flag."""

        status = classify_vector_side_status(
            has_matching_image=True,
            cif_load_failed=False,
            image_never_viewed=True,
            polygons_dirty=False,
            persist_highlight=True,
        )
        self.assertEqual(status, VectorSideListStatus.UNSEEN)

    def test_image_paint_status_requires_opened_seen(self) -> None:
        unopened = classify_image_side_paint_status(
            never_opened=True,
            polygons_dirty=False,
            persist_highlight=True,
        )
        self.assertEqual(unopened, ImageSideListPaintStatus.UNOPENED)
        modified = classify_image_side_paint_status(
            never_opened=False,
            polygons_dirty=True,
            persist_highlight=True,
        )
        self.assertEqual(modified, ImageSideListPaintStatus.MODIFIED)
        persisted = classify_image_side_paint_status(
            never_opened=False,
            polygons_dirty=False,
            persist_highlight=True,
        )
        self.assertEqual(persisted, ImageSideListPaintStatus.SAVED)

    def test_sidebar_hex_colors_follow_spec(self) -> None:
        self.assertIsNone(background_hex_vector_status(VectorSideListStatus.UNSEEN))
        self.assertEqual(
            background_hex_vector_status(VectorSideListStatus.MODIFIED),
            background_hex_image_paint_status(ImageSideListPaintStatus.MODIFIED),
        )
        self.assertEqual(background_hex_vector_status(VectorSideListStatus.SAVED), "#1e4a35")
        self.assertEqual(
            background_hex_vector_status(VectorSideListStatus.NO_MATCHING_IMAGE),
            background_hex_vector_status(VectorSideListStatus.LOAD_ERROR),
        )


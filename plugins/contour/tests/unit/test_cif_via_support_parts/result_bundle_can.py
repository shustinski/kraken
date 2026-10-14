"""test_cif_via_support: test result bundle can."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from contour.application.processing import DisplaySettings, SaveOptions
from contour.serializers import (
    save_result_bundle,
)

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon


class ResultBundleCanMixin:
    def test_result_bundle_can_save_cv_without_legacy_text_formats(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved = save_result_bundle(
                root,
                "sample.png",
                [polygon],
                image,
                DisplaySettings(),
                SaveOptions(save_cif=False, save_cv=True, save_preview=True),
            )

            self.assertEqual(set(saved), {"cv", "preview"})
            self.assertTrue(Path(saved["cv"]).exists())
            self.assertNotIn('"metadata"', Path(saved["cv"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(saved["preview"]).exists())
            self.assertFalse((root / "sample.csv").exists())
            self.assertFalse((root / "sample.txt").exists())
            self.assertFalse((root / "sample.svg").exists())

"""test_cif_via_support: test dataset export writes."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from contour.serializers import (
    export_dataset_frame,
)

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon


class DatasetExportWritesMixin:
    def test_dataset_export_writes_image_and_cif_subdirectories(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image_path = root / "frame_1.png"
            cv2.imwrite(str(image_path), image)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved_files = export_dataset_frame(root / "dataset", str(image_path), [polygon], image)

            saved_image = Path(saved_files["image"])
            saved_cif = Path(saved_files["cif"])
            self.assertEqual(saved_image.parent.name, "images")
            self.assertEqual(saved_cif.parent.name, "cif")
            self.assertTrue(saved_image.exists())
            self.assertTrue(saved_cif.exists())
            payload = saved_cif.read_text(encoding="utf-8")
            self.assertIn("( R frame_1.png );", payload)

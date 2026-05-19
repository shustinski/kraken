from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from contour.utils import _imread_unicode_safe, scan_image_files


class UtilsTests(unittest.TestCase):
    def test_imread_unicode_safe_skips_cv2_imread_for_non_ascii_paths(self) -> None:
        image_path = Path("D:/OZI/Нейронка/jpg/sample.jpg")
        decoded = np.zeros((2, 2), dtype=np.uint8)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("contour.utils.cv2.imread") as cv2_imread,
            patch(
                "contour.utils.np.fromfile",
                return_value=np.array([1, 2, 3], dtype=np.uint8),
            ) as np_fromfile,
            patch("contour.utils.cv2.imdecode", return_value=decoded) as cv2_imdecode,
        ):
            result = _imread_unicode_safe(image_path, 1)

        self.assertIs(result, decoded)
        cv2_imread.assert_not_called()
        np_fromfile.assert_called_once_with(str(image_path), dtype=np.uint8)
        cv2_imdecode.assert_called_once()

    def test_imread_unicode_safe_raises_before_cv2_imread_for_missing_file(self) -> None:
        missing_path = Path("missing-image.jpg")

        with patch("contour.utils.cv2.imread") as cv2_imread:
            with self.assertRaises(FileNotFoundError):
                _imread_unicode_safe(missing_path, 1)

        cv2_imread.assert_not_called()

    def test_scan_image_files_skips_underscore_prefixed_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            keep = root / "frame_001.png"
            hidden = root / "_frame_002.png"
            not_image = root / "notes.txt"
            keep.write_bytes(b"")
            hidden.write_bytes(b"")
            not_image.write_text("ignore", encoding="utf-8")

            self.assertEqual(scan_image_files(root), [str(keep)])


if __name__ == "__main__":
    unittest.main()

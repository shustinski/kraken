from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from contour.application.frame_drop import classify_dropped_paths


class FrameDropClassificationTests(unittest.TestCase):
    def test_classifies_images_vectors_and_ignores_other_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jpg = root / "frame_001.jpg"
            cif = root / "frame_001.cif"
            notes = root / "readme.txt"
            hidden = root / "_skip.jpg"
            jpg.write_bytes(b"")
            cif.write_bytes(b"")
            notes.write_text("ignore", encoding="utf-8")
            hidden.write_bytes(b"")

            payload = classify_dropped_paths([jpg, cif, notes, hidden])

            self.assertEqual(payload.image_paths, (str(jpg),))
            self.assertEqual(payload.vector_paths, (str(cif),))

    def test_folder_drop_collects_jpg_and_cif_without_nested_replace_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jpg = root / "frame_002.jpg"
            cif = root / "frame_002.cif"
            jpg.write_bytes(b"")
            cif.write_bytes(b"")

            payload = classify_dropped_paths([root])

            self.assertEqual(payload.image_paths, (str(jpg),))
            self.assertEqual(payload.vector_paths, (str(cif),))
            self.assertFalse(payload.is_empty())

    def test_empty_or_unsupported_payload_is_empty(self) -> None:
        with TemporaryDirectory() as directory:
            notes = Path(directory) / "notes.txt"
            notes.write_text("x", encoding="utf-8")
            payload = classify_dropped_paths([notes])
            self.assertTrue(payload.is_empty())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from contour.application.use_cases.workspace import (
    find_matching_cif_path,
    index_cif_directory,
    load_input_directory,
    normalize_image_selection,
)


class _FakeDirectoryEntry:
    def __init__(self, path: str, *, is_file: bool) -> None:
        normalized_path = Path(path)
        self._path = str(normalized_path)
        self.name = normalized_path.name
        self.stem = normalized_path.stem
        self.suffix = normalized_path.suffix
        self._is_file = is_file

    def is_file(self) -> bool:
        return self._is_file

    def __str__(self) -> str:
        return self._path


class WorkspaceUseCasesTests(unittest.TestCase):
    def test_normalize_image_selection_filters_unsupported_files(self) -> None:
        selected_paths = normalize_image_selection(
            ["./image.png", "./notes.txt", Path("./mask.JPG")],
            is_supported_image=lambda path: Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"},
        )

        self.assertEqual(selected_paths, ["image.png", "mask.JPG"])

    def test_load_input_directory_normalizes_directory_and_scanned_paths(self) -> None:
        calls: list[str] = []

        def scan_images(directory: str) -> list[str]:
            calls.append(directory)
            return [str(Path(directory) / "b.png"), str(Path(directory) / "a.jpg")]

        state = load_input_directory(".\\images", scan_images=scan_images)

        self.assertEqual(calls, ["images"])
        self.assertEqual(state.directory, "images")
        self.assertEqual(state.image_paths, ("images\\b.png", "images\\a.jpg"))

    def test_index_cif_directory_only_collects_cif_files(self) -> None:
        root = Path("images")
        entries = [
            _FakeDirectoryEntry("images/beta.CIF", is_file=True),
            _FakeDirectoryEntry("images/alpha.cif", is_file=True),
            _FakeDirectoryEntry("images/notes.txt", is_file=True),
            _FakeDirectoryEntry("images/nested", is_file=False),
        ]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.iterdir", return_value=entries),
        ):
            state = index_cif_directory(root)

        self.assertTrue(state.available)
        self.assertEqual(
            state.indexed_paths,
            {
                "alpha": "images\\alpha.cif",
                "beta": "images\\beta.CIF",
            },
        )

    def test_find_matching_cif_path_uses_case_insensitive_image_stem(self) -> None:
        indexed_paths = {"sample": "sample.cif"}

        result = find_matching_cif_path("Sample.PNG", indexed_paths)

        self.assertEqual(result, "sample.cif")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from ..dto import CifDirectoryState, InputDirectoryState

VECTOR_FILE_SUFFIXES = frozenset({".cif", ".cv"})


def _normalize_path(path: str | Path) -> str:
    return str(Path(path))


def normalize_image_selection(
    paths: Iterable[str | Path],
    *,
    is_supported_image: Callable[[str | Path], bool],
) -> list[str]:
    return [_normalize_path(path) for path in paths if is_supported_image(path)]


def load_input_directory(
    directory: str | Path,
    *,
    scan_images: Callable[[str], list[str]],
) -> InputDirectoryState:
    normalized_directory = _normalize_path(directory)
    return InputDirectoryState(
        directory=normalized_directory,
        image_paths=tuple(_normalize_path(path) for path in scan_images(normalized_directory)),
    )


def index_cif_directory(directory: str | Path) -> CifDirectoryState:
    normalized_directory = _normalize_path(directory)
    root = Path(normalized_directory)
    if not root.exists() or not root.is_dir():
        return CifDirectoryState(directory=normalized_directory)

    indexed_paths = {
        path.stem.lower(): str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in VECTOR_FILE_SUFFIXES
    }
    return CifDirectoryState(
        directory=normalized_directory,
        indexed_paths=indexed_paths,
        available=True,
    )


def find_matching_cif_path(image_path: str | Path, indexed_paths: Mapping[str, str]) -> str | None:
    image_stem = Path(image_path).stem.lower()
    return indexed_paths.get(image_stem)

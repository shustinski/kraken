"""Show a recognition result that was already produced when the file changes."""

from __future__ import annotations

from pathlib import Path

from ..domain import PolygonData


def _path_key(image_path: str) -> str:
    return str(Path(image_path))


class RecognitionResultCatalog:
    """Polygons produced by recognition, keyed by image path.

    Switching the file list reads this catalog. It does not run recognition again.
    """

    def __init__(self) -> None:
        self._polygons: dict[str, tuple[PolygonData, ...]] = {}

    def remember(self, image_path: str, polygons: list[PolygonData]) -> None:
        self._polygons[_path_key(image_path)] = tuple(polygon.clone() for polygon in polygons)

    def has_result(self, image_path: str) -> bool:
        return _path_key(image_path) in self._polygons

    def polygons_for(self, image_path: str) -> list[PolygonData] | None:
        stored = self._polygons.get(_path_key(image_path))
        if stored is None:
            return None
        return [polygon.clone() for polygon in stored]


def should_repeat_recognition(*, produced: bool) -> bool:
    """A produced result is shown as-is. Recognition runs only when none exists."""

    return not produced


def install_produced_recognition_for_switch(session, catalog: RecognitionResultCatalog, image_path: str) -> bool:
    produced = catalog.polygons_for(image_path)
    if produced is None:
        return False
    return bool(session.install_produced_recognition(image_path, produced))

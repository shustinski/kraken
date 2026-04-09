from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ...domain import PolygonData
from ..processing import BatchImageResult, ImageProcessingState
from ..use_cases.workspace import find_matching_cif_path, normalize_image_selection


@dataclass(frozen=True, slots=True)
class WorkspaceLoadResult:
    image_path: str
    state: ImageProcessingState | None
    cache_hit: bool = False
    reused_current_state: bool = False
    prepared_image_required: bool = False


class WorkspaceSession:
    def __init__(self) -> None:
        self._image_paths: list[str] = []
        self._current_image_path: str | None = None
        self._current_state: ImageProcessingState | None = None
        self._state_cache: dict[str, ImageProcessingState] = {}
        self._cif_paths_by_stem: dict[str, str] = {}

    @property
    def image_paths(self) -> tuple[str, ...]:
        return tuple(self._image_paths)

    @property
    def current_image_path(self) -> str | None:
        return self._current_image_path

    @property
    def current_state(self) -> ImageProcessingState | None:
        return self._current_state

    def replace_image_selection(
        self,
        paths: Iterable[str | Path],
        *,
        is_supported_image: Callable[[str | Path], bool],
    ) -> list[str]:
        self._image_paths = normalize_image_selection(paths, is_supported_image=is_supported_image)
        if not self._image_paths:
            self.clear_current_selection()
        return list(self._image_paths)

    def clear_current_selection(self) -> None:
        self._current_image_path = None
        self._current_state = None

    def set_cif_index(self, indexed_paths: Mapping[str, str]) -> None:
        self._cif_paths_by_stem = dict(indexed_paths)
        self._state_cache.clear()

    def clear_cif_index(self) -> None:
        self.set_cif_index({})

    def resolve_cif_path(self, image_path: str | Path) -> str | None:
        return find_matching_cif_path(image_path, self._cif_paths_by_stem)

    def load_image(
        self,
        path: str | Path,
        *,
        load_source_image: Callable[[str], Any],
        load_cif_overlay: Callable[[str], list[PolygonData]],
    ) -> WorkspaceLoadResult:
        image_path = str(Path(path))
        if self._current_image_path == image_path and self._current_state is not None and image_path in self._state_cache:
            return WorkspaceLoadResult(
                image_path=image_path,
                state=self._current_state,
                reused_current_state=True,
            )

        cached_state = self._state_cache.get(image_path)
        if cached_state is not None:
            self._current_image_path = image_path
            self._current_state = cached_state
            return WorkspaceLoadResult(
                image_path=image_path,
                state=cached_state,
                cache_hit=True,
                prepared_image_required=cached_state.preprocessed_image is None and cached_state.source_image is not None,
            )

        state = ImageProcessingState(
            image_path=image_path,
            source_image=load_source_image(image_path),
            polygons=load_cif_overlay(image_path),
        )
        self._state_cache[image_path] = state
        self._current_image_path = image_path
        self._current_state = state
        return WorkspaceLoadResult(
            image_path=image_path,
            state=state,
            prepared_image_required=state.source_image is not None,
        )

    def store_preprocessed_image(self, image_path: str, preprocessed_image: Any) -> bool:
        state = self._state_cache.get(image_path)
        if state is None:
            return False
        state.preprocessed_image = preprocessed_image
        return self._current_image_path == image_path and self._current_state is state

    def apply_processing_result(self, result: BatchImageResult) -> bool:
        new_state = ImageProcessingState(
            image_path=result.image_path,
            source_image=result.source_image,
            preprocessed_image=result.preprocessed_image,
            mask_image=result.mask_image,
            polygons=result.polygons,
        )
        self._state_cache[result.image_path] = new_state
        if self._current_image_path != result.image_path:
            return False
        self._current_state = new_state
        return True

    def update_current_polygons(self, polygons: list[PolygonData]) -> bool:
        if self._current_state is None or self._current_image_path is None:
            return False
        self._current_state.polygons = polygons
        self._state_cache[self._current_image_path] = self._current_state
        return True

    def current_display_image(self) -> Any | None:
        if self._current_state is None:
            return None
        if self._current_state.preprocessed_image is not None:
            return self._current_state.preprocessed_image
        return self._current_state.source_image

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain import PolygonData
from ..processing import BatchImageResult, ImageProcessingState
from ..use_cases.workspace import find_matching_cif_path, normalize_image_selection


def _norm_path_key(path: str | Path) -> str:
    return str(Path(path))


@dataclass(frozen=True, slots=True)
class WorkspaceLoadResult:
    image_path: str
    state: ImageProcessingState | None
    cache_hit: bool = False
    reused_current_state: bool = False
    prepared_image_required: bool = False
    vectors_only: bool = False


_POINT_SIGNATURE_SCALE = 1_000_000


def _normalize_polygon_points(points: list[tuple[float, float]]) -> tuple[tuple[int, int], ...]:
    scale = _POINT_SIGNATURE_SCALE
    normalized: list[tuple[int, int]] = []
    for x_coord, y_coord in points:
        if isinstance(x_coord, int) and isinstance(y_coord, int):
            normalized.append((x_coord * scale, y_coord * scale))
            continue
        normalized.append((int(round(float(x_coord) * scale)), int(round(float(y_coord) * scale))))
    return tuple(normalized)


def _sortable_optional_int(value: int | None) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    return (1, int(value))


def _polygon_signature(polygon: PolygonData) -> tuple[object, ...]:
    return (
        bool(polygon.is_hole),
        _sortable_optional_int(polygon.parent_id),
        str(polygon.category),
        str(polygon.shape_hint),
        _normalize_polygon_points(polygon.points),
    )


def _polygon_matches(first: PolygonData, second: PolygonData) -> bool:
    return (
        bool(first.is_hole) == bool(second.is_hole)
        and first.parent_id == second.parent_id
        and str(first.category) == str(second.category)
        and str(first.shape_hint) == str(second.shape_hint)
        and first.points == second.points
    )


def _polygons_equal(first: list[PolygonData], second: list[PolygonData]) -> bool:
    if len(first) != len(second):
        return False
    if all(_polygon_matches(left, right) for left, right in zip(first, second, strict=True)):
        return True
    first_signatures = [_polygon_signature(polygon) for polygon in first]
    second_signatures = [_polygon_signature(polygon) for polygon in second]
    first_signatures.sort()
    second_signatures.sort()
    return first_signatures == second_signatures


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

    @property
    def cif_paths_by_stem(self) -> dict[str, str]:
        return dict(self._cif_paths_by_stem)

    def cached_states(self) -> tuple[tuple[str, ImageProcessingState], ...]:
        return tuple((path, state) for path, state in self._state_cache.items())

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

    def clear_project(self) -> None:
        self._image_paths.clear()
        self.clear_current_selection()
        self._state_cache.clear()
        self._cif_paths_by_stem.clear()

    def set_cif_index(self, indexed_paths: Mapping[str, str]) -> None:
        """Update stem → vector path map; clear overlays but keep loaded source pixels."""

        self._cif_paths_by_stem = dict(indexed_paths)
        for state in self._state_cache.values():
            state.polygons = []
            state.loaded_cif_path = None
            state.reference_polygons = []
            state.polygons_dirty = None

    def merge_cif_paths(self, indexed_paths: Mapping[str, str]) -> None:
        """Update stem → CIF mapping; clears cache like :meth:`set_cif_index`."""

        merged = dict(self._cif_paths_by_stem)
        merged.update(dict(indexed_paths))
        self.set_cif_index(merged)

    def clear_cif_index(self) -> None:
        self.set_cif_index({})

    def resolve_cif_path(self, image_path: str | Path) -> str | None:
        return find_matching_cif_path(image_path, self._cif_paths_by_stem)

    def invalidate_image_states(self, image_paths: Iterable[str | Path]) -> None:
        keys = {_norm_path_key(p) for p in image_paths}
        for key in keys:
            self._state_cache.pop(key, None)
        if self._current_image_path in keys:
            self._current_state = None

    def sync_polygon_reference_to_current(self, image_path: str | Path) -> bool:
        """Set reference polygons to match current polygons after a successful save."""

        key = _norm_path_key(image_path)
        state = self._state_cache.get(key)
        if state is None:
            return False
        state.reference_polygons = [polygon.clone() for polygon in state.polygons]
        state.polygons_dirty = False
        return True

    def resolve_cached_load(self, path: str | Path) -> WorkspaceLoadResult | None:
        image_path = str(Path(path))
        if (
            self._current_image_path == image_path
            and self._current_state is not None
            and image_path in self._state_cache
        ):
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
                prepared_image_required=cached_state.preprocessed_image is None
                and cached_state.source_image is not None,
            )
        return None

    def load_image(
        self,
        path: str | Path,
        *,
        load_source_image: Callable[[str], Any],
        load_cif_overlay: Callable[[str], list[PolygonData]],
    ) -> WorkspaceLoadResult:
        image_path = str(Path(path))
        cached = self.resolve_cached_load(image_path)
        if cached is not None:
            return cached

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="contour-load") as executor:
            source_future = executor.submit(load_source_image, image_path)
            cif_future = executor.submit(load_cif_overlay, image_path)
            state = ImageProcessingState(
                image_path=image_path,
                source_image=source_future.result(),
                polygons=cif_future.result(),
            )
        return self._store_loaded_state(image_path, state)

    def apply_loaded_frame(
        self,
        image_path: str | Path,
        *,
        source_image: Any,
        polygons: list[PolygonData],
    ) -> WorkspaceLoadResult:
        path = str(Path(image_path))
        state = ImageProcessingState(
            image_path=path,
            source_image=source_image,
            polygons=list(polygons),
        )
        return self._store_loaded_state(path, state)

    def apply_frame_vectors(
        self,
        image_path: str | Path,
        *,
        polygons: list[PolygonData],
        loaded_cif_path: str | None = None,
    ) -> WorkspaceLoadResult | None:
        path = str(Path(image_path))
        state = self._state_cache.get(path)
        if state is None and self._current_image_path == path and self._current_state is not None:
            state = self._current_state
        if state is None or state.source_image is None:
            return None
        state.polygons = list(polygons)
        if loaded_cif_path is not None:
            state.loaded_cif_path = loaded_cif_path
        self._state_cache[path] = state
        if self._current_image_path == path:
            self._current_state = state
        return WorkspaceLoadResult(
            image_path=path,
            state=state,
            cache_hit=True,
            prepared_image_required=state.preprocessed_image is None,
            vectors_only=True,
        )

    def _store_loaded_state(self, image_path: str, state: ImageProcessingState) -> WorkspaceLoadResult:
        self._state_cache[image_path] = state
        self._current_image_path = image_path
        self._current_state = state
        return WorkspaceLoadResult(
            image_path=image_path,
            state=state,
            prepared_image_required=state.source_image is not None,
        )

    def store_preprocessed_image(
        self,
        image_path: str,
        preprocessed_image: Any,
        pipeline_config: dict[str, Any] | None = None,
    ) -> bool:
        state = self._state_cache.get(image_path)
        if state is None:
            return False
        state.preprocessed_image = preprocessed_image
        state.pipeline_config = None if pipeline_config is None else dict(pipeline_config)
        return self._current_image_path == image_path and self._current_state is state

    def apply_processing_result(self, result: BatchImageResult) -> bool:
        existing_state = self._state_cache.get(result.image_path)
        metal_layers = getattr(result, "metal_overlay_polygons", None) or {}
        new_state = ImageProcessingState(
            image_path=result.image_path,
            source_image=result.source_image,
            preprocessed_image=result.preprocessed_image,
            pipeline_config=None if result.pipeline_config is None else dict(result.pipeline_config),
            mask_image=result.mask_image,
            polygons=result.polygons,
            debug_candidates=list(result.debug_candidates),
            debug_gradient_maps=dict(result.debug_gradient_maps),
            metal_overlay_polygons={k: [p.clone() for p in v] for k, v in metal_layers.items()},
            loaded_cif_path=None if existing_state is None else existing_state.loaded_cif_path,
            reference_polygons=[]
            if existing_state is None
            else [polygon.clone() for polygon in existing_state.reference_polygons],
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
        self._current_state.polygons_dirty = None
        self._state_cache[self._current_image_path] = self._current_state
        return True

    def image_has_changes(self, image_path: str | Path) -> bool:
        state = self._state_cache.get(str(Path(image_path)))
        if state is None:
            return False
        dirty = state.polygons_dirty
        if dirty is None:
            dirty = not _polygons_equal(state.polygons, state.reference_polygons)
            state.polygons_dirty = dirty
        return dirty

    def current_image_has_changes(self) -> bool:
        if self._current_image_path is None:
            return False
        return self.image_has_changes(self._current_image_path)

    def current_display_image(self) -> Any | None:
        if self._current_state is None:
            return None
        if self._current_state.preprocessed_image is not None:
            return self._current_state.preprocessed_image
        return self._current_state.source_image

"""Vertical slice: 3×3 window → pairs → cycles → robust local poses."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

from cartograph.application.nominal import expected_pair_displacement
from cartograph.application.optimize_block import HuberOptimizerSettings, HuberTranslationOptimizer
from cartograph.application.ports import LocalBlockStore, LocalRegistrationCache, TileImageLoader
from cartograph.domain.coordinates import GridCoordinate, NominalCoordinate, Translation2D
from cartograph.domain.errors import RegistrationError
from cartograph.domain.registration import (
    FeaturePairRegistrar,
    PairHint,
    PairRegistrar,
    RegistrationParameters,
    RegistrationResult,
    RegistrationStatus,
)
from cartograph.domain.tiles import ImageBuffer, TileGrid
from cartograph.domain.topology import (
    CycleResidual,
    GraphEdge,
    LocalBlockSolution,
    LocalGraph,
    LocalWindow,
    cycle_measurement_residual,
    diagonal_edges,
    four_neighborhood_edges,
    select_window,
    unit_square_cycles,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalRegistrationRequest:
    grid: TileGrid
    images: Mapping[GridCoordinate, ImageBuffer]
    center: GridCoordinate
    placement: Mapping[GridCoordinate, NominalCoordinate]
    parameters: RegistrationParameters = RegistrationParameters()


@dataclass(frozen=True, slots=True)
class LocalRegistrationOutcome:
    window: LocalWindow
    solution: LocalBlockSolution
    from_cache: bool


class RegisterLocalWindow:
    """Orchestrate the first Cartograph compute unit: a sliding 3×3 block."""

    def __init__(
        self,
        registrar: PairRegistrar,
        *,
        optimizer: HuberTranslationOptimizer | None = None,
        cache: LocalRegistrationCache | None = None,
        store: LocalBlockStore | None = None,
        feature_fallback: FeaturePairRegistrar | None = None,
        image_loader: TileImageLoader | None = None,
    ) -> None:
        self._registrar = registrar
        self._optimizer = optimizer or HuberTranslationOptimizer()
        self._cache = cache
        self._store = store
        self._feature_fallback = feature_fallback
        self._image_loader = image_loader

    def execute(self, request: LocalRegistrationRequest) -> LocalRegistrationOutcome:
        window = select_window(request.grid, request.center)
        if not window.tiles:
            raise RegistrationError(f"no tiles in 3×3 window around {request.center}")
        images = dict(request.images)
        if self._image_loader is not None:
            for coord, tile in window.tiles.items():
                images.setdefault(coord, self._image_loader.load(tile))
        missing = [coord for coord in window.tiles if coord not in images]
        if missing:
            raise RegistrationError(f"missing images for window tiles: {missing}")

        key = parameter_hash(request.grid, window, request.parameters, request.placement)
        cached = self._read_cache(key)
        if cached is not None:
            return LocalRegistrationOutcome(window=window, solution=replace(cached, parameter_hash=key), from_cache=True)

        graph = self._build_graph(window, images, request)
        graph = apply_cycle_validation(graph, request.parameters)
        settings = HuberOptimizerSettings(
            delta_px=request.parameters.huber_delta_px,
            iterations=request.parameters.huber_iterations,
        )
        optimizer = HuberTranslationOptimizer(settings)
        solution = optimizer.optimize(graph)
        solution = replace(solution, parameter_hash=key)
        self._write_cache(key, solution)
        return LocalRegistrationOutcome(window=window, solution=solution, from_cache=False)

    def _build_graph(
        self,
        window: LocalWindow,
        images: Mapping[GridCoordinate, ImageBuffer],
        request: LocalRegistrationRequest,
    ) -> LocalGraph:
        coords = set(window.tiles)
        pairs = list(four_neighborhood_edges(coords))
        if request.parameters.include_diagonals:
            min_overlap = request.parameters.min_diagonal_overlap_px
            tile_w = next(iter(window.tiles.values())).width
            tile_h = next(iter(window.tiles.values())).height
            overlap_px_x = tile_w * request.grid.overlap_x
            overlap_px_y = tile_h * request.grid.overlap_y
            if overlap_px_x >= min_overlap and overlap_px_y >= min_overlap:
                pairs.extend(diagonal_edges(coords))

        edges: list[GraphEdge] = []
        for source, target in pairs:
            hint = PairHint(
                expected=expected_pair_displacement(window.tiles[source], window.tiles[target], request.placement),
                search_radius_px=request.parameters.search_radius_px,
                overlap_margin_px=request.parameters.overlap_margin_px,
            )
            result = self._registrar.register(images[source], images[target], hint)
            result = _with_pair(result, source, target)
            if (
                result.status in {RegistrationStatus.LOW_CONFIDENCE, RegistrationStatus.FAILED}
                and self._feature_fallback is not None
            ):
                fallback = _with_pair(self._feature_fallback.register(images[source], images[target], hint), source, target)
                _LOGGER.info("feature fallback used for %s -> %s status=%s", source, target, fallback.status)
                result = fallback
            weight = result.confidence if result.status is RegistrationStatus.OK else 0.0
            if result.status is RegistrationStatus.LOW_CONFIDENCE:
                weight = 0.25 * result.confidence
            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    measurement=Translation2D(result.transform.dx, result.transform.dy),
                    weight=weight,
                    result=result,
                )
            )
        return LocalGraph(center=request.center, nodes=tuple(sorted(coords)), edges=tuple(edges))

    def _read_cache(self, key: str) -> LocalBlockSolution | None:
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        if self._store is not None:
            stored = self._store.load(key)
            if stored is not None and self._cache is not None:
                self._cache.put(key, stored)
            return stored
        return None

    def _write_cache(self, key: str, solution: LocalBlockSolution) -> None:
        if self._cache is not None:
            self._cache.put(key, solution)
        if self._store is not None:
            self._store.save(key, solution)


def apply_cycle_validation(graph: LocalGraph, parameters: RegistrationParameters) -> LocalGraph:
    measurements = {(edge.source, edge.target): edge.measurement for edge in graph.edges if edge.result.is_usable}
    residuals: list[CycleResidual] = []
    dropped: set[tuple[GridCoordinate, GridCoordinate]] = set()
    updated_edges = list(graph.edges)
    for cycle in unit_square_cycles(set(graph.nodes)):
        residual = cycle_measurement_residual(measurements, cycle)
        if residual is None:
            continue
        excluded: tuple[GridCoordinate, GridCoordinate] | None = None
        if residual > parameters.cycle_residual_threshold_px:
            victim = _lowest_confidence_edge(updated_edges, cycle)
            if victim is not None:
                excluded = (victim.source, victim.target)
                dropped.add(excluded)
                _LOGGER.warning(
                    "cycle residual %.3f px exceeds %.3f; excluding edge %s -> %s",
                    residual,
                    parameters.cycle_residual_threshold_px,
                    victim.source,
                    victim.target,
                )
        residuals.append(CycleResidual(nodes=cycle, residual_px=residual, excluded_edge=excluded))
        for index, edge in enumerate(updated_edges):
            if _edge_in_cycle(edge, cycle):
                updated_edges[index] = edge.with_cycle_residual(residual)

    final_edges: list[GraphEdge] = []
    for edge in updated_edges:
        if (edge.source, edge.target) in dropped:
            result = replace(
                edge.result,
                status=RegistrationStatus.LOW_CONFIDENCE,
                message=edge.result.message or "excluded after high cycle residual",
            )
            final_edges.append(replace(edge, weight=0.0, result=result))
        else:
            final_edges.append(edge)
    return graph.with_edges(tuple(final_edges), tuple(residuals))


def parameter_hash(
    grid: TileGrid,
    window: LocalWindow,
    parameters: RegistrationParameters,
    placement: Mapping[GridCoordinate, NominalCoordinate],
) -> str:
    payload = {
        "grid_name": grid.name,
        "overlap_x": grid.overlap_x,
        "overlap_y": grid.overlap_y,
        "center": {"row": window.center.row, "col": window.center.col},
        "tiles": [
            {
                "row": tile.coord.row,
                "col": tile.coord.col,
                "source_id": tile.source_id,
                "width": tile.width,
                "height": tile.height,
                "stage": None if tile.stage is None else {"x": tile.stage.x, "y": tile.stage.y},
            }
            for tile in sorted(window.tiles.values(), key=lambda item: item.coord)
        ],
        "placement": [
            {"row": coord.row, "col": coord.col, "x": position.x, "y": position.y}
            for coord, position in sorted(placement.items())
            if coord in window.tiles
        ],
        "parameters": parameters.to_fingerprint_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_pair(result: RegistrationResult, source: GridCoordinate, target: GridCoordinate) -> RegistrationResult:
    return replace(result, source=source, target=target)


def _edge_in_cycle(edge: GraphEdge, cycle: tuple[GridCoordinate, ...]) -> bool:
    nodes = set(cycle)
    return edge.source in nodes and edge.target in nodes


def _lowest_confidence_edge(edges: list[GraphEdge], cycle: tuple[GridCoordinate, ...]) -> GraphEdge | None:
    candidates = [edge for edge in edges if _edge_in_cycle(edge, cycle) and edge.weight > 0.0]
    if not candidates:
        return None
    return min(candidates, key=lambda edge: edge.result.confidence)

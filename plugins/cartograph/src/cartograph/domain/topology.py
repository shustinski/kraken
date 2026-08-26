"""Sliding 3×3 compute window and the 12-edge 4-neighborhood local graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from .coordinates import GridCoordinate, Translation2D
from .registration import RegistrationResult, RegistrationStatus
from .tiles import Tile, TileGrid


WINDOW_OFFSETS: tuple[tuple[int, int], ...] = tuple((d_row, d_col) for d_row in (-1, 0, 1) for d_col in (-1, 0, 1))


@dataclass(frozen=True, slots=True)
class LocalWindow:
    """A 3×3 (or smaller at borders) compute window. Not a storage format."""

    center: GridCoordinate
    tiles: Mapping[GridCoordinate, Tile]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", dict(self.tiles))

    def coords(self) -> tuple[GridCoordinate, ...]:
        return tuple(sorted(self.tiles))

    def has_center(self) -> bool:
        return self.center in self.tiles


def select_window(grid: TileGrid, center: GridCoordinate) -> LocalWindow:
    tiles: dict[GridCoordinate, Tile] = {}
    for d_row, d_col in WINDOW_OFFSETS:
        coord = GridCoordinate(center.row + d_row, center.col + d_col) if center.row + d_row >= 0 and center.col + d_col >= 0 else None
        if coord is None:
            continue
        tile = grid.get(coord)
        if tile is not None:
            tiles[coord] = tile
    return LocalWindow(center=center, tiles=tiles)


def four_neighborhood_edges(coords: set[GridCoordinate]) -> tuple[tuple[GridCoordinate, GridCoordinate], ...]:
    """Primary 4-neighborhood edges directed right or down. Full 3×3 yields 12 edges."""

    edges: list[tuple[GridCoordinate, GridCoordinate]] = []
    for coord in sorted(coords):
        right = GridCoordinate(coord.row, coord.col + 1)
        down = GridCoordinate(coord.row + 1, coord.col)
        if right in coords:
            edges.append((coord, right))
        if down in coords:
            edges.append((coord, down))
    return tuple(edges)


def diagonal_edges(coords: set[GridCoordinate]) -> tuple[tuple[GridCoordinate, GridCoordinate], ...]:
    edges: list[tuple[GridCoordinate, GridCoordinate]] = []
    for coord in sorted(coords):
        diag = GridCoordinate(coord.row + 1, coord.col + 1)
        if diag in coords:
            edges.append((coord, diag))
    return tuple(edges)


def unit_square_cycles(coords: set[GridCoordinate]) -> tuple[tuple[GridCoordinate, ...], ...]:
    """A→B→E→D cycles on every filled 2×2 block."""

    cycles: list[tuple[GridCoordinate, ...]] = []
    for origin in sorted(coords):
        b = GridCoordinate(origin.row, origin.col + 1)
        e = GridCoordinate(origin.row + 1, origin.col + 1)
        d = GridCoordinate(origin.row + 1, origin.col)
        if b in coords and e in coords and d in coords:
            cycles.append((origin, b, e, d))
    return tuple(cycles)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: GridCoordinate
    target: GridCoordinate
    measurement: Translation2D
    weight: float
    result: RegistrationResult

    def with_weight(self, weight: float) -> GraphEdge:
        return replace(self, weight=weight)

    def with_cycle_residual(self, residual: float) -> GraphEdge:
        return replace(self, result=replace(self.result, cycle_residual=residual))


@dataclass(frozen=True, slots=True)
class CycleResidual:
    nodes: tuple[GridCoordinate, ...]
    residual_px: float
    excluded_edge: tuple[GridCoordinate, GridCoordinate] | None = None


@dataclass(frozen=True, slots=True)
class LocalGraph:
    center: GridCoordinate
    nodes: tuple[GridCoordinate, ...]
    edges: tuple[GraphEdge, ...]
    cycle_residuals: tuple[CycleResidual, ...] = ()

    def usable_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(edge for edge in self.edges if edge.weight > 0.0 and edge.result.is_usable)

    def with_edges(self, edges: tuple[GraphEdge, ...], cycle_residuals: tuple[CycleResidual, ...]) -> LocalGraph:
        return replace(self, edges=edges, cycle_residuals=cycle_residuals)


@dataclass(frozen=True, slots=True)
class LocalBlockSolution:
    center: GridCoordinate
    poses: Mapping[GridCoordinate, Translation2D]
    graph: LocalGraph
    excluded_edges: tuple[GraphEdge, ...] = ()
    status: RegistrationStatus = RegistrationStatus.OK
    message: str = ""
    parameter_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "poses", dict(self.poses))
        center_pose = self.poses.get(self.center)
        if center_pose is not None and (abs(center_pose.dx) > 1e-9 or abs(center_pose.dy) > 1e-9):
            raise ValueError("center tile pose must be (0, 0)")


def displacement_for_directed_edge(
    measurements: Mapping[tuple[GridCoordinate, GridCoordinate], Translation2D],
    source: GridCoordinate,
    target: GridCoordinate,
) -> Translation2D | None:
    direct = measurements.get((source, target))
    if direct is not None:
        return direct
    reverse = measurements.get((target, source))
    if reverse is not None:
        return reverse.negated()
    return None


def cycle_measurement_residual(
    measurements: Mapping[tuple[GridCoordinate, GridCoordinate], Translation2D],
    cycle: tuple[GridCoordinate, ...],
) -> float | None:
    """Spec cycle A→B→E→D→A: d_AB + d_BE - d_DE - d_AD."""

    if len(cycle) != 4:
        acc_x = 0.0
        acc_y = 0.0
        nodes = cycle + (cycle[0],)
        for source, target in zip(nodes, nodes[1:], strict=True):
            delta = displacement_for_directed_edge(measurements, source, target)
            if delta is None:
                return None
            acc_x += delta.dx
            acc_y += delta.dy
        return (acc_x * acc_x + acc_y * acc_y) ** 0.5

    a, b, e, d = cycle
    d_ab = displacement_for_directed_edge(measurements, a, b)
    d_be = displacement_for_directed_edge(measurements, b, e)
    d_de = displacement_for_directed_edge(measurements, d, e)
    d_ad = displacement_for_directed_edge(measurements, a, d)
    if d_ab is None or d_be is None or d_de is None or d_ad is None:
        return None
    residual = d_ab.plus(d_be).minus(d_de).minus(d_ad)
    return residual.magnitude


class LocalBlockOptimizer(Protocol):
    def optimize(self, graph: LocalGraph) -> LocalBlockSolution:
        """Solve p_j - p_i ≈ d_ij with p_center = (0, 0)."""

"""JSON sidecar persistence for local-block transforms and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.errors import PersistenceError
from cartograph.domain.registration import (
    LocalTransform,
    RegistrationMethod,
    RegistrationResult,
    RegistrationStatus,
    TransformKind,
)
from cartograph.domain.topology import GraphEdge, LocalBlockSolution, LocalGraph, CycleResidual

SCHEMA = "cartograph.local-block.v1"


class JsonLocalBlockStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self._root / f"{key}.json"

    def load(self, key: str) -> LocalBlockSolution | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(f"cannot read local block {path}: {exc}") from exc
        return solution_from_dict(payload)

    def save(self, key: str, solution: LocalBlockSolution) -> None:
        path = self.path_for(key)
        try:
            path.write_text(json.dumps(solution_to_dict(solution), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            raise PersistenceError(f"cannot write local block {path}: {exc}") from exc


class InMemoryRegistrationCache:
    def __init__(self) -> None:
        self._items: dict[str, LocalBlockSolution] = {}

    def get(self, key: str) -> LocalBlockSolution | None:
        return self._items.get(key)

    def put(self, key: str, solution: LocalBlockSolution) -> None:
        self._items[key] = solution


def solution_to_dict(solution: LocalBlockSolution) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "parameter_hash": solution.parameter_hash,
        "status": solution.status.value,
        "message": solution.message,
        "center": _coord(solution.center),
        "poses": [
            {"row": coord.row, "col": coord.col, "dx": pose.dx, "dy": pose.dy}
            for coord, pose in sorted(solution.poses.items())
        ],
        "edges": [_edge(edge) for edge in solution.graph.edges],
        "excluded_edges": [_edge(edge) for edge in solution.excluded_edges],
        "cycle_residuals": [
            {
                "nodes": [_coord(node) for node in item.nodes],
                "residual_px": item.residual_px,
                "excluded_edge": None
                if item.excluded_edge is None
                else {"source": _coord(item.excluded_edge[0]), "target": _coord(item.excluded_edge[1])},
            }
            for item in solution.graph.cycle_residuals
        ],
    }


def solution_from_dict(payload: Mapping[str, Any]) -> LocalBlockSolution:
    schema = str(payload.get("schema", ""))
    if schema != SCHEMA:
        raise PersistenceError(f"unsupported local-block schema: {schema}")
    center = _parse_coord(payload.get("center"))
    poses = {
        GridCoordinate(int(item["row"]), int(item["col"])): Translation2D(float(item["dx"]), float(item["dy"]))
        for item in payload.get("poses", ())
        if isinstance(item, Mapping)
    }
    edges = tuple(_parse_edge(item) for item in payload.get("edges", ()) if isinstance(item, Mapping))
    excluded = tuple(_parse_edge(item) for item in payload.get("excluded_edges", ()) if isinstance(item, Mapping))
    cycles = tuple(_parse_cycle(item) for item in payload.get("cycle_residuals", ()) if isinstance(item, Mapping))
    graph = LocalGraph(center=center, nodes=tuple(sorted(poses)), edges=edges, cycle_residuals=cycles)
    return LocalBlockSolution(
        center=center,
        poses=poses,
        graph=graph,
        excluded_edges=excluded,
        status=RegistrationStatus(str(payload.get("status", "ok"))),
        message=str(payload.get("message", "")),
        parameter_hash=str(payload.get("parameter_hash", "")),
    )


def _coord(coord: GridCoordinate) -> dict[str, int]:
    return {"row": coord.row, "col": coord.col}


def _parse_coord(payload: object) -> GridCoordinate:
    if not isinstance(payload, Mapping):
        raise PersistenceError("coordinate payload must be an object")
    return GridCoordinate(int(payload["row"]), int(payload["col"]))


def _edge(edge: GraphEdge) -> dict[str, Any]:
    result = edge.result
    return {
        "source": _coord(edge.source),
        "target": _coord(edge.target),
        "dx": edge.measurement.dx,
        "dy": edge.measurement.dy,
        "weight": edge.weight,
        "confidence": result.confidence,
        "phase_response": result.phase_response,
        "peak_ratio": result.peak_ratio,
        "raw_zncc": result.raw_zncc,
        "gradient_zncc": result.gradient_zncc,
        "expected_displacement_error": result.expected_displacement_error,
        "cycle_residual": result.cycle_residual,
        "method": result.method.value,
        "status": result.status.value,
        "message": result.message,
    }


def _parse_edge(payload: Mapping[str, Any]) -> GraphEdge:
    source = _parse_coord(payload.get("source"))
    target = _parse_coord(payload.get("target"))
    result = RegistrationResult(
        transform=LocalTransform(TransformKind.TRANSLATION, float(payload.get("dx", 0.0)), float(payload.get("dy", 0.0))),
        confidence=float(payload.get("confidence", 0.0)),
        phase_response=float(payload.get("phase_response", 0.0)),
        peak_ratio=float(payload.get("peak_ratio", 0.0)),
        raw_zncc=float(payload.get("raw_zncc", 0.0)),
        gradient_zncc=float(payload.get("gradient_zncc", 0.0)),
        expected_displacement_error=float(payload.get("expected_displacement_error", 0.0)),
        cycle_residual=None if payload.get("cycle_residual") is None else float(payload["cycle_residual"]),
        method=RegistrationMethod(str(payload.get("method", "phase_correlation"))),
        status=RegistrationStatus(str(payload.get("status", "ok"))),
        message=str(payload.get("message", "")),
        source=source,
        target=target,
    )
    return GraphEdge(
        source=source,
        target=target,
        measurement=Translation2D(float(payload.get("dx", 0.0)), float(payload.get("dy", 0.0))),
        weight=float(payload.get("weight", 0.0)),
        result=result,
    )


def _parse_cycle(payload: Mapping[str, Any]) -> CycleResidual:
    nodes = tuple(_parse_coord(item) for item in payload.get("nodes", ()) if isinstance(item, Mapping))
    raw_excluded = payload.get("excluded_edge")
    excluded = None
    if isinstance(raw_excluded, Mapping):
        excluded = (_parse_coord(raw_excluded.get("source")), _parse_coord(raw_excluded.get("target")))
    return CycleResidual(nodes=nodes, residual_px=float(payload.get("residual_px", 0.0)), excluded_edge=excluded)

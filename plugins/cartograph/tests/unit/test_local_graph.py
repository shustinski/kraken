from __future__ import annotations

import pytest

from cartograph.application.local_registration import apply_cycle_validation
from cartograph.application.optimize_block import HuberOptimizerSettings, HuberTranslationOptimizer
from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.registration import RegistrationMethod, RegistrationParameters, RegistrationStatus, translation_result
from cartograph.domain.topology import GraphEdge, LocalGraph


def _ok_result(dx: float, dy: float, confidence: float = 0.9):
    return translation_result(
        dx,
        dy,
        confidence=confidence,
        phase_response=0.8,
        peak_ratio=4.0,
        raw_zncc=0.9,
        gradient_zncc=0.85,
        expected_displacement_error=0.2,
        status=RegistrationStatus.OK,
        method=RegistrationMethod.PHASE_CORRELATION,
    )


def test_optimizer_keeps_center_at_origin() -> None:
    center = GridCoordinate(1, 1)
    right = GridCoordinate(1, 2)
    graph = LocalGraph(
        center=center,
        nodes=(center, right),
        edges=(
            GraphEdge(center, right, Translation2D(48.0, 0.0), 1.0, _ok_result(48.0, 0.0)),
        ),
    )
    solution = HuberTranslationOptimizer().optimize(graph)
    assert solution.poses[center] == Translation2D(0.0, 0.0)
    assert solution.poses[right].dx == pytest.approx(48.0, abs=1e-6)


def test_wrong_pairwise_edge_does_not_destroy_block() -> None:
    c = GridCoordinate(1, 1)
    a = GridCoordinate(0, 0)
    b = GridCoordinate(0, 1)
    d = GridCoordinate(1, 0)
    true = {
        (a, b): Translation2D(48.0, 0.0),
        (d, c): Translation2D(48.0, 0.0),
        (a, d): Translation2D(0.0, 48.0),
        (b, c): Translation2D(0.0, 48.0),
    }
    edges = [
        GraphEdge(source, target, delta, 0.9, _ok_result(delta.dx, delta.dy, confidence=0.9))
        for (source, target), delta in true.items()
    ]
    edges = [edge for edge in edges if not (edge.source == a and edge.target == b)]
    edges.append(GraphEdge(a, b, Translation2D(80.0, 12.0), 0.35, _ok_result(80.0, 12.0, confidence=0.35)))
    graph = LocalGraph(center=c, nodes=(a, b, c, d), edges=tuple(edges))
    graph = apply_cycle_validation(graph, RegistrationParameters(cycle_residual_threshold_px=2.0))
    solution = HuberTranslationOptimizer(HuberOptimizerSettings(delta_px=1.5, iterations=12)).optimize(graph)
    assert solution.poses[c] == Translation2D(0.0, 0.0)
    assert solution.poses[b].dx == pytest.approx(0.0, abs=2.0)
    assert solution.poses[d].dy == pytest.approx(0.0, abs=2.0)
    assert solution.poses[a].dx == pytest.approx(-48.0, abs=2.0)
    assert solution.poses[a].dy == pytest.approx(-48.0, abs=2.0)
    excluded = {(edge.source, edge.target) for edge in solution.excluded_edges}
    assert (a, b) in excluded or any(edge.weight == 0.0 and edge.source == a and edge.target == b for edge in graph.edges)

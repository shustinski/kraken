from __future__ import annotations

from pathlib import Path

from cartograph.application.optimize_block import HuberTranslationOptimizer
from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.registration import RegistrationMethod, RegistrationStatus, translation_result
from cartograph.domain.topology import GraphEdge, LocalGraph
from cartograph.infrastructure.persistence import JsonLocalBlockStore, solution_from_dict, solution_to_dict


def test_local_block_json_round_trip(tmp_path: Path) -> None:
    center = GridCoordinate(1, 1)
    right = GridCoordinate(1, 2)
    result = translation_result(
        40.0,
        1.0,
        confidence=0.8,
        phase_response=0.6,
        peak_ratio=3.0,
        raw_zncc=0.88,
        gradient_zncc=0.8,
        expected_displacement_error=0.4,
        status=RegistrationStatus.OK,
        method=RegistrationMethod.PHASE_CORRELATION,
        message="ok",
        source=center,
        target=right,
    )
    graph = LocalGraph(
        center=center,
        nodes=(center, right),
        edges=(GraphEdge(center, right, Translation2D(40.0, 1.0), 0.8, result),),
    )
    solution = HuberTranslationOptimizer().optimize(graph)
    store = JsonLocalBlockStore(tmp_path)
    store.save("abc", solution)
    restored = store.load("abc")
    assert restored is not None
    assert restored.center == center
    assert restored.poses[right].dx == solution.poses[right].dx
    assert restored.graph.edges[0].result.raw_zncc == 0.88
    assert solution_from_dict(solution_to_dict(solution)).parameter_hash == solution.parameter_hash

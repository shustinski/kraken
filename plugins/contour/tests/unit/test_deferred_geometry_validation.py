from __future__ import annotations

from pathlib import Path

from contour.adapters.qt.frame_load import GeometryValidationRunnable
from contour.application.services.workspace_session import WorkspaceSession
from contour.domain import PolygonData


def _square(polygon_id: int) -> PolygonData:
    return PolygonData(
        id=polygon_id,
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    )


def test_workspace_stores_prefetched_repair_reasons() -> None:
    session = WorkspaceSession()
    reasons = {1: ["overlapping"]}
    result = session.apply_loaded_frame(
        r"d:\frames\a.png",
        source_image="px",
        polygons=[_square(1)],
        make_current=False,
        polygons_needing_repair=reasons,
    )
    assert result.state is not None
    assert result.state.polygons_needing_repair == {1: ["overlapping"]}
    cached = session.cached_state(r"d:\frames\a.png")
    assert cached is not None
    assert cached.polygons_needing_repair == {1: ["overlapping"]}


def test_geometry_validation_runnable_emits_reasons() -> None:
    polygon = _square(1)
    calls: list[list[PolygonData]] = []

    def _scan(polygons: list[PolygonData]) -> dict[int, list[str]]:
        calls.append(polygons)
        return {1: ["self_intersecting"]}

    runnable = GeometryValidationRunnable(7, r"d:\frames\a.png", [polygon], _scan)
    received: list[tuple[int, str, dict[int, list[str]]]] = []

    def _on_result(req_id: int, path: str, reasons: object) -> None:
        received.append((req_id, path, dict(reasons)))

    runnable.signals.result.connect(_on_result)
    runnable.run()
    assert received == [(7, r"d:\frames\a.png", {1: ["self_intersecting"]})]
    assert len(calls) == 1
    assert calls[0][0].id == 1


def test_stale_geometry_validation_result_is_ignored_by_generation() -> None:
    """Mirrors the generation guard in ``_schedule_deferred_geometry_validation``."""

    current_generation = 3
    current_path = str(Path(r"d:\frames\a.png"))
    offered: list[str] = []
    stored: dict[int, list[str]] | None = None

    def _apply_if_current(req_id: int, path: str, reasons: dict[int, list[str]]) -> None:
        nonlocal stored
        if req_id != current_generation:
            return
        if str(Path(path)) != current_path:
            return
        stored = reasons
        if reasons:
            offered.append(path)

    _apply_if_current(2, current_path, {1: ["overlapping"]})
    assert stored is None
    assert offered == []
    _apply_if_current(3, current_path, {1: ["overlapping"]})
    assert stored == {1: ["overlapping"]}
    assert offered == [current_path]

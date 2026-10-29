"""Spatial reads preserve legacy ancestry and recover disposable cache state."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest
import sqlite3

from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore
from kraken_manager.infrastructure.filesystem import matrix_coordinates
from tests.test_work_orders import _project_with_frames, _service


def _filtered(service, project, layer, representation, bounds):
    return tuple(c for c in service.frame_cells(project.id, layer.id, representation.id)
                 if bounds[0] <= c.x <= bounds[2] and bounds[1] <= c.y <= bounds[3])


def test_viewport_does_not_materialize_the_entire_representation(tmp_path, monkeypatch):
    service = _service(tmp_path)
    _, _, project, layer, images, vectors = _project_with_frames(service, tmp_path / "source")
    expected = _filtered(service, project, layer, vectors, (1, 1, 1, 1))

    def refuse_full_read(*args, **kwargs):
        raise AssertionError("Viewport must select visible rows before decoding models")

    monkeypatch.setattr(SQLiteProjectionStore, "list_artifact_series", refuse_full_read)
    monkeypatch.setattr(SQLiteProjectionStore, "list_active_artifact_versions", refuse_full_read)
    assert service.frame_cells(project.id, layer.id, vectors.id, bounds=(1, 1, 1, 1)) == expected
    view = service.matrix_viewport(project.id, layer_id=layer.id,
        representation_ids=(images.id, vectors.id), x1=1, y1=1, x2=1, y2=1)
    assert len(view["cells"]) == 1 and view["cells"][0]["sha256"] == expected[0].sha256
    overview = service.matrix_viewport(project.id, layer_id=layer.id,
        representation_ids=(images.id, vectors.id), x1=1, y1=1, x2=2, y2=1, lod=1)
    assert sum(c["materialized_count"] for c in overview["aggregates"]) == 2


@pytest.mark.parametrize("parameters", [{}, {"x": True, "y": 1}, {"x": "1", "y": 1},
                                      {"x": 1.0, "y": 1}, {"x": 99, "y": 1}, {"x": 2, "y": 1}])
def test_changed_heads_and_legacy_coordinates_match_full_reads(tmp_path, parameters):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    projection = service._projection(project.id)
    first = service.frame_cells(project.id, layer.id, images.id)[0]
    original = projection.get_artifact_version(first.artifact_version_id)
    assert service.frame_cells(project.id, layer.id, images.id, bounds=(1, 1, 1, 1)) == (first,)
    changed = replace(original, id=str(uuid4()), parent_version_id=original.id, parameters=parameters)
    projection.save_artifact_version(changed, activate=True)
    for bounds in [(1, 1, 1, 1), (2, 1, 2, 1)]:
        assert service.frame_cells(project.id, layer.id, images.id, bounds=bounds) == _filtered(
            service, project, layer, images, bounds)
    projection.save_artifact_version(original, activate=True)
    assert service.frame_cells(project.id, layer.id, images.id, bounds=(1, 1, 1, 1)) == (first,)
    series = projection.get_artifact_series(original.series_id)
    projection.save_artifact_series(replace(series, archived=True))
    assert service.frame_cells(project.id, layer.id, images.id, bounds=(1, 1, 1, 1)) == ()


def test_duplicate_legacy_frame_winner_outside_region_suppresses_external_fallback(tmp_path, monkeypatch):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    projection = service._projection(project.id)
    original_cell = service.frame_cells(project.id, layer.id, images.id)[0]
    original = projection.get_artifact_version(original_cell.artifact_version_id)
    series = projection.get_artifact_series(original.series_id)
    legacy = replace(series, id="ffffffff-ffff-ffff-ffff-ffffffffffff", name="legacy duplicate")
    projection.save_artifact_series(legacy)
    projection.save_artifact_version(replace(original, id=str(uuid4()), series_id=legacy.id,
        parameters={"x": 2, "y": 1}), activate=True)
    projection.save_representation(replace(images, source=str(tmp_path / "external")))
    assert _filtered(service, project, layer, images, (1, 1, 1, 1)) == ()
    assert service.frame_cells(project.id, layer.id, images.id, bounds=(1, 1, 1, 1)) == ()

    def refuse_fallback(*args, **kwargs):
        raise AssertionError("Existing materialized rows elsewhere must suppress directory fallback")

    monkeypatch.setattr(service, "_external_directory_viewport", refuse_fallback)
    assert service.matrix_viewport(project.id, layer_id=layer.id, representation_ids=(images.id,),
        x1=1, y1=1, x2=1, y2=1)["cells"] == ()


def test_missing_spatial_table_and_entire_index_are_rebuilt_from_events(tmp_path):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    bounds = (1, 1, 2, 1)
    expected = service.frame_cells(project.id, layer.id, images.id, bounds=bounds)
    projection = service._projection(project.id)
    with projection._connect() as connection:
        connection.execute("DROP TABLE matrix_coordinates")
    assert service.frame_cells(project.id, layer.id, images.id, bounds=bounds) == expected
    projection.destroy_cache()
    assert service.frame_cells(project.id, layer.id, images.id, bounds=bounds) == expected


def test_two_readers_rebuild_one_stale_cache_without_partial_results(tmp_path):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    projection = service._projection(project.id)
    expected = projection.matrix_frame_rows(project.id, layer.id, images.id, (1, 1, 2, 1))
    with projection._connect() as connection:
        connection.execute("DELETE FROM index_metadata WHERE key = 'matrix_coordinates_v1'")
    start = Barrier(2)

    def read():
        start.wait(timeout=5)
        return projection.matrix_frame_rows(project.id, layer.id, images.id, (1, 1, 2, 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read) for _ in range(2)]
        assert all(future.result(timeout=10) == expected for future in futures)


@pytest.mark.parametrize("cycle", [False, True])
def test_missing_or_cyclic_parent_chain_does_not_invent_coordinates(tmp_path, cycle):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    projection = service._projection(project.id)
    first = service.frame_cells(project.id, layer.id, images.id)[0]
    original = projection.get_artifact_version(first.artifact_version_id)
    head_id, parent_id = str(uuid4()), str(uuid4())
    head = replace(original, id=head_id, parent_version_id=parent_id, parameters={})
    projection.save_artifact_version(head, activate=True)
    if cycle:
        projection.save_artifact_version(replace(original, id=parent_id, parent_version_id=head_id,
                                                parameters={}), activate=False)
    assert _filtered(service, project, layer, images, (1, 1, 1, 1)) == ()
    assert service.frame_cells(project.id, layer.id, images.id, bounds=(1, 1, 1, 1)) == ()


def test_failed_rebuild_rolls_back_and_next_read_recovers(tmp_path, monkeypatch):
    service = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    projection = service._projection(project.id)
    expected = projection.matrix_frame_rows(project.id, layer.id, images.id, (1, 1, 2, 1))
    with projection._connect() as connection:
        matrix_coordinates.invalidate(connection)
    with monkeypatch.context() as patch:
        patch.setattr(matrix_coordinates, "_REBUILD", "SELECT * FROM injected_missing_table")
        with pytest.raises(sqlite3.OperationalError, match="injected_missing_table"):
            projection.matrix_frame_rows(project.id, layer.id, images.id, (1, 1, 2, 1))
    assert projection.matrix_frame_rows(project.id, layer.id, images.id, (1, 1, 2, 1)) == expected

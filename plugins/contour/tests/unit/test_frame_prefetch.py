from __future__ import annotations

from pathlib import Path

from contour.application.frame_prefetch import neighborhood_indices
from contour.application.services import WorkspaceSession


def test_neighborhood_indices_include_two_list_neighbors() -> None:
    assert neighborhood_indices(4, 10, list_radius=2) == (4, 3, 5, 2, 6)


def test_neighborhood_indices_clamp_at_edges() -> None:
    assert neighborhood_indices(0, 5, list_radius=2) == (0, 1, 2)
    assert neighborhood_indices(4, 5, list_radius=2) == (4, 3, 2)


def test_neighborhood_indices_add_same_column_matrix_rows() -> None:
    assert neighborhood_indices(
        4,
        12,
        list_radius=2,
        matrix_enabled=True,
        columns=3,
        matrix_row_radius=2,
    ) == (4, 3, 5, 2, 6, 1, 7, 10)


def test_neighborhood_indices_skip_matrix_rows_when_disabled() -> None:
    assert neighborhood_indices(4, 12, matrix_enabled=False, columns=3) == (4, 3, 5, 2, 6)


def test_apply_loaded_frame_without_make_current_is_a_cache_hit_source() -> None:
    session = WorkspaceSession()
    session.replace_image_selection(["a.png", "b.png"], is_supported_image=lambda _path: True)
    session.apply_loaded_frame("a.png", source_image="pixels-a", polygons=[], make_current=True)
    session.apply_loaded_frame("b.png", source_image="pixels-b", polygons=[], make_current=False)

    assert session.current_image_path == str(Path("a.png"))
    assert session.has_cached_source("b.png")


def test_needs_vector_overlay_when_source_cached_but_cif_not_loaded() -> None:
    session = WorkspaceSession()
    session.replace_image_selection(["a.png"], is_supported_image=lambda _path: True)
    session.set_cif_index({"a": "/vectors/a.cif"})
    session.apply_loaded_frame("a.png", source_image="pixels", polygons=[], make_current=True)

    assert session.needs_vector_overlay("a.png")

    session.apply_frame_vectors("a.png", polygons=[], loaded_cif_path="/vectors/a.cif")
    assert not session.needs_vector_overlay("a.png")


def test_needs_vector_overlay_false_when_cleared_or_no_cif() -> None:
    session = WorkspaceSession()
    session.replace_image_selection(["a.png", "b.png"], is_supported_image=lambda _path: True)
    session.set_cif_index({"a": "/vectors/a.cif"})
    session.apply_loaded_frame("a.png", source_image="pixels-a", polygons=[], make_current=True)
    session.apply_loaded_frame("b.png", source_image="pixels-b", polygons=[], make_current=False)

    session.mark_vectors_cleared("a.png")
    assert not session.needs_vector_overlay("a.png")
    assert not session.needs_vector_overlay("b.png")


def test_widget_prefetch_enqueues_vectors_only_for_cached_source() -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    try:
        paths = [str(Path(f"frame_{index:02d}.png")) for index in range(5)]
        widget._workspace.replace_image_selection(paths, is_supported_image=lambda _path: True)
        widget._set_image_list_paths(paths)
        widget._workspace.set_cif_index(
            {Path(path).stem.lower(): f"{Path(path).stem}.cif" for path in paths}
        )
        widget._workspace.apply_loaded_frame(paths[2], source_image="px-current", polygons=[], make_current=True)
        for path in (paths[1], paths[3]):
            widget._workspace.apply_loaded_frame(path, source_image=f"px-{path}", polygons=[], make_current=False)

        enqueued: list[tuple[str, bool]] = []

        def _enqueue(image_path: str, generation: int, *, vectors_only: bool = False, **_kwargs) -> None:
            enqueued.append((str(Path(image_path)), bool(vectors_only)))

        widget._enqueue_prefetch_frame = _enqueue  # type: ignore[method-assign]
        widget._prefetch_frame_neighborhood()

        by_path = dict(enqueued)
        assert by_path[paths[1]] is True
        assert by_path[paths[3]] is True
        assert paths[2] not in by_path
        assert by_path[paths[0]] is False
        assert by_path[paths[4]] is False
    finally:
        widget.close()
        widget.deleteLater()


def test_widget_prefetch_paths_follow_list_and_matrix_neighbors() -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    try:
        paths = [f"frame_{index:02d}.png" for index in range(12)]
        widget._workspace.replace_image_selection(paths, is_supported_image=lambda _path: True)
        widget._set_image_list_paths([str(Path(path)) for path in paths])
        widget._workspace.apply_loaded_frame(
            str(Path(paths[4])), source_image="px", polygons=[], make_current=True
        )
        widget.show_frame_matrix_checkbox.setChecked(True)
        widget.neighbor_columns_spin.setValue(3)
        expected = tuple(str(Path(paths[index])) for index in (4, 3, 5, 2, 6, 1, 7, 10))
        assert widget._prefetch_neighborhood_paths() == expected
        widget.show_frame_matrix_checkbox.setChecked(False)
        expected_list = tuple(str(Path(paths[index])) for index in (4, 3, 5, 2, 6))
        assert widget._prefetch_neighborhood_paths() == expected_list
    finally:
        widget.close()
        widget.deleteLater()

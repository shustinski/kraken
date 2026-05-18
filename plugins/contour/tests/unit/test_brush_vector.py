"""Regression tests for vector brush strokes and booleans."""

from __future__ import annotations

from unittest.mock import patch

from shapely import unary_union
from shapely.geometry import Point

from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    apply_boolean,
    brush_stroke_geometry,
    densify_chain_with_new_vertex,
    filled_polygon_geometry,
    polygon_equivalent_preserved,
    region_geometry,
    shapely_to_polygon_data_list,
    tool_geometry,
)


def test_capsule_between_two_centers_positive_area() -> None:
    g = brush_stroke_geometry([(0.0, 0.0), (40.0, 0.0)], 16.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    assert getattr(g, "area", 0) > 1.0


def test_dense_multi_point_continuous_stroke() -> None:
    pts = [(0.0, 0.0), (120.0, 0.0), (120.0, 80.0)]

    mono = brush_stroke_geometry(pts, 14.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    pair_u = unary_union([brush_stroke_geometry([pts[0], pts[1]], 14.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)])
    merged = unary_union([pair_u, brush_stroke_geometry([pts[1], pts[2]], 14.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)])

    assert abs(float(mono.area) - float(merged.area)) <= max(25.0, 0.04 * float(mono.area))


def test_freehand_stroke_not_collapsed_to_start_end_line() -> None:
    points = [(0.0, 0.0), (80.0, 60.0), (160.0, 0.0)]
    freehand = brush_stroke_geometry(points, 16.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    straight = brush_stroke_geometry([points[0], points[-1]], 16.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)

    # A bent stroke must have a larger filled footprint than the direct chord.
    assert float(freehand.area) > float(straight.area) * 1.10


def test_sparse_points_stroke_fills_gap_with_capsule_geometry() -> None:
    sparse = brush_stroke_geometry([(0.0, 0.0), (300.0, 0.0)], 20.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)

    # Midpoint must be inside the swept capsule footprint.
    assert sparse.buffer(1e-7).contains(Point(150.0, 0.0))


def test_straight_brush_stroke_does_not_emit_collinear_side_vertices() -> None:
    stroke = brush_stroke_geometry([(0.0, 0.0), (300.0, 0.0)], 20.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    polys = shapely_to_polygon_data_list(stroke)
    assert len(polys) == 1

    top_side = [point for point in polys[0].points if abs(point[1] - 10.0) < 1e-6 and 0.0 <= point[0] <= 300.0]
    bottom_side = [point for point in polys[0].points if abs(point[1] + 10.0) < 1e-6 and 0.0 <= point[0] <= 300.0]

    assert len(top_side) == 2
    assert len(bottom_side) == 2


def test_densify_chain_keeps_initial_point_for_short_first_move() -> None:
    chain = [(40.0, 40.0)]
    out = densify_chain_with_new_vertex(chain, (40.05, 40.0), max_segment_length=6.0)

    assert len(out) >= 2
    assert out[0] == (40.0, 40.0)
    assert out[-1] == (40.05, 40.0)


def test_add_circle_one_point_equals_disk() -> None:
    blob = brush_stroke_geometry([(50.0, 50.0)], 20.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)

    circle_two = brush_stroke_geometry([(50.0, 50.0), (50.0, 50.0)], 20.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    symmetric = unary_union(blob.symmetric_difference(circle_two)).area

    assert symmetric <= 1e-6


def test_erase_circle_difference_smaller_than_frame() -> None:
    sq = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    outline = filled_polygon_geometry(sq)
    hole_tool = brush_stroke_geometry([(100.0, 100.0)], 36.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    carve, msg = apply_boolean(outline, hole_tool, subtract=True)

    assert msg is None

    carved_area = float(carve.area) if carve is not None else 0.0

    assert carved_area > 180.0 * 180.0
    assert carved_area < float(outline.area)


def test_difference_creates_hole_in_annulus_approximation() -> None:
    closed_ring = [(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0), (30.0, 30.0)]
    thick_outline = brush_stroke_geometry(closed_ring, 10.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    polygons_data = shapely_to_polygon_data_list(thick_outline)
    hole_items_flagged_true = [p for p in polygons_data if bool(p.is_hole)]
    assert hole_items_flagged_true


def test_many_extractions_keep_outer_outline() -> None:
    boundary = [(0.0, 0.0), (240.0, 0.0), (240.0, 240.0), (0.0, 240.0)]
    polygon_dict: dict[int, PolygonData] = {}
    perimeter_area, perimeter_len, perimeter_bbox = compute_polygon_metrics(boundary)
    polygon_dict[1] = PolygonData(
        id=1,
        points=[(float(x), float(y)) for x, y in boundary],
        area=perimeter_area,
        perimeter=perimeter_len,
        bbox=perimeter_bbox,
        is_hole=False,
        parent_id=None,
    )

    region_shape = region_geometry(polygon_dict, [1])

    chipped = unary_union(region_shape)
    offsets_x_y = [(40 + 18 * k, 40.0) for k in range(9)]
    offsets_x_y += [(120.0, 40 + 18 * k) for k in range(9)]
    offsets_x_y += [(120.0 + 22.0 * k, 210.0) for k in range(4)]

    for cx, cy in offsets_x_y:
        disk = tool_geometry([(cx, cy)], 8.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        nxt, fault = apply_boolean(chipped, disk, subtract=True)
        assert fault is None
        chipped = unary_union(nxt)

    chipped_area_est = float(chipped.area)
    boundary_area_square = perimeter_area if perimeter_area > 100.0 else 240.0 * 240.0

    assert chipped_area_est >= 0.35 * boundary_area_square


def test_nested_subtree_difference_does_not_raise() -> None:
    perimeter_big, perimeter_len_big, perimeter_bbox_big = compute_polygon_metrics(
        [(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)]
    )

    aperture_hole, aperture_len_h, aperture_bbox_h = compute_polygon_metrics(
        [(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)]
    )

    isle_solid_area, island_len_ring, bbox_island_outer = compute_polygon_metrics(
        [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]
    )

    collection: dict[int, PolygonData] = {
        1: PolygonData(
            id=1,
            points=[(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)],
            area=float(perimeter_big),
            perimeter=perimeter_len_big,
            bbox=perimeter_bbox_big,
            parent_id=None,
            is_hole=False,
        ),
        2: PolygonData(
            id=2,
            points=[(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)],
            area=float(aperture_hole),
            perimeter=aperture_len_h,
            bbox=aperture_bbox_h,
            parent_id=1,
            is_hole=True,
        ),
        3: PolygonData(
            id=3,
            points=[(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)],
            area=float(isle_solid_area),
            perimeter=island_len_ring,
            bbox=bbox_island_outer,
            parent_id=2,
            is_hole=False,
        ),
    }

    fused = region_geometry(collection, list(collection))
    incision = tool_geometry([(100.0, 100.0)], 22.0, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
    cut_out, glitch = apply_boolean(fused, incision, subtract=True)

    assert glitch is None
    assert cut_out is not None
    assert not cut_out.is_empty


def test_preserved_overlap_flags_symmetric_difference_tiny() -> None:

    verts_hole = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    area_polygon, peri_loop, perimeter_bbox_loop = compute_polygon_metrics(verts_hole)

    subject = PolygonData(
        id=9,
        points=list(verts_hole),
        is_hole=False,
        parent_id=None,
        area=area_polygon,
        perimeter=peri_loop,
        bbox=perimeter_bbox_loop,
    )

    phantom = subject.clone()

    assert polygon_equivalent_preserved(subject, [phantom]) is True


def test_invalid_boolean_result_keeps_polygon_unchanged() -> None:

    def fake_boolean(_base: object, _brush: object, *, subtract: bool = False):

        del subtract
        _ = (_base, _brush)
        return None, "simulated_topology_error"

    with patch("contour.graphics.editor_scene.apply_boolean", fake_boolean):
        from contour.graphics.editor_scene import PolygonEditorScene

        scene = PolygonEditorScene()
        frame = PolygonData(
            id=1,
            points=[(50.0, 50.0), (160.0, 50.0), (160.0, 160.0), (50.0, 160.0)],
            area=compute_polygon_metrics([(50.0, 50.0), (160.0, 50.0), (160.0, 160.0), (50.0, 160.0)])[0],
            perimeter=compute_polygon_metrics(
                [(50.0, 50.0), (160.0, 50.0), (160.0, 160.0), (50.0, 160.0)]
            )[1],
            bbox=compute_polygon_metrics([(50.0, 50.0), (160.0, 50.0), (160.0, 160.0), (50.0, 160.0)])[2],
            parent_id=None,
            is_hole=False,
        )
        scene.set_polygons([frame])
        polygons_before_scene = scene.get_polygons()

        scene.add_brush_stroke([(100.0, 100.0), (155.0, 100.0)], thickness=44.0, erase=False)

        polygons_after_scene = scene.get_polygons()

    assert polygons_before_scene[0].points == polygons_after_scene[0].points

"""Vector geometry for brush strokes: octagonal caps, booleans, and polygon conversion."""

from __future__ import annotations

from math import cos, hypot, radians, sin

from shapely import BufferCapStyle, BufferJoinStyle, make_valid, unary_union
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from ..application.polygon_antialiasing import _simplify_closed_ring
from ..domain import PolygonData, compute_polygon_metrics, integer_points
from ..domain.polygon_ring import collapse_redundant_polyline_vertices

QUAD_SEGS_BRUSH_DEFAULT = 8
BRUSH_RING_SIMPLIFY_EPSILON_PX = 1.0


def densify_polyline(points: list[tuple[float, float]], max_segment_length: float) -> list[tuple[float, float]]:
    """Insert intermediate vertices so consecutive points are never farther than ``max_segment_length``."""
    if len(points) < 2 or max_segment_length <= 1e-9:
        return list(points)
    out = [points[0]]
    for b in points[1:]:
        ax, ay = out[-1]
        bx, by = b
        distance = hypot(bx - ax, by - ay)
        if distance <= 1e-12:
            continue
        steps = max(1, int(distance / max_segment_length))
        ux, uy = (bx - ax) / distance, (by - ay) / distance
        for step in range(1, steps):
            fraction = step / steps
            out.append((ax + ux * fraction * distance, ay + uy * fraction * distance))
        out.append(b)
    return out


def densify_chain_with_new_vertex(
    chain: list[tuple[float, float]], new_vertex: tuple[float, float], max_segment_length: float
) -> list[tuple[float, float]]:
    if not chain:
        return [new_vertex]
    span = densify_polyline([chain[-1], new_vertex], max_segment_length=max_segment_length)
    # Keep the existing stroke tail and append only new points from the latest segment.
    return [*chain, *span[1:]]


def capsule_shape_between_two_points(ax: float, ay: float, bx: float, by: float, diameter: float) -> BaseGeometry:
    """Octagonal-capped stroke between two centres."""

    return brush_stroke_geometry([(ax, ay), (bx, by)], diameter, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)


def _octagon_points(center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (cx + cos(radians(angle)) * radius, cy + sin(radians(angle)) * radius)
        for angle in (0, 45, 90, 135, 180, 225, 270, 315)
    ]


def _half_octagon_cap(
    center: tuple[float, float], direction: tuple[float, float], radius: float
) -> list[tuple[float, float]]:
    cx, cy = center
    dx, dy = direction
    length = hypot(dx, dy)
    if length <= 1e-12:
        return _octagon_points(center, radius)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    return [
        (cx + nx * radius, cy + ny * radius),
        (cx + (ux + nx) * radius * 0.7071067811865476, cy + (uy + ny) * radius * 0.7071067811865476),
        (cx + ux * radius, cy + uy * radius),
        (cx + (ux - nx) * radius * 0.7071067811865476, cy + (uy - ny) * radius * 0.7071067811865476),
        (cx - nx * radius, cy - ny * radius),
    ]


def _stroke_endpoint_caps(cleaned: list[tuple[float, float]], radius: float) -> list[Polygon]:
    if len(cleaned) < 2 or cleaned[0] == cleaned[-1]:
        return []
    start = cleaned[0]
    next_point = cleaned[1]
    prev_point = cleaned[-2]
    end = cleaned[-1]
    start_dir = (start[0] - next_point[0], start[1] - next_point[1])
    end_dir = (end[0] - prev_point[0], end[1] - prev_point[1])
    return [
        Polygon(_half_octagon_cap(start, start_dir, radius)),
        Polygon(_half_octagon_cap(end, end_dir, radius)),
    ]


def brush_stroke_geometry(points: list[tuple[float, float]], diameter: float, *, quad_segs: int) -> BaseGeometry:
    del quad_segs
    radius = max(float(diameter) / 2.0, 0.5)

    cleaned: list[tuple[float, float]] = []
    for x_coord, y_coord in points:
        if cleaned and hypot(x_coord - cleaned[-1][0], y_coord - cleaned[-1][1]) < 1e-9:
            continue
        cleaned.append((float(x_coord), float(y_coord)))

    if not cleaned:
        return Polygon()

    if len(cleaned) == 1:
        gp = Polygon(_octagon_points(cleaned[0], radius))
        return unary_union(make_valid(gp))

    gp = LineString(cleaned).buffer(
        radius,
        quad_segs=1,
        cap_style=BufferCapStyle.flat,
        join_style=BufferJoinStyle.bevel,
    )
    return unary_union([make_valid(part) for part in (gp, *_stroke_endpoint_caps(cleaned, radius))])


def polygon_paint_footprint_geom(
    points: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> BaseGeometry:
    """Filled footprint for editor booleans, aligned with winding-fill display on CIF keyholes."""

    shell = _usable_ring_coords(points)
    if not shell:
        return Polygon()
    usable_holes = [hole for hole in (_usable_ring_coords(hole_points) for hole_points in holes or []) if hole]
    try:
        base = Polygon(shell, usable_holes) if usable_holes else Polygon(shell)
    except ValueError:
        return Polygon()
    if base.is_empty:
        return Polygon()
  # buffer(0) keeps self-intersecting CIF rings solid like Qt WindingFill; make_valid would carve holes.
    return unary_union(make_valid(base.buffer(0)))


def filled_polygon_geometry(points: list[tuple[float, float]]) -> BaseGeometry:
    if len(points) < 3:
        return Polygon()
    return polygon_paint_footprint_geom(points)


def tool_geometry(points: list[tuple[float, float]], thickness: float | None, *, quad_segs: int) -> BaseGeometry:
    if thickness is None:
        return filled_polygon_geometry(points)
    return brush_stroke_geometry(points, float(thickness), quad_segs=quad_segs)


def ring_coords_xyz(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    if points[0] == points[-1]:
        return list(points[:-1])
    return list(points)


def _dedupe_consecutive_ring_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coords:
        return []
    deduped: list[tuple[float, float]] = [coords[0]]
    for x_coord, y_coord in coords[1:]:
        last_x, last_y = deduped[-1]
        if hypot(float(x_coord) - last_x, float(y_coord) - last_y) < 1e-9:
            continue
        deduped.append((float(x_coord), float(y_coord)))
    return deduped


def _ring_signed_area(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for index, (x_coord, y_coord) in enumerate(coords):
        next_x, next_y = coords[(index + 1) % len(coords)]
        area += float(x_coord) * float(next_y) - float(next_x) * float(y_coord)
    return area * 0.5


def _usable_ring_coords(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    coords = _dedupe_consecutive_ring_coords(ring_coords_xyz(points))
    distinct = {(float(x_coord), float(y_coord)) for x_coord, y_coord in coords}
    if len(distinct) < 3:
        return []
    if abs(_ring_signed_area(coords)) <= 1e-6:
        return []
    return coords


def ring_is_valid_for_polygon(points: list[tuple[float, float]]) -> bool:
    return bool(_usable_ring_coords(points))


def _polygon_from_points(
    points: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> Polygon:
    shell = _usable_ring_coords(points)
    if not shell:
        return Polygon()

    usable_holes = [hole for hole in (_usable_ring_coords(hole_points) for hole_points in holes or []) if hole]
    try:
        return Polygon(shell, usable_holes)
    except ValueError:
        return Polygon()


def _subtree_geometry(
    node_id: int, polygons_by_id: dict[int, PolygonData], subset: frozenset[int], *, quad_segs: int
) -> BaseGeometry:
    poly = polygons_by_id[node_id]
    child_ids = sorted(cid for cid in subset if polygons_by_id[cid].parent_id == node_id)

    interior_rings: list[list[tuple[float, float]]] = []
    extra_parts: list[BaseGeometry] = []
    for cid in child_ids:
        cp = polygons_by_id[cid]
        if cp.is_hole:
            usable_hole = _usable_ring_coords(cp.points)
            if usable_hole:
                interior_rings.append(usable_hole)
            for nested_id in sorted(k for k in subset if polygons_by_id[k].parent_id == cid):
                nested_poly = polygons_by_id[nested_id]
                if nested_poly.is_hole:
                    continue
                part_nested = _subtree_geometry(nested_id, polygons_by_id, subset, quad_segs=quad_segs)
                if not part_nested.is_empty:
                    extra_parts.append(part_nested)
        else:
            part = _subtree_geometry(cid, polygons_by_id, subset, quad_segs=quad_segs)
            if not part.is_empty:
                extra_parts.append(part)

    if poly.is_hole:
        return unary_union(make_valid(_polygon_from_points(poly.points)))

    hull = polygon_paint_footprint_geom(poly.points, interior_rings)
    if not extra_parts:
        return unary_union(make_valid(hull))

    merged = unary_union([hull] + extra_parts)
    return unary_union(make_valid(merged))


def region_geometry(polygons_by_id: dict[int, PolygonData], polygon_ids_subset: list[int]) -> BaseGeometry:
    subset = frozenset(polygon_ids_subset)
    root_ids = sorted(
        pid
        for pid in subset
        if polygons_by_id[pid].parent_id is None or polygons_by_id[pid].parent_id not in subset
    )
    parts: list[BaseGeometry] = []
    for rid in root_ids:
        geo = _subtree_geometry(rid, polygons_by_id, subset, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        if not geo.is_empty:
            parts.append(geo)
    combined = unary_union(parts) if parts else Polygon()
    return unary_union(make_valid(combined))


def extract_polygonal_union(geom: BaseGeometry) -> BaseGeometry:
    """Drop points/lines and keep only polygon / multipolygon parts."""
    geom = unary_union(make_valid(geom))
    if geom.is_empty:
        return Polygon()

    stack: list[BaseGeometry] = [geom]
    fragments: list[BaseGeometry] = []
    while stack:
        g = unary_union(make_valid(stack.pop()))
        if g.is_empty:
            continue
        gt = getattr(g, "geom_type", "")
        if gt == "Polygon":
            fragments.append(g)
        elif gt == "MultiPolygon":
            fragments.extend(list(g.geoms))
        elif gt == "GeometryCollection":
            stack.extend(list(g.geoms))
        else:
            continue

    return unary_union(fragments) if fragments else Polygon()


def apply_boolean(
    base: BaseGeometry,
    tool: BaseGeometry,
    *,
    subtract: bool,
) -> tuple[BaseGeometry | None, str | None]:
    try:
        base_mv = unary_union(make_valid(base))
        tool_mv = unary_union(make_valid(tool))

        if base_mv.is_empty and not subtract:
            out = unary_union(make_valid(tool_mv))
        elif base_mv.is_empty:
            out = Polygon()
        elif subtract:
            out = unary_union(make_valid(base_mv.difference(tool_mv)))
        else:
            out = unary_union(make_valid(base_mv.union(tool_mv)))

        out = extract_polygonal_union(out)
        out = unary_union(make_valid(out))

        if not out.is_empty and not out.is_valid:
            try:
                from shapely.validation import explain_validity

                expl = explain_validity(out)
            except Exception:
                expl = "invalid geometry after boolean"

            repaired = extract_polygonal_union(unary_union(make_valid(out.buffer(0))))
            repaired = unary_union(make_valid(repaired))
            if repaired.is_empty or not repaired.is_valid:
                return None, expl
            out = repaired

        msg = geometry_validation_message(out)
        if msg is not None and not out.is_empty:
            return None, msg

        return out, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def geometry_validation_message(geom: BaseGeometry) -> str | None:
    if geom is None or geom.is_empty:
        return None

    stripped = unary_union(make_valid(extract_polygonal_union(geom)))
    if stripped.is_empty:
        return None

    if not stripped.is_valid:
        try:
            from shapely.validation import explain_validity

            return explain_validity(stripped)
        except Exception:
            return "geometry is not valid"

    gt = getattr(stripped, "geom_type", "")
    if gt not in ("Polygon", "MultiPolygon"):
        return f"unsupported result type after boolean: {gt}"
    return None


def _simplify_committed_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    simplified = _simplify_closed_ring(coords, BRUSH_RING_SIMPLIFY_EPSILON_PX)
    if len(simplified) < 3:
        return coords
    collapsed = collapse_redundant_polyline_vertices(
        integer_points(simplified),
        closed=True,
        min_vertices=3,
    )
    return collapsed if len(collapsed) >= 3 else coords


def simplify_polygonal_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Douglas–Peucker simplify tool rings without remeshing an existing boolean union."""

    polygons = shapely_to_polygon_data_list(geom)
    if not polygons:
        return Polygon()
    for polygon in polygons:
        polygon.points = _simplify_committed_ring(polygon.points)
        area, perimeter, bbox = compute_polygon_metrics(polygon.points)
        polygon.area = float(area)
        polygon.perimeter = float(perimeter)
        polygon.bbox = bbox
    by_id = {polygon.id: polygon for polygon in polygons}
    return region_geometry(by_id, [polygon.id for polygon in polygons])


def shapely_to_polygon_data_list(result: BaseGeometry) -> list[PolygonData]:
    polygons_out: list[PolygonData] = []
    geom = unary_union(make_valid(extract_polygonal_union(result)))
    if geom.is_empty:
        return []

    counter = {"next": 1}

    def allocate_id() -> int:
        value = counter["next"]
        counter["next"] += 1
        return value

    def push_polygon(poly: Polygon) -> None:
        coords = integer_points([(float(x), float(y)) for x, y in poly.exterior.coords[:-1]])
        coords = collapse_redundant_polyline_vertices(coords, closed=True, min_vertices=3)
        if not ring_is_valid_for_polygon(coords):
            return

        exterior_id = allocate_id()
        area, perimeter, bbox = compute_polygon_metrics(coords)

        polygons_out.append(
            PolygonData(
                id=exterior_id,
                points=coords,
                is_hole=False,
                parent_id=None,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )

        for interior_ring in poly.interiors:
            hcoords = integer_points([(float(x), float(y)) for x, y in interior_ring.coords[:-1]])
            hcoords = collapse_redundant_polyline_vertices(hcoords, closed=True, min_vertices=3)
            if not ring_is_valid_for_polygon(hcoords):
                continue

            hid = allocate_id()

            hole_area, perimeter, bbox_metrics = compute_polygon_metrics(hcoords)
            polygons_out.append(
                PolygonData(
                    id=hid,
                    points=hcoords,
                    is_hole=True,
                    parent_id=exterior_id,
                    area=hole_area,
                    perimeter=perimeter,
                    bbox=bbox_metrics,
                )
            )

    def walk(g: BaseGeometry) -> None:
        geom_u = unary_union(make_valid(extract_polygonal_union(g)))

        geom_u_type = getattr(geom_u, "geom_type", "")
        if geom_u_type == "Polygon":
            push_polygon(geom_u)
            return
        if geom_u_type == "MultiPolygon":
            for p in geom_u.geoms:
                push_polygon(p)
            return

        if geom_u_type != "GeometryCollection":
            return
        for child in getattr(geom_u, "geoms", ()):
            walk(child)

    walk(geom)

    polygons_out.sort(key=lambda polygon: polygon.id)
    return polygons_out


def polygon_equivalent_preserved(poly: PolygonData, preserved_polygons: list[PolygonData]) -> bool:
    """Detect rebuilt polygons that geometrically duplicate an untouched preserved polygon."""

    try:
        g = unary_union(make_valid(_polygon_from_points(poly.points).buffer(0)))
    except Exception:
        return False
    if g.is_empty:
        return False

    for preserved in preserved_polygons:
        try:
            h = unary_union(make_valid(_polygon_from_points(preserved.points).buffer(0)))
        except Exception:
            continue

        if h.is_empty:
            continue

        if poly.is_hole != preserved.is_hole:
            continue

        symmetric = unary_union(make_valid(g.symmetric_difference(h)))

        symmetric_area_sym = float(symmetric.area)

        denominator = max(1e-9, float(min(g.area, h.area)))

        tolerance = max(4.0, denominator * 1e-4)
        if symmetric_area_sym <= tolerance:
            return True

    return False


def bbox_intersects_geom_bounds(tool_bounds: tuple[float, float, float, float], polygon_bbox: tuple[int, int, int, int]) -> bool:

    minimum_x, minimum_y, maximum_x, maximum_y = tool_bounds

    padded_left, padded_top, padded_width, padded_height = polygon_bbox

    padded_right_coord = padded_left + padded_width
    padded_bottom_coord = padded_top + padded_height
    disjoint_x_coord = padded_right_coord < minimum_x or padded_left > maximum_x

    disjoint_y_coord = padded_bottom_coord < minimum_y or padded_top > maximum_y

    return not disjoint_x_coord and not disjoint_y_coord


def polygon_footprint_geom(polygon_points: list[tuple[float, float]]) -> BaseGeometry:
    return polygon_paint_footprint_geom(polygon_points)

"""Topology checks for closed/open polylines (no Qt / OpenCV dependency)."""

from __future__ import annotations

# --- simple polygon (editor & extraction): no self-intersection, at most one edge per
# vertex position on the ring (no "four edges" meeting in one point from a self-touching ring). ---

_POINT_EQ_EPS = 1e-5
_POINT_EQ_EPS_SQ = _POINT_EQ_EPS * _POINT_EQ_EPS
_SEG_EPS = 1e-7
# Dense mask-extracted rings can have thousands of vertices; full O(n²) checks are too slow.
TOPOLOGY_CHECK_MAX_VERTICES = 192


def _point_equal(a: tuple[float, float], b: tuple[float, float]) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy < _POINT_EQ_EPS_SQ


def _orient2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val > _SEG_EPS:
        return 1
    if val < -_SEG_EPS:
        return -1
    return 0


def _on_segment2(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
    if _orient2d(a, b, p) != 0:
        return False
    return (min(a[0], b[0]) - _SEG_EPS <= p[0] <= max(a[0], b[0]) + _SEG_EPS) and (
        min(a[1], b[1]) - _SEG_EPS <= p[1] <= max(a[1], b[1]) + _SEG_EPS
    )


def _only_single_shared_segment_endpoint(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    m = 0
    for u in (a, b):
        for v in (c, d):
            if _point_equal(u, v):
                m += 1
    return m == 1


def _segment_forbidden_for_simple_polygon(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """True if segments [ab] and [cd] violate a simple non-self-touching edge set."""
    if _only_single_shared_segment_endpoint(a, b, c, d):
        return False
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if o1 and o2 and o1 != o2 and o3 and o4 and o3 != o4:
        return True
    if o1 == 0 and _on_segment2(c, a, b):
        return not (_point_equal(c, a) or _point_equal(c, b))
    if o2 == 0 and _on_segment2(d, a, b):
        return not (_point_equal(d, a) or _point_equal(d, b))
    if o3 == 0 and _on_segment2(a, c, d):
        return not (_point_equal(a, c) or _point_equal(a, d))
    if o4 == 0 and _on_segment2(b, c, d):
        return not (_point_equal(b, c) or _point_equal(b, d))
    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        overlap_x = max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) + _SEG_EPS
        overlap_y = max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1])) + _SEG_EPS
        return overlap_x and overlap_y
    return False


def _closed_polygon_edges_share_vertex(num_vertices: int, ei: int, ej: int) -> bool:
    ai, aj = ei, (ei + 1) % num_vertices
    bi, bj = ej, (ej + 1) % num_vertices
    verts_a = {ai, aj}
    verts_b = {bi, bj}
    return bool(verts_a & verts_b)


def _normalized_closed_points_and_index(
    points: list[tuple[float, float]], vertex_index: int
) -> tuple[list[tuple[float, float]], int] | None:
    if vertex_index < 0 or vertex_index >= len(points):
        return None
    if len(points) > 3 and _point_equal(points[0], points[-1]):
        normalized = list(points[:-1])
        normalized_index = 0 if vertex_index == len(points) - 1 else vertex_index
        if normalized_index >= len(normalized):
            return None
        return normalized, normalized_index
    return list(points), vertex_index


def closed_ring_has_repeated_vertex(points: list[tuple[float, float]]) -> bool:
    """True when a vertex repeats and is not only the closing duplicate of a ring."""

    if len(points) < 2:
        return False
    open_points = list(points)
    if len(open_points) >= 2 and _point_equal(open_points[0], open_points[-1]):
        open_points = open_points[:-1]
    seen: set[tuple[int, int]] = set()
    inv_eps = 1.0 / _POINT_EQ_EPS
    for x_coord, y_coord in open_points:
        key = (int(round(float(x_coord) * inv_eps)), int(round(float(y_coord) * inv_eps)))
        if key in seen:
            return True
        seen.add(key)
    return False


def closed_ring_description_is_invalid(points: list[tuple[float, float]]) -> bool:
    """True for a truly invalid outline (e.g. bowtie), not a CIF keyhole bridge.

    CIF keyholes use a repeated vertex as a cut-line bridge; that is valid CIF and
    still reported by ``closed_ring_description_invalid_reason`` as ``repeated_vertex``
    for diagnostics / explicit keyhole repair, but is not treated as invalid here.
    """

    reason = closed_ring_description_invalid_reason(points)
    return reason is not None and reason != "repeated_vertex"


def closed_ring_description_invalid_reason(points: list[tuple[float, float]]) -> str | None:
    """Return a stable reason code, or ``None`` when the outline is a simple ring.

    Codes:
    - ``repeated_vertex`` — a non-closing vertex repeats (typical CIF keyhole bridge;
      diagnostic / explicit repair only; not an invalid description by itself)
    - ``self_intersecting`` — edges cross or the ring self-touches without a repeated vertex
    """

    if len(points) < 3:
        return None
    if closed_ring_has_repeated_vertex(points):
        return "repeated_vertex"
    if not is_valid_closed_polygon_ring(points):
        return "self_intersecting"
    return None


def is_valid_closed_polygon_ring(points: list[tuple[float, float]]) -> bool:
    """Closed polygon with edges (i, i+1 mod n); reject self-intersection / self-touch and
    duplicate vertices that are not consecutive on the ring."""
    n = len(points)
    if n < 3:
        return True
    if n > TOPOLOGY_CHECK_MAX_VERTICES:
        # GEOS uses a sweep-line topology check and remains practical for
        # mask-derived rings with thousands of vertices.
        from shapely.geometry import Polygon

        polygon = Polygon(points)
        return bool(polygon.is_valid and not polygon.is_empty and polygon.area > _SEG_EPS)
    inv_eps = 1.0 / _POINT_EQ_EPS
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (x_coord, y_coord) in enumerate(points):
        key = (int(round(x_coord * inv_eps)), int(round(y_coord * inv_eps)))
        for other_index in buckets.get(key, ()):
            if _point_equal(points[index], points[other_index]):
                if other_index + 1 != index and not (other_index == 0 and index == n - 1):
                    return False
        buckets.setdefault(key, []).append(index)
    for ei in range(n):
        a, b = points[ei], points[(ei + 1) % n]
        for ej in range(ei + 1, n):
            if _closed_polygon_edges_share_vertex(n, ei, ej):
                continue
            c, d = points[ej], points[(ej + 1) % n]
            if _segment_forbidden_for_simple_polygon(a, b, c, d):
                return False
    return True


def is_valid_closed_polygon_vertex_move(points: list[tuple[float, float]], vertex_index: int) -> bool:
    """Validate only the topology touched by one moved closed-ring vertex."""

    normalized = _normalized_closed_points_and_index(points, vertex_index)
    if normalized is None:
        return False
    pts, idx = normalized
    n = len(pts)
    if n < 3:
        return True

    moved = pts[idx]
    prev_idx = (idx - 1) % n
    next_idx = (idx + 1) % n
    for other_idx, other in enumerate(pts):
        if other_idx in (idx, prev_idx, next_idx):
            continue
        if _point_equal(moved, other):
            return False

    touched_edges = {prev_idx, idx}
    for edge_index in touched_edges:
        a, b = pts[edge_index], pts[(edge_index + 1) % n]
        for other_edge_index in range(n):
            if other_edge_index in touched_edges:
                continue
            if _closed_polygon_edges_share_vertex(n, edge_index, other_edge_index):
                continue
            c, d = pts[other_edge_index], pts[(other_edge_index + 1) % n]
            if _segment_forbidden_for_simple_polygon(a, b, c, d):
                return False
    return True


def is_valid_closed_polygon_edge_move(points: list[tuple[float, float]], edge_index: int) -> bool:
    """Validate only the topology touched by one translated closed-ring edge."""

    if edge_index < 0 or edge_index >= len(points):
        return False
    n = len(points)
    if n < 3:
        return True
    start_idx = edge_index
    end_idx = (edge_index + 1) % n
    moved_vertices = {start_idx, end_idx}
    if n > 3 and _point_equal(points[0], points[-1]):
        if 0 in moved_vertices or (n - 1) in moved_vertices:
            moved_vertices.update({0, n - 1})
    for other_idx, other in enumerate(points):
        if other_idx in moved_vertices:
            continue
        for moved_idx in moved_vertices:
            if _point_equal(points[moved_idx], other):
                return False

    touched_edges = {(start_idx - 1) % n, edge_index, end_idx}
    for touched_edge_index in touched_edges:
        a, b = points[touched_edge_index], points[(touched_edge_index + 1) % n]
        for other_edge_index in range(n):
            if other_edge_index in touched_edges:
                continue
            if _closed_polygon_edges_share_vertex(n, touched_edge_index, other_edge_index):
                continue
            c, d = points[other_edge_index], points[(other_edge_index + 1) % n]
            if _segment_forbidden_for_simple_polygon(a, b, c, d):
                return False
    return True


def _three_share_axis(
    prev_point: tuple[float, float],
    current_point: tuple[float, float],
    next_point: tuple[float, float],
) -> bool:
    same_x = (
        abs(prev_point[0] - current_point[0]) < _POINT_EQ_EPS
        and abs(current_point[0] - next_point[0]) < _POINT_EQ_EPS
    )
    same_y = (
        abs(prev_point[1] - current_point[1]) < _POINT_EQ_EPS
        and abs(current_point[1] - next_point[1]) < _POINT_EQ_EPS
    )
    return same_x or same_y


def _dedupe_consecutive_polyline_vertices(
    points: list[tuple[float, float]],
    *,
    closed: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []
    cleaned: list[tuple[float, float]] = [points[0]]
    for point in points[1:]:
        if not _point_equal(cleaned[-1], point):
            cleaned.append(point)
    if closed and len(cleaned) >= 2 and _point_equal(cleaned[0], cleaned[-1]):
        cleaned.pop()
    return cleaned


def collapse_redundant_polyline_vertices(
    points: list[tuple[float, float]],
    *,
    closed: bool = True,
    min_vertices: int = 3,
) -> list[tuple[float, float]]:
    """Drop consecutive duplicates and axis-aligned extra vertices.

    Two neighbors must not share both X and Y. Three neighbors must not share
    the same X or the same Y, which would leave a point that does not change
    the axis-aligned outline.
    """

    pts = _dedupe_consecutive_polyline_vertices(points, closed=closed)
    floor = max(0, int(min_vertices))
    while len(pts) > floor:
        kill: int | None = None
        if closed:
            count = len(pts)
            if count < 3:
                break
            for index in range(count):
                prev_point = pts[(index - 1) % count]
                current_point = pts[index]
                next_point = pts[(index + 1) % count]
                if _point_equal(prev_point, current_point) or _three_share_axis(
                    prev_point, current_point, next_point
                ):
                    kill = index
                    break
        else:
            if len(pts) < 3:
                break
            for index in range(1, len(pts) - 1):
                if _three_share_axis(pts[index - 1], pts[index], pts[index + 1]):
                    kill = index
                    break
        if kill is None:
            break
        pts.pop(kill)
        pts = _dedupe_consecutive_polyline_vertices(pts, closed=closed)
    return pts


def is_valid_open_polyline_last_edge(points: list[tuple[float, float]]) -> bool:
    """After appending the last point, the new edge must not cross earlier non-adjacent edges."""
    m = len(points)
    if m < 3:
        return True
    a, b = points[-2], points[-1]
    for i in range(0, m - 3):
        c, d = points[i], points[i + 1]
        if _segment_forbidden_for_simple_polygon(a, b, c, d):
            return False
    return True

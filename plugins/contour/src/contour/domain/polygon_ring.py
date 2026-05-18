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
        return not (
            _point_equal(a, c) or _point_equal(a, d) or _point_equal(b, c) or _point_equal(b, d)
        )
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


def is_valid_closed_polygon_ring(points: list[tuple[float, float]]) -> bool:
    """Closed polygon with edges (i, i+1 mod n); reject self-intersection / self-touch and
    duplicate vertices that are not consecutive on the ring."""
    n = len(points)
    if n < 3:
        return True
    if n > TOPOLOGY_CHECK_MAX_VERTICES:
        return True
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

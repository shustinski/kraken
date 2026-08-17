from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from neuralimage.targets.basic import _to_binary_mask, generate_skeleton_map


def _interior_valid(shape: tuple[int, int], border_ignore: int) -> np.ndarray:
    valid = np.ones(shape, dtype=np.float32)
    border = min(max(0, int(border_ignore)), max(0, min(shape) // 2))
    if border:
        valid[:border, :] = valid[-border:, :] = 0.0
        valid[:, :border] = valid[:, -border:] = 0.0
    return valid


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U, kernel)


def _polygon_vertices(mask: np.ndarray) -> np.ndarray:
    binary = _to_binary_mask(mask)
    vertices = np.zeros_like(binary, dtype=np.float32)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, max(0.75, min(3.0, perimeter * 0.005)), True)
        for x, y in approximation.reshape(-1, 2):
            if 0 <= int(y) < vertices.shape[0] and 0 <= int(x) < vertices.shape[1]:
                vertices[int(y), int(x)] = 1.0
    return vertices


def generate_vertex_map(mask: np.ndarray, *, border_ignore: int = 3) -> np.ndarray:
    vertices = _polygon_vertices(mask)
    return (vertices * _interior_valid(vertices.shape, border_ignore)).astype(np.float32)


def generate_corner_heatmap(mask: np.ndarray, *, sigma: float = 1.5, border_ignore: int = 3) -> np.ndarray:
    corner_map = generate_vertex_map(mask, border_ignore=border_ignore)
    if sigma > 0.0 and bool(corner_map.any()):
        corner_map = cv2.GaussianBlur(corner_map, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        peak = float(corner_map.max())
        if peak > 0.0:
            corner_map /= peak
    return corner_map.astype(np.float32)


def generate_endpoint_map(mask: np.ndarray, *, border_ignore: int = 3) -> np.ndarray:
    skeleton = (generate_skeleton_map(mask) > 0.5).astype(np.uint8)
    endpoints = ((_neighbour_count(skeleton) == 1) & (skeleton > 0)).astype(np.float32)
    return (endpoints * _interior_valid(endpoints.shape, border_ignore)).astype(np.float32)


def _junction_branch_angles(
    skeleton: np.ndarray,
    component: np.ndarray,
    *,
    center_xy: tuple[float, float],
) -> list[float]:
    component_mask = np.zeros_like(skeleton, dtype=np.uint8)
    component_mask[component[:, 0], component[:, 1]] = 1
    ring = cv2.dilate(component_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    branch_pixels = np.column_stack(np.where((ring > 0) & (skeleton > 0) & (component_mask == 0)))
    center_x, center_y = center_xy
    angles: list[float] = []
    for y, x in branch_pixels.tolist():
        angle = math.atan2(float(y) - center_y, float(x) - center_x)
        if all(abs(math.atan2(math.sin(angle - old), math.cos(angle - old))) > math.radians(25) for old in angles):
            angles.append(angle)
    return angles


def _classify_junction(angles: list[float]) -> int | None:
    if len(angles) >= 4:
        return 1  # X
    if len(angles) < 3:
        return None
    for index, first in enumerate(angles):
        for second in angles[index + 1:]:
            separation = abs(math.atan2(math.sin(first - second), math.cos(first - second)))
            if separation >= math.radians(150):
                return 0  # T
    return 2  # Y


def generate_junction_map(
    mask: np.ndarray,
    *,
    min_degree: int = 3,
    border_ignore: int = 3,
) -> np.ndarray:
    skeleton = (generate_skeleton_map(mask) > 0.5).astype(np.uint8)
    result = np.zeros((*skeleton.shape, 3), dtype=np.float32)
    candidates = ((_neighbour_count(skeleton) >= max(3, int(min_degree))) & (skeleton > 0)).astype(np.uint8)
    component_count, component_labels = cv2.connectedComponents(candidates, connectivity=8)
    valid = _interior_valid(skeleton.shape, border_ignore)
    for component_id in range(1, int(component_count)):
        points = np.column_stack(np.where(component_labels == component_id))
        if points.size == 0:
            continue
        center_y, center_x = points.mean(axis=0)
        x, y = int(round(float(center_x))), int(round(float(center_y)))
        if not (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]) or valid[y, x] <= 0.0:
            continue
        junction_class = _classify_junction(
            _junction_branch_angles(skeleton, points, center_xy=(float(center_x), float(center_y)))
        )
        if junction_class is not None:
            result[y, x, junction_class] = 1.0
    return result


def _local_axial_tangent(skeleton: np.ndarray, *, radius: int = 5) -> np.ndarray:
    height, width = skeleton.shape
    field = np.zeros((height, width, 2), dtype=np.float32)
    radius = max(1, int(radius))
    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        top, bottom = max(0, y - radius), min(height, y + radius + 1)
        left, right = max(0, x - radius), min(width, x + radius + 1)
        local_y, local_x = np.where(skeleton[top:bottom, left:right] > 0)
        if local_x.size < 2:
            continue
        points = np.column_stack((local_x + left - x, local_y + top - y)).astype(np.float32)
        eigenvalues, eigenvectors = np.linalg.eigh(points.T @ points)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        theta = math.atan2(float(direction[1]), float(direction[0]))
        field[y, x] = (math.cos(2.0 * theta), math.sin(2.0 * theta))
    return field


def generate_tangent_field(mask: np.ndarray, *, radius: int = 5) -> np.ndarray:
    return _local_axial_tangent((generate_skeleton_map(mask) > 0.5).astype(np.uint8), radius=radius)


def generate_orientation_field(mask: np.ndarray, *, bins: int = 0, radius: int = 5) -> np.ndarray:
    del bins  # Compatibility argument; axial vectors replace discontinuous angle bins.
    binary = _to_binary_mask(mask)
    skeleton = (generate_skeleton_map(binary) > 0.5).astype(np.uint8)
    tangent = _local_axial_tangent(skeleton, radius=radius)
    if not bool(skeleton.any()):
        return tangent
    nearest_indices = distance_transform_edt(
        skeleton == 0,
        return_distances=False,
        return_indices=True,
    )
    propagated = tangent[tuple(nearest_indices)]
    return (propagated * binary[..., None]).astype(np.float32)


def _skeleton_neighbours(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    y, x = point
    neighbours: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            candidate = (y + dy, x + dx)
            if not (dy or dx) or candidate not in pixels:
                continue
            # An 8-connected right-angle corner contains a redundant diagonal
            # edge. Keeping it turns the corner into a three-node junction and
            # prevents branch tracing from measuring its curvature. Preserve
            # diagonal-only conductors, but remove the diagonal when an
            # orthogonal route between the same pixels is present.
            if dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels):
                continue
            neighbours.append(candidate)
    return neighbours


def _trace_skeleton_paths(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    pixels = set(zip(*np.where(skeleton > 0)))
    neighbours = {point: _skeleton_neighbours(point, pixels) for point in pixels}
    critical = {point for point, adjacent in neighbours.items() if len(adjacent) != 2}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def trace(start: tuple[int, int], following: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, following]
        previous, current = start, following
        visited_edges.add(frozenset((previous, current)))
        while current not in critical:
            candidates = [point for point in neighbours[current] if point != previous]
            if not candidates:
                break
            next_point = candidates[0]
            edge = frozenset((current, next_point))
            if edge in visited_edges:
                break
            path.append(next_point)
            visited_edges.add(edge)
            previous, current = current, next_point
        return path

    for start in sorted(critical):
        for following in neighbours[start]:
            if frozenset((start, following)) not in visited_edges:
                paths.append(trace(start, following))

    # Closed loops have no degree!=2 point. Start each remaining edge once.
    for start in sorted(pixels):
        for following in neighbours[start]:
            if frozenset((start, following)) not in visited_edges:
                paths.append(trace(start, following))
    return paths


def generate_curvature_map(mask: np.ndarray, *, radius: int = 5) -> np.ndarray:
    skeleton = (generate_skeleton_map(mask) > 0.5).astype(np.uint8)
    curvature = np.zeros_like(skeleton, dtype=np.float32)
    lookahead = max(1, int(radius) // 2)
    for path in _trace_skeleton_paths(skeleton):
        if len(path) < (2 * lookahead + 1):
            continue
        points = np.asarray([(x, y) for y, x in path], dtype=np.float32)
        for index in range(lookahead, len(path) - lookahead):
            incoming = points[index] - points[index - lookahead]
            outgoing = points[index + lookahead] - points[index]
            incoming_norm = float(np.linalg.norm(incoming))
            outgoing_norm = float(np.linalg.norm(outgoing))
            if incoming_norm <= 1e-6 or outgoing_norm <= 1e-6:
                continue
            cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
            turn = math.acos(min(1.0, max(-1.0, cosine))) / math.pi
            y, x = path[index]
            curvature[y, x] = max(curvature[y, x], float(turn))
    return curvature.astype(np.float32)


def generate_topology_preservation_map(mask: np.ndarray) -> np.ndarray:
    binary = _to_binary_mask(mask)
    return np.stack((generate_skeleton_map(binary), generate_skeleton_map(1 - binary)), axis=-1).astype(np.float32)

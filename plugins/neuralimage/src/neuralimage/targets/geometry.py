from __future__ import annotations

import cv2
import numpy as np

from neuralimage.targets.basic import _to_binary_mask, generate_skeleton_map


def _normalize_vectors(field: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.sum(field ** 2, axis=-1, keepdims=True))
    magnitude = np.maximum(magnitude, 1e-6)
    return (field / magnitude).astype(np.float32)


def generate_vertex_map(mask: np.ndarray) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    if not bool(skeleton.any()):
        return skeleton
    skeleton_u8 = (skeleton > 0.5).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_u8, cv2.CV_8U, kernel)
    vertices = ((neighbor_count >= 12) & (skeleton_u8 > 0)).astype(np.float32)
    return vertices


def generate_corner_heatmap(mask: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    binary = _to_binary_mask(mask)
    if not bool(binary.any()):
        return np.zeros_like(binary, dtype=np.float32)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_map = np.zeros_like(binary, dtype=np.float32)
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for point in approx.reshape(-1, 2):
            x, y = int(point[0]), int(point[1])
            if 0 <= y < corner_map.shape[0] and 0 <= x < corner_map.shape[1]:
                corner_map[y, x] = 1.0
    if sigma > 0.0 and bool(corner_map.any()):
        corner_map = cv2.GaussianBlur(corner_map, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        if corner_map.max() > 0.0:
            corner_map = corner_map / corner_map.max()
    return corner_map.astype(np.float32)


def generate_endpoint_map(mask: np.ndarray) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    if not bool(skeleton.any()):
        return skeleton
    skeleton_u8 = (skeleton > 0.5).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_u8, cv2.CV_8U, kernel)
    endpoints = ((neighbor_count == 1) & (skeleton_u8 > 0)).astype(np.float32)
    return endpoints


def generate_junction_map(mask: np.ndarray, *, min_degree: int = 3) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    if not bool(skeleton.any()):
        return skeleton
    skeleton_u8 = (skeleton > 0.5).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_u8, cv2.CV_8U, kernel)
    junctions = ((neighbor_count >= int(min_degree)) & (skeleton_u8 > 0)).astype(np.float32)
    return junctions


def generate_orientation_field(mask: np.ndarray, *, bins: int = 36) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    height, width = skeleton.shape
    orientation = np.zeros((height, width), dtype=np.float32)
    if not bool(skeleton.any()):
        return orientation
    gx = cv2.Sobel(skeleton, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skeleton, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx)
    angle = np.mod(angle, np.pi)
    if bins > 0:
        bin_width = np.pi / float(bins)
        orientation = np.floor(angle / bin_width).astype(np.float32) / float(bins)
    else:
        orientation = (angle / np.pi).astype(np.float32)
    orientation *= skeleton
    return orientation.astype(np.float32)


def generate_tangent_field(mask: np.ndarray) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    height, width = skeleton.shape
    tangent = np.zeros((height, width, 2), dtype=np.float32)
    if not bool(skeleton.any()):
        return tangent
    gx = cv2.Sobel(skeleton, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skeleton, cv2.CV_32F, 0, 1, ksize=3)
    field = np.stack([gx, gy], axis=-1)
    field = _normalize_vectors(field)
    field *= skeleton[..., None]
    return field.astype(np.float32)


def generate_curvature_map(mask: np.ndarray) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    if not bool(skeleton.any()):
        return skeleton
    gx = cv2.Sobel(skeleton, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skeleton, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    numerator = np.abs(gx * gyy - gy * gxy)
    denominator = np.power(gx ** 2 + gy ** 2, 1.5) + 1e-6
    curvature = numerator / denominator
    curvature *= skeleton
    if curvature.max() > 0.0:
        curvature = curvature / curvature.max()
    return curvature.astype(np.float32)


def generate_topology_preservation_map(mask: np.ndarray) -> np.ndarray:
    skeleton = generate_skeleton_map(mask, iterations=12)
    endpoints = generate_endpoint_map(mask)
    junctions = generate_junction_map(mask)
    topology = np.clip(skeleton + endpoints + junctions, 0.0, 1.0)
    return topology.astype(np.float32)

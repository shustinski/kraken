from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any, Callable

import cv2
import numpy as np

from lib.data_interfaces import TechAugmentationParameters, build_tech_augmentation_config


def _shift_array(array: np.ndarray, dy: int, dx: int, fill_value: int | bool = 0) -> np.ndarray:
    shifted = np.full(array.shape, fill_value, dtype=array.dtype)
    height, width = array.shape

    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - max(0, int(dy)))
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - max(0, int(dx)))
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return shifted

    dst_y0 = max(0, int(dy))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, int(dx))
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _binary_dilation(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = max(0, int(kernel_size))
    if radius == 0:
        return mask.copy()
    dilated = np.zeros(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dilated |= _shift_array(mask, dy, dx, fill_value=False)
    return dilated


def _binary_erosion(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = max(0, int(kernel_size))
    if radius == 0:
        return mask.copy()
    eroded = np.ones(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            eroded &= _shift_array(mask, dy, dx, fill_value=False)
    return eroded


def _extract_single_channel(mask: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(mask)
    if array.ndim == 2:
        plane = array
    elif array.ndim == 3 and array.shape[0] == 1:
        plane = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        plane = array[:, :, 0]
    else:
        raise ValueError('TechVariationAugmentor expects a 2D mask or a single-channel tensor.')

    plane = np.asarray(plane)
    max_value = float(np.max(plane)) if plane.size else 1.0
    threshold_value = float(threshold if max_value <= 1.0 else threshold * max(1.0, max_value))
    binary = plane >= threshold_value
    return array, binary.astype(bool, copy=False)


def _restore_mask_shape(template: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(template)
    max_value = float(np.max(array)) if array.size else 1.0
    scale_value = 1.0 if max_value <= 1.0 else max_value
    restored = binary_mask.astype(np.float32) * scale_value

    if array.ndim == 2:
        return restored.astype(array.dtype, copy=False)
    if array.ndim == 3 and array.shape[0] == 1:
        return restored.astype(array.dtype, copy=False)[None, :, :]
    if array.ndim == 3 and array.shape[-1] == 1:
        return restored.astype(array.dtype, copy=False)[:, :, None]
    raise ValueError('Unsupported mask shape for TechVariationAugmentor output restoration.')


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def _u8_to_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.uint8) > 0


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _component_centroid(mask_u8: np.ndarray) -> tuple[float, float]:
    points = np.column_stack(np.nonzero(mask_u8 > 0))
    if points.size == 0:
        height, width = mask_u8.shape
        return height / 2.0, width / 2.0
    y_mean, x_mean = points.mean(axis=0)
    return float(y_mean), float(x_mean)


def _boundary_points(mask_u8: np.ndarray) -> np.ndarray:
    if not np.any(mask_u8):
        return np.empty((0, 2), dtype=np.int32)
    eroded = cv2.erode(mask_u8, _ellipse_kernel(1), iterations=1)
    boundary = cv2.subtract(mask_u8, eroded)
    return np.column_stack(np.nonzero(boundary > 0)).astype(np.int32, copy=False)


def _draw_polyline(shape: tuple[int, int], points_xy: list[tuple[int, int]], thickness: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    if len(points_xy) < 2:
        return canvas
    resolved_thickness = max(1, int(thickness))
    for start_point, end_point in zip(points_xy[:-1], points_xy[1:]):
        cv2.line(canvas, start_point, end_point, 255, thickness=resolved_thickness)
    return canvas


def _clip_point(x_coord: float, y_coord: float, width: int, height: int) -> tuple[int, int]:
    clipped_x = int(np.clip(round(x_coord), 0, max(0, width - 1)))
    clipped_y = int(np.clip(round(y_coord), 0, max(0, height - 1)))
    return clipped_x, clipped_y


def _random_point_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    first_index, second_index = random.sample(range(len(points)), 2)
    return points[first_index], points[second_index]


def _local_principal_direction(mask_u8: np.ndarray, point_y: int, point_x: int, radius: int = 4) -> tuple[float, float]:
    height, width = mask_u8.shape
    top = max(0, int(point_y) - int(radius))
    bottom = min(height, int(point_y) + int(radius) + 1)
    left = max(0, int(point_x) - int(radius))
    right = min(width, int(point_x) + int(radius) + 1)
    neighborhood = np.column_stack(np.nonzero(mask_u8[top:bottom, left:right] > 0)).astype(np.float32, copy=False)
    if len(neighborhood) < 2:
        return 1.0, 0.0

    neighborhood[:, 0] += float(top)
    neighborhood[:, 1] += float(left)
    centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    direction_y = float(principal[0])
    direction_x = float(principal[1])
    norm = math.hypot(direction_x, direction_y)
    if norm <= 1e-6:
        return 1.0, 0.0
    return direction_x / norm, direction_y / norm


class TechVariationAugmentor:
    """Generate topology-changing variations for binary metallization masks."""

    def __init__(self, config: TechAugmentationParameters | Mapping[str, Any] | None):
        self.config = build_tech_augmentation_config(config)
        self.available_ops: tuple[tuple[str, Callable[[np.ndarray], np.ndarray], float], ...] = (
            (
                'global_width',
                self.apply_global_width_variation,
                float(self.config.global_width.probability),
            ),
            (
                'scale_rethreshold',
                self.apply_scale_rethreshold,
                float(self.config.scale_rethreshold.probability),
            ),
            (
                'blur_threshold',
                self.apply_blur_threshold,
                float(self.config.blur_threshold.probability),
            ),
            (
                'boundary_aware',
                self.apply_boundary_aware_variation,
                float(self.config.boundary_aware.probability),
            ),
            (
                'local_morphology',
                self.apply_local_morphology,
                float(self.config.local_morphology.probability),
            ),
            (
                'gap_variation',
                self.apply_gap_open_close_variation,
                float(self.config.gap_variation.probability),
            ),
        )

    def __call__(self, mask: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply 1-3 topology-building operators to a binary mask."""

        template, original_binary = _extract_single_channel(mask, self.config.binarization_threshold)
        original_binary = original_binary.astype(bool, copy=True)
        augmented = original_binary.copy()
        applied_any = False

        if self.config.enabled and original_binary.size > 0:
            min_ops = max(1, int(self.config.min_operations))
            max_ops = max(min_ops, int(self.config.max_operations))
            eligible_ops = [item for item in self.available_ops if float(item[2]) > 0.0]
            if eligible_ops:
                op_count = min(len(eligible_ops), random.randint(min_ops, max_ops))
            else:
                op_count = 0
            for _name, operation, probability in random.sample(eligible_ops, k=op_count):
                if random.random() > probability:
                    continue
                candidate = operation(augmented)
                if self._is_reasonable_variation(augmented, candidate):
                    augmented = candidate
                    applied_any = True

        if not applied_any:
            augmented = original_binary

        original_out = _restore_mask_shape(template, original_binary)
        augmented_out = _restore_mask_shape(template, augmented)
        if self.config.debug_return_pair:
            return original_out, augmented_out
        return augmented_out

    def _connected_addition(
        self,
        mask: np.ndarray,
        geometry_builder: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        mask_u8 = _mask_to_u8(mask)
        addition = geometry_builder(mask_u8)
        if not np.any(addition):
            return mask.copy()
        added_pixels = np.count_nonzero((addition > 0) & (mask_u8 == 0))
        if added_pixels <= 0:
            return mask.copy()
        candidate = cv2.bitwise_or(mask_u8, addition)
        return _u8_to_bool(candidate)

    def _build_branch_addition(
        self,
        mask_u8: np.ndarray,
        *,
        length_range: tuple[int, int],
        thickness_range: tuple[int, int],
        angle_jitter_rad: float,
    ) -> np.ndarray:
        boundary = _boundary_points(mask_u8)
        if len(boundary) == 0:
            return np.zeros(mask_u8.shape, dtype=np.uint8)
        height, width = mask_u8.shape
        center_y, center_x = _component_centroid(mask_u8)

        for _ in range(32):
            point_y, point_x = boundary[random.randrange(len(boundary))]
            base_angle = math.atan2(float(point_y) - center_y, float(point_x) - center_x)
            angle = base_angle + random.uniform(-angle_jitter_rad, angle_jitter_rad)
            length = random.randint(int(length_range[0]), int(length_range[1]))
            thickness = random.randint(int(thickness_range[0]), int(thickness_range[1]))
            end_x, end_y = _clip_point(
                point_x + math.cos(angle) * length,
                point_y + math.sin(angle) * length,
                width,
                height,
            )
            addition = _draw_polyline(mask_u8.shape, [(int(point_x), int(point_y)), (end_x, end_y)], thickness)
            if np.count_nonzero((addition > 0) & (mask_u8 == 0)) < max(4, thickness * 2):
                continue
            overlap_ratio = float(np.count_nonzero((addition > 0) & (mask_u8 > 0))) / max(
                1,
                np.count_nonzero(addition),
            )
            if overlap_ratio > 0.55:
                continue
            return addition
        return np.zeros(mask_u8.shape, dtype=np.uint8)

    def _build_bridge_addition(
        self,
        mask_u8: np.ndarray,
        *,
        distance_range: tuple[int, int],
        thickness_range: tuple[int, int],
        allow_same_component: bool,
        use_waypoint: bool,
        waypoint_strength_range: tuple[float, float] = (0.35, 0.65),
    ) -> np.ndarray:
        boundary = _boundary_points(mask_u8)
        if len(boundary) < 2:
            return np.zeros(mask_u8.shape, dtype=np.uint8)

        component_count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        height, width = mask_u8.shape
        min_distance = max(2, int(distance_range[0]))
        max_distance = max(min_distance, int(distance_range[1]))

        for _ in range(48):
            pair = _random_point_pair(boundary)
            if pair is None:
                break
            first_point, second_point = pair
            first_y, first_x = (int(first_point[0]), int(first_point[1]))
            second_y, second_x = (int(second_point[0]), int(second_point[1]))

            if not allow_same_component and component_count > 2:
                if labels[first_y, first_x] == labels[second_y, second_x]:
                    continue
            elif not allow_same_component and labels[first_y, first_x] == labels[second_y, second_x]:
                continue

            delta_y = float(second_y - first_y)
            delta_x = float(second_x - first_x)
            distance = math.hypot(delta_x, delta_y)
            if distance < float(min_distance) or distance > float(max_distance):
                continue

            thickness = random.randint(int(thickness_range[0]), int(thickness_range[1]))
            points_xy: list[tuple[int, int]] = [(first_x, first_y)]
            if use_waypoint and distance > 1.0:
                midpoint_x = (first_x + second_x) / 2.0
                midpoint_y = (first_y + second_y) / 2.0
                normal_x = -delta_y / distance
                normal_y = delta_x / distance
                offset = distance * random.uniform(*waypoint_strength_range)
                if random.random() < 0.5:
                    offset *= -1.0
                waypoint_x, waypoint_y = _clip_point(
                    midpoint_x + normal_x * offset,
                    midpoint_y + normal_y * offset,
                    width,
                    height,
                )
                points_xy.append((waypoint_x, waypoint_y))
            points_xy.append((second_x, second_y))

            addition = _draw_polyline(mask_u8.shape, points_xy, thickness)
            if np.count_nonzero((addition > 0) & (mask_u8 == 0)) < max(5, thickness * 2):
                continue
            overlap_ratio = float(np.count_nonzero((addition > 0) & (mask_u8 > 0))) / max(
                1,
                np.count_nonzero(addition),
            )
            if overlap_ratio > 0.6:
                continue
            return addition
        return np.zeros(mask_u8.shape, dtype=np.uint8)

    def apply_global_width_variation(self, mask: np.ndarray) -> np.ndarray:
        """Add a new branch that extends the existing routing graph."""

        cfg = self.config.global_width
        height, width = mask.shape
        branch_length = max(4, int(round(max(height, width) * 0.18)))
        kernel_radius = max(1, random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1])))
        thickness = max(1, kernel_radius * 2 - 1)
        return self._connected_addition(
            mask,
            lambda mask_u8: self._build_branch_addition(
                mask_u8,
                length_range=(max(4, branch_length // 2), max(6, branch_length)),
                thickness_range=(thickness, thickness + 1),
                angle_jitter_rad=math.pi / 3.0,
            ),
        )

    def apply_scale_rethreshold(self, mask: np.ndarray) -> np.ndarray:
        """Create a shortcut between nearby conductors instead of resizing the mask."""

        cfg = self.config.scale_rethreshold
        max_dim = max(mask.shape)
        deviation = max(abs(float(cfg.scale_range[0]) - 1.0), abs(float(cfg.scale_range[1]) - 1.0))
        max_gap = max(6, int(round(max_dim * max(0.12, deviation * 1.6))))
        return self._connected_addition(
            mask,
            lambda mask_u8: self._build_bridge_addition(
                mask_u8,
                distance_range=(3, max_gap),
                thickness_range=(1, 3),
                allow_same_component=True,
                use_waypoint=False,
            ),
        )

    def apply_blur_threshold(self, mask: np.ndarray) -> np.ndarray:
        """Insert a new bypass loop attached to the existing routing."""

        cfg = self.config.blur_threshold
        max_dim = max(mask.shape)
        max_span = max(8, int(round(max_dim * max(0.16, float(cfg.blur_radius_range[1]) * 0.12))))
        return self._connected_addition(
            mask,
            lambda mask_u8: self._build_bridge_addition(
                mask_u8,
                distance_range=(max(4, max_span // 3), max_span),
                thickness_range=(1, 3),
                allow_same_component=True,
                use_waypoint=True,
                waypoint_strength_range=(0.25, 0.55),
            ),
        )

    def apply_boundary_aware_variation(self, mask: np.ndarray) -> np.ndarray:
        """Extend an existing trace segment without drawing standalone polygons."""

        cfg = self.config.boundary_aware
        max_band = max(1, int(cfg.band_width_range[1]))
        branch_length = max(4, max_band * 4)
        thickness = max(1, int(cfg.smoothing_kernel_size))

        def _builder(mask_u8: np.ndarray) -> np.ndarray:
            boundary = _boundary_points(mask_u8)
            if len(boundary) == 0:
                return np.zeros(mask_u8.shape, dtype=np.uint8)
            height, width = mask_u8.shape

            for _ in range(32):
                point_y, point_x = boundary[random.randrange(len(boundary))]
                direction_x, direction_y = _local_principal_direction(mask_u8, int(point_y), int(point_x))
                if random.random() < 0.5:
                    direction_x *= -1.0
                    direction_y *= -1.0
                angle = math.atan2(direction_y, direction_x) + random.uniform(-math.pi / 10.0, math.pi / 10.0)
                length = random.randint(max(3, branch_length // 2), max(5, branch_length))
                end_x, end_y = _clip_point(
                    point_x + math.cos(angle) * length,
                    point_y + math.sin(angle) * length,
                    width,
                    height,
                )
                addition = _draw_polyline(
                    mask_u8.shape,
                    [(int(point_x), int(point_y)), (end_x, end_y)],
                    max(1, thickness),
                )
                added_pixels = np.count_nonzero((addition > 0) & (mask_u8 == 0))
                if added_pixels < max(4, thickness * 2):
                    continue
                overlap_ratio = float(np.count_nonzero((addition > 0) & (mask_u8 > 0))) / max(
                    1,
                    np.count_nonzero(addition),
                )
                if overlap_ratio > 0.7:
                    continue
                return addition
            return np.zeros(mask_u8.shape, dtype=np.uint8)

        return self._connected_addition(
            mask,
            _builder,
        )

    def apply_local_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Locally reroute a fragment by drawing a detour polyline inside a random ROI."""

        cfg = self.config.local_morphology
        height, width = mask.shape
        roi_ratio = random.uniform(float(cfg.roi_size_ratio_range[0]), float(cfg.roi_size_ratio_range[1]))
        roi_size = max(8, int(round(max(height, width) * roi_ratio)))
        thickness = max(1, random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1])))

        def _builder(mask_u8: np.ndarray) -> np.ndarray:
            points = np.column_stack(np.nonzero(mask_u8 > 0))
            if len(points) == 0:
                return np.zeros(mask_u8.shape, dtype=np.uint8)
            center_y, center_x = points[random.randrange(len(points))]
            top = max(0, int(center_y) - roi_size // 2)
            left = max(0, int(center_x) - roi_size // 2)
            bottom = min(height, top + roi_size)
            right = min(width, left + roi_size)
            roi = mask_u8[top:bottom, left:right]
            roi_addition = self._build_bridge_addition(
                roi,
                distance_range=(max(3, roi_size // 5), max(5, roi_size // 2)),
                thickness_range=(thickness, thickness + 1),
                allow_same_component=True,
                use_waypoint=True,
                waypoint_strength_range=(0.2, 0.45),
            )
            addition = np.zeros(mask_u8.shape, dtype=np.uint8)
            addition[top:bottom, left:right] = roi_addition
            return addition

        return self._connected_addition(mask, _builder)

    def apply_gap_open_close_variation(self, mask: np.ndarray) -> np.ndarray:
        """Close narrow gaps by inserting a short conductive bridge."""

        cfg = self.config.gap_variation
        kernel_size = max(1, random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1])))
        return self._connected_addition(
            mask,
            lambda mask_u8: self._build_bridge_addition(
                mask_u8,
                distance_range=(2, max(4, kernel_size * 4)),
                thickness_range=(kernel_size, kernel_size + 1),
                allow_same_component=False,
                use_waypoint=False,
            ),
        )

    def _is_reasonable_variation(self, reference: np.ndarray, candidate: np.ndarray) -> bool:
        """Reject implausible topology changes before committing them."""

        if candidate.shape != reference.shape:
            return False
        if reference.any() and not candidate.any():
            return False
        if (~reference).any() and candidate.all():
            return False
        if np.count_nonzero(candidate) < np.count_nonzero(reference):
            return False

        changed_ratio = float(np.mean(reference != candidate))
        if changed_ratio > float(self.config.max_changed_pixels_ratio):
            return False

        foreground_delta = abs(float(candidate.mean()) - float(reference.mean()))
        if foreground_delta > float(self.config.max_foreground_ratio_delta):
            return False

        return True

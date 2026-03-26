from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - depends on environment.
    cv2 = None

from lib.data_interfaces import PCBDefectParameters, build_pcb_defect_parameters


@dataclass(frozen=True)
class _ArrayFormat:
    dtype: Any
    scale: float
    chw: bool
    squeeze_channel: bool
    channels: int


class PCBDefectAugmentor:
    """Generate structurally plausible synthetic PCB defects on image/mask pairs."""

    _DEFECT_ORDER: tuple[str, ...] = (
        'break',
        'short',
        'missing_copper',
        'excess_copper',
        'pinhole',
        'spurious_copper',
        'via',
        'misalignment',
    )

    def __init__(self, config: PCBDefectParameters | dict[str, Any] | Any):
        self.config = build_pcb_defect_parameters(config)
        if self.config.enabled and cv2 is None:
            raise ImportError(
                'PCB defect augmentation requires OpenCV (cv2). '
                'Install opencv-python or disable pcb_defects.enabled.'
            )
        self._defect_handlers = {
            'break': self._apply_break_defect,
            'short': self._apply_short_defect,
            'missing_copper': self._apply_missing_copper_defect,
            'excess_copper': self._apply_excess_copper_defect,
            'pinhole': self._apply_pinhole_defect,
            'spurious_copper': self._apply_spurious_copper_defect,
            'via': self._apply_via_defect,
            'misalignment': self._apply_misalignment_defect,
        }
        weights = []
        names = []
        for name in self._DEFECT_ORDER:
            weight = max(0.0, float(self.config.defect_probabilities.get(name, 0.0)))
            if weight <= 0.0:
                continue
            names.append(name)
            weights.append(weight)
        if not names:
            names = list(self._DEFECT_ORDER)
            weights = [1.0] * len(names)
        self._weighted_defect_names = tuple(names)
        self._weighted_defect_probabilities = np.asarray(weights, dtype=np.float64)
        self._weighted_defect_probabilities /= self._weighted_defect_probabilities.sum()
        self._enabled_defect_count = int(len(self._weighted_defect_names))

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        seed: int | None = None,
        return_debug: bool = False,
    ):
        """Apply one or more PCB defects."""

        normalized_image, image_format = self._prepare_image(image)
        normalized_mask, mask_format = self._prepare_mask(mask)

        if not self.config.enabled:
            empty_mask = np.zeros(normalized_image.shape[:2], dtype=np.uint8)
            return self._finalize_outputs(
                normalized_image,
                normalized_image.copy(),
                empty_mask,
                image_format=image_format,
                mask_format=mask_format,
                return_debug=return_debug,
            )

        if cv2 is None:
            raise ImportError(
                'PCB defect augmentation requires OpenCV (cv2). '
                'Install opencv-python or disable pcb_defects.enabled.'
            )

        original_copper = self._resolve_copper_mask(normalized_image, normalized_mask)
        original_copper_u8 = self._to_u8_mask(original_copper)
        rng = np.random.default_rng(seed)

        if np.count_nonzero(original_copper_u8) < max(1, int(self.config.min_component_area)):
            return self._finalize_outputs(
                normalized_image,
                normalized_image.copy(),
                np.zeros_like(original_copper_u8, dtype=np.uint8),
                image_format=image_format,
                mask_format=mask_format,
                return_debug=return_debug,
            )

        if float(rng.random()) >= float(self.config.defect_probability):
            return self._finalize_outputs(
                normalized_image,
                normalized_image.copy(),
                np.zeros_like(original_copper_u8, dtype=np.uint8),
                image_format=image_format,
                mask_format=mask_format,
                return_debug=return_debug,
            )

        augmented_copper = original_copper_u8.copy()
        target_count = int(
            rng.integers(int(self.config.min_defects), int(self.config.max_defects) + 1)
        )
        applied_count = 0
        max_attempts = (
            max(1, int(self.config.max_attempts_per_defect))
            * max(1, target_count)
            * max(1, self._enabled_defect_count)
        )
        attempts = 0

        while applied_count < target_count and attempts < max_attempts:
            cycle_applied = False
            for defect_name in self._iter_defect_attempt_order(rng):
                if attempts >= max_attempts:
                    break
                attempts += 1
                handler = self._defect_handlers[defect_name]
                updated = handler(augmented_copper, rng)
                if updated is None:
                    continue
                if np.array_equal(updated, augmented_copper):
                    continue
                augmented_copper = updated
                applied_count += 1
                cycle_applied = True
                break
            if not cycle_applied:
                break

        defect_mask = cv2.absdiff(original_copper_u8, augmented_copper)
        if np.count_nonzero(defect_mask) == 0:
            augmented_image = normalized_image.copy()
        else:
            augmented_image = self._render_augmented_image(
                normalized_image,
                original_copper_u8,
                augmented_copper,
                defect_mask,
            )

        return self._finalize_outputs(
            normalized_image,
            augmented_image,
            defect_mask,
            image_format=image_format,
            mask_format=mask_format,
            return_debug=return_debug,
        )

    @staticmethod
    def _prepare_image(image: np.ndarray) -> tuple[np.ndarray, _ArrayFormat]:
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError('PCBDefectAugmentor expects 2D or 3D images.')

        chw = bool(
            array.ndim == 3
            and array.shape[0] in (1, 3)
            and array.shape[1] > 4
            and array.shape[2] > 4
        )
        squeeze_channel = bool(array.ndim == 2)
        if squeeze_channel:
            hwc = array[..., None]
            channels = 1
        elif chw:
            hwc = np.transpose(array, (1, 2, 0))
            channels = int(array.shape[0])
        else:
            hwc = array
            channels = int(array.shape[2])

        if np.issubdtype(array.dtype, np.integer):
            scale = 255.0
        else:
            max_value = float(np.max(hwc)) if hwc.size else 0.0
            scale = 255.0 if max_value > 1.5 else 1.0

        normalized = hwc.astype(np.float32, copy=True)
        if scale > 1.0:
            normalized /= float(scale)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        return normalized, _ArrayFormat(
            dtype=array.dtype,
            scale=scale,
            chw=chw,
            squeeze_channel=squeeze_channel,
            channels=channels,
        )

    @staticmethod
    def _prepare_mask(mask: np.ndarray | None) -> tuple[np.ndarray | None, _ArrayFormat | None]:
        if mask is None:
            return None, None
        array = np.asarray(mask)
        if array.ndim not in (2, 3):
            raise ValueError('PCBDefectAugmentor expects 2D or 3D masks.')
        chw = bool(
            array.ndim == 3
            and array.shape[0] == 1
            and array.shape[1] > 4
            and array.shape[2] > 4
        )
        squeeze_channel = bool(array.ndim == 2)
        if squeeze_channel:
            normalized = array.astype(np.float32, copy=True)[..., None]
            channels = 1
        elif chw:
            normalized = np.transpose(array, (1, 2, 0)).astype(np.float32, copy=True)
            channels = int(array.shape[0])
        else:
            normalized = array.astype(np.float32, copy=True)
            channels = int(array.shape[2])

        scale = 255.0 if np.max(normalized) > 1.5 else 1.0
        if scale > 1.0:
            normalized /= float(scale)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        return normalized, _ArrayFormat(
            dtype=array.dtype,
            scale=scale,
            chw=chw,
            squeeze_channel=squeeze_channel,
            channels=channels,
        )

    @staticmethod
    def _restore_image(image: np.ndarray, fmt: _ArrayFormat) -> np.ndarray:
        restored = np.clip(image, 0.0, 1.0)
        if fmt.scale > 1.0:
            restored = np.round(restored * fmt.scale)
        restored = restored.astype(fmt.dtype, copy=False)
        if fmt.squeeze_channel:
            return restored[..., 0]
        if fmt.chw:
            return np.transpose(restored, (2, 0, 1))
        return restored

    def _restore_mask(
        self,
        mask_u8: np.ndarray,
        *,
        image_format: _ArrayFormat,
        mask_format: _ArrayFormat | None,
    ) -> np.ndarray:
        target_format = mask_format or _ArrayFormat(
            dtype=np.float32 if image_format.scale <= 1.0 else np.uint8,
            scale=1.0 if image_format.scale <= 1.0 else 255.0,
            chw=image_format.chw,
            squeeze_channel=image_format.squeeze_channel,
            channels=1,
        )
        mask_float = (mask_u8.astype(np.float32) / 255.0)[..., None]
        if target_format.scale > 1.0:
            mask_float = np.round(mask_float * target_format.scale)
        restored = mask_float.astype(target_format.dtype, copy=False)
        if target_format.squeeze_channel:
            return restored[..., 0]
        if target_format.chw:
            return np.transpose(restored, (2, 0, 1))
        return restored

    def _finalize_outputs(
        self,
        original_image: np.ndarray,
        augmented_image: np.ndarray,
        defect_mask_u8: np.ndarray,
        *,
        image_format: _ArrayFormat,
        mask_format: _ArrayFormat | None,
        return_debug: bool,
    ):
        restored_original = self._restore_image(original_image, image_format)
        restored_augmented = self._restore_image(augmented_image, image_format)
        restored_mask = self._restore_mask(
            defect_mask_u8,
            image_format=image_format,
            mask_format=mask_format,
        )
        if return_debug:
            return restored_original, restored_augmented, restored_mask
        return restored_augmented, restored_mask

    def _iter_defect_attempt_order(self, rng: np.random.Generator) -> tuple[str, ...]:
        if self._enabled_defect_count <= 1:
            return self._weighted_defect_names
        ordered = rng.choice(
            self._weighted_defect_names,
            size=self._enabled_defect_count,
            replace=False,
            p=self._weighted_defect_probabilities,
        )
        return tuple(str(item) for item in ordered.tolist())

    def _resolve_copper_mask(
        self,
        image_hwc: np.ndarray,
        mask_hwc: np.ndarray | None,
    ) -> np.ndarray:
        if mask_hwc is not None and bool(self.config.use_input_mask):
            return np.max(mask_hwc, axis=2) > 0.5

        grayscale = self._to_grayscale_u8(image_hwc)
        _threshold, bright = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        dark = cv2.bitwise_not(bright)
        bright_ratio = float(np.count_nonzero(bright)) / float(bright.size)
        dark_ratio = float(np.count_nonzero(dark)) / float(dark.size)
        if 0.01 <= bright_ratio <= 0.75 and not (0.01 <= dark_ratio <= 0.75):
            selected = bright
        elif 0.01 <= dark_ratio <= 0.75 and not (0.01 <= bright_ratio <= 0.75):
            selected = dark
        else:
            target_coverage = 0.35
            selected = bright if abs(bright_ratio - target_coverage) <= abs(dark_ratio - target_coverage) else dark
        kernel = self._ellipse_kernel(1)
        selected = cv2.morphologyEx(selected, cv2.MORPH_OPEN, kernel, iterations=1)
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=1)
        return selected > 0

    @staticmethod
    def _to_grayscale_u8(image_hwc: np.ndarray) -> np.ndarray:
        if image_hwc.shape[2] == 1:
            return np.clip(np.round(image_hwc[..., 0] * 255.0), 0.0, 255.0).astype(np.uint8)
        rgb = np.clip(np.round(image_hwc[..., :3] * 255.0), 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _to_u8_mask(mask: np.ndarray) -> np.ndarray:
        return np.where(mask, 255, 0).astype(np.uint8)

    @staticmethod
    def _ellipse_kernel(radius: int) -> np.ndarray:
        size = max(1, int(radius)) * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _random_coordinate(candidate_mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
        flat_candidates = np.flatnonzero(candidate_mask)
        if flat_candidates.size == 0:
            return None
        flat_index = int(flat_candidates[int(rng.integers(0, flat_candidates.size))])
        y, x = np.unravel_index(flat_index, candidate_mask.shape)
        return int(x), int(y)

    @staticmethod
    def _edge_mask(mask_u8: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask_u8) == 0:
            return np.zeros_like(mask_u8, dtype=bool)
        eroded = cv2.erode(mask_u8, PCBDefectAugmentor._ellipse_kernel(1), iterations=1)
        return (mask_u8 > 0) & (eroded == 0)

    @staticmethod
    def _estimate_local_orientation(mask_u8: np.ndarray, center_xy: tuple[int, int], radius: int) -> float:
        center_x, center_y = center_xy
        radius = max(3, int(radius))
        height, width = mask_u8.shape
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(width, center_x + radius + 1)
        y2 = min(height, center_y + radius + 1)
        window = mask_u8[y1:y2, x1:x2]
        coords = np.argwhere(window > 0)
        if coords.shape[0] < 8:
            return 0.0
        xy = np.empty((coords.shape[0], 2), dtype=np.float32)
        xy[:, 0] = coords[:, 1].astype(np.float32) + float(x1 - center_x)
        xy[:, 1] = coords[:, 0].astype(np.float32) + float(y1 - center_y)
        covariance = np.cov(xy, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        major_vector = eigvecs[:, int(np.argmax(eigvals))]
        return float(math.atan2(float(major_vector[1]), float(major_vector[0])))

    @staticmethod
    def _draw_rotated_rectangle(
        canvas: np.ndarray,
        *,
        center_xy: tuple[int, int],
        size_xy: tuple[int, int],
        angle_rad: float,
        value: int,
    ) -> None:
        rect = (
            (float(center_xy[0]), float(center_xy[1])),
            (float(max(1, size_xy[0])), float(max(1, size_xy[1]))),
            float(math.degrees(angle_rad)),
        )
        points = cv2.boxPoints(rect)
        cv2.fillConvexPoly(canvas, np.round(points).astype(np.int32), int(value))

    @staticmethod
    def _draw_irregular_blob(
        canvas: np.ndarray,
        *,
        center_xy: tuple[int, int],
        radius_xy: tuple[int, int],
        angle_rad: float,
        rng: np.random.Generator,
        value: int,
    ) -> None:
        point_count = int(rng.integers(5, 9))
        base_angles = np.linspace(0.0, 2.0 * math.pi, point_count, endpoint=False)
        base_angles += float(rng.uniform(0.0, 2.0 * math.pi / max(1, point_count)))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        radius_x = max(1.0, float(radius_xy[0]))
        radius_y = max(1.0, float(radius_xy[1]))
        points: list[list[int]] = []
        for theta in base_angles:
            scale_x = float(rng.uniform(0.7, 1.25))
            scale_y = float(rng.uniform(0.7, 1.25))
            local_x = math.cos(theta) * radius_x * scale_x
            local_y = math.sin(theta) * radius_y * scale_y
            rot_x = local_x * cos_a - local_y * sin_a
            rot_y = local_x * sin_a + local_y * cos_a
            points.append([
                int(round(center_xy[0] + rot_x)),
                int(round(center_xy[1] + rot_y)),
            ])
        cv2.fillConvexPoly(canvas, np.asarray(points, dtype=np.int32), int(value))

    def _apply_break_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        candidate = (mask_u8 > 0) & (distance >= 1.0)
        point = self._random_coordinate(candidate, rng)
        if point is None:
            return None
        center_x, center_y = point
        local_half_width = max(1.0, float(distance[center_y, center_x]))
        tangent_angle = self._estimate_local_orientation(
            mask_u8,
            point,
            radius=max(4, int(round(local_half_width * 3.0))),
        )
        gap_width = int(
            rng.integers(
                int(self.config.break_width_range[0]),
                int(self.config.break_width_range[1]) + 1,
            )
        )
        remove_mask = np.zeros_like(mask_u8)
        self._draw_rotated_rectangle(
            remove_mask,
            center_xy=point,
            size_xy=(gap_width, max(3, int(round(local_half_width * 2.6)))),
            angle_rad=tangent_angle,
            value=255,
        )
        updated = mask_u8.copy()
        updated[remove_mask > 0] = 0
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_short_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        component_count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        if component_count <= 2:
            return None
        max_distance = int(self.config.short_bridge_distance_range[1])
        min_distance = int(self.config.short_bridge_distance_range[0])
        background_distance = cv2.distanceTransform((mask_u8 == 0).astype(np.uint8), cv2.DIST_L2, 3)
        candidate = (
            (mask_u8 == 0)
            & (background_distance >= max(1, min_distance // 2))
            & (background_distance <= max_distance)
        )
        for _ in range(max(6, int(self.config.max_attempts_per_defect))):
            point = self._random_coordinate(candidate, rng)
            if point is None:
                return None
            center_x, center_y = point
            x1 = max(0, center_x - max_distance - 1)
            y1 = max(0, center_y - max_distance - 1)
            x2 = min(mask_u8.shape[1], center_x + max_distance + 2)
            y2 = min(mask_u8.shape[0], center_y + max_distance + 2)
            local_labels = labels[y1:y2, x1:x2]
            unique_components = np.unique(local_labels[local_labels > 0])
            if unique_components.size < 2:
                continue

            endpoints: list[tuple[float, int, int, int]] = []
            for component_id in unique_components:
                coords = np.argwhere(local_labels == component_id)
                if coords.size == 0:
                    continue
                global_coords = coords + np.asarray([[y1, x1]], dtype=np.int32)
                distances = (
                    (global_coords[:, 0] - center_y) ** 2
                    + (global_coords[:, 1] - center_x) ** 2
                )
                nearest_index = int(np.argmin(distances))
                nearest_y = int(global_coords[nearest_index, 0])
                nearest_x = int(global_coords[nearest_index, 1])
                endpoints.append((
                    float(math.sqrt(float(distances[nearest_index]))),
                    int(component_id),
                    nearest_x,
                    nearest_y,
                ))

            if len(endpoints) < 2:
                continue
            endpoints.sort(key=lambda item: item[0])
            (_, first_id, first_x, first_y), (_, second_id, second_x, second_y) = endpoints[:2]
            if first_id == second_id:
                continue
            endpoint_distance = math.hypot(float(first_x - second_x), float(first_y - second_y))
            if endpoint_distance < float(min_distance) or endpoint_distance > float(max_distance * 1.5):
                continue
            bridge = np.zeros_like(mask_u8)
            thickness = int(rng.integers(1, 4))
            cv2.line(bridge, (first_x, first_y), (second_x, second_y), 255, thickness=thickness)
            updated = cv2.bitwise_or(mask_u8, bridge)
            if np.count_nonzero(updated != mask_u8) > 0:
                return updated
        return None

    def _apply_missing_copper_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        edge = self._edge_mask(mask_u8)
        point = self._random_coordinate(edge, rng)
        if point is None:
            return None
        tangent_angle = self._estimate_local_orientation(mask_u8, point, radius=7)
        remove = np.zeros_like(mask_u8)
        radius_x = int(
            rng.integers(
                int(self.config.missing_copper_radius_range[0]),
                int(self.config.missing_copper_radius_range[1]) + 1,
            )
        )
        radius_y = max(1, int(round(radius_x * float(rng.uniform(0.5, 1.2)))))
        self._draw_irregular_blob(
            remove,
            center_xy=point,
            radius_xy=(radius_x, radius_y),
            angle_rad=tangent_angle,
            rng=rng,
            value=255,
        )
        updated = mask_u8.copy()
        updated[remove > 0] = 0
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_excess_copper_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        edge = self._edge_mask(mask_u8)
        point = self._random_coordinate(edge, rng)
        if point is None:
            return None
        tangent_angle = self._estimate_local_orientation(mask_u8, point, radius=7)
        add = np.zeros_like(mask_u8)
        radius_x = int(
            rng.integers(
                int(self.config.excess_copper_radius_range[0]),
                int(self.config.excess_copper_radius_range[1]) + 1,
            )
        )
        radius_y = max(1, int(round(radius_x * float(rng.uniform(0.5, 1.1)))))
        self._draw_irregular_blob(
            add,
            center_xy=point,
            radius_xy=(radius_x, radius_y),
            angle_rad=tangent_angle,
            rng=rng,
            value=255,
        )
        updated = cv2.bitwise_or(mask_u8, add)
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_pinhole_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        candidate = mask_u8 > 0
        point = self._random_coordinate(candidate, rng)
        if point is None:
            return None
        center_x, center_y = point
        local_radius_limit = max(0, int(math.floor(float(distance[center_y, center_x]) - 0.5)))
        if local_radius_limit < 1:
            return None
        max_radius = min(int(self.config.pinhole_radius_range[1]), local_radius_limit)
        min_radius = min(int(self.config.pinhole_radius_range[0]), max_radius)
        if max_radius < 1:
            return None
        radius = int(rng.integers(max(1, min_radius), max_radius + 1))
        updated = mask_u8.copy()
        cv2.circle(updated, point, radius, 0, thickness=-1)
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_spurious_copper_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        background_distance = cv2.distanceTransform((mask_u8 == 0).astype(np.uint8), cv2.DIST_L2, 3)
        min_sep = max(2, int(self.config.spurious_copper_radius_range[1]) + 1)
        max_sep = min_sep + 6
        candidate = (mask_u8 == 0) & (background_distance >= min_sep) & (background_distance <= max_sep)
        point = self._random_coordinate(candidate, rng)
        if point is None:
            return None
        radius = int(
            rng.integers(
                int(self.config.spurious_copper_radius_range[0]),
                int(self.config.spurious_copper_radius_range[1]) + 1,
            )
        )
        add = np.zeros_like(mask_u8)
        self._draw_irregular_blob(
            add,
            center_xy=point,
            radius_xy=(radius, max(1, int(round(radius * 0.9)))),
            angle_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
            rng=rng,
            value=255,
        )
        contact = cv2.bitwise_and(cv2.dilate(mask_u8, self._ellipse_kernel(1), iterations=1), add)
        if np.count_nonzero(contact) > 0:
            return None
        updated = cv2.bitwise_or(mask_u8, add)
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_via_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        via_candidates = self._detect_via_holes(mask_u8)
        if not via_candidates:
            return None
        component_mask, center_xy, radius = via_candidates[int(rng.integers(0, len(via_candidates)))]
        mode = str(rng.choice(['shift', 'resize', 'partial']))
        updated = mask_u8.copy()

        if mode == 'shift':
            shift_min, shift_max = self.config.via_shift_range
            dx = int(rng.integers(int(shift_min), int(shift_max) + 1))
            dy = int(rng.integers(int(shift_min), int(shift_max) + 1))
            if rng.random() < 0.5:
                dx *= -1
            if rng.random() < 0.5:
                dy *= -1
            if dx == 0 and dy == 0:
                dx = 1
            updated[component_mask] = 255
            shifted_center = (
                int(np.clip(center_xy[0] + dx, 0, mask_u8.shape[1] - 1)),
                int(np.clip(center_xy[1] + dy, 0, mask_u8.shape[0] - 1)),
            )
            cv2.circle(updated, shifted_center, radius, 0, thickness=-1)
        elif mode == 'resize':
            delta_min, delta_max = self.config.via_size_delta_range
            delta = int(rng.integers(int(delta_min), int(delta_max) + 1))
            if rng.random() < 0.5:
                delta *= -1
            updated[component_mask] = 255
            resized_radius = max(1, int(radius + delta))
            cv2.circle(updated, center_xy, resized_radius, 0, thickness=-1)
        else:
            fill = np.zeros_like(mask_u8)
            self._draw_irregular_blob(
                fill,
                center_xy=center_xy,
                radius_xy=(max(1, radius - 1), max(1, int(round(radius * 0.75)))),
                angle_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
                rng=rng,
                value=255,
            )
            updated = cv2.bitwise_or(updated, fill)

        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _apply_misalignment_defect(
        self,
        mask_u8: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        foreground = mask_u8 > 0
        point = self._random_coordinate(foreground, rng)
        if point is None:
            return None
        image_h, image_w = mask_u8.shape
        scale_min, scale_max = self.config.misalignment_roi_scale_range
        roi_w = max(12, int(round(image_w * float(rng.uniform(scale_min, scale_max)))))
        roi_h = max(12, int(round(image_h * float(rng.uniform(scale_min, scale_max)))))
        center_x, center_y = point
        x1 = int(np.clip(center_x - roi_w // 2, 0, max(0, image_w - roi_w)))
        y1 = int(np.clip(center_y - roi_h // 2, 0, max(0, image_h - roi_h)))
        x2 = min(image_w, x1 + roi_w)
        y2 = min(image_h, y1 + roi_h)

        shift_min, shift_max = self.config.misalignment_shift_range
        dx = int(rng.integers(int(shift_min), int(shift_max) + 1))
        dy = int(rng.integers(int(shift_min), int(shift_max) + 1))
        if rng.random() < 0.5:
            dx *= -1
        if rng.random() < 0.5:
            dy *= -1
        if dx == 0 and dy == 0:
            dx = 1

        roi = mask_u8[y1:y2, x1:x2]
        transform = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        shifted = cv2.warpAffine(
            roi,
            transform,
            (roi.shape[1], roi.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        region = np.zeros_like(roi)
        cv2.ellipse(
            region,
            (roi.shape[1] // 2, roi.shape[0] // 2),
            (max(2, roi.shape[1] // 2 - 1), max(2, roi.shape[0] // 2 - 1)),
            0.0,
            0.0,
            360.0,
            255,
            thickness=-1,
        )
        updated_roi = roi.copy()
        updated_roi[region > 0] = shifted[region > 0]
        updated = mask_u8.copy()
        updated[y1:y2, x1:x2] = updated_roi
        return updated if np.count_nonzero(updated != mask_u8) > 0 else None

    def _detect_via_holes(
        self,
        mask_u8: np.ndarray,
    ) -> list[tuple[np.ndarray, tuple[int, int], int]]:
        inverse = (mask_u8 == 0).astype(np.uint8)
        components, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        min_area, max_area = self.config.via_hole_area_range
        candidates: list[tuple[np.ndarray, tuple[int, int], int]] = []
        height, width = mask_u8.shape
        for component_id in range(1, components):
            x, y, w, h, area = stats[component_id]
            if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
                continue
            if area < int(min_area) or area > int(max_area):
                continue
            component_mask = labels == component_id
            component_u8 = np.where(component_mask, 255, 0).astype(np.uint8)
            contours, _hierarchy = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perimeter = float(cv2.arcLength(contours[0], closed=True))
            circularity = float((4.0 * math.pi * float(area)) / max(perimeter * perimeter, 1e-6))
            if circularity < 0.2:
                continue
            surround = cv2.dilate(component_u8, self._ellipse_kernel(1), iterations=1)
            ring_overlap = np.count_nonzero((surround > 0) & (mask_u8 > 0))
            if ring_overlap <= 0:
                continue
            center_x = int(round(float(centroids[component_id][0])))
            center_y = int(round(float(centroids[component_id][1])))
            radius = max(1, int(round(math.sqrt(float(area) / math.pi))))
            candidates.append((component_mask, (center_x, center_y), radius))
        return candidates

    def _render_augmented_image(
        self,
        image_hwc: np.ndarray,
        original_copper_u8: np.ndarray,
        augmented_copper_u8: np.ndarray,
        defect_mask_u8: np.ndarray,
    ) -> np.ndarray:
        original_copper = original_copper_u8 > 0
        augmented_copper = augmented_copper_u8 > 0
        image = image_hwc.copy()
        if image.shape[2] == 1 and self._is_binary_like(image):
            image[..., 0] = augmented_copper.astype(np.float32)
            return image

        copper_mean = self._masked_channel_mean(image, original_copper, fallback=0.85)
        background_mean = self._masked_channel_mean(image, ~original_copper, fallback=0.15)
        removed = original_copper & (~augmented_copper)
        added = augmented_copper & (~original_copper)

        target = image.copy()
        if np.count_nonzero(removed) > 0:
            target[removed] = background_mean
        if np.count_nonzero(added) > 0:
            target[added] = copper_mean

        alpha = cv2.GaussianBlur(
            defect_mask_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=0.9,
            sigmaY=0.9,
        )
        alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
        blended = (image * (1.0 - alpha)) + (target * alpha)
        np.clip(blended, 0.0, 1.0, out=blended)
        return blended.astype(np.float32, copy=False)

    @staticmethod
    def _masked_channel_mean(
        image_hwc: np.ndarray,
        mask: np.ndarray,
        *,
        fallback: float,
    ) -> np.ndarray:
        if np.count_nonzero(mask) <= 0:
            return np.full((image_hwc.shape[2],), float(fallback), dtype=np.float32)
        selected = image_hwc[mask]
        if selected.ndim == 1:
            selected = selected[:, None]
        return selected.mean(axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _is_binary_like(image_hwc: np.ndarray) -> bool:
        quantized = np.unique(np.round(np.clip(image_hwc, 0.0, 1.0) * 255.0).astype(np.uint8))
        return bool(quantized.size <= 4)

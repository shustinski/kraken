"""synthetic_topology: allow candidate overlap."""
from __future__ import annotations
import random
import cv2
import numpy as np


class AllowCandidateOverlapMixin:
    def _build_pcb_diagonal_branch_endpoint(
        self,
        start_xy: tuple[int, int],
        *,
        width: int,
        height: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        diagonal_length = max(6, int(round(min(width, height) * rng.uniform(0.07, 0.18))))
        dx = diagonal_length if rng.random() < 0.5 else -diagonal_length
        dy = diagonal_length if rng.random() < 0.5 else -diagonal_length
        return (
            int(np.clip(start_xy[0] + dx, 0, width - 1)),
            int(np.clip(start_xy[1] + dy, 0, height - 1)),
        )

    def _resolve_branch_fractions(self, target: int) -> list[float]:
        branch_count = max(1, int(target))
        return [float(index + 1) / float(branch_count + 1) for index in range(branch_count)]

    @staticmethod
    def _point_on_segment(
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
        fraction: float,
    ) -> tuple[int, int]:
        fraction = float(min(max(fraction, 0.0), 1.0))
        return (
            int(round(start_xy[0] + ((end_xy[0] - start_xy[0]) * fraction))),
            int(round(start_xy[1] + ((end_xy[1] - start_xy[1]) * fraction))),
        )

    @staticmethod
    def _midpoint(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round((start_xy[0] + end_xy[0]) / 2.0)),
            int(round((start_xy[1] + end_xy[1]) / 2.0)),
        )

    @staticmethod
    def _is_boundary_anchor(anchor_xy: tuple[int, int], shape_hw: tuple[int, int]) -> bool:
        x, y = anchor_xy
        height, width = shape_hw
        return x <= 1 or y <= 1 or x >= (width - 2) or y >= (height - 2)

    def _ring_fits(self, mask_u8: np.ndarray, center_xy: tuple[int, int], outer_radius: int) -> bool:
        center_x, center_y = center_xy
        height, width = mask_u8.shape
        x1 = max(0, center_x - outer_radius - 1)
        y1 = max(0, center_y - outer_radius - 1)
        x2 = min(width, center_x + outer_radius + 2)
        y2 = min(height, center_y + outer_radius + 2)
        if x2 <= x1 or y2 <= y1:
            return False
        return bool(np.count_nonzero(mask_u8[y1:y2, x1:x2]) > 0)

    @staticmethod
    def _allow_candidate_overlap(
        occupied_mask: np.ndarray,
        candidate_mask: np.ndarray,
        canvas_shape: tuple[int, int],
        allowed_points: list[tuple[int, int]],
        *,
        allowance_radius: int,
    ) -> bool:
        overlap = (occupied_mask > 0) & (candidate_mask > 0)
        if not np.any(overlap):
            return True
        allowed_mask = np.zeros(canvas_shape, dtype=np.uint8)
        radius = max(1, int(allowance_radius))
        for point_x, point_y in allowed_points:
            cv2.circle(allowed_mask, (int(point_x), int(point_y)), radius, 255, thickness=-1)
        return not np.any(overlap & (allowed_mask == 0))

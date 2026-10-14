"""synthetic_topology: decorate pcb terminals."""
from __future__ import annotations
import math
import random
import cv2
import numpy as np


class DecoratePcbTerminalsMixin:
    def _decorate_terminals(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
        family: str,
        rng: random.Random,
    ) -> np.ndarray:
        if int(np.count_nonzero(mask_u8)) <= 0 or not anchors:
            return mask_u8
        if self._domain() == 'ic':
            return self._decorate_ic_contacts(mask_u8, anchors)
        return self._decorate_pcb_terminals(mask_u8, anchors, family, rng)

    def _decorate_pcb_terminals(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
        family: str,
        rng: random.Random,
    ) -> np.ndarray:
        updated = mask_u8.copy()
        pad_radius = max(2, self._width_range()[1] + 1)
        boundary_anchors = [anchor for anchor in anchors if self._is_boundary_anchor(anchor, updated.shape)]
        inner_anchors = [anchor for anchor in anchors if anchor not in boundary_anchors]
        if not boundary_anchors:
            boundary_anchors = anchors[:]
        rng.shuffle(boundary_anchors)
        rng.shuffle(inner_anchors)

        pad_count = max(2, min(len(boundary_anchors), max(2, int(round(self.params.trace_count * 0.35)))))
        for anchor in boundary_anchors[:pad_count]:
            cv2.circle(updated, anchor, pad_radius, 255, thickness=-1)

        via_count_range = self.params.via_count_range or self._default_via_count_range(family)
        via_min = max(0, int(via_count_range[0]))
        via_max = max(via_min, int(via_count_range[1]))
        via_target = 0 if via_max <= 0 else rng.randint(via_min, via_max)
        via_added = 0
        via_candidates = inner_anchors if inner_anchors else boundary_anchors
        for anchor in via_candidates:
            if via_added >= via_target:
                break
            outer_radius = max(3, pad_radius + rng.randint(0, 2))
            inner_radius = max(1, outer_radius // 2)
            if not self._ring_fits(updated, anchor, outer_radius):
                continue
            cv2.circle(updated, anchor, outer_radius, 255, thickness=-1)
            cv2.circle(updated, anchor, inner_radius, 0, thickness=-1)
            via_added += 1
        return updated

    def _decorate_ic_contacts(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
    ) -> np.ndarray:
        updated = mask_u8.copy()
        bump_half = max(1, self._width_range()[1] + 1)
        if not anchors:
            return updated
        target_count = max(2, min(len(anchors), int(round(self.params.trace_count * 0.6))))
        if len(anchors) <= target_count:
            selected_anchors = anchors
        else:
            sample_indexes = np.linspace(0, len(anchors) - 1, target_count)
            selected_anchors = [anchors[int(round(index))] for index in sample_indexes]
        for anchor_x, anchor_y in selected_anchors:
            x1 = max(0, anchor_x - bump_half)
            y1 = max(0, anchor_y - bump_half)
            x2 = min(updated.shape[1], anchor_x + bump_half + 1)
            y2 = min(updated.shape[0], anchor_y + bump_half + 1)
            updated[y1:y2, x1:x2] = 255
        return updated

    def _default_via_count_range(self, family: str) -> tuple[int, int]:
        if self._domain() == 'ic':
            return (0, 0)
        if family == 'cell_array':
            upper = max(2, min(8, int(round(self.params.trace_count / 2.0))))
            return (2, upper)
        if family == 'tree':
            return (1, max(2, min(5, int(round(self.params.trace_count / 3.0)))))
        return (1, max(2, min(4, int(round(self.params.trace_count / 4.0)))))

    def _resolve_parallel_offsets(
        self,
        height: int,
        width: int,
        orientation: str,
        spacing: int,
        requested: int,
    ) -> list[int]:
        margin = self._effective_margin(height, width)
        if orientation == 'horizontal':
            axis_length = max(8, height - (margin * 2))
            count = max(1, min(requested, max(1, axis_length // max(1, spacing))))
            if count == 1:
                return [height // 2]
            start = margin + max(0, (axis_length - ((count - 1) * spacing)) // 2)
            return [int(np.clip(start + (index * spacing), margin, height - margin - 1)) for index in range(count)]
        if orientation == 'vertical':
            axis_length = max(8, width - (margin * 2))
            count = max(1, min(requested, max(1, axis_length // max(1, spacing))))
            if count == 1:
                return [width // 2]
            start = margin + max(0, (axis_length - ((count - 1) * spacing)) // 2)
            return [int(np.clip(start + (index * spacing), margin, width - margin - 1)) for index in range(count)]
        normal_limit = max(6, int(round((min(height, width) / 2.0) - margin)))
        count = max(1, min(requested, max(1, int((normal_limit * 2) // max(1, spacing)))))
        if count == 1:
            return [0]
        start = -normal_limit + max(0, ((normal_limit * 2) - ((count - 1) * spacing)) // 2)
        return [int(np.clip(start + (index * spacing), -normal_limit, normal_limit)) for index in range(count)]

    def _full_span_segment(
        self,
        height: int,
        width: int,
        orientation: str,
        offset: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if orientation == 'horizontal':
            y = int(np.clip(offset, 0, height - 1))
            return (0, y), (width - 1, y)
        if orientation == 'vertical':
            x = int(np.clip(offset, 0, width - 1))
            return (x, 0), (x, height - 1)
        angle_deg = 45.0 if orientation == 'diag_pos' else -45.0
        angle_rad = math.radians(angle_deg)
        direction = (math.cos(angle_rad), math.sin(angle_rad))
        normal = (-direction[1], direction[0])
        center_x = ((width - 1) / 2.0) + (normal[0] * float(offset))
        center_y = ((height - 1) / 2.0) + (normal[1] * float(offset))
        extent = max(width, height) * 2.0
        start = (
            int(round(center_x - (direction[0] * extent))),
            int(round(center_y - (direction[1] * extent))),
        )
        end = (
            int(round(center_x + (direction[0] * extent))),
            int(round(center_y + (direction[1] * extent))),
        )
        clipped = cv2.clipLine((0, 0, width, height), start, end)
        if not clipped[0]:
            return None
        return tuple(int(v) for v in clipped[1]), tuple(int(v) for v in clipped[2])

    def _random_parallel_offset(
        self,
        height: int,
        width: int,
        orientation: str,
        rng: random.Random,
    ) -> int:
        margin = self._effective_margin(height, width)
        if orientation == 'horizontal':
            return rng.randint(margin, max(margin + 1, height - margin - 1))
        if orientation == 'vertical':
            return rng.randint(margin, max(margin + 1, width - margin - 1))
        normal_limit = max(6, int(round((min(height, width) / 2.0) - margin)))
        return rng.randint(-normal_limit, normal_limit)

    def _build_axis_branch_endpoint(
        self,
        start_xy: tuple[int, int],
        *,
        length: int,
        direction: int,
        axis: str,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        if axis == 'x':
            return (
                int(np.clip(start_xy[0] + (int(direction) * int(length)), 0, width - 1)),
                int(start_xy[1]),
            )
        return (
            int(start_xy[0]),
            int(np.clip(start_xy[1] + (int(direction) * int(length)), 0, height - 1)),
        )

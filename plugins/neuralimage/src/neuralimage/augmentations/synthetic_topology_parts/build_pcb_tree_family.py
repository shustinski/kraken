"""synthetic_topology: build pcb tree family."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildPcbTreeFamilyMixin:
    def _build_pcb_parallel_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        orientation = str(rng.choices(
            ('horizontal', 'vertical', 'diag_pos', 'diag_neg'),
            weights=(0.32, 0.18, 0.25, 0.25),
            k=1,
        )[0])
        requested = max(2, int(self.params.trace_count))
        max_width = self._width_range()[1]
        spacing = max(8, (max_width * 3) + max(2, int(self.params.trace_clearance)) + 3)
        for offset in self._resolve_parallel_offsets(height, width, orientation, spacing, requested):
            trace_half_width = rng.randint(*self._width_range())
            segment = self._full_span_segment(height, width, orientation, offset)
            if segment is None:
                continue
            candidate_mask = np.zeros_like(mask_u8)
            cv2.line(candidate_mask, segment[0], segment[1], 255, thickness=(trace_half_width * 2 + 1))
            if np.any((candidate_mask > 0) & (self._build_guard_mask(mask_u8) > 0)):
                continue
            mask_u8 = np.maximum(mask_u8, candidate_mask)
            anchors.extend((segment[0], segment[1], self._midpoint(segment[0], segment[1])))
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_pcb_independent_family_mask(height, width, rng)
        return mask_u8, anchors

    def _build_pcb_independent_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        requested_trace_count = max(1, int(self.params.trace_count))
        for _trace_index in range(requested_trace_count):
            candidate_mask, candidate_anchors = self._build_pcb_route_component(
                height,
                width,
                rng,
                occupied_guard_mask=self._build_guard_mask(mask_u8),
            )
            if candidate_mask is None:
                continue
            mask_u8 = np.maximum(mask_u8, candidate_mask)
            anchors.extend(candidate_anchors)
        return mask_u8, anchors

    def _build_pcb_tree_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        trace_half_width = rng.randint(*self._width_range())
        horizontal = bool(rng.getrandbits(1))
        margin = self._effective_margin(height, width)
        if horizontal:
            center_y = rng.randint(margin, max(margin + 1, height - margin - 1))
            trunk_start = (0, center_y)
            trunk_end = (width - 1, center_y)
        else:
            center_x = rng.randint(margin, max(margin + 1, width - margin - 1))
            trunk_start = (center_x, 0)
            trunk_end = (center_x, height - 1)
        cv2.line(mask_u8, trunk_start, trunk_end, 255, thickness=(trace_half_width * 2 + 1))
        anchors.extend((trunk_start, trunk_end))

        branch_target = max(2, min(18, int(self.params.trace_count)))
        for branch_fraction in self._resolve_branch_fractions(branch_target):
            branch_half_width = rng.randint(*self._width_range())
            branch_start = self._point_on_segment(trunk_start, trunk_end, branch_fraction)
            if horizontal:
                branch_end = self._build_axis_branch_endpoint(
                    branch_start,
                    length=max(10, int(round(height * rng.uniform(0.18, 0.44)))),
                    direction=-1 if rng.random() < 0.5 else 1,
                    axis='y',
                    width=width,
                    height=height,
                )
            else:
                branch_end = self._build_axis_branch_endpoint(
                    branch_start,
                    length=max(10, int(round(width * rng.uniform(0.18, 0.44)))),
                    direction=-1 if rng.random() < 0.5 else 1,
                    axis='x',
                    width=width,
                    height=height,
                )
            cv2.line(mask_u8, branch_start, branch_end, 255, thickness=(branch_half_width * 2 + 1))
            if rng.random() < 0.6:
                diagonal_end = self._build_pcb_diagonal_branch_endpoint(
                    branch_end,
                    width=width,
                    height=height,
                    rng=rng,
                )
                cv2.line(mask_u8, branch_end, diagonal_end, 255, thickness=(branch_half_width * 2 + 1))
                anchors.append(diagonal_end)
            anchors.append(branch_end)
        return mask_u8, anchors

    def _build_pcb_cell_array_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        margin = self._effective_margin(height, width)
        motif_count = max(2, int(self.params.trace_count))
        cols = max(2, int(round(np.sqrt(motif_count))))
        rows = max(1, int(np.ceil(float(motif_count) / float(cols))))
        cell_w = max(22, (width - (margin * 2)) // max(1, cols))
        cell_h = max(22, (height - (margin * 2)) // max(1, rows))
        motif_index = 0
        for row in range(rows):
            for col in range(cols):
                if motif_index >= motif_count:
                    break
                x1 = margin + (col * cell_w)
                y1 = margin + (row * cell_h)
                x2 = min(width - margin, x1 + cell_w)
                y2 = min(height - margin, y1 + cell_h)
                candidate_mask, candidate_anchors = self._build_pcb_cell_motif(mask_u8.shape, x1, y1, x2, y2, rng)
                if candidate_mask is None:
                    motif_index += 1
                    continue
                if np.any((candidate_mask > 0) & (self._build_guard_mask(mask_u8) > 0)):
                    motif_index += 1
                    continue
                mask_u8 = np.maximum(mask_u8, candidate_mask)
                anchors.extend(candidate_anchors)
                motif_index += 1
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_pcb_parallel_family_mask(height, width, rng)
        return mask_u8, anchors

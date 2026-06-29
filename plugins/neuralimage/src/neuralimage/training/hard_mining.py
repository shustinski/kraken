from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from neuralimage.targets.basic import generate_boundary_map, generate_skeleton_map


def compute_geometry_difficulty_score(mask: np.ndarray) -> float:
    """Score patch difficulty from geometry: thin wires, boundaries, small structures."""
    binary = (np.asarray(mask) >= 0.5).astype(np.uint8)
    if not bool(binary.any()):
        return 0.0
    boundary = generate_boundary_map(binary)
    skeleton = generate_skeleton_map(binary)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thinness = 1.0 - np.clip(distance / max(1.0, distance.max()), 0.0, 1.0)
    component_count, _ = cv2.connectedComponents(binary)
    density = float(binary.mean())
    boundary_score = float(boundary.mean())
    skeleton_score = float(skeleton.mean())
    thin_score = float(thinness.mean())
    small_component_bonus = 0.1 * max(0, int(component_count) - 1)
    return float(
        0.35 * boundary_score
        + 0.25 * skeleton_score
        + 0.25 * thin_score
        + 0.10 * density
        + small_component_bonus
    )


@dataclass
class DifficultyPatchSampler:
    """Offline/online patch difficulty scoring for hard example mining."""

    geometry_weight: float = 0.5
    loss_weight: float = 0.5

    def combine_scores(self, *, geometry_score: float, loss_score: float) -> float:
        geometry = float(max(0.0, geometry_score))
        loss = float(max(0.0, loss_score))
        return self.geometry_weight * geometry + self.loss_weight * loss

    def score_mask(self, mask: np.ndarray, *, loss_score: float = 0.0) -> float:
        return self.combine_scores(geometry_score=compute_geometry_difficulty_score(mask), loss_score=loss_score)


class OfflineHardDatasetBuilder:
    """Build an offline hard-example index from geometry difficulty scores."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def build(self, samples: list[tuple[str, np.ndarray]], *, top_fraction: float = 0.25) -> Path:
        scored: list[tuple[str, float]] = []
        sampler = DifficultyPatchSampler()
        for sample_id, mask in samples:
            scored.append((sample_id, sampler.score_mask(mask)))
        scored.sort(key=lambda item: item[1], reverse=True)
        keep_count = max(1, int(len(scored) * float(top_fraction)))
        selected = scored[:keep_count]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f'{sample_id}\t{score:.6f}' for sample_id, score in selected]
        self.output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return self.output_path

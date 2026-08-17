from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from neuralimage.targets.basic import generate_boundary_map, generate_skeleton_map
from neuralimage.targets.geometry import generate_junction_map


@dataclass(frozen=True)
class GeometryDifficultyFeatures:
    thin_wire_length: float
    boundary_density: float
    small_component_density: float
    junction_density: float
    routing_density: float
    compact_component_density: float

    @property
    def score(self) -> float:
        return float(
            0.25 * self.thin_wire_length
            + 0.20 * self.boundary_density
            + 0.15 * self.small_component_density
            + 0.15 * self.junction_density
            + 0.15 * self.routing_density
            + 0.10 * self.compact_component_density
        )


def compute_geometry_difficulty_features(mask: np.ndarray) -> GeometryDifficultyFeatures:
    binary = (np.asarray(mask) >= 0.5).astype(np.uint8)
    if binary.ndim == 3:
        binary = np.squeeze(binary)
    if not bool(binary.any()):
        return GeometryDifficultyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pixels = float(binary.size)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = generate_skeleton_map(binary) > 0.5
    thin_skeleton = skeleton & (distance <= 2.5)
    boundary = generate_boundary_map(binary) > 0.5
    junctions = generate_junction_map(binary).max(axis=-1) > 0.5
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if component_count > 1 else np.empty(0)
    small_components = int((component_areas <= 64).sum())
    compact_components = 0
    for component_id in range(1, component_count):
        component = (labels == component_id).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        area = float(stats[component_id, cv2.CC_STAT_AREA])
        if perimeter > 0.0 and 4.0 * np.pi * area / (perimeter * perimeter) > 0.55:
            compact_components += 1
    return GeometryDifficultyFeatures(
        thin_wire_length=float(thin_skeleton.sum() / pixels),
        boundary_density=float(boundary.sum() / pixels),
        small_component_density=float(min(1.0, small_components / 16.0)),
        junction_density=float(min(1.0, junctions.sum() / max(1.0, pixels * 0.01))),
        routing_density=float(binary.mean()),
        compact_component_density=float(min(1.0, compact_components / 16.0)),
    )


def compute_geometry_difficulty_score(mask: np.ndarray) -> float:
    return compute_geometry_difficulty_features(mask).score


@dataclass
class DifficultyPatchSampler:
    geometry_weight: float = 0.5
    loss_weight: float = 0.5
    exploration_floor: float = 0.1
    score_clip: float = 5.0

    def combine_scores(self, *, geometry_score: float, loss_score: float) -> float:
        geometry = float(np.clip(geometry_score, 0.0, self.score_clip))
        loss = float(np.clip(loss_score, 0.0, self.score_clip))
        combined = self.geometry_weight * geometry + self.loss_weight * loss
        return float(max(self.exploration_floor, combined))

    def score_mask(self, mask: np.ndarray, *, loss_score: float = 0.0) -> float:
        return self.combine_scores(geometry_score=compute_geometry_difficulty_score(mask), loss_score=loss_score)


class OfflineHardDatasetBuilder:
    """Write ranked patch metadata without duplicating source image crops."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def build(
        self,
        samples: list[tuple[str, np.ndarray]],
        *,
        top_fraction: float = 0.25,
        historical_losses: dict[str, float] | None = None,
        rois: dict[str, tuple[int, int, int, int]] | None = None,
        frame_ids: dict[str, str] | None = None,
    ) -> Path:
        sampler = DifficultyPatchSampler()
        rows: list[dict[str, object]] = []
        for sample_id, mask in samples:
            features = compute_geometry_difficulty_features(mask)
            loss = float((historical_losses or {}).get(sample_id, 0.0))
            rows.append({
                'sample_id': sample_id,
                'frame': (frame_ids or {}).get(sample_id, sample_id),
                'roi': list((rois or {}).get(sample_id, (0, 0, int(mask.shape[-1]), int(mask.shape[-2])))),
                'geometry': asdict(features),
                'historical_loss': loss,
                'score': sampler.combine_scores(geometry_score=features.score, loss_score=loss),
            })
        rows.sort(key=lambda row: float(row['score']), reverse=True)
        keep_count = min(len(rows), max(1, int(np.ceil(len(rows) * np.clip(top_fraction, 0.0, 1.0))))) if rows else 0
        selected = rows[:keep_count]
        for rank, row in enumerate(selected, start=1):
            row['rank'] = rank
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + '.tmp')
        temporary.write_text('\n'.join(json.dumps(row, separators=(',', ':')) for row in selected) + ('\n' if selected else ''), encoding='utf-8')
        os.replace(temporary, self.output_path)
        csv_path = self.output_path.with_suffix('.csv')
        csv_temporary = csv_path.with_suffix('.csv.tmp')
        with csv_temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('rank', 'sample_id', 'frame', 'roi', 'historical_loss', 'score'))
            writer.writeheader()
            for row in selected:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        os.replace(csv_temporary, csv_path)
        return self.output_path

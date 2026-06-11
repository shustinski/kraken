"""Raster layer builders for comparison results."""
from __future__ import annotations

import numpy as np

from .geometry import boundary_mask
from .models import RasterLayer
from .skeleton import skeletonize_mask


def pairwise_raster_layers(
    *,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    probability_a: np.ndarray | None,
    probability_b: np.ndarray | None,
    threshold: float,
    include_standard_layers: bool = True,
    include_skeleton_layers: bool = True,
    boundary_a: np.ndarray | None = None,
    boundary_b: np.ndarray | None = None,
    skeleton_a: np.ndarray | None = None,
    skeleton_b: np.ndarray | None = None,
) -> tuple[RasterLayer, ...]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    common = a & b
    a_only = a & ~b
    b_only = b & ~a
    xor = a ^ b
    union = a | b
    layers: list[RasterLayer] = [
        RasterLayer("mask_common", "A and B", common.astype(np.float32), 0.55, False, "pixel"),
        RasterLayer("mask_a_only", "A only", a_only.astype(np.float32), 0.75, True, "pixel"),
        RasterLayer("mask_b_only", "B only", b_only.astype(np.float32), 0.75, True, "pixel"),
        RasterLayer("mask_xor", "A xor B", xor.astype(np.float32), 0.85, True, "pixel"),
        RasterLayer("mask_union", "A or B", union.astype(np.float32), 0.35, False, "pixel"),
    ]
    if include_standard_layers:
        prepared_boundary_a = boundary_mask(a) if boundary_a is None else np.asarray(boundary_a, dtype=bool)
        prepared_boundary_b = boundary_mask(b) if boundary_b is None else np.asarray(boundary_b, dtype=bool)
        layers.extend(
            [
                RasterLayer("boundary_a", "Boundary A", prepared_boundary_a.astype(np.float32), 0.9, False, "geometry"),
                RasterLayer("boundary_b", "Boundary B", prepared_boundary_b.astype(np.float32), 0.9, False, "geometry"),
            ]
        )
        if include_skeleton_layers:
            prepared_skeleton_a = skeletonize_mask(a) if skeleton_a is None else np.asarray(skeleton_a, dtype=bool)
            prepared_skeleton_b = skeletonize_mask(b) if skeleton_b is None else np.asarray(skeleton_b, dtype=bool)
            layers.extend(
                [
                    RasterLayer("skeleton_a", "Skeleton A", prepared_skeleton_a.astype(np.float32), 0.9, False, "skeleton"),
                    RasterLayer("skeleton_b", "Skeleton B", prepared_skeleton_b.astype(np.float32), 0.9, False, "skeleton"),
                    RasterLayer("skeleton_xor", "Skeleton xor", np.logical_xor(prepared_skeleton_a, prepared_skeleton_b).astype(np.float32), 0.9, False, "skeleton"),
                ]
            )
    if include_standard_layers and probability_a is not None and probability_b is not None:
        pa = np.clip(np.asarray(probability_a, dtype=np.float32), 0.0, 1.0)
        pb = np.clip(np.asarray(probability_b, dtype=np.float32), 0.0, 1.0)
        layers.extend(
            [
                RasterLayer("soft_abs_difference", "|P_A - P_B|", np.abs(pa - pb).astype(np.float32), 0.85, True, "soft_confidence"),
                RasterLayer("soft_signed_difference", "P_A - P_B", np.asarray((pa - pb + 1.0) * 0.5, dtype=np.float32), 0.65, False, "soft_confidence"),
                RasterLayer("threshold_crossing_map", "Threshold crossing", np.logical_xor(pa >= float(threshold), pb >= float(threshold)).astype(np.float32), 0.8, False, "soft_confidence"),
            ]
        )
    return tuple(layers)


def ensemble_raster_layers(vote_map: np.ndarray, consensus_mask: np.ndarray, uncertainty: np.ndarray) -> tuple[RasterLayer, ...]:
    return (
        RasterLayer("vote_map", "Vote map", np.asarray(vote_map, dtype=np.float32), 0.70, True, "ensemble"),
        RasterLayer("consensus_mask", "Consensus mask", np.asarray(consensus_mask, dtype=bool).astype(np.float32), 0.55, True, "ensemble"),
        RasterLayer("ensemble_uncertainty", "Ensemble uncertainty", np.asarray(uncertainty, dtype=np.float32), 0.75, True, "ensemble"),
    )

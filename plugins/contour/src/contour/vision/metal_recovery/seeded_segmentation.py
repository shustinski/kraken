"""Conductor segmentation: watershed, random walker, graph cut, reconstruction, closed boundary."""

from __future__ import annotations

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from .closed_boundary import closed_boundary_mask
from .gradient_watershed import (
    ConductorSeeds,
    GradientWatershedConfig,
    build_conductor_seeds,
    gradient_watershed_mask,
)
from .structural_watershed import structural_watershed_mask, structural_watershed_config_from_object

_SEEDED_MAX_SIDE = 640
_RANDOM_WALKER_BETA = 90.0
_RANDOM_WALKER_ITERATIONS = 160
_GRAB_CUT_ITERATIONS = 5


def _empty_mask(gray: np.ndarray) -> np.ndarray:
    return np.zeros(gray.shape[:2], dtype=np.uint8)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _downsample_marker(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Conservatively retain every full-resolution hard marker."""
    if mask.shape == (height, width):
        return ensure_binary_mask(mask)
    coverage = cv2.resize(
        (mask > 0).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return np.where(coverage > 0.0, 255, 0).astype(np.uint8)


def _prepare_working_image(
    gray: np.ndarray,
    seeds: ConductorSeeds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample large frames so iterative solvers stay interactive."""
    height, width = int(gray.shape[0]), int(gray.shape[1])
    longest = max(height, width)
    core = seeds.core_seeds
    groove = seeds.groove_seeds
    if longest <= _SEEDED_MAX_SIDE:
        working = gray
    else:
        scale = _SEEDED_MAX_SIDE / float(longest)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        working = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        core = _downsample_marker(core, new_width, new_height)
        groove = _downsample_marker(groove, new_width, new_height)
    background = groove > 0
    foreground = (core > 0) & ~background
    return working, (foreground.astype(np.uint8) * 255), (background.astype(np.uint8) * 255)


_MARKER_SEPARATION_PX = 3
_RANDOM_WALKER_EDGE_SCALE = 32.0


def _separated_markers(
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    max_radius: int = _MARKER_SEPARATION_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull dense core/gap labels apart so the unlabeled band can be solved.

    Watershed-style seeds often partition the whole frame.  Random Walker and
    GrabCut then have no free pixels, so β and iteration counts cannot change
    the mask.  Eroding both classes leaves a corridor along the intensity edge.
    """
    core = ensure_binary_mask(core_seeds)
    groove = ensure_binary_mask(groove_seeds)
    groove = cv2.bitwise_and(groove, cv2.bitwise_not(core))
    if max_radius <= 0 or not np.any(core) or not np.any(groove):
        return core, groove
    for radius in range(int(max_radius), 0, -1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        shrunk_core = cv2.erode(core, kernel)
        shrunk_groove = cv2.erode(groove, kernel)
        shrunk_groove = cv2.bitwise_and(shrunk_groove, cv2.bitwise_not(shrunk_core))
        if np.any(shrunk_core) and np.any(shrunk_groove):
            return shrunk_core, shrunk_groove
    return core, groove


def reconstruction_from_seeds(
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    erode_px: int = 0,
) -> np.ndarray:
    """Geodesic fill of metal cores, blocked by gap seeds."""
    cores = ensure_binary_mask(core_seeds)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(cores, kernel)
        if np.any(eroded):
            cores = eroded
    allowed = np.where(groove_seeds > 0, 0, 255).astype(np.uint8)
    count, labels = cv2.connectedComponents((allowed > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return cores
    core_hits = np.bincount(
        labels.ravel(),
        weights=(cores > 0).ravel().astype(np.float64),
        minlength=count,
    )
    keep = core_hits > 0
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def random_walker_from_seeds(
    gray: np.ndarray,
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    beta: float = _RANDOM_WALKER_BETA,
    max_iterations: int = _RANDOM_WALKER_ITERATIONS,
) -> np.ndarray:
    """Solve the random-walker Laplace equation by vectorized Jacobi iteration."""
    core, groove = _separated_markers(core_seeds, groove_seeds)
    image = ensure_uint8(gray).astype(np.float32)
    height, width = image.shape
    foreground = core > 0
    background = (groove > 0) & ~foreground
    if not np.any(foreground) or not np.any(background):
        return ensure_binary_mask(core_seeds)

    probability = np.full((height, width), 0.5, dtype=np.float32)
    probability[foreground] = 1.0
    probability[background] = 0.0
    fixed = foreground | background

    inv_scale_sq = 1.0 / (_RANDOM_WALKER_EDGE_SCALE * _RANDOM_WALKER_EDGE_SCALE)
    strength = max(1.0, float(beta))
    delta_x = image[:, 1:] - image[:, :-1]
    weight_x = np.exp(-strength * delta_x * delta_x * inv_scale_sq).astype(np.float32)
    delta_y = image[1:, :] - image[:-1, :]
    weight_y = np.exp(-strength * delta_y * delta_y * inv_scale_sq).astype(np.float32)

    for _ in range(max(1, int(max_iterations))):
        numerator = np.zeros((height, width), dtype=np.float32)
        denominator = np.zeros((height, width), dtype=np.float32)
        numerator[:, 1:] += weight_x * probability[:, :-1]
        denominator[:, 1:] += weight_x
        numerator[:, :-1] += weight_x * probability[:, 1:]
        denominator[:, :-1] += weight_x
        numerator[1:, :] += weight_y * probability[:-1, :]
        denominator[1:, :] += weight_y
        numerator[:-1, :] += weight_y * probability[1:, :]
        denominator[:-1, :] += weight_y
        updated = numerator / np.maximum(denominator, 1e-8)
        next_probability = np.where(fixed, probability, updated)
        if float(np.max(np.abs(next_probability - probability))) < 1e-4:
            probability = next_probability
            break
        probability = next_probability
    return ensure_binary_mask((probability >= 0.5).astype(np.uint8) * 255)


def graph_cut_from_seeds(
    gray: np.ndarray,
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    iterations: int = _GRAB_CUT_ITERATIONS,
) -> np.ndarray:
    """GrabCut with hard core/gap seeds and a reconstructed interior as the FG prior."""
    core, groove = _separated_markers(core_seeds, groove_seeds)
    foreground = core > 0
    background = (groove > 0) & ~foreground
    if not np.any(foreground) or not np.any(background):
        return ensure_binary_mask(core_seeds)

    grown = reconstruction_from_seeds(core, groove)
    kernel = np.ones((3, 3), np.uint8)
    band = cv2.bitwise_xor(cv2.dilate(grown, kernel), cv2.erode(grown, kernel))
    mask = np.full(gray.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[grown > 0] = cv2.GC_FGD
    mask[band > 0] = cv2.GC_PR_FGD
    mask[background] = cv2.GC_BGD
    mask[foreground] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR),
            mask,
            (0, 0, int(gray.shape[1]), int(gray.shape[0])),
            background_model,
            foreground_model,
            _GRAB_CUT_ITERATIONS if iterations <= 0 else int(iterations),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return ensure_binary_mask(grown)

    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result[foreground] = 255
    result[background] = 0
    return ensure_binary_mask(result)


def seeded_segmentation_mask(
    gray: np.ndarray,
    strategy: str,
    config: GradientWatershedConfig,
) -> np.ndarray:
    """Dispatch a seeded conductor algorithm; unknown tokens fall back to watershed."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return _empty_mask(source)
    if strategy == "gradient_watershed":
        return gradient_watershed_mask(source, config)
    if strategy == "structural_watershed":
        return structural_watershed_mask(
            source,
            config,
            structural_watershed_config_from_object(config),
        )
    if strategy == "closed_boundary":
        return closed_boundary_mask(source, config)

    seeds = build_conductor_seeds(source, config)
    if seeds is None:
        return _empty_mask(source)
    if not seeds.has_both_classes:
        return seeds.fallback_mask()

    working, core, groove = _prepare_working_image(source, seeds)
    if not np.any(core) or not np.any(groove):
        return seeds.fallback_mask()

    if strategy == "random_walker":
        mask = random_walker_from_seeds(
            working,
            core,
            groove,
            beta=float(config.random_walker_beta),
            max_iterations=int(config.random_walker_iterations),
        )
    elif strategy == "graph_cut":
        mask = graph_cut_from_seeds(
            working,
            core,
            groove,
            iterations=int(config.graph_cut_iterations),
        )
    else:
        mask = reconstruction_from_seeds(
            core,
            groove,
            erode_px=int(config.reconstruction_erode_px),
        )
    restored = ensure_binary_mask(
        _resize_mask(mask, int(source.shape[1]), int(source.shape[0]))
    )
    # The iterative solve may be coarse, but its full-resolution hard evidence
    # is authoritative.  Reapply it after interpolation so a confirmed 1–2 px
    # separating groove or narrow conductor cannot disappear permanently.
    restored[seeds.core_seeds > 0] = 255
    restored[seeds.groove_seeds > 0] = 0
    return ensure_binary_mask(restored)

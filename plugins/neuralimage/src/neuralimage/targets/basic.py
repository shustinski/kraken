from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, skeletonize, thin


def _to_binary_mask(mask: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError('Expected a single-channel 2D mask.')
    max_value = float(np.max(array)) if array.size else 1.0
    threshold_value = threshold if max_value <= 1.0 else threshold * max(1.0, max_value)
    return (array >= threshold_value).astype(np.uint8)


def generate_boundary_map(mask: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    binary = _to_binary_mask(mask)
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 1)
    return boundary.astype(np.float32)


def generate_skeleton_map(mask: np.ndarray, *, iterations: int = 0) -> np.ndarray:
    binary = _to_binary_mask(mask)
    if not bool(binary.any()):
        return np.zeros_like(binary, dtype=np.float32)
    iteration_limit = int(iterations)
    skeleton = (
        thin(binary.astype(bool), max_num_iter=iteration_limit)
        if iteration_limit > 0
        else skeletonize(binary.astype(bool), method='zhang')
    )
    return skeleton.astype(np.float32, copy=False)


def generate_distance_transform(mask: np.ndarray, *, clip: float = 32.0) -> np.ndarray:
    binary = _to_binary_mask(mask)
    if not bool(binary.any()):
        return np.zeros_like(binary, dtype=np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    clip = max(float(clip), 1e-6)
    return (np.clip(distance, 0.0, clip) / clip).astype(np.float32)


def generate_signed_distance_field(mask: np.ndarray, *, clip: float = 32.0) -> np.ndarray:
    binary = _to_binary_mask(mask)
    clip = max(float(clip), 1e-6)
    if not bool(binary.any()):
        background = np.ones_like(binary, dtype=np.uint8)
        outside = cv2.distanceTransform(background, cv2.DIST_L2, 5)
        clipped = np.clip(outside, 0.0, float(clip))
        return (-clipped / float(clip)).astype(np.float32)
    if bool(binary.all()):
        return np.ones_like(binary, dtype=np.float32)

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    sdf = inside.astype(np.float32) - outside.astype(np.float32)
    sdf = np.clip(sdf, -float(clip), float(clip)) / float(clip)
    return sdf.astype(np.float32)


def generate_local_thickness_map(mask: np.ndarray, *, max_thickness: float = 64.0) -> np.ndarray:
    binary = _to_binary_mask(mask)
    if not bool(binary.any()):
        return np.zeros_like(binary, dtype=np.float32)
    skeleton, radius = medial_axis(binary.astype(bool), return_distance=True, rng=0)
    if not bool(skeleton.any()):
        return np.zeros_like(binary, dtype=np.float32)
    nearest_indices = distance_transform_edt(
        ~skeleton,
        return_distances=False,
        return_indices=True,
    )
    max_thickness = max(float(max_thickness), 1e-6)
    medial_thickness = radius.astype(np.float32) * skeleton * 2.0
    thickness = medial_thickness[tuple(nearest_indices)] * binary
    return (np.clip(thickness, 0.0, max_thickness) / max_thickness).astype(np.float32)

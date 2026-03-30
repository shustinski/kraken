from __future__ import annotations

import cv2
import numpy as np
import torch

from lib.data_interfaces import RANDOM_ARTIFACT_TYPES

ARTIFACT_TYPES: tuple[str, ...] = RANDOM_ARTIFACT_TYPES


def _make_master_seed(seed: int | None = None) -> int:
    if seed is not None:
        return int(seed)
    return int(np.random.randint(0, 2**31 - 1))


def _spawn_subseed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _scale_value(patch_scale: float, value: float, *, minimum: float) -> float:
    scale = max(16.0, float(patch_scale)) / 256.0
    return max(float(minimum), float(value) * scale)


def _scaled_range(
    patch_scale: float,
    low: float,
    high: float,
    *,
    min_low: float,
    min_high: float,
) -> tuple[float, float]:
    scaled_low = _scale_value(patch_scale, low, minimum=min_low)
    scaled_high = _scale_value(patch_scale, high, minimum=max(min_high, scaled_low + 1e-3))
    return float(scaled_low), float(max(scaled_high, scaled_low + 1e-3))


def _kernel_size(size: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    image_h, image_w = int(size[0]), int(size[1])
    kx = max(1, min(int(width), image_w))
    ky = max(1, min(int(height), image_h))
    if kx % 2 == 0:
        kx = max(1, kx - 1)
    if ky % 2 == 0:
        ky = max(1, ky - 1)
    return kx, ky


def _ellipse_kernel(size: tuple[int, int], width: int, height: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _kernel_size(size, width, height))


def _smoothstep01(values: np.ndarray) -> np.ndarray:
    clamped = np.clip(values, 0.0, 1.0)
    return clamped * clamped * (3.0 - (2.0 * clamped))


def _sample_artifact_parameters(artifact_type: str, patch_scale: float) -> dict[str, object]:
    if artifact_type == 'dust':
        return {
            'n_blobs': (3, 7),
            'n_blobs_range': (3, 7),
            'radius_range': _scaled_range(patch_scale, 10.0, 28.0, min_low=3.0, min_high=6.0),
            'elongation': (0.7, 1.3),
            'threshold': (0.23, 0.33),
            'threshold_range': (0.23, 0.33),
            'anisotropy': (1.0, 1.0),
            'satellites': True,
            'weights': (0.68, 0.18, 0.08, 0.06),
            'angle_range': (0.0, 180.0),
            'postprocess': 'close_open_3',
            'microvoid_range': (0, 3),
            'alpha_out_scale': _scaled_range(patch_scale, 1.2, 2.4, min_low=0.7, min_high=1.0),
            'alpha_in_depth': 0.18,
            'intensity_range_255': (150.0, 235.0),
            'intensity_mix': (0.42, 0.28, 0.18, 0.12),
            'edge_delta': (-18.0, -8.0),
            'halo_strength': (6.0, 14.0),
            'halo_alpha': (0.10, 0.24),
            'halo_sign_mode': 'random',
            'gray_level_range': (0.84, 1.00),
        }
    if artifact_type == 'resist_residue':
        return {
            'n_blobs': (4, 9),
            'n_blobs_range': (4, 9),
            'radius_range': _scaled_range(patch_scale, 10.0, 28.0, min_low=3.0, min_high=6.0),
            'elongation': (0.8, 1.7),
            'threshold': (0.18, 0.28),
            'threshold_range': (0.18, 0.28),
            'anisotropy': (1.0, 1.6),
            'satellites': True,
            'weights': (0.50, 0.20, 0.12, 0.18),
            'angle_range': (-25.0, 25.0),
            'postprocess': 'close_5x3',
            'microvoid_range': (1, 5),
            'alpha_out_scale': _scaled_range(patch_scale, 2.2, 4.0, min_low=1.0, min_high=1.6),
            'alpha_in_depth': 0.35,
            'intensity_range_255': (105.0, 195.0),
            'intensity_mix': (0.22, 0.24, 0.18, 0.36),
            'edge_delta': (6.0, 16.0),
            'halo_strength': (8.0, 18.0),
            'halo_alpha': (0.18, 0.35),
            'halo_sign_mode': 'positive',
            'gray_level_range': (0.45, 0.90),
        }
    if artifact_type == 'etch_residue':
        return {
            'n_blobs': (3, 6),
            'n_blobs_range': (3, 6),
            'radius_range': _scaled_range(patch_scale, 12.0, 34.0, min_low=4.0, min_high=8.0),
            'elongation': (0.9, 2.2),
            'threshold': (0.19, 0.27),
            'threshold_range': (0.19, 0.27),
            'anisotropy': (1.0, 2.2),
            'satellites': False,
            'weights': (0.46, 0.18, 0.10, 0.26),
            'angle_range': (-25.0, 25.0),
            'postprocess': 'erode_dilate_5x3',
            'microvoid_range': (1, 6),
            'alpha_out_scale': _scaled_range(patch_scale, 1.5, 2.8, min_low=0.8, min_high=1.1),
            'alpha_in_depth': 0.22,
            'intensity_range_255': (60.0, 145.0),
            'intensity_mix': (0.34, 0.18, 0.18, 0.30),
            'edge_delta': (6.0, 16.0),
            'halo_strength': (8.0, 18.0),
            'halo_alpha': (0.18, 0.35),
            'halo_sign_mode': 'positive',
            'gray_level_range': (0.40, 0.82),
        }
    if artifact_type == 'particle_cluster':
        return {
            'n_blobs': (5, 12),
            'n_blobs_range': (5, 12),
            'radius_range': _scaled_range(patch_scale, 4.0, 14.0, min_low=1.5, min_high=4.0),
            'elongation': (0.8, 1.4),
            'threshold': (0.20, 0.30),
            'threshold_range': (0.20, 0.30),
            'anisotropy': (1.0, 1.0),
            'satellites': True,
            'weights': (0.74, 0.14, 0.10, 0.02),
            'angle_range': (0.0, 180.0),
            'postprocess': 'close_open_3',
            'microvoid_range': (0, 2),
            'alpha_out_scale': _scaled_range(patch_scale, 1.0, 2.0, min_low=0.6, min_high=0.9),
            'alpha_in_depth': 0.14,
            'intensity_range_255': (55.0, 135.0),
            'intensity_mix': (0.55, 0.22, 0.15, 0.08),
            'edge_delta': (-18.0, -8.0),
            'halo_strength': (6.0, 14.0),
            'halo_alpha': (0.10, 0.24),
            'halo_sign_mode': 'random',
            'gray_level_range': (0.35, 0.75),
        }
    if artifact_type == 'flake':
        return {
            'n_blobs': (2, 5),
            'n_blobs_range': (2, 5),
            'radius_range': _scaled_range(patch_scale, 18.0, 54.0, min_low=7.0, min_high=14.0),
            'elongation': (0.8, 2.0),
            'threshold': (0.24, 0.35),
            'threshold_range': (0.24, 0.35),
            'anisotropy': (1.0, 1.8),
            'satellites': False,
            'weights': (0.62, 0.16, 0.10, 0.12),
            'angle_range': (-25.0, 25.0),
            'postprocess': 'close_5',
            'microvoid_range': (0, 4),
            'alpha_out_scale': _scaled_range(patch_scale, 1.0, 1.8, min_low=0.6, min_high=0.9),
            'alpha_in_depth': 0.10,
            'intensity_range_255': (175.0, 248.0),
            'intensity_mix': (0.58, 0.20, 0.14, 0.08),
            'edge_delta': (-18.0, -8.0),
            'halo_strength': (6.0, 14.0),
            'halo_alpha': (0.10, 0.24),
            'halo_sign_mode': 'random',
            'gray_level_range': (0.90, 1.00),
        }
    raise ValueError(f'Unsupported artifact type: {artifact_type}')


def fractal_noise(
    shape: tuple[int, int],
    octaves: int = 5,
    persistence: float = 0.55,
    anisotropy: tuple[float, float] = (1.0, 1.0),
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width), np.float32)
    amplitude = 1.0
    amplitude_sum = 0.0
    ay, ax = anisotropy

    for octave in range(max(1, int(octaves))):
        small_h = max(2, int(height / ((2 ** (octave + 1)) * max(float(ay), 1e-3))))
        small_w = max(2, int(width / ((2 ** (octave + 1)) * max(float(ax), 1e-3))))
        small = rng.random((small_h, small_w), dtype=np.float32)
        interpolation = cv2.INTER_CUBIC if octave < 2 else cv2.INTER_LINEAR
        upsampled = cv2.resize(small, (width, height), interpolation=interpolation)
        out += amplitude * upsampled
        amplitude_sum += amplitude
        amplitude *= float(persistence)

    out /= max(amplitude_sum, 1e-6)
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    return np.clip(out, 0.0, 1.0)


def directional_streak_noise(
    shape: tuple[int, int],
    angle_deg: float = 0.0,
    sigma_long: float = 9.0,
    sigma_short: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(shape[0]), int(shape[1])
    base = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=float(sigma_long), sigmaY=float(sigma_short))

    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        blurred,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    rotated -= rotated.min()
    rotated /= max(rotated.max(), 1e-6)
    return rotated


def soft_union_blobs(
    size: tuple[int, int],
    n_blobs: int = 5,
    radius_range: tuple[float, float] = (8.0, 30.0),
    elongation: tuple[float, float] = (0.7, 1.5),
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(size[0]), int(size[1])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.zeros((height, width), np.float32)

    for _ in range(max(1, int(n_blobs))):
        cx = rng.uniform(width * 0.18, width * 0.82)
        cy = rng.uniform(height * 0.18, height * 0.82)
        radius = rng.uniform(radius_range[0], radius_range[1])
        ex = rng.uniform(elongation[0], elongation[1])
        ey = rng.uniform(elongation[0], elongation[1])
        rx = max(1e-3, radius * ex)
        ry = max(1e-3, radius * ey)
        angle = rng.uniform(0.0, np.pi)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x = xx - cx
        y = yy - cy
        xr = (x * cos_a) + (y * sin_a)
        yr = (-x * sin_a) + (y * cos_a)
        blob = np.exp(-0.5 * ((xr / rx) ** 2 + (yr / ry) ** 2))
        field += rng.uniform(0.7, 1.4) * blob

    field /= max(field.max(), 1e-6)
    return field


def keep_main_components(mask: np.ndarray, allow_satellites: bool = False) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1] + 1

    result = np.zeros_like(mask)
    main = int(order[0])
    result[labels == main] = 255

    if allow_satellites:
        main_area = stats[main, cv2.CC_STAT_AREA]
        for idx in order[1:4]:
            if stats[int(idx), cv2.CC_STAT_AREA] > 0.03 * main_area:
                result[labels == int(idx)] = 255

    return result


def make_semiconductor_defect_mask(
    size: tuple[int, int] = (256, 256),
    defect_type: str = 'dust',
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = _sample_artifact_parameters(defect_type, float(max(size)))

    n_blobs_low, n_blobs_high = params['n_blobs_range']
    n_blobs = int(rng.integers(int(n_blobs_low), int(n_blobs_high)))
    threshold_low, threshold_high = params['threshold_range']
    threshold = float(rng.uniform(float(threshold_low), float(threshold_high)))

    base = soft_union_blobs(
        size=size,
        n_blobs=n_blobs,
        radius_range=tuple(params['radius_range']),
        elongation=tuple(params['elongation']),
        seed=_spawn_subseed(rng),
    )
    coarse = fractal_noise(
        size,
        octaves=4,
        persistence=0.60,
        anisotropy=tuple(params['anisotropy']),
        seed=_spawn_subseed(rng),
    )
    fine = fractal_noise(
        size,
        octaves=6,
        persistence=0.52,
        anisotropy=(1.0, 1.0),
        seed=_spawn_subseed(rng),
    )
    sigma_long = rng.uniform(*_scaled_range(float(max(size)), 7.0, 16.0, min_low=1.2, min_high=2.2))
    sigma_short = rng.uniform(*_scaled_range(float(max(size)), 0.8, 2.2, min_low=0.5, min_high=0.8))
    angle = rng.uniform(*tuple(params['angle_range']))
    streaks = directional_streak_noise(
        size,
        angle_deg=float(angle),
        sigma_long=float(sigma_long),
        sigma_short=float(sigma_short),
        seed=_spawn_subseed(rng),
    )

    weight_base, weight_coarse, weight_fine, weight_streaks = tuple(params['weights'])
    field = (
        (float(weight_base) * base)
        + (float(weight_coarse) * coarse)
        + (float(weight_fine) * fine)
        + (float(weight_streaks) * streaks)
    )
    field = cv2.GaussianBlur(field, (0, 0), _scale_value(float(max(size)), 1.0, minimum=0.45))
    mask = (field > threshold).astype(np.uint8) * 255

    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge_noise = fractal_noise(size, octaves=5, persistence=0.58, seed=_spawn_subseed(rng))
    mask[(gradient > 0) & (edge_noise < 0.16)] = 0
    mask[(gradient > 0) & (edge_noise > 0.88)] = 255

    postprocess = str(params['postprocess'])
    if postprocess == 'close_open_3':
        kernel = _ellipse_kernel(size, 3, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif postprocess == 'close_5x3':
        kernel = _ellipse_kernel(size, 5, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif postprocess == 'erode_dilate_5x3':
        kernel = _ellipse_kernel(size, 5, 3)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
    elif postprocess == 'close_5':
        kernel = _ellipse_kernel(size, 5, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = keep_main_components(mask, allow_satellites=bool(params['satellites']))
    return mask


def add_microvoids(mask: np.ndarray, defect_type: str = 'dust', seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask

    params = _sample_artifact_parameters(defect_type, float(max(mask.shape)))
    holes = np.zeros_like(mask)
    void_low, void_high = params['microvoid_range']
    hole_count = int(rng.integers(int(void_low), int(void_high)))
    radius_low, radius_high = _scaled_range(float(max(mask.shape)), 2.0, 7.0, min_low=1.0, min_high=2.0)

    for _ in range(hole_count):
        index = int(rng.integers(0, len(xs)))
        cx = int(xs[index])
        cy = int(ys[index])
        rx = int(rng.integers(max(1, int(round(radius_low))), max(2, int(round(radius_high)) + 1)))
        ry = int(max(1, round(rx * rng.uniform(0.6, 1.6))))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(holes, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    holes = cv2.GaussianBlur(holes, (0, 0), _scale_value(float(max(mask.shape)), 1.0, minimum=0.5))
    result = mask.copy()
    result[holes > 90] = 0
    return result


def alpha_from_mask(mask: np.ndarray, defect_type: str = 'dust', seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = _sample_artifact_parameters(defect_type, float(max(mask.shape)))

    dist_in = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    alpha = np.zeros(mask.shape, np.float32)
    inside = mask > 0
    alpha[inside] = 1.0

    out_scale = float(rng.uniform(*tuple(params['alpha_out_scale'])))
    in_depth = float(params['alpha_in_depth'])
    edge_falloff = rng.uniform(
        *_scaled_range(float(max(mask.shape)), 1.2, 3.0, min_low=0.7, min_high=1.1)
    )

    alpha[~inside] = 0.40 * _stable_exp_decay(dist_out[~inside], out_scale)
    alpha[inside] *= 1.0 - in_depth * _stable_exp_decay(dist_in[inside], edge_falloff)

    noise = fractal_noise(mask.shape, octaves=5, persistence=0.56, seed=_spawn_subseed(rng))
    alpha *= 0.72 + (0.28 * noise)
    alpha = cv2.GaussianBlur(alpha, (0, 0), _scale_value(float(max(mask.shape)), 0.8, minimum=0.4))
    return np.clip(alpha, 0.0, 1.0)


def intensity_model(mask: np.ndarray, defect_type: str = 'dust', seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = _sample_artifact_parameters(defect_type, float(max(mask.shape)))
    height, width = mask.shape

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) > 0.0:
        dist = dist / float(dist.max())

    coarse = fractal_noise((height, width), octaves=4, persistence=0.60, seed=_spawn_subseed(rng))
    fine = fractal_noise((height, width), octaves=6, persistence=0.52, seed=_spawn_subseed(rng))
    streak = directional_streak_noise(
        (height, width),
        angle_deg=float(rng.uniform(*tuple(params['angle_range']))),
        sigma_long=float(rng.uniform(*_scaled_range(float(max(mask.shape)), 8.0, 14.0, min_low=1.5, min_high=2.3))),
        sigma_short=float(_scale_value(float(max(mask.shape)), 1.2, minimum=0.6)),
        seed=_spawn_subseed(rng),
    )

    dist_weight, coarse_weight, fine_weight, streak_weight = tuple(params['intensity_mix'])
    density = (
        (float(dist_weight) * dist)
        + (float(coarse_weight) * coarse)
        + (float(fine_weight) * fine)
        + (float(streak_weight) * streak)
    )
    density = np.clip(density, 0.0, 1.0)

    low, high = tuple(params['intensity_range_255'])
    intensity = low + ((high - low) * density)

    micro = fractal_noise((height, width), octaves=5, persistence=0.5, seed=_spawn_subseed(rng))
    intensity += (micro - 0.5) * rng.uniform(12.0, 30.0)

    edges = cv2.Canny(mask, 50, 120).astype(np.float32) / 255.0
    edge_low, edge_high = tuple(params['edge_delta'])
    intensity += edges * rng.uniform(edge_low, edge_high)

    intensity = np.clip(intensity, 0.0, 255.0).astype(np.uint8)
    intensity[mask == 0] = 0
    return intensity


def add_optical_halo(rgba: np.ndarray, defect_type: str = 'dust', seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = _sample_artifact_parameters(defect_type, float(max(rgba.shape[:2])))

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    mask = (alpha > 0.06).astype(np.uint8) * 255

    dist_out = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    halo_decay = rng.uniform(
        *_scaled_range(float(max(rgba.shape[:2])), 1.2, 3.2, min_low=0.8, min_high=1.3)
    )
    ring = _stable_exp_decay(dist_out, halo_decay)
    ring *= (mask == 0).astype(np.float32)

    halo_strength = float(rng.uniform(*tuple(params['halo_strength'])))
    halo_alpha = float(rng.uniform(*tuple(params['halo_alpha'])))
    halo_sign_mode = str(params['halo_sign_mode'])
    halo_sign = 1.0 if halo_sign_mode == 'positive' else (-1.0 if rng.random() < 0.6 else 1.0)

    halo = halo_sign * halo_strength * ring
    halo_a = halo_alpha * ring

    for channel in range(3):
        rgb[..., channel] = np.clip(rgb[..., channel] + halo, 0.0, 255.0)

    out = np.zeros_like(rgba)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = (np.clip(alpha + halo_a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def _fallback_artifact_rgba(size: tuple[int, int], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(size[0]), int(size[1])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = rng.uniform(width * 0.35, width * 0.65)
    cy = rng.uniform(height * 0.35, height * 0.65)
    rx = rng.uniform(max(1.0, width * 0.08), max(2.0, width * 0.22))
    ry = rng.uniform(max(1.0, height * 0.08), max(2.0, height * 0.22))
    blob = np.exp(-0.5 * (((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2))
    alpha = cv2.GaussianBlur(blob.astype(np.float32), (0, 0), _scale_value(float(max(size)), 0.8, minimum=0.4))
    alpha = np.clip(alpha, 0.0, 1.0)
    intensity = np.clip((90.0 + (90.0 * blob)).astype(np.uint8), 0, 255)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = intensity
    rgba[..., 1] = intensity
    rgba[..., 2] = intensity
    rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
    return rgba


def _edge_falloff_mask(size: tuple[int, int], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(size[0]), int(size[1])
    yy, xx = np.indices((height, width), dtype=np.float32)
    distance_to_edge = np.minimum(
        np.minimum(xx, (float(width) - 1.0) - xx),
        np.minimum(yy, (float(height) - 1.0) - yy),
    )
    patch_scale = float(max(size))
    feather_px = min(
        max(1.0, patch_scale * float(rng.uniform(0.24, 0.36))),
        max(1.0, patch_scale * 0.45),
    )
    edge_noise = fractal_noise(size, octaves=4, persistence=0.60, seed=_spawn_subseed(rng))
    irregular_edge = distance_to_edge + ((edge_noise - 0.5) * feather_px * 0.7)
    return _smoothstep01(irregular_edge / max(feather_px, 1e-6))


def _stable_exp_decay(distance: np.ndarray, scale: float) -> np.ndarray:
    safe_scale = np.float32(max(float(scale), 1e-3))
    exponent = -distance.astype(np.float32, copy=False) / safe_scale
    exponent = np.clip(exponent, -60.0, 0.0)
    return np.exp(exponent).astype(np.float32, copy=False)


def generate_semiconductor_defect_rgba(
    size: tuple[int, int] = (256, 256),
    defect_type: str = 'dust',
    crop: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    master_seed = _make_master_seed(seed)
    rng = np.random.default_rng(master_seed)

    rgba = np.zeros((int(size[0]), int(size[1]), 4), dtype=np.uint8)
    for _ in range(3):
        mask = make_semiconductor_defect_mask(
            size=size,
            defect_type=defect_type,
            seed=_spawn_subseed(rng),
        )
        mask = add_microvoids(
            mask,
            defect_type=defect_type,
            seed=_spawn_subseed(rng),
        )
        intensity = intensity_model(
            mask,
            defect_type=defect_type,
            seed=_spawn_subseed(rng),
        )
        alpha = alpha_from_mask(
            mask,
            defect_type=defect_type,
            seed=_spawn_subseed(rng),
        )

        rgba = np.zeros((int(size[0]), int(size[1]), 4), dtype=np.uint8)
        rgba[..., 0] = intensity
        rgba[..., 1] = intensity
        rgba[..., 2] = intensity
        rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
        rgba = add_optical_halo(
            rgba,
            defect_type=defect_type,
            seed=_spawn_subseed(rng),
        )
        if int(rgba[..., 3].max()) > 6:
            break
    else:
        rgba = _fallback_artifact_rgba(size, seed=_spawn_subseed(rng))

    if crop:
        ys, xs = np.where(rgba[..., 3] > 6)
        if len(xs) > 0:
            pad = 5
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(int(size[1]), int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(int(size[0]), int(ys.max()) + pad + 1)
            rgba = rgba[y0:y1, x0:x1]

    return rgba


def generate_random_artifact_patch(
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    artifact_types: tuple[str, ...] | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    available_types = tuple(artifact_types or ARTIFACT_TYPES)
    if not available_types:
        raise ValueError('At least one artifact type must be enabled.')

    rng = np.random.default_rng(seed)
    artifact_type = available_types[int(rng.integers(0, len(available_types)))]
    rgba = generate_semiconductor_defect_rgba(
        size=(int(height), int(width)),
        defect_type=artifact_type,
        crop=False,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    intensity = rgba[..., 0].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    edge_falloff = _edge_falloff_mask((int(height), int(width)), seed=int(rng.integers(0, 2**31 - 1)))
    params = _sample_artifact_parameters(artifact_type, float(max(height, width)))
    gray_level = float(rng.uniform(*tuple(params.get('gray_level_range', (0.35, 0.92)))))
    alpha_scale = float(rng.uniform(0.45, 1.0))
    intensity *= gray_level
    alpha *= edge_falloff ** 1.5
    alpha *= alpha_scale
    alpha = np.clip(alpha, 0.0, 1.0)
    intensity = np.clip(intensity, 0.0, 1.0)

    if channels <= 1:
        overlay_np = intensity[None, ...]
    else:
        overlay_np = np.repeat(intensity[None, ...], int(channels), axis=0)

    overlay = torch.from_numpy(np.ascontiguousarray(overlay_np)).to(device=device, dtype=dtype)
    alpha_tensor = torch.from_numpy(np.ascontiguousarray(alpha[None, ...])).to(device=device, dtype=dtype)
    return overlay, alpha_tensor

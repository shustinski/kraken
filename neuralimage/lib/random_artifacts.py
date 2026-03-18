from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


ARTIFACT_TYPES: tuple[str, ...] = (
    'dust',
    'resist_residue',
    'etch_residue',
    'particle_cluster',
    'flake',
)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_value = tensor.min()
    max_value = tensor.max()
    scale = max_value - min_value
    if float(scale.abs().item()) <= 1e-6:
        return torch.zeros_like(tensor)
    return (tensor - min_value) / scale


def _gaussian_blur2d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image

    if image.ndim == 2:
        batch = image.unsqueeze(0).unsqueeze(0)
        restore_shape = 'hw'
    elif image.ndim == 3:
        batch = image.unsqueeze(0)
        restore_shape = 'chw'
    elif image.ndim == 4:
        batch = image
        restore_shape = 'nchw'
    else:
        raise ValueError(f'Unsupported tensor rank for blur: {image.ndim}')

    channels = int(batch.shape[1])
    radius = max(1, int(round(float(sigma) * 3.0)))
    coords = torch.arange(-radius, radius + 1, device=batch.device, dtype=batch.dtype)
    kernel = torch.exp(-(coords ** 2) / max(2.0 * float(sigma) * float(sigma), 1e-6))
    kernel = kernel / kernel.sum()
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    blurred = F.pad(batch, (radius, radius, 0, 0), mode='replicate')
    blurred = F.conv2d(blurred, kernel_x, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode='replicate')
    blurred = F.conv2d(blurred, kernel_y, groups=channels)

    if restore_shape == 'hw':
        return blurred[0, 0]
    if restore_shape == 'chw':
        return blurred[0]
    return blurred


def _coordinate_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij',
    )
    return yy, xx


def _fractal_noise(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    octaves: int = 5,
    persistence: float = 0.55,
    anisotropy: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    out = torch.zeros((height, width), device=device, dtype=dtype)
    amplitude = 1.0
    amplitude_sum = 0.0
    ay = max(0.1, float(anisotropy[0]))
    ax = max(0.1, float(anisotropy[1]))

    for octave in range(max(1, int(octaves))):
        small_h = max(2, int(round(height / ((2 ** (octave + 1)) * ay))))
        small_w = max(2, int(round(width / ((2 ** (octave + 1)) * ax))))
        small = torch.rand((1, 1, small_h, small_w), device=device, dtype=dtype)
        upsampled = F.interpolate(small, size=(height, width), mode='bilinear', align_corners=False)[0, 0]
        out = out + (float(amplitude) * upsampled)
        amplitude_sum += float(amplitude)
        amplitude *= float(persistence)

    if amplitude_sum > 0.0:
        out = out / float(amplitude_sum)
    return _normalize_tensor(_gaussian_blur2d(out, sigma=0.8))


def _soft_union_blobs(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_blobs: int,
    radius_range: tuple[float, float],
    elongation: tuple[float, float],
) -> torch.Tensor:
    yy, xx = _coordinate_grid(height, width, device=device, dtype=dtype)
    field = torch.zeros((height, width), device=device, dtype=dtype)

    for _ in range(max(1, int(n_blobs))):
        cx = np.random.uniform(width * 0.18, width * 0.82)
        cy = np.random.uniform(height * 0.18, height * 0.82)
        radius = np.random.uniform(*radius_range)
        rx = max(1.0, radius * np.random.uniform(*elongation))
        ry = max(1.0, radius * np.random.uniform(*elongation))
        angle = np.random.uniform(0.0, math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = xx - float(cx)
        y = yy - float(cy)
        xr = (x * cos_a) + (y * sin_a)
        yr = (-x * sin_a) + (y * cos_a)
        blob = torch.exp(-0.5 * (((xr / float(rx)) ** 2) + ((yr / float(ry)) ** 2)))
        field = field + (float(np.random.uniform(0.7, 1.4)) * blob)

    return _normalize_tensor(field)


def _directional_streak_noise(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    angle_deg: float,
    sigma_long: float,
    sigma_short: float,
) -> torch.Tensor:
    yy, xx = _coordinate_grid(height, width, device=device, dtype=dtype)
    center_x = (float(width) - 1.0) * 0.5
    center_y = (float(height) - 1.0) * 0.5
    theta = math.radians(float(angle_deg))
    cos_a = math.cos(theta)
    sin_a = math.sin(theta)
    projected = ((xx - center_x) * cos_a) + ((yy - center_y) * sin_a)
    base_frequency = np.random.uniform(0.12, 0.35) / max(float(sigma_short), 0.8)
    phase_a = np.random.uniform(0.0, math.tau)
    phase_b = np.random.uniform(0.0, math.tau)
    stripes = (
        0.5
        + (0.25 * torch.sin(projected * float(base_frequency) + float(phase_a)))
        + (0.25 * torch.sin(projected * float(base_frequency * 1.8) + float(phase_b)))
    )
    stripes = _normalize_tensor(stripes)
    coarse = _fractal_noise(
        height,
        width,
        device=device,
        dtype=dtype,
        octaves=4,
        persistence=0.6,
        anisotropy=(1.0, max(1.0, float(sigma_long) / max(float(sigma_short), 0.8))),
    )
    streaks = stripes * (0.55 + (0.45 * coarse))
    return _normalize_tensor(_gaussian_blur2d(streaks, sigma=max(0.6, float(sigma_short))))


def _sample_artifact_parameters(artifact_type: str, patch_scale: float) -> dict[str, object]:
    if artifact_type == 'dust':
        return {
            'n_blobs': np.random.randint(3, 7),
            'radius_range': (max(2.0, 0.12 * patch_scale), max(4.0, 0.30 * patch_scale)),
            'elongation': (0.7, 1.3),
            'threshold': (0.23, 0.33),
            'anisotropy': (1.0, 1.0),
            'weights': (0.68, 0.18, 0.08, 0.06),
            'angle_range': (0.0, 180.0),
            'intensity_range': (0.28, 0.68),
            'alpha_scale': (0.35, 0.65),
            'micro_shift': 0.09,
            'max_microvoids': 2,
            'halo_sigma': (1.0, 1.8),
            'halo_alpha': (0.02, 0.08),
        }
    if artifact_type == 'resist_residue':
        return {
            'n_blobs': np.random.randint(4, 9),
            'radius_range': (max(3.0, 0.16 * patch_scale), max(6.0, 0.38 * patch_scale)),
            'elongation': (0.8, 1.7),
            'threshold': (0.18, 0.28),
            'anisotropy': (1.0, 1.6),
            'weights': (0.50, 0.20, 0.12, 0.18),
            'angle_range': (-25.0, 25.0),
            'intensity_range': (0.44, 0.86),
            'alpha_scale': (0.42, 0.78),
            'micro_shift': 0.12,
            'max_microvoids': 4,
            'halo_sigma': (1.4, 2.4),
            'halo_alpha': (0.05, 0.12),
        }
    if artifact_type == 'etch_residue':
        return {
            'n_blobs': np.random.randint(3, 7),
            'radius_range': (max(4.0, 0.18 * patch_scale), max(8.0, 0.42 * patch_scale)),
            'elongation': (0.9, 2.2),
            'threshold': (0.19, 0.27),
            'anisotropy': (1.0, 2.2),
            'weights': (0.46, 0.18, 0.10, 0.26),
            'angle_range': (-25.0, 25.0),
            'intensity_range': (0.20, 0.58),
            'alpha_scale': (0.35, 0.72),
            'micro_shift': 0.13,
            'max_microvoids': 5,
            'halo_sigma': (1.2, 2.2),
            'halo_alpha': (0.04, 0.10),
        }
    if artifact_type == 'particle_cluster':
        return {
            'n_blobs': np.random.randint(5, 12),
            'radius_range': (max(1.5, 0.08 * patch_scale), max(4.0, 0.18 * patch_scale)),
            'elongation': (0.8, 1.4),
            'threshold': (0.20, 0.30),
            'anisotropy': (1.0, 1.0),
            'weights': (0.74, 0.14, 0.10, 0.02),
            'angle_range': (0.0, 180.0),
            'intensity_range': (0.18, 0.54),
            'alpha_scale': (0.38, 0.70),
            'micro_shift': 0.08,
            'max_microvoids': 1,
            'halo_sigma': (0.9, 1.6),
            'halo_alpha': (0.02, 0.06),
        }
    return {
        'n_blobs': np.random.randint(2, 5),
        'radius_range': (max(5.0, 0.22 * patch_scale), max(10.0, 0.48 * patch_scale)),
        'elongation': (0.8, 2.0),
        'threshold': (0.24, 0.35),
        'anisotropy': (1.0, 1.8),
        'weights': (0.62, 0.16, 0.10, 0.12),
        'angle_range': (-25.0, 25.0),
        'intensity_range': (0.32, 0.80),
        'alpha_scale': (0.40, 0.75),
        'micro_shift': 0.10,
        'max_microvoids': 3,
        'halo_sigma': (1.0, 1.8),
        'halo_alpha': (0.03, 0.08),
    }


def _apply_microvoids(mask: torch.Tensor, *, max_holes: int) -> torch.Tensor:
    hole_count = int(np.random.randint(0, max(0, int(max_holes)) + 1))
    if hole_count <= 0:
        return mask

    coords = torch.nonzero(mask > float(mask.mean().item()), as_tuple=False)
    if int(coords.shape[0]) == 0:
        return mask

    height, width = int(mask.shape[0]), int(mask.shape[1])
    yy, xx = _coordinate_grid(height, width, device=mask.device, dtype=mask.dtype)
    holes = torch.zeros_like(mask)

    for _ in range(hole_count):
        center_index = int(np.random.randint(0, int(coords.shape[0])))
        cy = float(coords[center_index, 0].item())
        cx = float(coords[center_index, 1].item())
        rx = np.random.uniform(1.2, max(1.6, width * 0.12))
        ry = np.random.uniform(1.2, max(1.6, height * 0.12))
        angle = np.random.uniform(0.0, math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = xx - cx
        y = yy - cy
        xr = (x * cos_a) + (y * sin_a)
        yr = (-x * sin_a) + (y * cos_a)
        blob = torch.exp(-0.5 * (((xr / float(rx)) ** 2) + ((yr / float(ry)) ** 2)))
        holes = torch.maximum(holes, blob)

    holes = _gaussian_blur2d(holes, sigma=0.45)
    return torch.clamp(mask * (1.0 - (0.8 * holes)), min=0.0, max=1.0)


def generate_random_artifact_patch(
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    artifact_type = ARTIFACT_TYPES[int(np.random.randint(0, len(ARTIFACT_TYPES)))]
    patch_scale = float(max(height, width))
    params = _sample_artifact_parameters(artifact_type, patch_scale)

    base = _soft_union_blobs(
        height,
        width,
        device=device,
        dtype=work_dtype,
        n_blobs=int(params['n_blobs']),
        radius_range=tuple(params['radius_range']),
        elongation=tuple(params['elongation']),
    )
    coarse = _fractal_noise(
        height,
        width,
        device=device,
        dtype=work_dtype,
        octaves=4,
        persistence=0.60,
        anisotropy=tuple(params['anisotropy']),
    )
    fine = _fractal_noise(
        height,
        width,
        device=device,
        dtype=work_dtype,
        octaves=6,
        persistence=0.52,
    )
    angle = float(np.random.uniform(*tuple(params['angle_range'])))
    streaks = _directional_streak_noise(
        height,
        width,
        device=device,
        dtype=work_dtype,
        angle_deg=angle,
        sigma_long=np.random.uniform(7.0, 16.0),
        sigma_short=np.random.uniform(0.8, 2.2),
    )

    weight_base, weight_coarse, weight_fine, weight_streaks = tuple(params['weights'])
    field = (
        (float(weight_base) * base)
        + (float(weight_coarse) * coarse)
        + (float(weight_fine) * fine)
        + (float(weight_streaks) * streaks)
    )
    field = _normalize_tensor(_gaussian_blur2d(field, sigma=0.8))

    threshold = float(np.random.uniform(*tuple(params['threshold'])))
    mask = torch.sigmoid((field - threshold) * 12.0)
    mask = _apply_microvoids(mask, max_holes=int(params['max_microvoids']))
    edge_noise = _fractal_noise(height, width, device=device, dtype=work_dtype, octaves=4, persistence=0.58)
    alpha = mask * (0.70 + (0.30 * edge_noise))
    alpha = _gaussian_blur2d(alpha, sigma=0.7)
    alpha = torch.clamp(alpha * float(np.random.uniform(*tuple(params['alpha_scale']))), min=0.0, max=0.92)

    density = torch.clamp(
        (0.44 * mask) + (0.28 * coarse) + (0.18 * fine) + (0.10 * streaks),
        min=0.0,
        max=1.0,
    )
    intensity_low, intensity_high = tuple(params['intensity_range'])
    texture = float(intensity_low) + ((float(intensity_high) - float(intensity_low)) * density)
    micro_texture = _fractal_noise(height, width, device=device, dtype=work_dtype, octaves=5, persistence=0.50)
    texture = torch.clamp(
        texture + ((micro_texture - 0.5) * float(params['micro_shift'])),
        min=0.0,
        max=1.0,
    )

    halo = torch.clamp(
        _gaussian_blur2d(mask, sigma=float(np.random.uniform(*tuple(params['halo_sigma'])))) - mask,
        min=0.0,
        max=1.0,
    )
    alpha = torch.clamp(alpha + (halo * float(np.random.uniform(*tuple(params['halo_alpha'])))), min=0.0, max=0.95)

    overlay = texture.unsqueeze(0).repeat(max(1, int(channels)), 1, 1)
    if int(channels) >= 3:
        channel_scale = 1.0 + ((torch.rand((int(channels), 1, 1), device=device, dtype=work_dtype) - 0.5) * 0.10)
        overlay = torch.clamp(overlay * channel_scale, min=0.0, max=1.0)

    return overlay.to(dtype=dtype), alpha.unsqueeze(0).to(dtype=dtype)

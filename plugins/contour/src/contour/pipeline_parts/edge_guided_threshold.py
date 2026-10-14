"""pipeline: edge guided threshold."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.edge_detection import (
    build_gradient_elevation,
    combined_elevation,
)
from contour.edge_detection import (
    auto_canny as _ed_auto_canny,
)
from contour.edge_detection import (
    laplacian_of_gaussian as _ed_log,
)
from contour.edge_detection import (
    phase_congruency as _ed_phase_congruency,
)
from contour.edge_detection import (
    ridge_response as _ed_ridge_response,
)
from contour.edge_detection import (
    scharr_magnitude as _ed_scharr_magnitude,
)
from contour.edge_detection import (
    structured_edges as _ed_structured_edges,
)
from contour.utils import ensure_binary_mask, ensure_uint8

from contour.pipeline_parts.module_state import _COMBINED_PRESETS

from contour.pipeline_parts.edge_contour_refined_mask import (
    _base_threshold_mask,
    _edge_contour_refined_mask,
    _edge_elevation,
)

from contour.pipeline_parts.watershed_split import (
    _as_gray,
    _binary_fill_holes,
    _kernel,
    _odd,
    _resize_interpolation,
)

def _edge_guided_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    gray = _as_gray(image)
    base_mask = ensure_binary_mask(_base_threshold_mask(gray, parameters))
    correction_radius = max(0, int(parameters.get("correction_radius", 2)))
    if correction_radius <= 0 or cv2.countNonZero(base_mask) == 0:
        return base_mask

    kernel_size = correction_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground_seed = cv2.erode(base_mask, kernel, iterations=1)
    background_seed = cv2.erode(cv2.bitwise_not(base_mask), kernel, iterations=1)
    if cv2.countNonZero(foreground_seed) == 0 or cv2.countNonZero(background_seed) == 0:
        return base_mask

    markers = np.zeros(gray.shape[:2], dtype=np.int32)
    markers[background_seed > 0] = 1
    _component_count, foreground_labels = cv2.connectedComponents(
        (foreground_seed > 0).astype(np.uint8), connectivity=8
    )
    markers[foreground_labels > 0] = foreground_labels[foreground_labels > 0] + 1

    elevation = _edge_elevation(gray, parameters)
    marker_image = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(marker_image, markers)
    refined = np.where(markers > 1, 255, 0).astype(np.uint8)

    dilated = cv2.dilate(base_mask, kernel, iterations=1)
    eroded = cv2.erode(base_mask, kernel, iterations=1)
    correction_band = cv2.bitwise_xor(dilated, eroded)
    contour_refined = _edge_contour_refined_mask(gray, base_mask, correction_band, correction_radius, parameters)
    if contour_refined is not None:
        if bool(parameters.get("fill_holes", True)):
            contour_refined = _binary_fill_holes(contour_refined, {})
        return contour_refined

    result = base_mask.copy()
    result[correction_band > 0] = refined[correction_band > 0]
    if bool(parameters.get("fill_holes", True)):
        result = _binary_fill_holes(result, {})
    return result

def _morph_op(image: np.ndarray, op_code: int, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.morphologyEx(
        image,
        op_code,
        _kernel(parameters),
        iterations=max(1, int(parameters.get("iterations", 1))),
    )

def _erode(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.erode(image, _kernel(parameters), iterations=max(1, int(parameters.get("iterations", 1))))

def _dilate(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.dilate(image, _kernel(parameters), iterations=max(1, int(parameters.get("iterations", 1))))

def _canny(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    aperture_size = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
    return cv2.Canny(
        image,
        float(parameters.get("threshold1", 50)),
        float(parameters.get("threshold2", 150)),
        apertureSize=aperture_size,
        L2gradient=bool(parameters.get("l2gradient", False)),
    )

def _invert(image: np.ndarray, _: dict[str, Any]) -> np.ndarray:
    return cv2.bitwise_not(image)

def _resize(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    target_width = int(parameters.get("width", width))
    target_height = int(parameters.get("height", height))
    keep_aspect = bool(parameters.get("keep_aspect", True))
    if keep_aspect:
        if target_width <= 0 and target_height > 0:
            target_width = max(1, round(width * (target_height / height)))
        elif target_height <= 0 and target_width > 0:
            target_height = max(1, round(height * (target_width / width)))
    target_width = max(1, target_width if target_width > 0 else width)
    target_height = max(1, target_height if target_height > 0 else height)
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=_resize_interpolation(str(parameters.get("interpolation", "linear"))),
    )

def _scale_resize(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(0.05, float(parameters.get("scale", 1.0)))
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=_resize_interpolation(str(parameters.get("interpolation", "linear"))),
    )

def _crop(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    x_coord = max(0, int(parameters.get("x", 0)))
    y_coord = max(0, int(parameters.get("y", 0)))
    width = max(1, int(parameters.get("width", image.shape[1] - x_coord)))
    height = max(1, int(parameters.get("height", image.shape[0] - y_coord)))
    return image[y_coord : min(image.shape[0], y_coord + height), x_coord : min(image.shape[1], x_coord + width)].copy()

def _sharpen(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    amount = float(parameters.get("amount", 1.5))
    sigma = max(0.0, float(parameters.get("sigma", 1.0)))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def _scharr_edges(image: np.ndarray, _parameters: dict[str, Any]) -> np.ndarray:
    return _ed_scharr_magnitude(_as_gray(image))

def _auto_canny(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    sigma = float(parameters.get("sigma", 0.33))
    aperture = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
    l2gradient = bool(parameters.get("l2gradient", True))
    return _ed_auto_canny(_as_gray(image), sigma=sigma, aperture_size=aperture, l2gradient=l2gradient)

def _log_edges(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    sigma_small = max(0.4, float(parameters.get("sigma_small", 1.0)))
    sigma_mid = max(sigma_small + 0.1, float(parameters.get("sigma_mid", 2.0)))
    sigma_large = max(sigma_mid + 0.1, float(parameters.get("sigma_large", 3.5)))
    return _ed_log(_as_gray(image), sigmas=(sigma_small, sigma_mid, sigma_large))

def _ridge_edges(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    sigma = max(0.5, float(parameters.get("sigma", 1.5)))
    return _ed_ridge_response(_as_gray(image), sigma=sigma)

def _structured_edge_operation(image: np.ndarray, _parameters: dict[str, Any]) -> np.ndarray:
    return _ed_structured_edges(image)

def _phase_congruency_edges(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return _ed_phase_congruency(
        _as_gray(image),
        num_scales=max(1, int(parameters.get("num_scales", 4))),
        num_orientations=max(1, int(parameters.get("num_orientations", 6))),
        min_wavelength=max(1.5, float(parameters.get("min_wavelength", 3.0))),
        scale_factor=max(1.2, float(parameters.get("scale_factor", 2.1))),
    )

def _combined_edges(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    preset = str(parameters.get("preset", "default") or "default")
    methods = _COMBINED_PRESETS.get(preset, _COMBINED_PRESETS["default"])
    return combined_elevation(_as_gray(image), methods)

def _edge_dispatch(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    method = str(parameters.get("method", "sobel") or "sobel")
    return build_gradient_elevation(_as_gray(image), method)

def _denoise(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = ensure_uint8(image)
    kwargs = dict(
        h=max(1.0, float(parameters.get("h", 10.0))),
        templateWindowSize=_odd(int(parameters.get("template_window_size", 7)), minimum=3),
        searchWindowSize=_odd(int(parameters.get("search_window_size", 21)), minimum=3),
    )
    if image.ndim == 3 and image.shape[2] == 3:
        if (
            np.array_equal(image[..., 0], image[..., 1])
            and np.array_equal(image[..., 0], image[..., 2])
        ):
            gray = np.ascontiguousarray(image[..., 0])
            denoised = cv2.fastNlMeansDenoising(gray, None, **kwargs)
            return cv2.merge((denoised, denoised, denoised))
        return cv2.fastNlMeansDenoisingColored(
            image, None, kwargs["h"], kwargs["h"], kwargs["templateWindowSize"], kwargs["searchWindowSize"]
        )
    return cv2.fastNlMeansDenoising(image, None, **kwargs)

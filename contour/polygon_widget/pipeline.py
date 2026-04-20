from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .application.processing import OperationParameterSpec, PipelineStepConfig
from .i18n import choice_label, operation_name, parameter_label, tr
from .utils import ensure_binary_mask, ensure_uint8


OperationCallable = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


@dataclass(frozen=True, slots=True)
class OperationDescriptor:
    type_name: str
    display_name: str
    parameters: tuple[OperationParameterSpec, ...]
    handler: OperationCallable

    def default_parameters(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.parameters}


_OPERATIONS: dict[str, OperationDescriptor] = {}


def register_operation(descriptor: OperationDescriptor) -> None:
    _OPERATIONS[descriptor.type_name] = descriptor


def get_operation_descriptor(operation_name: str) -> OperationDescriptor:
    try:
        return _OPERATIONS[operation_name]
    except KeyError as exc:
        raise KeyError(tr("unsupported_pipeline_operation", operation=operation_name)) from exc


def available_operations() -> list[OperationDescriptor]:
    return list(_OPERATIONS.values())


def get_operation_display_name(operation_key: str, language: str | None = None) -> str:
    descriptor = get_operation_descriptor(operation_key)
    return operation_name(descriptor.type_name, default=descriptor.display_name, language=language)


def get_parameter_display_label(spec: OperationParameterSpec, language: str | None = None) -> str:
    return parameter_label(spec.name, default=spec.label, language=language)


def get_choice_display_label(parameter_name: str, value: str, language: str | None = None) -> str:
    return choice_label(parameter_name, value, default=value, language=language)


class PreprocessingPipeline:
    def __init__(self, steps: list[PipelineStepConfig] | None = None) -> None:
        self.steps = [step.clone() for step in (steps or [])]

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = ensure_uint8(image)
        for index, step in enumerate(self.steps):
            if not step.enabled:
                continue
            descriptor = get_operation_descriptor(step.operation)
            try:
                result = descriptor.handler(result, dict(step.parameters))
                result = ensure_uint8(result)
            except Exception as exc:
                raise RuntimeError(
                    tr(
                        "pipeline_step_failed",
                        index=index + 1,
                        step=operation_name(descriptor.type_name, default=step.name, language=None),
                        error=exc,
                    )
                ) from exc
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PreprocessingPipeline":
        payload = payload or {}
        steps = [PipelineStepConfig.from_dict(item) for item in payload.get("steps", [])]
        return cls(steps)

    @staticmethod
    def create_step(operation_name: str) -> PipelineStepConfig:
        descriptor = get_operation_descriptor(operation_name)
        return PipelineStepConfig(
            operation=descriptor.type_name,
            name=descriptor.display_name,
            enabled=True,
            parameters=descriptor.default_parameters(),
        )


def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _threshold_mode(name: str) -> int:
    mapping = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    return mapping.get(name, cv2.THRESH_BINARY)


def _adaptive_mode(name: str) -> int:
    return cv2.ADAPTIVE_THRESH_GAUSSIAN_C if name == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C


def _morph_shape(name: str) -> int:
    mapping = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return mapping.get(name, cv2.MORPH_RECT)


def _resize_interpolation(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(name, cv2.INTER_LINEAR)


def _as_gray(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
    if data.ndim == 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return data


def _as_bgr(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data


def _color_selection_entries(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = parameters.get("selected_colors", [])
    if not isinstance(raw_entries, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        rgb = entry.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            continue
        try:
            parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
        except (TypeError, ValueError):
            continue
        result.append(
            {
                "rgb": parsed_rgb,
                "enabled": bool(entry.get("enabled", True)),
            }
        )
    return result


def _color_binarize(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    bgr = _as_bgr(image)
    delta = max(0, min(255, int(parameters.get("delta", 10))))
    color_entries = [entry for entry in _color_selection_entries(parameters) if entry.get("enabled", True)]
    if not color_entries:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for entry in color_entries:
        rgb = np.asarray(entry["rgb"], dtype=np.int16)
        lower = np.clip(rgb - delta, 0, 255).astype(np.uint8)
        upper = np.clip(rgb + delta, 0, 255).astype(np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(rgb_image, lower, upper))
    return mask


def _binary_fill_holes(image: np.ndarray, _parameters: dict[str, Any]) -> np.ndarray:
    mask = ensure_binary_mask(image)
    flood = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _binary_filter_components(image: np.ndarray, parameters: dict[str, Any], *, metric: str) -> np.ndarray:
    mask = ensure_binary_mask(image)
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask)
    minimum = float(parameters.get(f"min_component_{metric}", 0.0) or 0.0)
    maximum_raw = float(parameters.get(f"max_component_{metric}", 0.0) or 0.0)
    maximum = maximum_raw if maximum_raw > 0 else None
    for label_index in range(1, num_labels):
        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        if metric == "area":
            value = float(cv2.countNonZero(component_mask))
        else:
            contours, _hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            value = float(sum(cv2.arcLength(contour, True) for contour in contours))
        if value < minimum:
            continue
        if maximum is not None and value > maximum:
            continue
        result[labels == label_index] = 255
    return result


def _binary_filter_area(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return _binary_filter_components(image, parameters, metric="area")


def _binary_filter_perimeter(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return _binary_filter_components(image, parameters, metric="perimeter")


def _watershed_split(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    mask = ensure_binary_mask(image)
    if cv2.countNonZero(mask) == 0:
        return mask

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return mask

    distance_ratio = min(0.95, max(0.05, float(parameters.get("distance_ratio", 0.35))))
    local_max_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_max = distance >= cv2.dilate(distance, local_max_kernel, iterations=1) - 1e-6
    sure_fg = np.where(local_max & (distance >= (max_distance * distance_ratio)), 255, 0).astype(np.uint8)

    min_peak_area = max(0, int(parameters.get("min_peak_area", 0) or 0))
    if min_peak_area > 0:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((sure_fg > 0).astype(np.uint8), connectivity=8)
        filtered = np.zeros_like(sure_fg)
        for label_index in range(1, count):
            if int(stats[label_index, cv2.CC_STAT_AREA]) >= min_peak_area:
                filtered[labels == label_index] = 255
        sure_fg = filtered

    marker_count, _marker_labels = cv2.connectedComponents((sure_fg > 0).astype(np.uint8), connectivity=8)
    if marker_count <= 2:
        return mask

    kernel = _kernel(parameters)
    sure_bg = cv2.dilate(mask, kernel, iterations=max(1, int(parameters.get("background_iterations", 1))))
    unknown = cv2.subtract(sure_bg, sure_fg)

    _marker_count, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8), connectivity=8)
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0

    marker_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(marker_image, markers)

    result = np.zeros_like(mask)
    result[markers > 1] = 255
    return result


def _kernel(parameters: dict[str, Any]) -> np.ndarray:
    size = _odd(int(parameters.get("kernel_size", 3)), minimum=1)
    shape = _morph_shape(str(parameters.get("shape", "rect")))
    return cv2.getStructuringElement(shape, (size, size))


def _gaussian_blur(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    kernel_size = _odd(int(parameters.get("kernel_size", 5)), minimum=1)
    sigma_x = float(parameters.get("sigma_x", 0.0))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)


def _median_blur(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.medianBlur(image, _odd(int(parameters.get("kernel_size", 5)), minimum=3))


def _bilateral(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        d=max(1, int(parameters.get("diameter", 9))),
        sigmaColor=float(parameters.get("sigma_color", 75.0)),
        sigmaSpace=float(parameters.get("sigma_space", 75.0)),
    )


def _clahe(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    tile_grid = max(1, int(parameters.get("tile_grid_size", 8)))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(parameters.get("clip_limit", 2.0))),
        tileGridSize=(tile_grid, tile_grid),
    )
    return clahe.apply(image)


def _histogram_equalization(image: np.ndarray, _: dict[str, Any]) -> np.ndarray:
    return cv2.equalizeHist(_as_gray(image))


def _brightness_contrast(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.convertScaleAbs(
        image,
        alpha=float(parameters.get("alpha", 1.0)),
        beta=float(parameters.get("beta", 0.0)),
    )


def _gamma_correction(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    gamma = max(0.01, float(parameters.get("gamma", 1.0)))
    table = np.array([((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def _threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    _, result = cv2.threshold(
        image,
        float(parameters.get("threshold", 127)),
        float(parameters.get("max_value", 255)),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
    )
    return result


def _adaptive_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    return cv2.adaptiveThreshold(
        image,
        float(parameters.get("max_value", 255)),
        _adaptive_mode(str(parameters.get("adaptive_method", "gaussian"))),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
        _odd(int(parameters.get("block_size", 11)), minimum=3),
        float(parameters.get("c_value", 2.0)),
    )


def _otsu_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    _, result = cv2.threshold(
        image,
        0,
        255,
        _threshold_mode(str(parameters.get("threshold_type", "binary"))) | cv2.THRESH_OTSU,
    )
    return result


def _binary_threshold_mode(parameters: dict[str, Any]) -> int:
    threshold_type = str(parameters.get("threshold_type", "binary"))
    if threshold_type == "binary_inv":
        return cv2.THRESH_BINARY_INV
    return cv2.THRESH_BINARY


def _base_threshold_mask(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    threshold_mode = str(parameters.get("threshold_mode", "otsu"))
    threshold_type = _binary_threshold_mode(parameters)
    max_value = float(parameters.get("max_value", 255))
    if threshold_mode == "adaptive":
        return cv2.adaptiveThreshold(
            image,
            max_value,
            _adaptive_mode(str(parameters.get("adaptive_method", "gaussian"))),
            threshold_type,
            _odd(int(parameters.get("block_size", 11)), minimum=3),
            float(parameters.get("c_value", 2.0)),
        )
    if threshold_mode == "manual":
        _, result = cv2.threshold(
            image,
            float(parameters.get("threshold", 127)),
            max_value,
            threshold_type,
        )
        return result
    _, result = cv2.threshold(image, 0, max_value, threshold_type | cv2.THRESH_OTSU)
    return result


def _edge_elevation(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    detector = str(parameters.get("edge_detector", "canny"))
    aperture_size = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    if detector == "canny":
        edges = cv2.Canny(
            blurred,
            float(parameters.get("threshold1", 50)),
            float(parameters.get("threshold2", 150)),
            apertureSize=aperture_size,
            L2gradient=bool(parameters.get("l2gradient", False)),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.GaussianBlur(edges, (3, 3), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=aperture_size)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=aperture_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if float(magnitude.max()) <= 0.0:
        return np.zeros_like(image, dtype=np.uint8)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _edge_mask(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    detector = str(parameters.get("edge_detector", "canny"))
    if detector == "canny":
        aperture_size = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
        return cv2.Canny(
            cv2.GaussianBlur(image, (3, 3), 0),
            float(parameters.get("threshold1", 50)),
            float(parameters.get("threshold2", 150)),
            apertureSize=aperture_size,
            L2gradient=bool(parameters.get("l2gradient", False)),
        )

    elevation = _edge_elevation(image, parameters)
    if cv2.countNonZero(elevation) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    nonzero = elevation[elevation > 0]
    percentile = max(1.0, min(99.0, float(parameters.get("edge_percentile", 80.0))))
    threshold_value = float(np.percentile(nonzero, percentile)) if nonzero.size else 255.0
    return np.where(elevation >= threshold_value, 255, 0).astype(np.uint8)


def _edge_contour_refined_mask(
    image: np.ndarray,
    base_mask: np.ndarray,
    correction_band: np.ndarray,
    correction_radius: int,
    parameters: dict[str, Any],
) -> np.ndarray | None:
    edges = _edge_mask(image, parameters)
    if cv2.countNonZero(edges) == 0:
        return None

    close_radius = max(1, correction_radius)
    kernel_size = close_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    edge_regions = np.zeros_like(base_mask)
    base_area = float(cv2.countNonZero(base_mask))
    min_overlap = max(1.0, base_area * 0.02)
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        candidate = np.zeros_like(base_mask)
        cv2.fillPoly(candidate, [contour], 255)
        overlap = float(cv2.countNonZero(cv2.bitwise_and(candidate, base_mask)))
        if overlap < min_overlap:
            continue
        edge_regions = cv2.bitwise_or(edge_regions, candidate)

    if cv2.countNonZero(edge_regions) == 0:
        return None

    result = base_mask.copy()
    result[correction_band > 0] = edge_regions[correction_band > 0]
    if cv2.countNonZero(result) == 0:
        return None
    return result


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
    _component_count, foreground_labels = cv2.connectedComponents((foreground_seed > 0).astype(np.uint8), connectivity=8)
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
            target_width = max(1, int(round(width * (target_height / height))))
        elif target_height <= 0 and target_width > 0:
            target_height = max(1, int(round(height * (target_width / width))))
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
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
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


def _denoise(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = ensure_uint8(image)
    kwargs = dict(
        h=max(1.0, float(parameters.get("h", 10.0))),
        templateWindowSize=_odd(int(parameters.get("template_window_size", 7)), minimum=3),
        searchWindowSize=_odd(int(parameters.get("search_window_size", 21)), minimum=3),
    )
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, kwargs["h"], kwargs["h"], kwargs["templateWindowSize"], kwargs["searchWindowSize"])
    return cv2.fastNlMeansDenoising(image, None, **kwargs)


def _register_builtin_operations() -> None:
    if _OPERATIONS:
        return

    common_kernel_specs = (
        OperationParameterSpec("kernel_size", "Kernel size", "int", 3, minimum=1, maximum=99, step=2),
        OperationParameterSpec("iterations", "Iterations", "int", 1, minimum=1, maximum=20, step=1),
        OperationParameterSpec("shape", "Kernel shape", "choice", "rect", options=["rect", "ellipse", "cross"]),
    )

    descriptors = [
        OperationDescriptor(
            "gaussian_blur",
            "Gaussian Blur",
            (
                OperationParameterSpec("kernel_size", "Kernel size", "int", 5, minimum=1, maximum=99, step=2),
                OperationParameterSpec("sigma_x", "Sigma X", "float", 0.0, minimum=0.0, maximum=50.0, step=0.1),
            ),
            _gaussian_blur,
        ),
        OperationDescriptor(
            "median_blur",
            "Median Blur",
            (OperationParameterSpec("kernel_size", "Kernel size", "int", 5, minimum=3, maximum=99, step=2),),
            _median_blur,
        ),
        OperationDescriptor(
            "bilateral_filter",
            "Bilateral Filter",
            (
                OperationParameterSpec("diameter", "Diameter", "int", 9, minimum=1, maximum=50, step=1),
                OperationParameterSpec("sigma_color", "Sigma color", "float", 75.0, minimum=1.0, maximum=200.0, step=1.0),
                OperationParameterSpec("sigma_space", "Sigma space", "float", 75.0, minimum=1.0, maximum=200.0, step=1.0),
            ),
            _bilateral,
        ),
        OperationDescriptor(
            "clahe",
            "CLAHE",
            (
                OperationParameterSpec("clip_limit", "Clip limit", "float", 2.0, minimum=0.1, maximum=16.0, step=0.1),
                OperationParameterSpec("tile_grid_size", "Tile grid", "int", 8, minimum=1, maximum=32, step=1),
            ),
            _clahe,
        ),
        OperationDescriptor("histogram_equalization", "Histogram Equalization", (), _histogram_equalization),
        OperationDescriptor(
            "brightness_contrast",
            "Brightness / Contrast",
            (
                OperationParameterSpec("alpha", "Contrast", "float", 1.0, minimum=0.1, maximum=4.0, step=0.05),
                OperationParameterSpec("beta", "Brightness", "float", 0.0, minimum=-255.0, maximum=255.0, step=1.0),
            ),
            _brightness_contrast,
        ),
        OperationDescriptor(
            "gamma_correction",
            "Gamma Correction",
            (OperationParameterSpec("gamma", "Gamma", "float", 1.0, minimum=0.1, maximum=5.0, step=0.05),),
            _gamma_correction,
        ),
        OperationDescriptor(
            "color_binarize",
            "Color Binarize",
            (
                OperationParameterSpec("delta", "Color delta", "int", 10, minimum=0, maximum=255, step=1),
            ),
            _color_binarize,
        ),
        OperationDescriptor(
            "threshold",
            "Threshold",
            (
                OperationParameterSpec("threshold", "Threshold", "float", 127.0, minimum=0.0, maximum=255.0, step=1.0),
                OperationParameterSpec("max_value", "Max value", "float", 255.0, minimum=1.0, maximum=255.0, step=1.0),
                OperationParameterSpec(
                    "threshold_type",
                    "Type",
                    "choice",
                    "binary",
                    options=["binary", "binary_inv", "trunc", "tozero", "tozero_inv"],
                ),
            ),
            _threshold,
        ),
        OperationDescriptor(
            "adaptive_threshold",
            "Adaptive Threshold",
            (
                OperationParameterSpec("max_value", "Max value", "float", 255.0, minimum=1.0, maximum=255.0, step=1.0),
                OperationParameterSpec("adaptive_method", "Method", "choice", "gaussian", options=["mean", "gaussian"]),
                OperationParameterSpec("threshold_type", "Type", "choice", "binary", options=["binary", "binary_inv"]),
                OperationParameterSpec("block_size", "Block size", "int", 11, minimum=3, maximum=99, step=2),
                OperationParameterSpec("c_value", "C", "float", 2.0, minimum=-50.0, maximum=50.0, step=0.5),
            ),
            _adaptive_threshold,
        ),
        OperationDescriptor(
            "otsu_threshold",
            "Otsu Threshold",
            (OperationParameterSpec("threshold_type", "Type", "choice", "binary", options=["binary", "binary_inv"]),),
            _otsu_threshold,
        ),
        OperationDescriptor(
            "edge_guided_threshold",
            "Edge-guided Threshold",
            (
                OperationParameterSpec("threshold_mode", "Base mode", "choice", "otsu", options=["otsu", "adaptive", "manual"]),
                OperationParameterSpec("threshold", "Threshold", "float", 127.0, minimum=0.0, maximum=255.0, step=1.0),
                OperationParameterSpec("max_value", "Max value", "float", 255.0, minimum=1.0, maximum=255.0, step=1.0),
                OperationParameterSpec("threshold_type", "Type", "choice", "binary", options=["binary", "binary_inv"]),
                OperationParameterSpec("adaptive_method", "Method", "choice", "gaussian", options=["mean", "gaussian"]),
                OperationParameterSpec("block_size", "Block size", "int", 11, minimum=3, maximum=99, step=2),
                OperationParameterSpec("c_value", "C", "float", 2.0, minimum=-50.0, maximum=50.0, step=0.5),
                OperationParameterSpec("edge_detector", "Edge detector", "choice", "canny", options=["canny", "sobel"]),
                OperationParameterSpec("edge_percentile", "Sobel percentile", "float", 80.0, minimum=1.0, maximum=99.0, step=1.0),
                OperationParameterSpec("correction_radius", "Correction radius", "int", 2, minimum=0, maximum=25, step=1),
                OperationParameterSpec("threshold1", "Threshold 1", "float", 50.0, minimum=0.0, maximum=500.0, step=1.0),
                OperationParameterSpec("threshold2", "Threshold 2", "float", 150.0, minimum=0.0, maximum=500.0, step=1.0),
                OperationParameterSpec("aperture_size", "Aperture", "int", 3, minimum=3, maximum=7, step=2),
                OperationParameterSpec("l2gradient", "L2 gradient", "bool", False),
                OperationParameterSpec("fill_holes", "Fill holes", "bool", True),
            ),
            _edge_guided_threshold,
        ),
        OperationDescriptor("binary_fill_holes", "Fill Holes", (), _binary_fill_holes),
        OperationDescriptor(
            "binary_filter_area",
            "Filter By Area",
            (
                OperationParameterSpec("min_component_area", "Min area", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0),
                OperationParameterSpec("max_component_area", "Max area", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0),
            ),
            _binary_filter_area,
        ),
        OperationDescriptor(
            "binary_filter_perimeter",
            "Filter By Perimeter",
            (
                OperationParameterSpec("min_component_perimeter", "Min perimeter", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0),
                OperationParameterSpec("max_component_perimeter", "Max perimeter", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0),
            ),
            _binary_filter_perimeter,
        ),
        OperationDescriptor(
            "watershed_split",
            "Watershed Split",
            (
                OperationParameterSpec("distance_ratio", "Distance ratio", "float", 0.35, minimum=0.05, maximum=0.95, step=0.01),
                OperationParameterSpec("min_peak_area", "Min peak area", "int", 0, minimum=0, maximum=1_000_000, step=1),
                OperationParameterSpec("kernel_size", "Kernel size", "int", 3, minimum=1, maximum=99, step=2),
                OperationParameterSpec("shape", "Kernel shape", "choice", "ellipse", options=["rect", "ellipse", "cross"]),
                OperationParameterSpec("background_iterations", "Background iterations", "int", 1, minimum=1, maximum=20, step=1),
            ),
            _watershed_split,
        ),
        OperationDescriptor("morph_open", "Morphological Open", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_OPEN, params)),
        OperationDescriptor("morph_close", "Morphological Close", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_CLOSE, params)),
        OperationDescriptor("erode", "Erode", common_kernel_specs, _erode),
        OperationDescriptor("dilate", "Dilate", common_kernel_specs, _dilate),
        OperationDescriptor("gradient", "Gradient", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_GRADIENT, params)),
        OperationDescriptor("tophat", "TopHat", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_TOPHAT, params)),
        OperationDescriptor("blackhat", "BlackHat", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_BLACKHAT, params)),
        OperationDescriptor(
            "canny",
            "Canny",
            (
                OperationParameterSpec("threshold1", "Threshold 1", "float", 50.0, minimum=0.0, maximum=500.0, step=1.0),
                OperationParameterSpec("threshold2", "Threshold 2", "float", 150.0, minimum=0.0, maximum=500.0, step=1.0),
                OperationParameterSpec("aperture_size", "Aperture", "int", 3, minimum=3, maximum=7, step=2),
                OperationParameterSpec("l2gradient", "L2 gradient", "bool", False),
            ),
            _canny,
        ),
        OperationDescriptor("invert", "Invert", (), _invert),
        OperationDescriptor(
            "resize",
            "Resize",
            (
                OperationParameterSpec("width", "Width", "int", 0, minimum=0, maximum=20000, step=1),
                OperationParameterSpec("height", "Height", "int", 0, minimum=0, maximum=20000, step=1),
                OperationParameterSpec("keep_aspect", "Keep aspect", "bool", True),
                OperationParameterSpec(
                    "interpolation",
                    "Interpolation",
                    "choice",
                    "linear",
                    options=["nearest", "linear", "area", "cubic", "lanczos"],
                ),
            ),
            _resize,
        ),
        OperationDescriptor(
            "scale_resize",
            "Scale Resize",
            (
                OperationParameterSpec("scale", "Scale", "float", 1.0, minimum=0.05, maximum=16.0, step=0.05),
                OperationParameterSpec(
                    "interpolation",
                    "Interpolation",
                    "choice",
                    "linear",
                    options=["nearest", "linear", "area", "cubic", "lanczos"],
                ),
            ),
            _scale_resize,
        ),
        OperationDescriptor(
            "crop",
            "Crop",
            (
                OperationParameterSpec("x", "X", "int", 0, minimum=0, maximum=20000, step=1),
                OperationParameterSpec("y", "Y", "int", 0, minimum=0, maximum=20000, step=1),
                OperationParameterSpec("width", "Width", "int", 512, minimum=1, maximum=20000, step=1),
                OperationParameterSpec("height", "Height", "int", 512, minimum=1, maximum=20000, step=1),
            ),
            _crop,
        ),
        OperationDescriptor(
            "sharpen",
            "Sharpen",
            (
                OperationParameterSpec("amount", "Amount", "float", 1.5, minimum=0.0, maximum=5.0, step=0.1),
                OperationParameterSpec("sigma", "Sigma", "float", 1.0, minimum=0.0, maximum=10.0, step=0.1),
            ),
            _sharpen,
        ),
        OperationDescriptor(
            "denoise",
            "Denoise",
            (
                OperationParameterSpec("h", "Strength", "float", 10.0, minimum=1.0, maximum=50.0, step=0.5),
                OperationParameterSpec("template_window_size", "Template window", "int", 7, minimum=3, maximum=21, step=2),
                OperationParameterSpec("search_window_size", "Search window", "int", 21, minimum=3, maximum=31, step=2),
            ),
            _denoise,
        ),
    ]
    for descriptor in descriptors:
        register_operation(descriptor)


_register_builtin_operations()

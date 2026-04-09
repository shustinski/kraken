from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .application.processing import OperationParameterSpec, PipelineStepConfig
from .i18n import choice_label, operation_name, parameter_label, tr
from .utils import ensure_uint8


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
    tile_grid = max(1, int(parameters.get("tile_grid_size", 8)))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(parameters.get("clip_limit", 2.0))),
        tileGridSize=(tile_grid, tile_grid),
    )
    return clahe.apply(image)


def _histogram_equalization(image: np.ndarray, _: dict[str, Any]) -> np.ndarray:
    return cv2.equalizeHist(image)


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
    _, result = cv2.threshold(
        image,
        float(parameters.get("threshold", 127)),
        float(parameters.get("max_value", 255)),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
    )
    return result


def _adaptive_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image,
        float(parameters.get("max_value", 255)),
        _adaptive_mode(str(parameters.get("adaptive_method", "gaussian"))),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
        _odd(int(parameters.get("block_size", 11)), minimum=3),
        float(parameters.get("c_value", 2.0)),
    )


def _otsu_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    _, result = cv2.threshold(
        image,
        0,
        255,
        _threshold_mode(str(parameters.get("threshold_type", "binary"))) | cv2.THRESH_OTSU,
    )
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
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=max(1.0, float(parameters.get("h", 10.0))),
        templateWindowSize=_odd(int(parameters.get("template_window_size", 7)), minimum=3),
        searchWindowSize=_odd(int(parameters.get("search_window_size", 21)), minimum=3),
    )


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

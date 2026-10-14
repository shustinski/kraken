"""pipeline: register builtin operations."""
from __future__ import annotations
import cv2
from ..application.processing import OperationParameterSpec
from contour.edge_detection import (
    EDGE_METHOD_CHOICES,
)

from contour.pipeline_parts.module_state import (
    _COMBINED_PRESETS,
    _OPERATIONS,
)

from contour.pipeline_parts.edge_contour_refined_mask import (
    _adaptive_threshold,
    _brightness,
    _brightness_contrast,
    _clahe,
    _contrast,
    _gamma_correction,
    _histogram_equalization,
    _otsu_threshold,
    _threshold,
)

from contour.pipeline_parts.edge_guided_threshold import (
    _auto_canny,
    _canny,
    _combined_edges,
    _crop,
    _denoise,
    _dilate,
    _edge_dispatch,
    _edge_guided_threshold,
    _erode,
    _invert,
    _log_edges,
    _morph_op,
    _phase_congruency_edges,
    _resize,
    _ridge_edges,
    _scale_resize,
    _scharr_edges,
    _sharpen,
    _structured_edge_operation,
)

from contour.pipeline_parts.operation_descriptor import (
    OperationDescriptor,
    register_operation,
)

from contour.pipeline_parts.watershed_split import (
    _bilateral,
    _binary_fill_holes,
    _binary_filter_area,
    _binary_filter_perimeter,
    _color_binarize,
    _gaussian_blur,
    _median_blur,
    _watershed_split,
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
                OperationParameterSpec(
                    "sigma_color", "Sigma color", "float", 75.0, minimum=1.0, maximum=200.0, step=1.0
                ),
                OperationParameterSpec(
                    "sigma_space", "Sigma space", "float", 75.0, minimum=1.0, maximum=200.0, step=1.0
                ),
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
            public=False,
        ),
        OperationDescriptor(
            "brightness",
            "Brightness",
            (
                OperationParameterSpec(
                    "brightness",
                    "Brightness",
                    "float",
                    0.0,
                    minimum=-255.0,
                    maximum=255.0,
                    step=1.0,
                ),
            ),
            _brightness,
        ),
        OperationDescriptor(
            "contrast",
            "Contrast",
            (
                OperationParameterSpec(
                    "contrast",
                    "Contrast",
                    "float",
                    1.0,
                    minimum=0.1,
                    maximum=4.0,
                    step=0.05,
                ),
            ),
            _contrast,
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
            (OperationParameterSpec("delta", "Color delta", "int", 10, minimum=0, maximum=255, step=1),),
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
                OperationParameterSpec(
                    "threshold_mode", "Base mode", "choice", "otsu", options=["otsu", "adaptive", "manual"]
                ),
                OperationParameterSpec("threshold", "Threshold", "float", 127.0, minimum=0.0, maximum=255.0, step=1.0),
                OperationParameterSpec("max_value", "Max value", "float", 255.0, minimum=1.0, maximum=255.0, step=1.0),
                OperationParameterSpec("threshold_type", "Type", "choice", "binary", options=["binary", "binary_inv"]),
                OperationParameterSpec("adaptive_method", "Method", "choice", "gaussian", options=["mean", "gaussian"]),
                OperationParameterSpec("block_size", "Block size", "int", 11, minimum=3, maximum=99, step=2),
                OperationParameterSpec("c_value", "C", "float", 2.0, minimum=-50.0, maximum=50.0, step=0.5),
                OperationParameterSpec("edge_detector", "Edge detector", "choice", "canny", options=["canny", "sobel"]),
                OperationParameterSpec(
                    "edge_percentile", "Sobel percentile", "float", 80.0, minimum=1.0, maximum=99.0, step=1.0
                ),
                OperationParameterSpec(
                    "correction_radius", "Correction radius", "int", 2, minimum=0, maximum=25, step=1
                ),
                OperationParameterSpec(
                    "threshold1", "Threshold 1", "float", 50.0, minimum=0.0, maximum=500.0, step=1.0
                ),
                OperationParameterSpec(
                    "threshold2", "Threshold 2", "float", 150.0, minimum=0.0, maximum=500.0, step=1.0
                ),
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
                OperationParameterSpec(
                    "min_component_area", "Min area", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0
                ),
                OperationParameterSpec(
                    "max_component_area", "Max area", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0
                ),
            ),
            _binary_filter_area,
        ),
        OperationDescriptor(
            "binary_filter_perimeter",
            "Filter By Perimeter",
            (
                OperationParameterSpec(
                    "min_component_perimeter", "Min perimeter", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0
                ),
                OperationParameterSpec(
                    "max_component_perimeter", "Max perimeter", "float", 0.0, minimum=0.0, maximum=1_000_000.0, step=1.0
                ),
            ),
            _binary_filter_perimeter,
        ),
        OperationDescriptor(
            "watershed_split",
            "Watershed Split",
            (
                OperationParameterSpec(
                    "distance_ratio", "Distance ratio", "float", 0.35, minimum=0.05, maximum=0.95, step=0.01
                ),
                OperationParameterSpec(
                    "min_peak_area", "Min peak area", "int", 0, minimum=0, maximum=1_000_000, step=1
                ),
                OperationParameterSpec("kernel_size", "Kernel size", "int", 3, minimum=1, maximum=99, step=2),
                OperationParameterSpec(
                    "shape", "Kernel shape", "choice", "ellipse", options=["rect", "ellipse", "cross"]
                ),
                OperationParameterSpec(
                    "background_iterations", "Background iterations", "int", 1, minimum=1, maximum=20, step=1
                ),
            ),
            _watershed_split,
        ),
        OperationDescriptor(
            "morph_open",
            "Morphological Open",
            common_kernel_specs,
            lambda image, params: _morph_op(image, cv2.MORPH_OPEN, params),
        ),
        OperationDescriptor(
            "morph_close",
            "Morphological Close",
            common_kernel_specs,
            lambda image, params: _morph_op(image, cv2.MORPH_CLOSE, params),
        ),
        OperationDescriptor("erode", "Erode", common_kernel_specs, _erode),
        OperationDescriptor("dilate", "Dilate", common_kernel_specs, _dilate),
        OperationDescriptor(
            "gradient",
            "Gradient",
            common_kernel_specs,
            lambda image, params: _morph_op(image, cv2.MORPH_GRADIENT, params),
        ),
        OperationDescriptor(
            "tophat", "TopHat", common_kernel_specs, lambda image, params: _morph_op(image, cv2.MORPH_TOPHAT, params)
        ),
        OperationDescriptor(
            "blackhat",
            "BlackHat",
            common_kernel_specs,
            lambda image, params: _morph_op(image, cv2.MORPH_BLACKHAT, params),
        ),
        OperationDescriptor(
            "canny",
            "Canny",
            (
                OperationParameterSpec(
                    "threshold1", "Threshold 1", "float", 50.0, minimum=0.0, maximum=500.0, step=1.0
                ),
                OperationParameterSpec(
                    "threshold2", "Threshold 2", "float", 150.0, minimum=0.0, maximum=500.0, step=1.0
                ),
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
                OperationParameterSpec(
                    "template_window_size", "Template window", "int", 7, minimum=3, maximum=21, step=2
                ),
                OperationParameterSpec("search_window_size", "Search window", "int", 21, minimum=3, maximum=31, step=2),
            ),
            _denoise,
        ),
        OperationDescriptor(
            "scharr_edges",
            "Scharr Edges",
            (),
            _scharr_edges,
        ),
        OperationDescriptor(
            "auto_canny",
            "Auto Canny",
            (
                OperationParameterSpec("sigma", "Sigma", "float", 0.33, minimum=0.0, maximum=1.0, step=0.01),
                OperationParameterSpec("aperture_size", "Aperture", "int", 3, minimum=3, maximum=7, step=2),
                OperationParameterSpec("l2gradient", "L2 gradient", "bool", True),
            ),
            _auto_canny,
        ),
        OperationDescriptor(
            "log_edges",
            "Laplacian of Gaussian",
            (
                OperationParameterSpec("sigma_small", "Sigma small", "float", 1.0, minimum=0.4, maximum=8.0, step=0.1),
                OperationParameterSpec("sigma_mid", "Sigma mid", "float", 2.0, minimum=0.5, maximum=12.0, step=0.1),
                OperationParameterSpec("sigma_large", "Sigma large", "float", 3.5, minimum=0.5, maximum=16.0, step=0.1),
            ),
            _log_edges,
        ),
        OperationDescriptor(
            "ridge_edges",
            "Ridge Response",
            (OperationParameterSpec("sigma", "Sigma", "float", 1.5, minimum=0.5, maximum=8.0, step=0.1),),
            _ridge_edges,
        ),
        OperationDescriptor(
            "structured_edges",
            "Structured Edges",
            (),
            _structured_edge_operation,
        ),
        OperationDescriptor(
            "phase_congruency",
            "Phase Congruency",
            (
                OperationParameterSpec("num_scales", "Scales", "int", 4, minimum=1, maximum=8, step=1),
                OperationParameterSpec("num_orientations", "Orientations", "int", 6, minimum=1, maximum=12, step=1),
                OperationParameterSpec(
                    "min_wavelength", "Min wavelength", "float", 3.0, minimum=1.5, maximum=16.0, step=0.1
                ),
                OperationParameterSpec(
                    "scale_factor", "Scale factor", "float", 2.1, minimum=1.2, maximum=4.0, step=0.1
                ),
            ),
            _phase_congruency_edges,
        ),
        OperationDescriptor(
            "combined_edges",
            "Combined Edges",
            (
                OperationParameterSpec(
                    "preset",
                    "Preset",
                    "choice",
                    "default",
                    options=list(_COMBINED_PRESETS.keys()),
                ),
            ),
            _combined_edges,
        ),
        OperationDescriptor(
            "edge_method",
            "Edge Detector",
            (
                OperationParameterSpec(
                    "method",
                    "Method",
                    "choice",
                    "sobel",
                    options=list(EDGE_METHOD_CHOICES),
                ),
            ),
            _edge_dispatch,
        ),
    ]
    for descriptor in descriptors:
        register_operation(descriptor)

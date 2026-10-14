"""PIPELINE_OPERATION_GROUPS."""
from __future__ import annotations
from contour.graphics_view import EditorTool

PIPELINE_OPERATION_GROUPS: tuple[tuple[str, tuple[str, str], tuple[str, ...]], ...] = (
    (
        "smoothing",
        ("Сглаживание и шум", "Smoothing and noise"),
        ("gaussian_blur", "median_blur", "bilateral_filter", "denoise"),
    ),
    (
        "contrast",
        ("Контраст и тон", "Contrast and tone"),
        ("clahe", "histogram_equalization", "brightness", "contrast", "gamma_correction", "sharpen"),
    ),
    (
        "thresholding",
        ("Пороговая обработка", "Thresholding"),
        ("color_binarize", "threshold", "adaptive_threshold", "otsu_threshold", "edge_guided_threshold", "invert"),
    ),
    (
        "binary",
        ("Бинарные фильтры", "Binary filters"),
        ("binary_fill_holes", "binary_filter_area", "binary_filter_perimeter", "watershed_split"),
    ),
    (
        "morphology",
        ("Морфология", "Morphology"),
        ("morph_open", "morph_close", "erode", "dilate", "gradient", "tophat", "blackhat"),
    ),
    (
        "geometry",
        ("Геометрия и границы", "Geometry and edges"),
        ("canny", "resize", "scale_resize", "crop"),
    ),
    (
        "advanced_edges",
        ("Современные детекторы границ", "Modern edge detectors"),
        (
            "scharr_edges",
            "auto_canny",
            "log_edges",
            "ridge_edges",
            "structured_edges",
            "phase_congruency",
            "combined_edges",
            "edge_method",
        ),
    ),
)

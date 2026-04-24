"""Built-in preprocessing pipeline presets for common image conditions."""

from __future__ import annotations


def _low_noise_contrast_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "operation": "clahe",
                "name": "CLAHE",
                "enabled": True,
                "parameters": {"clip_limit": 2.0, "tile_grid_size": 8},
            },
            {
                "operation": "edge_guided_threshold",
                "name": "Edge-guided Threshold",
                "enabled": True,
                "parameters": {
                    "threshold_mode": "otsu",
                    "threshold_type": "binary",
                    "edge_detector": "sobel",
                    "edge_percentile": 82.0,
                    "correction_radius": 2,
                    "fill_holes": True,
                },
            },
            {
                "operation": "morph_open",
                "name": "Morphological Open",
                "enabled": True,
                "parameters": {"kernel_size": 3, "iterations": 1, "shape": "ellipse"},
            },
        ]
    }


def _noisy_texture_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "operation": "median_blur",
                "name": "Median Blur",
                "enabled": True,
                "parameters": {"kernel_size": 5},
            },
            {
                "operation": "clahe",
                "name": "CLAHE",
                "enabled": True,
                "parameters": {"clip_limit": 1.8, "tile_grid_size": 8},
            },
            {
                "operation": "edge_guided_threshold",
                "name": "Edge-guided Threshold",
                "enabled": True,
                "parameters": {
                    "threshold_mode": "adaptive",
                    "adaptive_method": "gaussian",
                    "block_size": 15,
                    "c_value": 2.0,
                    "threshold_type": "binary",
                    "edge_detector": "canny",
                    "threshold1": 45.0,
                    "threshold2": 130.0,
                    "correction_radius": 2,
                    "fill_holes": True,
                },
            },
            {
                "operation": "binary_filter_area",
                "name": "Filter By Area",
                "enabled": True,
                "parameters": {"min_component_area": 10.0, "max_component_area": 0.0},
            },
        ]
    }


def _uneven_illumination_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "operation": "gamma_correction",
                "name": "Gamma Correction",
                "enabled": True,
                "parameters": {"gamma": 0.85},
            },
            {
                "operation": "clahe",
                "name": "CLAHE",
                "enabled": True,
                "parameters": {"clip_limit": 2.5, "tile_grid_size": 10},
            },
            {
                "operation": "edge_guided_threshold",
                "name": "Edge-guided Threshold",
                "enabled": True,
                "parameters": {
                    "threshold_mode": "adaptive",
                    "adaptive_method": "gaussian",
                    "block_size": 19,
                    "c_value": 3.0,
                    "threshold_type": "binary",
                    "edge_detector": "sobel",
                    "edge_percentile": 80.0,
                    "correction_radius": 3,
                    "fill_holes": True,
                },
            },
            {
                "operation": "morph_close",
                "name": "Morphological Close",
                "enabled": True,
                "parameters": {"kernel_size": 3, "iterations": 1, "shape": "ellipse"},
            },
        ]
    }


def _weak_blurred_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "operation": "bilateral_filter",
                "name": "Bilateral Filter",
                "enabled": True,
                "parameters": {"diameter": 9, "sigma_color": 60.0, "sigma_space": 60.0},
            },
            {
                "operation": "sharpen",
                "name": "Sharpen",
                "enabled": True,
                "parameters": {"amount": 1.2, "sigma": 1.1},
            },
            {
                "operation": "edge_guided_threshold",
                "name": "Edge-guided Threshold",
                "enabled": True,
                "parameters": {
                    "threshold_mode": "otsu",
                    "threshold_type": "binary",
                    "edge_detector": "canny",
                    "threshold1": 40.0,
                    "threshold2": 120.0,
                    "correction_radius": 2,
                    "fill_holes": True,
                },
            },
            {
                "operation": "morph_close",
                "name": "Morphological Close",
                "enabled": True,
                "parameters": {"kernel_size": 5, "iterations": 1, "shape": "ellipse"},
            },
        ]
    }


def built_in_pipeline_presets(language: str) -> dict[str, dict[str, object]]:
    if language == "ru":
        return {
            "SEM clean metal": _low_noise_contrast_payload(),
            "SEM noisy grain": _noisy_texture_payload(),
            "SEM weak vias on traces": _weak_blurred_payload(),
            "SEM uneven illumination": _uneven_illumination_payload(),
        }
    return {
        "SEM clean metal": _low_noise_contrast_payload(),
        "SEM noisy grain": _noisy_texture_payload(),
        "SEM weak vias on traces": _weak_blurred_payload(),
        "SEM uneven illumination": _uneven_illumination_payload(),
    }


__all__ = ["built_in_pipeline_presets"]

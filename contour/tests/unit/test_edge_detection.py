from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.edge_detection import (
    EDGE_METHOD_AUTO_CANNY,
    EDGE_METHOD_CHOICES,
    EDGE_METHOD_COMBINED,
    EDGE_METHOD_LOG,
    EDGE_METHOD_PHASE_CONGRUENCY,
    EDGE_METHOD_SCHARR,
    EDGE_METHOD_SOBEL,
    EDGE_METHOD_STRUCTURED,
    auto_canny,
    build_gradient_elevation,
    combined_elevation,
    gradient_color_map,
    laplacian_magnitude,
    laplacian_of_gaussian,
    normalize_edge_method,
    phase_congruency,
    ridge_response,
    scharr_magnitude,
    sobel_magnitude,
    structured_edges,
)


def _make_bright_spot_image(size: int = 96, radius: int = 12) -> np.ndarray:
    image = np.full((size, size), 40, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), radius, 230, thickness=-1)
    return image


def _make_ridge_image(size: int = 96, ridge_width: int = 4) -> np.ndarray:
    image = np.full((size, size), 30, dtype=np.uint8)
    y_start = size // 2 - ridge_width // 2
    y_end = y_start + ridge_width
    image[y_start:y_end, :] = 220
    return image


def _make_color_image(size: int = 96) -> np.ndarray:
    color = cv2.cvtColor(_make_bright_spot_image(size), cv2.COLOR_GRAY2BGR)
    return color


class EdgeDetectionMethodsTests(unittest.TestCase):
    def test_all_methods_return_uint8_of_same_shape(self) -> None:
        image = _make_bright_spot_image()
        for method in EDGE_METHOD_CHOICES:
            with self.subTest(method=method):
                elevation = build_gradient_elevation(image, method)
                self.assertIsInstance(elevation, np.ndarray)
                self.assertEqual(elevation.dtype, np.uint8)
                self.assertEqual(elevation.shape, image.shape)
                self.assertGreaterEqual(int(elevation.max()), 1)

    def test_normalize_edge_method_accepts_aliases_and_unknown(self) -> None:
        self.assertEqual(normalize_edge_method("canny"), EDGE_METHOD_AUTO_CANNY)
        self.assertEqual(normalize_edge_method("phase"), EDGE_METHOD_PHASE_CONGRUENCY)
        self.assertEqual(normalize_edge_method("ml"), EDGE_METHOD_STRUCTURED)
        self.assertEqual(normalize_edge_method(""), EDGE_METHOD_SOBEL)
        self.assertEqual(normalize_edge_method("does-not-exist"), EDGE_METHOD_SOBEL)
        self.assertEqual(normalize_edge_method(None), EDGE_METHOD_SOBEL)

    def test_sobel_and_scharr_highlight_a_bright_disc(self) -> None:
        image = _make_bright_spot_image()
        cy, cx = image.shape[0] // 2, image.shape[1] // 2

        for producer in (sobel_magnitude, scharr_magnitude):
            with self.subTest(op=producer.__name__):
                elevation = producer(image)
                disc_interior = elevation[cy - 3 : cy + 3, cx - 3 : cx + 3].mean()
                disc_edge = elevation[cy - 12 : cy + 12, cx - 12 : cx + 12].max()
                flat_region = elevation[:20, :20].mean()
                self.assertLess(disc_interior, disc_edge)
                self.assertLess(flat_region, disc_edge)

    def test_auto_canny_returns_binary_edge_map(self) -> None:
        image = _make_bright_spot_image()
        edges = auto_canny(image)
        unique_values = np.unique(edges)
        self.assertTrue(
            np.array_equal(unique_values, np.array([0], dtype=np.uint8))
            or np.array_equal(unique_values, np.array([0, 255], dtype=np.uint8))
            or set(int(v) for v in unique_values).issubset({0, 255})
        )
        self.assertEqual(edges.dtype, np.uint8)

    def test_log_responds_stronger_to_blob_than_to_flat_area(self) -> None:
        image = _make_bright_spot_image()
        elevation = laplacian_of_gaussian(image)
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        blob_window = elevation[cy - 14 : cy + 14, cx - 14 : cx + 14]
        flat_window = elevation[:20, :20]
        self.assertGreater(float(blob_window.max()), float(flat_window.mean()) + 20)

    def test_laplacian_output_is_in_valid_range(self) -> None:
        image = _make_bright_spot_image()
        elevation = laplacian_magnitude(image)
        self.assertEqual(elevation.dtype, np.uint8)
        self.assertGreaterEqual(int(elevation.min()), 0)
        self.assertLessEqual(int(elevation.max()), 255)

    def test_ridge_response_peaks_on_horizontal_ridge(self) -> None:
        image = _make_ridge_image()
        elevation = ridge_response(image)
        ridge_row = image.shape[0] // 2
        along_ridge = elevation[ridge_row - 1 : ridge_row + 2, :].mean()
        far_from_ridge = elevation[:6, :].mean()
        self.assertGreater(along_ridge, far_from_ridge)

    def test_structured_edges_falls_back_without_contrib(self) -> None:
        image = _make_bright_spot_image()
        elevation = structured_edges(image)
        self.assertEqual(elevation.dtype, np.uint8)
        self.assertEqual(elevation.shape, image.shape)
        self.assertGreater(int(elevation.max()), 0)

    def test_phase_congruency_runs_on_small_image(self) -> None:
        image = _make_bright_spot_image(size=64, radius=8)
        elevation = phase_congruency(image, num_scales=2, num_orientations=4)
        self.assertEqual(elevation.dtype, np.uint8)
        self.assertEqual(elevation.shape, image.shape)

    def test_combined_ensemble_dominates_individual_methods(self) -> None:
        image = _make_bright_spot_image()
        scharr = scharr_magnitude(image).astype(np.int32)
        log_map = laplacian_of_gaussian(image).astype(np.int32)
        combined = combined_elevation(image, [EDGE_METHOD_SCHARR, EDGE_METHOD_LOG]).astype(np.int32)
        self.assertTrue(np.all(combined >= scharr))
        self.assertTrue(np.all(combined >= log_map))

    def test_combined_handles_unknown_methods_and_empty_list(self) -> None:
        image = _make_bright_spot_image()
        default = combined_elevation(image, [])
        nonsense = combined_elevation(image, ["unknown", "also-unknown", EDGE_METHOD_COMBINED])
        self.assertEqual(default.shape, image.shape)
        self.assertEqual(default.dtype, np.uint8)
        self.assertEqual(nonsense.shape, image.shape)
        self.assertEqual(nonsense.dtype, np.uint8)

    def test_gradient_color_map_returns_bgr_uint8(self) -> None:
        image = _make_bright_spot_image()
        color = gradient_color_map(image, EDGE_METHOD_SCHARR)
        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(color.ndim, 3)
        self.assertEqual(color.shape[:2], image.shape)
        self.assertEqual(color.shape[2], 3)

    def test_build_gradient_elevation_accepts_color_input(self) -> None:
        color = _make_color_image()
        elevation = build_gradient_elevation(color, EDGE_METHOD_SOBEL)
        self.assertEqual(elevation.shape, color.shape[:2])
        self.assertEqual(elevation.dtype, np.uint8)

    def test_build_gradient_elevation_unknown_method_defaults_to_sobel(self) -> None:
        image = _make_bright_spot_image()
        expected = sobel_magnitude(image)
        actual = build_gradient_elevation(image, "mysterious-method")
        self.assertTrue(np.array_equal(actual, expected))

    def test_empty_image_returns_empty_uint8_map(self) -> None:
        empty = np.zeros((0, 0), dtype=np.uint8)
        for method in EDGE_METHOD_CHOICES:
            with self.subTest(method=method):
                elevation = build_gradient_elevation(empty, method)
                self.assertEqual(elevation.dtype, np.uint8)
                self.assertEqual(elevation.size, 0)


if __name__ == "__main__":
    unittest.main()

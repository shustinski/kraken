"""benchmark_metal_segmentation: build benchmark cases."""
from __future__ import annotations
import numpy as np

from benchmark_metal_segmentation_parts.render_sem import (
    BenchmarkCase,
    _draw_line,
    _draw_rect,
    _render_sem,
)

def build_benchmark_cases() -> list[BenchmarkCase]:
    shape = (256, 256)
    cases: list[BenchmarkCase] = []

    def add(
        name: str,
        category: str,
        labels: np.ndarray,
        **render_options: object,
    ) -> None:
        image = _render_sem(labels, seed=100 + len(cases), **render_options)
        cases.append(BenchmarkCase(name, category, image, labels))

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (32, 48, 224, 208))
    add("large_isolated", "scale", labels)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((20, 30), (235, 226)), 4)
    add("narrow_4px", "scale", labels, metal=88.0, rim=150.0, rim_width=1)

    for gap in (1, 2, 3):
        labels = np.zeros(shape, np.int32)
        left_end = 126 - gap // 2
        right_start = left_end + gap
        _draw_rect(labels, 1, (36, 12, left_end, 244))
        _draw_rect(labels, 2, (right_start, 12, 220, 244))
        add(
            f"parallel_gap_{gap}px",
            "close_conductors",
            labels,
            elevated_regions=((slice(12, 244), slice(left_end, right_start), 105.0),),
        )

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (24, 30, 232, 226))
    add(
        "wide_dark_center",
        "dark_interior",
        labels,
        metal=104.0,
        dark_center=49.0,
        dark_center_depth=13.0,
    )

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (34, 35, 222, 221))
    add("bright_rims", "rim", labels, metal=76.0, rim=224.0, rim_width=3)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((18, 70), (238, 70), (238, 190), (35, 190)), 10)
    add("weak_contrast", "contrast", labels, metal=72.0, rim=92.0, rim_width=1)

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (0, 52, 116, 184))
    add("touches_one_border", "border", labels, dark_center=58.0, dark_center_depth=11.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((0, 128), (128, 128), (128, 0)), 28)
    add("crosses_multiple_borders", "border", labels, dark_center=57.0, dark_center_depth=9.0)

    labels = np.zeros(shape, np.int32)
    for object_id, x_coord in enumerate((28, 63, 98, 133, 168, 203), start=1):
        _draw_line(labels, object_id, ((x_coord, 0), (x_coord, 256)), 9)
    add("parallel_bundle", "topology", labels, metal=92.0, rim=178.0, rim_width=2)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((20, 128), (236, 128)), 20)
    _draw_line(labels, 1, ((128, 22), (128, 234)), 20)
    add("junction", "topology", labels, dark_center=62.0, dark_center_depth=7.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((15, 55), (240, 55)), 11)
    _draw_line(labels, 2, ((15, 155), (240, 205)), 13)
    add("noisy_background", "noise", labels, noise_sigma=11.0, rim=175.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((8, 45), (245, 45)), 14)
    _draw_line(labels, 2, ((8, 210), (245, 210)), 14)
    add("illumination_drift", "illumination", labels, illumination_amplitude=30.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((15, 80), (240, 80)), 8)
    _draw_line(labels, 2, ((15, 145), (240, 145)), 16)
    add("blurred_boundaries", "blur", labels, blur_sigma=2.0, rim=170.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((0, 35), (218, 35), (218, 112)), 7)
    _draw_line(labels, 2, ((18, 124), (238, 124)), 17)
    _draw_line(labels, 3, ((18, 143), (238, 143)), 17)
    _draw_rect(labels, 4, (32, 174, 224, 256))
    add(
        "combined_stress",
        "combined",
        labels,
        metal=91.0,
        rim=185.0,
        dark_center=54.0,
        dark_center_depth=9.0,
        noise_sigma=8.0,
        illumination_amplitude=22.0,
        blur_sigma=1.4,
        elevated_regions=((slice(124, 144), slice(18, 238), 101.0),),
    )
    return cases

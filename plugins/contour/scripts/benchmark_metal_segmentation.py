"""Deterministic topology-sensitive benchmark for classical metal segmentation.

The synthetic scenes intentionally contain blurred edges, shot/read noise,
illumination drift, bright rims, dark conductor interiors, and elevated narrow
gaps.  They are not intended to replace CIF golden tests; they isolate failure
classes whose ground-truth object topology is known exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from contour.vision.metal_recovery.pipeline_stages import _segment  # noqa: E402

STRATEGIES = (
    "legacy_otsu",
    "local_adaptive",
    "gradient_watershed",
    "random_walker",
    "graph_cut",
    "reconstruction",
    "closed_boundary",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    category: str
    image: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    iou: float
    precision: float
    recall: float
    boundary_f1: float
    expected_components: int
    predicted_components: int
    false_merges: int
    false_splits: int
    elapsed_ms: float


def _draw_line(labels: np.ndarray, object_id: int, points: Iterable[tuple[int, int]], width: int) -> None:
    vertices = np.asarray(tuple(points), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(labels, [vertices], False, int(object_id), thickness=int(width), lineType=cv2.LINE_8)


def _draw_rect(labels: np.ndarray, object_id: int, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    labels[y0:y1, x0:x1] = int(object_id)


def _render_sem(
    labels: np.ndarray,
    *,
    seed: int,
    substrate: float = 46.0,
    metal: float = 105.0,
    rim: float = 205.0,
    rim_width: int = 2,
    dark_center: float | None = None,
    dark_center_depth: float = 8.0,
    noise_sigma: float = 4.0,
    illumination_amplitude: float = 8.0,
    blur_sigma: float = 0.9,
    elevated_regions: tuple[tuple[slice, slice, float], ...] = (),
) -> np.ndarray:
    height, width = labels.shape
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    illumination = illumination_amplitude * (
        0.55 * np.sin(xx / max(24.0, width * 0.31))
        + 0.45 * np.cos(yy / max(24.0, height * 0.37))
    )
    image = np.full(labels.shape, substrate, dtype=np.float32) + illumination
    foreground = labels > 0
    image[foreground] = metal + 0.35 * illumination[foreground]

    if dark_center is not None:
        distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
        center = foreground & (distance >= float(dark_center_depth))
        image[center] = float(dark_center) + 0.25 * illumination[center]

    if rim_width > 0:
        kernel_size = 2 * int(rim_width) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior_edge = (
            cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        ) & foreground
        image[interior_edge] = float(rim)

    for rows, columns, value in elevated_regions:
        region = labels[rows, columns] == 0
        patch = image[rows, columns]
        patch[region] = float(value)

    rng = np.random.default_rng(seed)
    image += rng.normal(0.0, float(noise_sigma), size=image.shape).astype(np.float32)
    # A scaled Poisson draw approximates shot noise without overwhelming the
    # deliberately weak contrast cases.
    image = rng.poisson(np.clip(image, 0.0, 255.0) * 4.0).astype(np.float32) * 0.25
    if blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


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


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def _boundary_f1(predicted: np.ndarray, expected: np.ndarray, tolerance_px: int = 2) -> float:
    pred_boundary = _boundary(predicted)
    true_boundary = _boundary(expected)
    if not pred_boundary.any() and not true_boundary.any():
        return 1.0
    kernel_size = 2 * max(0, int(tolerance_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_near_true = cv2.dilate(true_boundary.astype(np.uint8), kernel) > 0
    true_near_pred = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(pred_boundary & pred_near_true)) / max(
        1, int(np.count_nonzero(pred_boundary))
    )
    recall = float(np.count_nonzero(true_boundary & true_near_pred)) / max(
        1, int(np.count_nonzero(true_boundary))
    )
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def _topology_errors(predicted: np.ndarray, expected_labels: np.ndarray) -> tuple[int, int, int]:
    predicted_count, predicted_labels = cv2.connectedComponents(
        (predicted > 0).astype(np.uint8), connectivity=8
    )
    false_merges = 0
    for predicted_id in range(1, predicted_count):
        overlaps = expected_labels[predicted_labels == predicted_id]
        object_ids = np.unique(overlaps[overlaps > 0])
        material_overlaps = sum(
            int(np.count_nonzero(overlaps == object_id))
            >= max(3, round(0.01 * np.count_nonzero(expected_labels == object_id)))
            for object_id in object_ids
        )
        false_merges += max(0, material_overlaps - 1)

    false_splits = 0
    for object_id in range(1, int(expected_labels.max()) + 1):
        overlaps = predicted_labels[expected_labels == object_id]
        component_ids = np.unique(overlaps[overlaps > 0])
        material_overlaps = sum(
            int(np.count_nonzero(overlaps == component_id))
            >= max(3, round(0.01 * np.count_nonzero(expected_labels == object_id)))
            for component_id in component_ids
        )
        false_splits += max(0, material_overlaps - 1)
    return predicted_count - 1, false_merges, false_splits


def measure_segmentation(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    *,
    elapsed_ms: float,
) -> SegmentationMetrics:
    predicted_active = predicted > 0
    expected_active = expected_labels > 0
    true_positive = int(np.count_nonzero(predicted_active & expected_active))
    false_positive = int(np.count_nonzero(predicted_active & ~expected_active))
    false_negative = int(np.count_nonzero(~predicted_active & expected_active))
    predicted_components, false_merges, false_splits = _topology_errors(
        predicted, expected_labels
    )
    return SegmentationMetrics(
        iou=true_positive / max(1, true_positive + false_positive + false_negative),
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        boundary_f1=_boundary_f1(predicted, expected_active.astype(np.uint8) * 255),
        expected_components=int(expected_labels.max()),
        predicted_components=predicted_components,
        false_merges=false_merges,
        false_splits=false_splits,
        elapsed_ms=float(elapsed_ms),
    )


def resolution_probe(source_width: int = 2000, working_width: int = 640) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for feature in ("gap", "conductor"):
        background, foreground = ((255, 0) if feature == "gap" else (0, 255))
        for width_px in range(1, 9):
            coverage: list[float] = []
            vanished = 0
            for phase in range(10):
                mask = np.full((32, source_width), background, np.uint8)
                start = source_width // 2 + phase
                mask[:, start : start + width_px] = foreground
                working_height = max(1, round(mask.shape[0] * working_width / source_width))
                reduced = cv2.resize(
                    mask, (working_width, working_height), interpolation=cv2.INTER_NEAREST
                )
                restored = cv2.resize(
                    reduced, (source_width, mask.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                retained = float(np.mean(restored[:, start : start + width_px] == foreground))
                coverage.append(retained)
                vanished += int(retained == 0.0)
            results.append(
                {
                    "feature": feature,
                    "width_px": width_px,
                    "mean_retained_fraction": float(np.mean(coverage)),
                    "vanished_phases_of_10": vanished,
                }
            )
    return results


def run_benchmark(strategies: Iterable[str] = STRATEGIES) -> dict[str, object]:
    config = GradientWatershedConfig()
    case_results: dict[str, dict[str, dict[str, float | int]]] = {}
    strategy_names = tuple(strategies)
    for case in build_benchmark_cases():
        per_strategy: dict[str, dict[str, float | int]] = {}
        for strategy in strategy_names:
            started = perf_counter()
            mask, _polarity, _selected = _segment(
                case.image,
                strategy=strategy,
                watershed_config=config,
            )
            elapsed_ms = (perf_counter() - started) * 1000.0
            per_strategy[strategy] = asdict(
                measure_segmentation(mask, case.labels, elapsed_ms=elapsed_ms)
            )
        case_results[case.name] = per_strategy

    aggregates: dict[str, dict[str, float | int]] = {}
    for strategy in strategy_names:
        metrics = [case_results[name][strategy] for name in case_results]
        aggregates[strategy] = {
            "mean_iou": float(np.mean([item["iou"] for item in metrics])),
            "mean_precision": float(np.mean([item["precision"] for item in metrics])),
            "mean_recall": float(np.mean([item["recall"] for item in metrics])),
            "mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in metrics])),
            "false_merges": int(sum(item["false_merges"] for item in metrics)),
            "false_splits": int(sum(item["false_splits"] for item in metrics)),
            "elapsed_ms": float(sum(item["elapsed_ms"] for item in metrics)),
        }
    return {
        "schema_version": 1,
        "case_count": len(case_results),
        "strategies": list(strategy_names),
        "config": asdict(config),
        "aggregates": aggregates,
        "cases": case_results,
        "resolution_probe": resolution_probe(),
    }


def _markdown(report: dict[str, object]) -> str:
    aggregates = report["aggregates"]
    assert isinstance(aggregates, dict)
    lines = [
        "| strategy | mean IoU | precision | recall | boundary F1 | merges | splits | ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, raw_metrics in aggregates.items():
        assert isinstance(raw_metrics, dict)
        lines.append(
            f"| {strategy} | {raw_metrics['mean_iou']:.3f} | "
            f"{raw_metrics['mean_precision']:.3f} | {raw_metrics['mean_recall']:.3f} | "
            f"{raw_metrics['mean_boundary_f1']:.3f} | {raw_metrics['false_merges']} | "
            f"{raw_metrics['false_splits']} | {raw_metrics['elapsed_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--output", type=Path, help="write the full JSON report to this path")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = parser.parse_args()
    report = run_benchmark(args.strategies)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json else _markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

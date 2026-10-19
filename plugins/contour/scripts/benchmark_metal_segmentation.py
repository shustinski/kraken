"""Deterministic topology-sensitive benchmark for classical metal segmentation.

The synthetic scenes intentionally contain blurred edges, shot/read noise,
illumination drift, bright rims, dark conductor interiors, and elevated narrow
gaps.  They are not intended to replace CIF golden tests; they isolate failure
classes whose ground-truth object topology is known exactly.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PLUGIN_ROOT, PLUGIN_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings  # noqa: E402
from contour.domain import PolygonData  # noqa: E402
from contour.serializers import load_polygons_cif  # noqa: E402
from contour.ui.metal_presets import standard_metal_preset_payload  # noqa: E402
from contour.vision.metal_recovery import (  # noqa: E402
    detect_metalization,
    metal_recovery_config_from_settings,
)
from contour.vision.metal_recovery.detector import _segmentation_config_from_recovery  # noqa: E402
from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
    gradient_watershed_mask_from_seeds,
    intensity_class_limits,
    selective_conductor_recovery,
)
from contour.vision.metal_recovery.pipeline_stages import (  # noqa: E402
    IMPLEMENTED_NEW_STRATEGIES,
    _segment,
    build_metal_segmentation_mask_staged,
)
from contour.vision.preprocessing import (  # noqa: E402
    metal_preprocess_config_from_settings,
    preprocess_for_sem,
)

from scripts.benchmark_metal_segmentation_parts.module_state import (
    REAL_DATASET_ROOT,
    SOURCE_ROOT,
    STRATEGIES,
    EVALUATION_BORDER_CROP_PX,
    BORDER_AUDIT_FRAMES,
)
from scripts.benchmark_metal_segmentation_parts.render_sem import (
    BenchmarkCase,
    SegmentationMetrics,
    SeedDiagnostics,
    _draw_line,
    _draw_rect,
    _render_sem,
)
from scripts.benchmark_metal_segmentation_parts.build_benchmark_cases import build_benchmark_cases
from scripts.benchmark_metal_segmentation_parts.rasterize_polygon_labels import (
    _preprocess_real_sem,
    _rasterize_polygon_labels,
    _rasterize_reference_labels,
    build_real_benchmark_cases,
    _boundary,
    evaluation_crop_slices,
    crop_evaluation_region,
    relabel_connected_components,
    prepare_evaluation_masks,
    _component_count,
    _component_records,
    components_removed_by_crop,
    labels_for_component_audit,
    _boundary_f1,
)
from scripts.benchmark_metal_segmentation_parts.topology_errors import _topology_errors
from scripts.benchmark_metal_segmentation_parts.measure_segmentation import measure_segmentation
from scripts.benchmark_metal_segmentation_parts.seed_diagnostics import seed_diagnostics
from scripts.benchmark_metal_segmentation_parts.ui_recovery_config import (
    resolution_probe,
    _write_diagnostic_overlay,
    _rasterize_detected_polygons,
    _ui_recovery_config,
)
from scripts.benchmark_metal_segmentation_parts.aggregate_strategy_metrics import _aggregate_strategy_metrics
from scripts.benchmark_metal_segmentation_parts.evaluation_record import _evaluation_record
from scripts.benchmark_metal_segmentation_parts.run_benchmark import run_benchmark
from scripts.benchmark_metal_segmentation_parts.write_csv import (
    _write_csv,
    _format_metric,
)
from scripts.benchmark_metal_segmentation_parts.markdown import _markdown
from scripts.benchmark_metal_segmentation_parts.default_evaluation_border_crop import _default_evaluation_border_crop_px
from scripts.benchmark_metal_segmentation_parts.main import main


if __name__ == "__main__":
    raise SystemExit(main())

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    STRATEGIES,
    EVALUATION_BORDER_CROP_PX,
    BORDER_AUDIT_FRAMES,
    BenchmarkCase,
    SegmentationMetrics,
    SeedDiagnostics,
    _draw_line,
    _draw_rect,
    _render_sem,
    build_benchmark_cases,
    _preprocess_real_sem,
    _rasterize_polygon_labels,
    _rasterize_reference_labels,
    build_real_benchmark_cases,
    _boundary,
    evaluation_crop_slices,
    crop_evaluation_region,
    relabel_connected_components,
    prepare_evaluation_masks,
    _component_count,
    _component_records,
    components_removed_by_crop,
    labels_for_component_audit,
    _boundary_f1,
    _topology_errors,
    measure_segmentation,
    seed_diagnostics,
    resolution_probe,
    _write_diagnostic_overlay,
    _rasterize_detected_polygons,
    _ui_recovery_config,
    _aggregate_strategy_metrics,
    _evaluation_record,
    run_benchmark,
    _write_csv,
    _format_metric,
    _markdown,
    _default_evaluation_border_crop_px,
    main,
)

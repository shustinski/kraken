"""The new cell-error result in the shape the matrix, the frame window and the old export read.

Each error region becomes one ``GridCellAnalysisResult`` with its outline (the object, not a
box) and the old reason code of its class; conductor zones become ``conductor_zone`` records.
Detailed reasons go into the feature snapshot as ``reason:<code>`` flags. Touching the frame
border is a flag (``touches_frame_border``), not a defect.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from . import error_classes as ec
from .cell_error_analysis import ErrorAnalysisResult, ErrorRegion
from .grid_anomaly import (
    CONTOUR_AREA_FEATURE,
    GridCellAnalysisResult,
    GridDamageSeverityThresholds,
    GridFrameAnalysisResult,
    _damage_score,
)

# Old reason code of each class. SPLIT and UNKNOWN are new codes.
CLASS_REASONS: dict[int, str] = {
    ec.BAD_GEOMETRY: "broken_geometry",
    ec.DEBRIS: "small_artifact",
    ec.MERGE: "merged_contour",
    ec.SPLIT: "split_cell",
    ec.UNKNOWN: "unknown_anomaly",
    ec.IGNORE: "conductor_zone",
}
MAX_OUTLINE_POINTS = 96


def _outline(region: ErrorRegion) -> tuple[tuple[int, int], ...]:
    piece = region.mask().astype(np.uint8)
    found = cv2.findContours(np.pad(piece, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        return ()
    points = max(contours, key=cv2.contourArea).reshape(-1, 2) - 1 + np.array(region.bbox[:2])
    step = max(1, len(points) // MAX_OUTLINE_POINTS)
    return tuple((int(x), int(y)) for x, y in points[::step])


def _cell(region: ErrorRegion, row: int) -> GridCellAnalysisResult:
    features = dict(region.feature_snapshot)
    features[CONTOUR_AREA_FEATURE] = float(region.pixel_area)
    for reason in region.reasons:
        features[f"reason:{reason}"] = 1.0
    is_zone = region.error_class == ec.IGNORE
    is_normal = region.error_class == ec.OK
    return GridCellAnalysisResult(
        row=int(row),
        col=0,
        bbox=tuple(int(value) for value in region.bbox),
        centroid=(float(region.centroid[0]), float(region.centroid[1])),
        contour_id=int(region.instance_id or row + 1),
        status="zone" if is_zone else ("normal" if is_normal else "broken"),
        score=0.0 if is_zone or is_normal else float(region.score),
        reasons=() if is_normal else (CLASS_REASONS[region.error_class],),
        mean_confidence=features.get("confidence_uncertainty_mean"),
        uncertain_pixel_ratio=features.get("confidence_uncertain_share"),
        border_uncertainty=features.get("confidence_border_uncertainty"),
        feature_snapshot=tuple(sorted(features.items())),
        outline=_outline(region),
    )


def to_grid_frame_result(
    result: ErrorAnalysisResult, *, severity: GridDamageSeverityThresholds | None = None
) -> GridFrameAnalysisResult:
    cells = [_cell(region, index) for index, region in enumerate(result.regions)]
    normals = [_cell(region, len(cells) + index) for index, region in enumerate(result.normal_regions)]
    zones = [_cell(region, len(cells) + len(normals) + index) for index, region in enumerate(result.ignore_regions)]
    normal = len(normals)
    total = normal + len(cells)
    damage = _damage_score(cells + normals, max(1, total)) if cells else 0.0
    height, width = result.input_shape
    ready = result.normal_model_status == "ok"
    return GridFrameAnalysisResult(
        frame_id=result.frame_id,
        frame_path=result.frame_path,
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=int(total),
        detected_cells=int(total),
        normal_cells=normal,
        suspicious_cells=0,
        broken_cells=len(cells),
        missing_cells=0,
        artifact_cells=0,
        damage_score=float(damage),
        severity_level=(severity or GridDamageSeverityThresholds()).level_for_score(damage) if ready else result.normal_model_status,
        grid_detected=ready,
        per_cell_results=tuple(cells + normals + zones),
        component_count=int(result.diagnostics.get("components", 0)),
        cell_width=int(round(float(result.diagnostics.get("cell_width", 0.0)))),
        cell_height=int(round(float(result.diagnostics.get("cell_height", 0.0)))),
        model_file_status="" if ready else result.normal_model_status,
    )

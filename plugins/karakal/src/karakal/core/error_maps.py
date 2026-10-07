"""Machine maps of a cell-error analysis, their lossless files, a preview and the manifest.

Maps (all the size of the input mask):

- ``class_map`` uint8: the class of every pixel of a suspicious object (whole objects, not
  boxes), 0 elsewhere, 255 for IGNORE;
- ``score_map`` float32 in [0, 1]; on disk as uint16 PNG (value * 65535);
- ``instance_map`` uint16 (uint32 when more objects): the region id of each object pixel.

Machine maps are written only as PNG/TIFF/NPZ, never as JPEG. The RGB preview is for
people and follows the class palette; the class number is what downstream code reads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from . import error_classes as ec
from .cell_error_analysis import RESULT_SCHEMA_VERSION, ErrorAnalysisResult, ErrorRegion

LOSSLESS_FORMATS = ("png", "tif", "tiff", "npz")


def _paint(target: np.ndarray, regions: Iterable[ErrorRegion], value_of) -> None:
    rows, cols = target.shape[:2]
    for region in regions:
        x, y, w, h = region.bbox
        piece = region.mask()
        x1, y1 = min(cols, x + w), min(rows, y + h)
        piece = piece[: y1 - y, : x1 - x]
        view = target[y:y1, x:x1]
        view[piece] = value_of(region)


def _priority(region: ErrorRegion) -> int:
    order = ec.CLASS_PRIORITY
    return len(order) - order.index(region.error_class) if region.error_class in order else 0


def render_class_map(result: ErrorAnalysisResult) -> np.ndarray:
    class_map = np.zeros(result.input_shape, dtype=np.uint8)
    # Lower priority first, so the class the order prefers is painted last.
    _paint(class_map, sorted(result.regions, key=_priority), lambda region: region.error_class)
    _paint(class_map, result.ignore_regions, lambda _region: ec.IGNORE)
    return class_map


def render_score_map(result: ErrorAnalysisResult) -> np.ndarray:
    score_map = np.zeros(result.input_shape, dtype=np.float32)
    _paint(score_map, sorted(result.regions, key=lambda region: region.score), lambda region: np.float32(region.score))
    return score_map


def render_instance_map(result: ErrorAnalysisResult) -> np.ndarray:
    dtype = np.uint16 if len(result.regions) <= np.iinfo(np.uint16).max else np.uint32
    instance_map = np.zeros(result.input_shape, dtype=dtype)
    _paint(instance_map, result.regions, lambda region: region.instance_id)
    return instance_map


def _hex_bgr(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[4:6], 16), int(text[2:4], 16), int(text[0:2], 16)


def render_preview(result: ErrorAnalysisResult, mask: np.ndarray | None = None) -> np.ndarray:
    """BGR picture for people: the mask in gray, objects in their class colors."""

    class_map = render_class_map(result)
    base = np.zeros(result.input_shape + (3,), dtype=np.uint8)
    if mask is not None:
        from .mask_normalization import binarize_mask

        base[binarize_mask(mask)] = (90, 90, 90)
    for value, color in ec.ERROR_CLASS_COLORS.items():
        if value == ec.OK:
            continue
        selected = class_map == value
        if value == ec.IGNORE:
            base[selected] = (base[selected] * 0.5 + np.array(_hex_bgr(color)) * 0.5).astype(np.uint8)
        else:
            base[selected] = _hex_bgr(color)
    return base


def _write(path: Path, image: np.ndarray) -> Path:
    extension = path.suffix.lower().lstrip(".")
    if extension not in LOSSLESS_FORMATS or extension == "npz":
        raise ValueError(f"machine maps are written lossless (png/tif), not {extension or 'no extension'}")
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return path


def manifest(result: ErrorAnalysisResult, *, analysis_parameters: dict[str, Any] | None = None,
             calibration_fingerprint: str = "") -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "algorithm_version": result.algorithm_version,
        "frame_id": result.frame_id,
        "frame_path": result.frame_path,
        "mode": result.mode,
        "inputs": dict(result.inputs),
        "normal_model_id": result.normal_model_id,
        "normal_model_status": result.normal_model_status,
        "calibration_fingerprint": str(calibration_fingerprint or ""),
        "input_shape": list(result.input_shape),
        "binarization": dict(result.binarization),
        "classes": {str(value): name for value, name in ec.ERROR_CLASS_NAMES.items()},
        "palette": {str(value): color for value, color in ec.ERROR_CLASS_COLORS.items()},
        "maps": {
            "class_map": {"dtype": "uint8", "ignore": ec.IGNORE},
            "score_map": {"dtype": "uint16", "scale": 65535, "range": [0.0, 1.0]},
            "instance_map": {"dtype": "uint16"},
        },
        "parameters": dict(analysis_parameters or {}),
        "regions": [
            {
                "instance_id": region.instance_id,
                "class": region.class_name,
                "score": round(float(region.score), 6),
                "bbox": list(region.bbox),
                "centroid": [round(float(value), 2) for value in region.centroid],
                "pixel_area": region.pixel_area,
                "component_ids": list(region.component_ids),
                "reasons": list(region.reasons),
                "evidence_sources": list(region.evidence_sources),
                "touches_border": bool(region.touches_border),
                "features": {key: round(float(value), 6) for key, value in region.feature_snapshot},
            }
            for region in result.regions
        ],
        "ignore_regions": [
            {"instance_id": region.instance_id, "bbox": list(region.bbox), "pixel_area": region.pixel_area}
            for region in result.ignore_regions
        ],
        "diagnostics": dict(result.diagnostics),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def export_maps(
    result: ErrorAnalysisResult,
    folder: Path | str,
    stem: str,
    *,
    mask: np.ndarray | None = None,
    analysis_parameters: dict[str, Any] | None = None,
    calibration_fingerprint: str = "",
) -> dict[str, Path]:
    """Write class/score/instance maps (PNG), the preview (PNG) and the manifest (JSON)."""

    root = Path(folder)
    written = {
        "class_map": _write(root / f"{stem}_class.png", render_class_map(result)),
        "score_map": _write(
            root / f"{stem}_score.png", np.round(np.clip(render_score_map(result), 0.0, 1.0) * 65535.0).astype(np.uint16)
        ),
        "instance_map": _write(root / f"{stem}_instance.png", render_instance_map(result)),
        "preview": _write(root / f"{stem}_preview.png", render_preview(result, mask)),
    }
    target = root / f"{stem}_manifest.json"
    target.write_text(
        json.dumps(manifest(result, analysis_parameters=analysis_parameters, calibration_fingerprint=calibration_fingerprint), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    written["manifest"] = target
    return written

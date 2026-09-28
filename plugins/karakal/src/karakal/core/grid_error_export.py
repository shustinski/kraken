"""Export grid-cell defects as filled rectangles on a black JPEG."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .exports import _export_worker_count, _grid_defect_reason_rgb
from .grid_anomaly import _DEFECT_REASON_PRIORITY, GridFrameAnalysisResult


_EXCLUDED_EXPORT_TYPES = {"conductor_zone", "normal"}


@dataclass(slots=True)
class GridErrorExportFrame:
    file_name: str
    mask_name: str
    layer_name: str
    result: GridFrameAnalysisResult | None


@dataclass(slots=True)
class GridErrorExportReport:
    output_dir: Path
    written: tuple[Path, ...] = ()
    skipped_empty: tuple[str, ...] = ()
    missing_results: tuple[str, ...] = ()
    cancelled: bool = False


def export_type_slug(selected_types: tuple[str, ...] | list[str], *, all_types: bool) -> str:
    if all_types:
        return "all"
    names = [str(item) for item in selected_types if str(item)]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return "types"


def render_grid_error_image(
    result: GridFrameAnalysisResult,
    selected_types: tuple[str, ...] | list[str] | set[str],
    *,
    color_mode: str = "color",
) -> np.ndarray:
    """Filled cell rectangles on black. Color follows only the selected types."""

    selected = {str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES}
    height = max(1, int(getattr(result, "image_height", 0) or 1))
    width = max(1, int(getattr(result, "image_width", 0) or 1))
    grayscale = str(color_mode).lower() in {"bw", "gray", "grayscale"}
    image = np.zeros((height, width), dtype=np.uint8) if grayscale else np.zeros((height, width, 3), dtype=np.uint8)
    cells = []
    for cell in getattr(result, "per_cell_results", ()) or ():
        if str(getattr(cell, "status", "") or "") == "normal":
            continue
        reasons = tuple(
            str(reason)
            for reason in (getattr(cell, "reasons", ()) or ())
            if str(reason) and str(reason) not in _EXCLUDED_EXPORT_TYPES
        )
        if not any(reason in selected for reason in reasons):
            continue
        rank = max((_DEFECT_REASON_PRIORITY.get(reason, 10) for reason in reasons if reason in selected), default=0)
        cells.append((rank, cell, reasons))
    cells.sort(key=lambda item: item[0])
    for _rank, cell, reasons in cells:
        x = max(0, min(width - 1, int(getattr(cell, "left", 0))))
        y = max(0, min(height - 1, int(getattr(cell, "top", 0))))
        w = max(1, int(getattr(cell, "width", 1)))
        h = max(1, int(getattr(cell, "height", 1)))
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 <= x or y1 <= y:
            continue
        if grayscale:
            image[y:y1, x:x1] = 255
            continue
        red, green, blue = _grid_defect_reason_rgb(reasons, selected_types=selected)
        image[y:y1, x:x1] = (blue, green, red)
    return image


def frame_has_selected_errors(
    result: GridFrameAnalysisResult,
    selected_types: tuple[str, ...] | list[str] | set[str],
) -> bool:
    selected = {str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES}
    for cell in getattr(result, "per_cell_results", ()) or ():
        if str(getattr(cell, "status", "") or "") == "normal":
            continue
        reasons = {str(reason) for reason in (getattr(cell, "reasons", ()) or ()) if str(reason)}
        if reasons & selected:
            return True
    return False


def reason_counts(
    result: GridFrameAnalysisResult,
    selected_types: tuple[str, ...] | list[str] | set[str],
) -> dict[str, int]:
    selected = [str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES]
    counts = {item: 0 for item in selected}
    for cell in getattr(result, "per_cell_results", ()) or ():
        if str(getattr(cell, "status", "") or "") == "normal":
            continue
        reasons = {str(reason) for reason in (getattr(cell, "reasons", ()) or ()) if str(reason)}
        for reason in selected:
            if reason in reasons:
                counts[reason] += 1
    return counts


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        100,
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
    ]
    ok, encoded = cv2.imencode(".jpg", image, params)
    if not ok:
        raise OSError(f"failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def export_grid_cell_defect_frames(
    frames: tuple[GridErrorExportFrame, ...] | list[GridErrorExportFrame],
    output_dir: Path,
    selected_types: tuple[str, ...] | list[str],
    *,
    color_mode: str = "color",
    skip_empty: bool = False,
    all_types: bool = False,
    algorithm_version: str = "",
    calibration_fingerprint: str = "",
    type_colors: dict[str, str] | None = None,
    progress=None,
    cancelled=None,
) -> GridErrorExportReport:
    """Write one JPEG per frame. ``cancelled`` is a zero-arg callable."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES)
    written: list[Path] = []
    skipped_empty: list[str] = []
    missing: list[str] = []
    summary_rows: list[dict[str, object]] = []
    pending_writes: list[tuple[Path, np.ndarray]] = []
    total = max(1, len(frames))
    for index, frame in enumerate(frames, start=1):
        if cancelled is not None and cancelled():
            return GridErrorExportReport(
                output_dir=output_dir,
                written=tuple(written),
                skipped_empty=tuple(skipped_empty),
                missing_results=tuple(missing),
                cancelled=True,
            )
        label = f"{frame.mask_name}/{frame.layer_name}/{frame.file_name}"
        if progress is not None:
            progress(index - 1, total, label)
        if frame.result is None:
            missing.append(label)
            summary_rows.append(
                {
                    "frame": frame.file_name,
                    "mask": frame.mask_name,
                    "layer": frame.layer_name,
                    "status": "no_result",
                    **{item: 0 for item in selected},
                }
            )
            continue
        counts = reason_counts(frame.result, selected)
        has_errors = any(counts.values())
        if skip_empty and not has_errors:
            skipped_empty.append(label)
            summary_rows.append(
                {
                    "frame": frame.file_name,
                    "mask": frame.mask_name,
                    "layer": frame.layer_name,
                    "status": "skipped_empty",
                    **counts,
                }
            )
            continue
        image = render_grid_error_image(frame.result, selected, color_mode=color_mode)
        destination = output_dir / frame.mask_name / frame.layer_name / frame.file_name
        pending_writes.append((destination, image))
        written.append(destination)
        summary_rows.append(
            {
                "frame": frame.file_name,
                "mask": frame.mask_name,
                "layer": frame.layer_name,
                "status": "written",
                **counts,
            }
        )
    worker_count = _export_worker_count(None, len(pending_writes))
    if pending_writes:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            list(pool.map(lambda item: _write_jpeg(item[0], item[1]), pending_writes))
    if progress is not None:
        progress(total, total, "")
    colors = dict(type_colors or {})
    legend = {
        "types": list(selected),
        "colors": {item: colors.get(item, "") for item in selected},
        "color_mode": "bw" if str(color_mode).lower() in {"bw", "gray", "grayscale"} else "color",
        "algorithm_version": str(algorithm_version or ""),
        "calibration_fingerprint": str(calibration_fingerprint or ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_types": bool(all_types),
    }
    (output_dir / "legend.json").write_text(json.dumps(legend, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = ["frame", "mask", "layer", "status", *selected]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    report_lines = ["written:"]
    report_lines.extend(str(path) for path in written)
    report_lines.append("missing:")
    report_lines.extend(missing)
    report_lines.append("skipped_empty:")
    report_lines.extend(skipped_empty)
    (output_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return GridErrorExportReport(
        output_dir=output_dir,
        written=tuple(written),
        skipped_empty=tuple(skipped_empty),
        missing_results=tuple(missing),
    )


@dataclass(slots=True)
class GridErrorExportChoices:
    selected_types: tuple[str, ...]
    all_types: bool
    color_mode: str
    layers: tuple[str, ...]
    masks: tuple[str, ...]
    frame_scope: str
    skip_empty: bool
    output_dir: Path
    notes: list[str] = field(default_factory=list)

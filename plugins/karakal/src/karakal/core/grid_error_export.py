"""Export grid-cell defects as filled rectangles on a black image."""
from __future__ import annotations

import csv
import json
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .exports import _export_worker_count, _grid_defect_reason_rgb
from .grid_anomaly import _DEFECT_REASON_PRIORITY, GridFrameAnalysisResult


_EXCLUDED_EXPORT_TYPES = {"conductor_zone", "normal"}
_SAFE_FOLDER_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


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
    run_dir: Path | None = None


def export_type_slug(selected_types: tuple[str, ...] | list[str], *, all_types: bool) -> str:
    if all_types:
        return "all"
    names = [str(item) for item in selected_types if str(item)]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return "types"


def safe_export_folder_name(title: str, *, fallback: str = "layer") -> str:
    """Keep Cyrillic and spaces; strip only characters Windows rejects."""

    text = _SAFE_FOLDER_RE.sub("_", str(title or "").strip())
    text = text.strip(" .")
    return text or fallback


def color_mode_folder_name(
    image_format: str,
    color_mode: str,
    *,
    single_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    fmt = str(image_format or "jpg").upper()
    if str(color_mode).lower() in {"filter", "color"}:
        return f"{fmt}_filter-colors"
    red, green, blue = (int(value) for value in single_color[:3])
    if (red, green, blue) == (255, 255, 255):
        return f"{fmt}_single-white"
    return f"{fmt}_single-#{red:02x}{green:02x}{blue:02x}"


def unique_run_dir(output_dir: Path, *, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M")
    base = Path(output_dir) / f"Karakal_export_{stamp}"
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = Path(output_dir) / f"Karakal_export_{stamp}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def render_grid_error_image(
    result: GridFrameAnalysisResult,
    selected_types: tuple[str, ...] | list[str] | set[str],
    *,
    color_mode: str = "color",
    single_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Filled cell rectangles on black. Color follows only the selected types."""

    selected = {str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES}
    height = max(1, int(getattr(result, "image_height", 0) or 1))
    width = max(1, int(getattr(result, "image_width", 0) or 1))
    mode = str(color_mode).lower()
    grayscale = mode in {"bw", "gray", "grayscale"} or (
        mode == "single" and (single_color is None or tuple(int(v) for v in single_color[:3]) == (255, 255, 255))
    )
    image = np.zeros((height, width), dtype=np.uint8) if grayscale else np.zeros((height, width, 3), dtype=np.uint8)
    paint_bgr = None
    if mode == "single" and not grayscale and single_color is not None:
        red, green, blue = (int(value) for value in single_color[:3])
        paint_bgr = (blue, green, red)
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
        if paint_bgr is not None:
            image[y:y1, x:x1] = paint_bgr
            continue
        red, green, blue = _grid_defect_reason_rgb(reasons, selected_types=selected)
        image[y:y1, x:x1] = (blue, green, red)
    return image


def _write_image(path: Path, image: np.ndarray, image_format: str) -> None:
    fmt = str(image_format or "jpg").strip().lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"jpg", "png", "bmp", "tif", "tiff"}:
        fmt = "jpg"
    extension = ".tiff" if fmt in {"tif", "tiff"} else f".{fmt}"
    # Always match the requested format — source stems often end in .jpg.
    path = path.with_suffix(extension)
    params: list[int] = []
    if fmt == "jpg":
        params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            100,
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
        ]
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise OSError(f"failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    _write_image(path, image, "jpg")


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
    image_format: str = "jpg",
    single_color: tuple[int, int, int] | None = None,
    progress=None,
    cancelled=None,
) -> GridErrorExportReport:
    """Write one image per frame. ``cancelled`` is a zero-arg callable.

    Legacy callers pass ``color_mode="bw"|"color"``. The wizard uses
    ``color_mode="single"|"filter"`` with an optional ``single_color``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES)
    render_mode = str(color_mode)
    paint_color = single_color
    if render_mode.lower() in {"bw", "gray", "grayscale"}:
        render_mode = "single"
        paint_color = (255, 255, 255)
    elif render_mode.lower() == "color":
        render_mode = "filter"
    written: list[Path] = []
    skipped_empty: list[str] = []
    missing: list[str] = []
    summary_rows: list[dict[str, object]] = []
    total = max(1, len(frames))
    # Each frame is written right after it is drawn. Holding every drawn frame
    # until the end cost a full-size image per frame (tens of GB on big runs).
    worker_count = _export_worker_count(None, len(frames))
    in_flight: deque[Future] = deque()
    pool = ThreadPoolExecutor(max_workers=worker_count)
    try:
        for index, frame in enumerate(frames, start=1):
            if cancelled is not None and cancelled():
                while in_flight:
                    in_flight.popleft().result()
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
            image = render_grid_error_image(
                frame.result,
                selected,
                color_mode=render_mode,
                single_color=paint_color,
            )
            fmt = str(image_format or "jpg").strip().lower().lstrip(".")
            if fmt == "jpeg":
                fmt = "jpg"
            if fmt not in {"jpg", "png", "bmp", "tif", "tiff"}:
                fmt = "jpg"
            extension = ".tiff" if fmt in {"tif", "tiff"} else f".{fmt}"
            destination = (output_dir / frame.mask_name / frame.layer_name / frame.file_name).with_suffix(extension)
            in_flight.append(pool.submit(_write_image, destination, image, image_format))
            written.append(destination)
            del image
            while len(in_flight) > 2 * worker_count:
                in_flight.popleft().result()
            summary_rows.append(
                {
                    "frame": frame.file_name,
                    "mask": frame.mask_name,
                    "layer": frame.layer_name,
                    "status": "written",
                    **counts,
                }
            )
        while in_flight:
            in_flight.popleft().result()
    finally:
        pool.shutdown(wait=True)
    if progress is not None:
        progress(total, total, "")
    colors = dict(type_colors or {})
    legend = {
        "types": list(selected),
        "colors": {item: colors.get(item, "") for item in selected},
        "color_mode": render_mode,
        "image_format": str(image_format or "jpg"),
        "single_color": list(paint_color) if paint_color is not None else None,
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


@dataclass(slots=True)
class GridExportLayerSpec:
    key: str
    title: str
    folder_name: str
    source_path: str = ""
    analysis_layer: str = "binary"


def export_grid_error_wizard_run(
    frames: tuple[GridErrorExportFrame, ...] | list[GridErrorExportFrame],
    output_dir: Path,
    selected_types: tuple[str, ...] | list[str],
    layers: tuple[GridExportLayerSpec, ...] | list[GridExportLayerSpec],
    *,
    image_format: str = "jpg",
    color_mode: str = "single",
    single_color: tuple[int, int, int] = (255, 255, 255),
    skip_empty: bool = False,
    algorithm_version: str = "",
    calibration_fingerprint: str = "",
    type_colors: dict[str, str] | None = None,
    progress=None,
    cancelled=None,
) -> GridErrorExportReport:
    """Write a dated run folder: format+color / model / frames."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = unique_run_dir(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    mode_folder = color_mode_folder_name(image_format, color_mode, single_color=single_color)
    selected = tuple(str(item) for item in selected_types if str(item) and str(item) not in _EXCLUDED_EXPORT_TYPES)
    used_folder_names: set[str] = set()
    folder_for_key: dict[str, str] = {}
    for layer in layers:
        name = safe_export_folder_name(layer.folder_name or layer.title, fallback=layer.key or "layer")
        candidate = name
        index = 2
        while candidate.lower() in used_folder_names:
            candidate = f"{name}_{index}"
            index += 1
        used_folder_names.add(candidate.lower())
        folder_for_key[str(layer.key)] = candidate

    flat_frames = [
        GridErrorExportFrame(
            file_name=frame.file_name,
            mask_name=mode_folder,
            layer_name=folder_for_key.get(str(frame.mask_name), safe_export_folder_name(frame.mask_name)),
            result=frame.result,
        )
        for frame in frames
    ]
    report = export_grid_cell_defect_frames(
        flat_frames,
        run_dir,
        selected,
        color_mode=color_mode,
        single_color=single_color,
        image_format=image_format,
        skip_empty=skip_empty,
        all_types=False,
        algorithm_version=algorithm_version,
        calibration_fingerprint=calibration_fingerprint,
        type_colors=type_colors,
        progress=progress,
        cancelled=cancelled,
    )
    legend_path = run_dir / "legend.json"
    legend: dict[str, object] = {}
    if legend_path.is_file():
        try:
            legend = json.loads(legend_path.read_text(encoding="utf-8"))
        except (TypeError, ValueError, OSError):
            legend = {}
    legend.update(
        {
            "run_dir": str(run_dir),
            "image_format": str(image_format),
            "color_mode": str(color_mode),
            "single_color": list(single_color),
            "layers": [
                {
                    "key": layer.key,
                    "title": layer.title,
                    "folder": folder_for_key.get(layer.key, layer.folder_name),
                    "source_path": layer.source_path,
                    "analysis_layer": layer.analysis_layer,
                }
                for layer in layers
            ],
            "folder_naming": "cyrillic_kept",
        }
    )
    legend_path.write_text(json.dumps(legend, ensure_ascii=False, indent=2), encoding="utf-8")
    return GridErrorExportReport(
        output_dir=report.output_dir,
        written=report.written,
        skipped_empty=report.skipped_empty,
        missing_results=report.missing_results,
        cancelled=report.cancelled,
        run_dir=run_dir,
    )

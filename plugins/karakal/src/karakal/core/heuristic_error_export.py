"""Error maps of the heuristic check: its findings as filled rectangles on black, one image per frame.

Colours by group; where findings overlap the more serious group is painted on top: gross defects
over topology over dirt over edge shape. Beside the maps, ``findings.csv`` lists every finding
with its box in frame pixels, and every frame without findings or not rated with its verdict.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .domain import BuildResult, FrameRecord
from .grid_error_export import _write_image
from .heuristic_check import (
    CONTAMINATION_REASONS,
    GEOMETRY_REASONS,
    GROSS_REASONS,
    TOPOLOGY_REASONS,
    VERDICT_UNREADABLE,
)

ERROR_MAPS_FOLDER = "Ошибки"
FINDINGS_FILE = "findings.csv"
# Painted from the bottom up: a later group covers an earlier one.
GROUPS: tuple[tuple[str, tuple[str, ...], tuple[int, int, int]], ...] = (
    ("geometry", GEOMETRY_REASONS, (0, 200, 255)),
    ("dirt", CONTAMINATION_REASONS, (255, 165, 0)),
    ("topology", TOPOLOGY_REASONS, (255, 40, 40)),
    ("gross", GROSS_REASONS, (255, 0, 255)),
)
_GROUP_OF = {code: name for name, codes, _rgb in GROUPS for code in codes}
_RANK = {name: index for index, (name, _codes, _rgb) in enumerate(GROUPS)}
_RGB = {name: rgb for name, _codes, rgb in GROUPS}


@dataclass(frozen=True, slots=True)
class HeuristicExportReport:
    output_dir: Path
    written: tuple[Path, ...]
    without_findings: tuple[str, ...]
    unrated: tuple[str, ...]
    not_computed: tuple[str, ...]
    findings_csv: Path | None
    cancelled: bool = False


def _extension(image_format: str) -> str:
    fmt = str(image_format or "png").strip().lower().lstrip(".")
    fmt = "jpg" if fmt == "jpeg" else fmt
    if fmt not in {"jpg", "png", "bmp", "tif", "tiff"}:
        fmt = "jpg"
    return ".tiff" if fmt in {"tif", "tiff"} else f".{fmt}"


def finding_group(code: str) -> str | None:
    """Group of a reason code on the error map; None for reasons without a place (run, capture)."""

    return _GROUP_OF.get(str(code))


def _frame_size(record: FrameRecord, check) -> tuple[int, int]:
    values = dict(getattr(check, "feature_values", {}) or {})
    height, width = int(values.get("frame_height", 0) or 0), int(values.get("frame_width", 0) or 0)
    if height > 0 and width > 0:
        return height, width
    # Results computed before the size was kept: read it from the photo.
    path = next(iter((record.model_mask_paths or {}).values()), "") or record.original_path or ""
    if path and cv2 is not None:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return int(image.shape[0]), int(image.shape[1])
    return 0, 0


def _findings(check) -> tuple:
    """Every finding with a place: the full regions when the result has them, else its reasons."""

    regions = tuple(getattr(check, "regions", ()) or ())
    source = regions or tuple(getattr(check, "reasons", ()) or ())
    return tuple(
        reason
        for reason in source
        if finding_group(getattr(reason, "code", "")) is not None and getattr(reason, "bbox", None) is not None
    )


def _pixel_box(bbox, height: int, width: int) -> tuple[int, int, int, int]:
    x, y, w, h = (float(value) for value in bbox)
    x0 = max(0, min(width - 1, int(round(x * width))))
    y0 = max(0, min(height - 1, int(round(y * height))))
    x1 = max(x0 + 1, min(width, int(round((x + w) * width))))
    y1 = max(y0 + 1, min(height, int(round((y + h) * height))))
    return x0, y0, x1, y1


def render_heuristic_error_map(check, height: int, width: int) -> np.ndarray:
    """BGR image of ``height`` x ``width``: findings filled in their group colours on black."""

    image = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    boxes = []
    for reason in _findings(check):
        group = str(finding_group(reason.code))
        boxes.append((_RANK[group], group, _pixel_box(reason.bbox, height, width)))
    for _rank, group, (x0, y0, x1, y1) in sorted(boxes, key=lambda item: item[0]):
        red, green, blue = _RGB[group]
        image[y0:y1, x0:x1] = (blue, green, red)
    return image


def export_heuristic_error_maps(
    build_result: BuildResult,
    output_dir: Path | str,
    *,
    image_format: str = "png",
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> HeuristicExportReport:
    """One error map per frame with findings into ``output_dir``/Ошибки, and findings.csv beside."""

    output_dir = Path(output_dir)
    maps_dir = output_dir / ERROR_MAPS_FOLDER
    maps_dir.mkdir(parents=True, exist_ok=True)
    records = tuple(build_result.records)
    written: list[Path] = []
    without_findings: list[str] = []
    unrated: list[str] = []
    not_computed: list[str] = []
    rows: list[list[object]] = []
    total = max(1, len(records))
    for index, record in enumerate(records, start=1):
        if cancelled is not None and cancelled():
            return HeuristicExportReport(
                output_dir, tuple(written), tuple(without_findings), tuple(unrated), tuple(not_computed), None, True
            )
        key = str(record.key)
        if progress is not None:
            progress(index - 1, total, key)
        check = None if record.summary is None else getattr(record.summary, "heuristic_check", None)
        if check is None:
            not_computed.append(key)
            continue
        verdict = str(getattr(check, "verdict", ""))
        risk = getattr(check, "total_risk", None)
        risk_text = "" if risk is None else f"{float(risk):.1f}"
        if verdict == VERDICT_UNREADABLE:
            unrated.append(key)
            rows.append([key, verdict, risk_text, ",".join(getattr(check, "capture_issues", ()) or ()), "", "", "", "", "", ""])
            continue
        height, width = _frame_size(record, check)
        placed = list(_findings(check))
        if not placed or height <= 0 or width <= 0:
            without_findings.append(key)
            rows.append([key, verdict, risk_text, "", "", "", "", "", "", ""])
            continue
        for reason in placed:
            x0, y0, x1, y1 = _pixel_box(reason.bbox, height, width)
            rows.append(
                [
                    key,
                    verdict,
                    risk_text,
                    reason.code,
                    finding_group(reason.code),
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    f"{float(reason.contribution):.2f}",
                ]
            )
        target = maps_dir / Path(key).stem
        _write_image(target, render_heuristic_error_map(check, height, width), image_format)
        written.append(target.with_suffix(_extension(image_format)))
    if progress is not None:
        progress(total, total, "")
    findings_csv = output_dir / FINDINGS_FILE
    with findings_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "verdict", "risk", "code", "group", "x", "y", "width", "height", "strength"])
        writer.writerows(rows)
    return HeuristicExportReport(
        output_dir, tuple(written), tuple(without_findings), tuple(unrated), tuple(not_computed), findings_csv
    )


__all__ = [
    "ERROR_MAPS_FOLDER",
    "FINDINGS_FILE",
    "GROUPS",
    "HeuristicExportReport",
    "export_heuristic_error_maps",
    "finding_group",
    "render_heuristic_error_map",
]

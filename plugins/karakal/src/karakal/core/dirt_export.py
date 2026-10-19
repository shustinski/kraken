"""Files of the dirty regions of a run. Qt-free.

For every frame, in ``dirt/<layer>/``:

- ``<frame>.dirt.json``: the regions (box and outline in frame pixels, dirtiness, level, parts,
  edge flag, what made the dirt);
- ``<frame>.dirt.png``: a frame-sized image holding each pixel's region id (0 outside);
- ``<frame>.dirt_view.jpg``: a picture for people, the mask with the regions drawn on it.

``dirt/dirt_regions.csv`` lists every region of the run, one row each; its outline is one cell
of ``x:y`` points joined by ``;``.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .dirt_regions import (
    DirtConfig,
    FrameDirtResult,
    dirt_layers,
    read_gray,
    region_label_image,
    render_dirt_view,
)

DIRT_FOLDER = "dirt"
CSV_NAME = "dirt_regions.csv"
CSV_FIELDS = (
    "frame",
    "layer",
    "id",
    "level",
    "dirtiness",
    "x",
    "y",
    "w",
    "h",
    "parts",
    "area_px",
    "dirty_area_px",
    "edge",
    "contour",
)


@dataclass(slots=True)
class DirtExportFrame:
    file_name: str
    layer_folder: str
    result: FrameDirtResult | None


@dataclass(slots=True)
class DirtExportReport:
    output_dir: Path
    written: tuple[Path, ...] = ()
    skipped_empty: tuple[str, ...] = ()
    missing_results: tuple[str, ...] = ()
    regions: int = 0
    cancelled: bool = False


def region_payload(region) -> dict:
    x, y, w, h = region.bbox
    return {
        "id": int(region.id),
        "level": str(region.level),
        "dirtiness": round(float(region.dirtiness), 3),
        "peak": round(float(region.peak), 3),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "contour": [[int(px), int(py)] for px, py in region.contour],
        "parts": int(region.parts),
        "area_px": int(region.area_px),
        "dirty_area_px": int(region.dirty_area_px),
        "edge": bool(region.edge),
        "signals": {
            "blur_px": int(region.blur_px),
            "unsure_px": int(region.unsure_px),
            "nick_px": int(region.nick_px),
            "crumb_px": int(region.crumb_px),
        },
    }


def frame_payload(result: FrameDirtResult, *, frame: str = "", layer: str = "") -> dict:
    return {
        "frame": str(frame or result.frame_id),
        "layer": str(layer),
        "width": int(result.width),
        "height": int(result.height),
        "source": {
            "mask": str(result.mask_path),
            "confidence": str(result.confidence_path) if result.used_confidence else None,
        },
        "settings": dict(result.settings or {}),
        "regions": [region_payload(region) for region in result.regions],
    }


def csv_rows(result: FrameDirtResult, *, frame: str = "", layer: str = "") -> list[dict]:
    rows = []
    for region in result.regions:
        x, y, w, h = region.bbox
        rows.append(
            {
                "frame": str(frame or result.frame_id),
                "layer": str(layer),
                "id": int(region.id),
                "level": str(region.level),
                "dirtiness": round(float(region.dirtiness), 3),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "parts": int(region.parts),
                "area_px": int(region.area_px),
                "dirty_area_px": int(region.dirty_area_px),
                "edge": int(bool(region.edge)),
                "contour": ";".join(f"{int(px)}:{int(py)}" for px, py in region.contour),
            }
        )
    return rows


def _write_bytes(path: Path, extension: str, image: np.ndarray, params: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(extension, image, params or [])
    if not ok:
        raise OSError(f"cannot encode {path}")
    encoded.tofile(str(path))


def write_frame_dirt_files(
    result: FrameDirtResult,
    folder: Path,
    file_stem: str,
    *,
    layer: str = "",
    config: DirtConfig | None = None,
    write_view: bool = True,
) -> tuple[Path, ...]:
    """The JSON, the label image and (when the mask can be read) the picture of one frame."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{file_stem}.dirt.json"
    json_path.write_text(
        json.dumps(frame_payload(result, frame=file_stem, layer=layer), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    written = [json_path]
    if result.width > 0 and result.height > 0:
        label_path = folder / f"{file_stem}.dirt.png"
        _write_bytes(label_path, ".png", region_label_image(result))
        written.append(label_path)
    if write_view:
        mask = read_gray(result.mask_path)
        if mask is not None:
            confidence = read_gray(result.confidence_path) if result.used_confidence else None
            if confidence is not None and confidence.shape != mask.shape:
                confidence = None
            view = render_dirt_view(mask, result, dirt_layers(mask, confidence, config))
            view_path = folder / f"{file_stem}.dirt_view.jpg"
            _write_bytes(view_path, ".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            written.append(view_path)
    return tuple(written)


def export_dirt_frames(
    frames: tuple[DirtExportFrame, ...] | list[DirtExportFrame],
    output_dir: Path,
    *,
    skip_empty: bool = False,
    write_view: bool = True,
    config: DirtConfig | None = None,
    progress=None,
    cancelled=None,
    workers: int = 4,
) -> DirtExportReport:
    """Write ``dirt/`` under ``output_dir``. ``cancelled`` is a zero-arg callable."""

    root = Path(output_dir) / DIRT_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []
    missing: list[str] = []
    rows: list[dict] = []
    total = max(1, len(frames))
    in_flight: deque[Future] = deque()
    stopped = False
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for index, frame in enumerate(frames):
            if cancelled is not None and cancelled():
                stopped = True
                break
            label = f"{frame.layer_folder}/{frame.file_name}"
            if progress is not None:
                progress(index, total, label)
            result = frame.result
            if result is None or result.error:
                missing.append(label)
                continue
            if skip_empty and not result.regions:
                skipped.append(label)
                continue
            stem = Path(frame.file_name).stem
            rows.extend(csv_rows(result, frame=stem, layer=frame.layer_folder))
            in_flight.append(
                pool.submit(
                    write_frame_dirt_files,
                    result,
                    root / frame.layer_folder,
                    stem,
                    layer=frame.layer_folder,
                    config=config,
                    write_view=write_view,
                )
            )
            while len(in_flight) > 4 * max(1, int(workers)):
                written.extend(in_flight.popleft().result())
        while in_flight:
            written.extend(in_flight.popleft().result())
    with (root / CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    written.append(root / CSV_NAME)
    if progress is not None and not stopped:
        progress(total, total, "")
    return DirtExportReport(
        output_dir=root,
        written=tuple(written),
        skipped_empty=tuple(skipped),
        missing_results=tuple(missing),
        regions=len(rows),
        cancelled=stopped,
    )


def read_contour(cell: str) -> list[tuple[int, int]]:
    """The outline of a ``dirt_regions.csv`` row back as points."""

    points = []
    for item in str(cell or "").split(";"):
        if ":" in item:
            x, y = item.split(":", 1)
            points.append((int(x), int(y)))
    return points

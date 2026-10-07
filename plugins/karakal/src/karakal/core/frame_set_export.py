"""Export one frame set: chosen files per frame plus a summary for the whole set.

Layout of one export::

    <export folder>/                default Karakal_export_<date>, the user may rename it
      <set name>/
        Исходные снимки/            source image of every frame (copied as is)
        Маски/<model>/              mask of the layer (copied as is)
        Confidence/<model>/         confidence map of the layer (copied as is)
        Ошибки/<model>/             defects as coloured cells on black; with
                                    ``fill_frames`` the rest of the run is here
                                    too, as black frames (no errors to take)
        Ошибки на маске/<model>/    defects outlined over the mask
        Карты ошибок/<model>/       machine maps per frame (lossless PNG): <frame>_class.png
                                    (uint8, 255 = IGNORE), <frame>_score.png (uint16),
                                    <frame>_instance.png (uint16), <frame>_preview.png and
                                    <frame>_manifest.json; whole objects, not boxes
        холст.png                   the whole set as one picture
        результаты.csv              one row per frame

The model folder level is left out when only one model is exported.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .grid_error_export import _write_image, render_grid_error_image, safe_export_folder_name, unique_run_dir

ITEM_SOURCE = "source"
ITEM_MASK = "mask"
ITEM_CONFIDENCE = "confidence"
ITEM_ERRORS = "errors"
ITEM_ERRORS_OVER_MASK = "errors_over_mask"
ITEM_ERROR_MAPS = "error_maps"
FRAME_ITEMS = (ITEM_SOURCE, ITEM_MASK, ITEM_CONFIDENCE, ITEM_ERRORS, ITEM_ERRORS_OVER_MASK, ITEM_ERROR_MAPS)

ITEM_FOLDERS = {
    ITEM_SOURCE: "Исходные снимки",
    ITEM_MASK: "Маски",
    ITEM_CONFIDENCE: "Confidence",
    ITEM_ERRORS: "Ошибки",
    ITEM_ERRORS_OVER_MASK: "Ошибки на маске",
    ITEM_ERROR_MAPS: "Карты ошибок",
}
# Files written per frame and model for the machine maps.
_MAP_FILES_PER_FRAME = 5
_MAPS_BYTES_PER_PIXEL = 0.05
CANVAS_FILE = "холст.png"
CSV_FILE = "результаты.csv"
CANVAS_TILE_PX = 12
CANVAS_MAX_SIDE_PX = 8000

# Size guesses for files that do not exist yet (bytes per image pixel).
_ERRORS_BYTES_PER_PIXEL = 0.03
_BLACK_BYTES_PER_PIXEL = 0.002
_OVERLAY_BYTES_PER_PIXEL = 0.25


@dataclass(slots=True)
class FrameSetExportModel:
    model_id: str
    title: str


@dataclass(slots=True)
class FrameSetExportFrame:
    key: str
    name: str
    original_path: str = ""
    mask_paths: dict[str, str] = field(default_factory=dict)
    confidence_paths: dict[str, str] = field(default_factory=dict)
    results: dict[str, object] = field(default_factory=dict)
    quality: float | None = None
    row: int = -1
    column: int = -1
    participating: bool = True
    defect_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class FrameSetExportPlan:
    set_name: str
    layer_title: str
    frames: tuple[FrameSetExportFrame, ...]
    models: tuple[FrameSetExportModel, ...]
    items: frozenset[str]
    canvas: bool = False
    table: bool = False
    error_types: tuple[str, ...] = ()
    defect_labels: dict[str, str] = field(default_factory=dict)
    canvas_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    matrix_size: tuple[int, int] = (0, 0)
    rules_text: tuple[str, ...] = ()
    # Run frames outside the set, written to the errors folder as black frames.
    fill_frames: tuple[FrameSetExportFrame, ...] = ()
    # Machine maps: the run's normal-cell profile per model and the analysis settings of the run.
    normal_profiles: dict[str, object] = field(default_factory=dict)
    analysis_config: object | None = None
    calibration_fingerprint: str = ""


@dataclass(slots=True)
class FrameSetExportPreview:
    lines: tuple[str, ...]
    file_count: int
    approx_bytes: int


@dataclass(slots=True)
class FrameSetExportReport:
    run_dir: Path
    set_dir: Path
    written: int = 0
    missing: tuple[str, ...] = ()
    cancelled: bool = False


def _model_folders(plan: FrameSetExportPlan) -> dict[str, str]:
    if len(plan.models) <= 1:
        return {model.model_id: "" for model in plan.models}
    used: set[str] = set()
    folders: dict[str, str] = {}
    for model in plan.models:
        name = safe_export_folder_name(model.title, fallback=model.model_id or "model")
        candidate, index = name, 2
        while candidate.lower() in used:
            candidate = f"{name}_{index}"
            index += 1
        used.add(candidate.lower())
        folders[model.model_id] = candidate
    return folders


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size if path else 0
    except OSError:
        return 0


def _sample_size(paths: list[str]) -> int:
    """Average size of a few existing files, so the preview stays fast on big runs."""

    sizes = [size for size in (_file_size(path) for path in paths[:8]) if size > 0]
    return int(sum(sizes) / len(sizes)) if sizes else 0


def _frame_size(frame: FrameSetExportFrame) -> tuple[int, int] | None:
    """Image height and width from the frame's grid result."""

    for result in frame.results.values():
        height = int(getattr(result, "image_height", 0) or 0)
        width = int(getattr(result, "image_width", 0) or 0)
        if height > 0 and width > 0:
            return height, width
    return None


def _image_pixels(frame: FrameSetExportFrame) -> int:
    size = _frame_size(frame)
    return size[0] * size[1] if size is not None else 0


def _run_frame_size(plan: FrameSetExportPlan) -> tuple[int, int]:
    """Size for black frames that have no result of their own."""

    for frame in plan.frames + plan.fill_frames:
        size = _frame_size(frame)
        if size is not None:
            return size
    return 1, 1


def _count_and_size(plan: FrameSetExportPlan, item: str, model_id: str | None) -> tuple[int, int]:
    frames = plan.frames
    if item == ITEM_SOURCE:
        paths = [frame.original_path for frame in frames if frame.original_path]
        return len(paths), _sample_size(paths) * len(paths)
    if item in {ITEM_MASK, ITEM_CONFIDENCE}:
        source = "mask_paths" if item == ITEM_MASK else "confidence_paths"
        paths = [getattr(frame, source).get(str(model_id), "") for frame in frames]
        paths = [path for path in paths if path]
        return len(paths), _sample_size(paths) * len(paths)
    if item == ITEM_ERROR_MAPS:
        with_mask = [frame for frame in frames if frame.mask_paths.get(str(model_id))]
        pixels = _image_pixels(with_mask[0]) if with_mask else 0
        return len(with_mask) * _MAP_FILES_PER_FRAME, int(pixels * _MAPS_BYTES_PER_PIXEL) * len(with_mask)
    with_result = [frame for frame in frames if frame.results.get(str(model_id)) is not None]
    if item == ITEM_ERRORS_OVER_MASK:
        with_result = [frame for frame in with_result if frame.mask_paths.get(str(model_id))]
    per_pixel = _ERRORS_BYTES_PER_PIXEL if item == ITEM_ERRORS else _OVERLAY_BYTES_PER_PIXEL
    pixels = _image_pixels(with_result[0]) if with_result else 0
    count, size = len(with_result), int(pixels * per_pixel) * len(with_result)
    if item == ITEM_ERRORS and plan.fill_frames:
        height, width = _run_frame_size(plan)
        count += len(plan.fill_frames)
        size += int(height * width * _BLACK_BYTES_PER_PIXEL) * len(plan.fill_frames)
    return count, size


def format_size(size_bytes: int) -> str:
    megabytes = float(size_bytes) / (1024.0 * 1024.0)
    if megabytes >= 1024.0:
        return f"~{megabytes / 1024.0:.1f} ГБ"
    if megabytes >= 10.0:
        return f"~{megabytes:.0f} МБ"
    return f"~{max(0.1, megabytes):.1f} МБ"


def _run_dir(output_dir: Path, run_name: str | None) -> Path:
    """New export folder: the user's name, or the dated default; never an existing one."""

    name = safe_export_folder_name(run_name or "", fallback="")
    if not name:
        return unique_run_dir(output_dir)
    candidate, index = output_dir / name, 2
    while candidate.exists():
        candidate = output_dir / f"{name}_{index}"
        index += 1
    return candidate


def preview_frame_set_export(
    plan: FrameSetExportPlan, output_dir: Path | str, run_name: str | None = None
) -> FrameSetExportPreview:
    """Folder tree with file counts and sizes, before anything is written."""

    set_folder = safe_export_folder_name(plan.set_name, fallback="выборка")
    run_folder = safe_export_folder_name(run_name or "", fallback="Karakal_export_<дата>")
    rows: list[tuple[str, str]] = [
        (f"{Path(output_dir)}\\", ""),
        (f"  {run_folder}\\", ""),
        (f"    {set_folder}\\", f"{len(plan.frames)} кадров"),
    ]
    model_folders = _model_folders(plan)
    total_files = 0
    total_bytes = 0

    def files_text(item: str, count: int, size: int) -> str:
        text = f"{count} файлов, {format_size(size)}"
        if item == ITEM_ERRORS and plan.fill_frames:
            text += f" (из них чёрных: {len(plan.fill_frames)})"
        return text

    for item in FRAME_ITEMS:
        if item not in plan.items:
            continue
        folder = ITEM_FOLDERS[item]
        if item == ITEM_SOURCE or len(plan.models) <= 1:
            model_id = None if item == ITEM_SOURCE else (plan.models[0].model_id if plan.models else "")
            count, size = _count_and_size(plan, item, model_id)
            rows.append((f"      {folder}\\", files_text(item, count, size)))
            total_files += count
            total_bytes += size
            continue
        rows.append((f"      {folder}\\", ""))
        for model in plan.models:
            count, size = _count_and_size(plan, item, model.model_id)
            rows.append((f"        {model_folders[model.model_id]}\\", files_text(item, count, size)))
            total_files += count
            total_bytes += size
    if plan.canvas:
        rows.append((f"      {CANVAS_FILE}", "1 файл"))
        total_files += 1
        total_bytes += 200_000
    if plan.table:
        rows.append((f"      {CSV_FILE}", f"{len(plan.frames)} строк"))
        total_files += 1
        total_bytes += 120 * len(plan.frames)
    width = max(len(name) for name, info in rows if info) if any(info for _name, info in rows) else 0
    lines = tuple(f"{name.ljust(width)}  {info}" if info else name for name, info in rows)
    return FrameSetExportPreview(lines, total_files, total_bytes)


def _copy(source: str, target_dir: Path) -> bool:
    if not source:
        return False
    path = Path(source)
    if not path.is_file():
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target_dir / path.name)
    return True


def _read_gray(path: str) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def render_errors_over_mask(mask: np.ndarray, errors_bgr: np.ndarray) -> np.ndarray:
    """Dim mask with defect cells tinted and outlined in their type colour."""

    if mask.shape[:2] != errors_bgr.shape[:2]:
        errors_bgr = cv2.resize(errors_bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    base = cv2.cvtColor((mask.astype(np.float32) * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    painted = np.any(errors_bgr > 0, axis=2)
    blended = base.copy()
    blended[painted] = (base[painted].astype(np.float32) * 0.45 + errors_bgr[painted].astype(np.float32) * 0.55).astype(
        np.uint8
    )
    edges = cv2.morphologyEx(painted.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    blended[edges] = errors_bgr[edges]
    return blended


def render_canvas(plan: FrameSetExportPlan) -> np.ndarray:
    """The matrix with only this set's frames coloured, the rest left dark."""

    rows, columns = plan.matrix_size
    rows = max(1, int(rows))
    columns = max(1, int(columns))
    tile = max(1, min(CANVAS_TILE_PX, CANVAS_MAX_SIDE_PX // max(rows, columns)))
    gap = 1 if tile >= 4 else 0
    canvas = np.full((rows * tile, columns * tile, 3), 24, dtype=np.uint8)
    for frame in plan.frames:
        if frame.row < 0 or frame.column < 0 or frame.row >= rows or frame.column >= columns:
            continue
        red, green, blue = plan.canvas_colors.get(frame.key, (200, 200, 200))
        top = frame.row * tile
        left = frame.column * tile
        canvas[top : top + tile - gap, left : left + tile - gap] = (blue, green, red)
    return canvas


def _write_table(plan: FrameSetExportPlan, path: Path) -> None:
    types = list(plan.error_types)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["кадр", "ключ", "строка", "столбец", "качество_%", "участвует"]
            + [plan.defect_labels.get(item, item) for item in types]
        )
        for frame in plan.frames:
            quality = "" if frame.quality is None else f"{frame.quality * 100.0:.1f}"
            writer.writerow(
                [frame.name, frame.key, frame.row + 1, frame.column + 1, quality, "да" if frame.participating else "нет"]
                + [int(frame.defect_counts.get(item, 0)) for item in types]
            )


def _write_error_maps(plan: FrameSetExportPlan, frame: FrameSetExportFrame, model_id: str, target: Path) -> int:
    """Analyze the frame again with the run's bank and write its machine maps; 0 when it cannot."""

    from .cell_error_analysis import analyze_cell_errors
    from .error_maps import export_maps
    from .grid_anomaly import GridDamageAnalysisConfig, _cell_error_config

    profile = plan.normal_profiles.get(model_id)
    bank = getattr(profile, "normal_bank", None)
    mask_path = frame.mask_paths.get(model_id, "")
    mask = _read_gray(mask_path) if mask_path and bank is not None else None
    if mask is None:
        return 0
    confidence_path = frame.confidence_paths.get(model_id, "")
    confidence = _read_gray(confidence_path) if confidence_path else None
    config = plan.analysis_config if isinstance(plan.analysis_config, GridDamageAnalysisConfig) else GridDamageAnalysisConfig()
    cell_config = _cell_error_config(config.normalized())
    analysis = analyze_cell_errors(
        mask, bank, confidence=confidence, config=cell_config, frame_id=frame.key, frame_path=mask_path
    )
    written = export_maps(
        analysis,
        target,
        Path(frame.name).stem or frame.key,
        mask=mask,
        analysis_parameters={
            "geometry_sensitivity": cell_config.geometry_sensitivity,
            "merge_sensitivity": cell_config.merge_sensitivity,
            "debris_min_area_px": cell_config.debris_min_area_px,
        },
        calibration_fingerprint=plan.calibration_fingerprint,
    )
    return len(written)


def export_frame_set(
    plan: FrameSetExportPlan,
    output_dir: Path | str,
    *,
    run_name: str | None = None,
    progress=None,
    cancelled=None,
) -> FrameSetExportReport:
    """Write the export described by ``plan`` into a new folder (``run_name`` or a dated one)."""

    run_dir = _run_dir(Path(output_dir), run_name)
    set_dir = run_dir / safe_export_folder_name(plan.set_name, fallback="выборка")
    set_dir.mkdir(parents=True, exist_ok=True)
    report = FrameSetExportReport(run_dir=run_dir, set_dir=set_dir)
    model_folders = _model_folders(plan)
    missing: list[str] = []
    written = 0
    fill_frames = plan.fill_frames if ITEM_ERRORS in plan.items else ()
    total = len(plan.frames) + len(fill_frames)
    for index, frame in enumerate(plan.frames):
        if cancelled is not None and cancelled():
            report.cancelled = True
            break
        frame_missing = False
        if ITEM_SOURCE in plan.items:
            if _copy(frame.original_path, set_dir / ITEM_FOLDERS[ITEM_SOURCE]):
                written += 1
            else:
                frame_missing = True
        for model in plan.models:
            model_dir = model_folders.get(model.model_id, "")
            mask_path = frame.mask_paths.get(model.model_id, "")
            result = frame.results.get(model.model_id)
            if ITEM_MASK in plan.items:
                target = set_dir / ITEM_FOLDERS[ITEM_MASK] / model_dir
                if _copy(mask_path, target):
                    written += 1
                else:
                    frame_missing = True
            if ITEM_CONFIDENCE in plan.items:
                target = set_dir / ITEM_FOLDERS[ITEM_CONFIDENCE] / model_dir
                if _copy(frame.confidence_paths.get(model.model_id, ""), target):
                    written += 1
                else:
                    frame_missing = True
            errors_bgr = None
            if result is not None and (ITEM_ERRORS in plan.items or ITEM_ERRORS_OVER_MASK in plan.items):
                errors_bgr = render_grid_error_image(result, plan.error_types, color_mode="color")
                if errors_bgr.ndim == 2:
                    errors_bgr = cv2.cvtColor(errors_bgr, cv2.COLOR_GRAY2BGR)
            stem = Path(frame.name).stem or frame.key
            if ITEM_ERRORS in plan.items:
                if errors_bgr is None:
                    frame_missing = True
                else:
                    _write_image(set_dir / ITEM_FOLDERS[ITEM_ERRORS] / model_dir / f"{stem}.png", errors_bgr, "png")
                    written += 1
            if ITEM_ERRORS_OVER_MASK in plan.items:
                mask = _read_gray(mask_path) if mask_path else None
                if errors_bgr is None or mask is None:
                    frame_missing = True
                else:
                    overlay = render_errors_over_mask(mask, errors_bgr)
                    _write_image(set_dir / ITEM_FOLDERS[ITEM_ERRORS_OVER_MASK] / model_dir / f"{stem}.png", overlay, "png")
                    written += 1
            if ITEM_ERROR_MAPS in plan.items:
                count = _write_error_maps(plan, frame, model.model_id, set_dir / ITEM_FOLDERS[ITEM_ERROR_MAPS] / model_dir)
                if count:
                    written += count
                else:
                    frame_missing = True
        if frame_missing:
            missing.append(frame.name)
        if progress is not None:
            progress(index + 1, total, frame.name)
    if fill_frames and not report.cancelled:
        default_size = _run_frame_size(plan)
        for index, frame in enumerate(fill_frames, start=len(plan.frames)):
            if cancelled is not None and cancelled():
                report.cancelled = True
                break
            height, width = _frame_size(frame) or default_size
            black = np.zeros((height, width, 3), dtype=np.uint8)
            stem = Path(frame.name).stem or frame.key
            for model in plan.models:
                model_dir = model_folders.get(model.model_id, "")
                _write_image(set_dir / ITEM_FOLDERS[ITEM_ERRORS] / model_dir / f"{stem}.png", black, "png")
                written += 1
            if progress is not None:
                progress(index + 1, total, frame.name)
    if plan.canvas and not report.cancelled:
        _write_image(set_dir / CANVAS_FILE, render_canvas(plan), "png")
        written += 1
    if plan.table and not report.cancelled:
        _write_table(plan, set_dir / CSV_FILE)
        written += 1
    report.written = written
    report.missing = tuple(missing)
    return report

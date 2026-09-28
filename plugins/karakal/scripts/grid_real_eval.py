"""Evaluate cell-defect inspection on real SEM frames. Qt-free.

Uses the same call the application makes for the binary mask layer:
``analyze_grid_frame_path(mask, confidence_path=..., use_cache=False)`` with the
balanced inspection config from ``KarakalPresenter._grid_damage_config_from_payload``.

Nothing is written under the repository or the data root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from karakal.core.grid_anomaly import (  # noqa: E402
    GRID_DAMAGE_REASON_TYPES,
    GridDamageAnalysisConfig,
    analyze_grid_frame_path,
    configure_grid_worker_process,
)
from karakal.core.grid_calibration import reference_pairs  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"F:\test_test_test")
FRAME_STEM = "TEST_DIRECT_{index:04d}.jpg"
BALANCED_SLIDERS = {
    "fill_sensitivity": 60,
    "debris_sensitivity": 75,
    "geometry_sensitivity": 40,
    "disagreement_sensitivity": 50,
    "merge_sensitivity": 35,
}
SLIDER_KEYS = tuple(BALANCED_SLIDERS)
REASON_COLORS_BGR = {
    "filled_cell": (82, 64, 235),
    "partial_filled_cell": (74, 153, 242),
    "small_artifact": (153, 72, 236),
    "broken_geometry": (248, 189, 56),
    "merged_contour": (247, 85, 168),
    "edge_clipped_cell": (21, 204, 250),
    "class_conflict": (72, 29, 225),
}
REASON_PRIORITY = (
    "merged_contour",
    "broken_geometry",
    "small_artifact",
    "filled_cell",
    "partial_filled_cell",
    "edge_clipped_cell",
    "class_conflict",
)
ROLE_DIRS = {
    "direct": ("NN_res/cells/Direct_result_test", "NN_res/cells/Direct_Confidence_test"),
    "inv": ("NN_res/cells_inv/inv_result_test", "NN_res/cells_inv/Confidence_inv_test"),
}


def _slider_unit(values: dict, key: str, default: int) -> float:
    try:
        return max(0.0, min(1.0, float(values.get(key, default)) / 100.0))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default) / 100.0))


def _example_pairs(value: object) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    if not isinstance(value, list):
        return ()
    pairs = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("features"), dict):
            continue
        features = item["features"]
        pairs.append(
            (
                str(item.get("label") or ""),
                tuple((str(key), float(number)) for key, number in features.items()),
            )
        )
    return tuple(pairs)


def app_config_from_payload(payload: dict | None) -> GridDamageAnalysisConfig:
    """Mirror ``KarakalPresenter._grid_damage_config_from_payload`` without Qt."""

    values = dict(BALANCED_SLIDERS)
    incoming = dict(payload or {})
    for key in SLIDER_KEYS:
        if key in incoming:
            values[key] = int(incoming[key])
    fill = _slider_unit(values, "fill_sensitivity", 60)
    debris = _slider_unit(values, "debris_sensitivity", 75)
    merge = _slider_unit(values, "merge_sensitivity", 35)
    geometry = _slider_unit(values, "geometry_sensitivity", 40)
    disagreement = _slider_unit(values, "disagreement_sensitivity", 50)
    enabled = incoming.get("enabled_reason_types") or incoming.get("enabled_error_types")
    if enabled is None:
        enabled_reason_types = GRID_DAMAGE_REASON_TYPES
    else:
        enabled_reason_types = tuple(str(item) for item in enabled)
    reference = incoming.get("calibration_reference")
    config = GridDamageAnalysisConfig(
        include_debug_payload=False,
        debug=False,
        blur_radius=1,
        min_contour_area=max(2.0, 14.0 - 10.0 * debris),
        min_cell_size=2,
        filled_ratio_delta=max(0.04, 0.42 - 0.34 * fill),
        filled_ratio_absolute=max(0.20, 0.72 - 0.30 * fill),
        bad_score_threshold=max(0.25, 0.92 - 0.50 * disagreement),
        merged_size_ratio=max(1.10, 1.95 - 0.75 * merge),
        merged_area_ratio=max(1.10, 1.95 - 0.78 * merge),
        geometry_solidity_limit=max(0.40, min(0.95, 0.50 + 0.40 * geometry)),
        geometry_iou_threshold=0.55,
        centroid_mismatch_ratio=0.55,
        enabled_reason_types=enabled_reason_types,
        scoring_mode="calibrated",
        fill_sensitivity=int(values["fill_sensitivity"]),
        debris_sensitivity=int(values["debris_sensitivity"]),
        geometry_sensitivity=int(values["geometry_sensitivity"]),
        merge_sensitivity=int(values["merge_sensitivity"]),
        calibration_reference=reference_pairs(reference if isinstance(reference, dict) else None),
        calibration_examples=_example_pairs(incoming.get("calibration_examples")),
        example_influence=float(incoming.get("example_influence") or 0.5),
        calibration_fingerprint=str(incoming.get("calibration_fingerprint") or ""),
    ).normalized()
    overrides = {}
    known = set(asdict(config))
    for key, value in incoming.items():
        if key in known and key not in SLIDER_KEYS and key not in {
            "calibration_reference",
            "calibration_examples",
            "enabled_reason_types",
            "enabled_error_types",
        }:
            overrides[key] = value
    if overrides:
        config = replace(config, **overrides).normalized()
    return config


def load_gray(path: Path) -> np.ndarray | None:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    if encoded.size == 0:
        return None
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return np.asarray(image, dtype=np.uint8)


def frame_indexes(spec: str | None, sample: int | None, total: int = 184) -> list[int]:
    if spec:
        indexes = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                indexes.extend(range(min(start, end), max(start, end) + 1))
            else:
                indexes.append(int(part))
        unique = [index for index in dict.fromkeys(indexes) if 1 <= index <= total]
        if not unique:
            raise SystemExit("no frames in range 1..184")
        return unique
    count = 18 if sample is None else int(sample)
    count = max(1, min(total, count))
    if count >= total:
        return list(range(1, total + 1))
    if count == 1:
        return [1]
    picked = [round(step * (total - 1) / (count - 1)) + 1 for step in range(count)]
    return list(dict.fromkeys(picked))


def discover_jobs(data_root: Path, indexes: list[int], roles: tuple[str, ...]) -> list[tuple[str, str, str, str]]:
    source_dir = data_root / "source_test"
    jobs = []
    for index in indexes:
        name = FRAME_STEM.format(index=index)
        source = source_dir / name
        if not source.is_file():
            continue
        for role in roles:
            mask_rel, confidence_rel = ROLE_DIRS[role]
            mask = data_root / mask_rel / name
            confidence = data_root / confidence_rel / name
            if mask.is_file():
                jobs.append((f"{index:04d}", role, str(mask), str(confidence if confidence.is_file() else "")))
    return jobs


def _primary_reason(reasons: tuple[str, ...] | list[str]) -> str:
    present = {str(reason) for reason in reasons}
    for reason in REASON_PRIORITY:
        if reason in present:
            return reason
    return next(iter(present), "unknown")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _compact_result(frame_id: str, role: str, result, elapsed_ms: float, error: str = "") -> dict:
    if result is None:
        return {
            "frame": frame_id,
            "role": role,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": error or "decode_error",
            "grid_detected": False,
            "reasons": {},
            "defects": [],
        }
    reason_counts: dict[str, int] = {}
    defects = []
    feature_groups: dict[str, dict[str, list[float]]] = {}
    for cell in result.per_cell_results:
        reasons = tuple(str(reason) for reason in cell.reasons)
        label = _primary_reason(reasons) if reasons else "normal"
        bucket = feature_groups.setdefault(label, {})
        for key, value in cell.feature_snapshot:
            bucket.setdefault(str(key), []).append(float(value))
        if not reasons:
            continue
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        defects.append(
            {
                "bbox": [int(value) for value in cell.bbox[:4]],
                "reasons": list(reasons),
                "score": round(float(cell.score), 4),
                "status": str(cell.status),
            }
        )
    feature_medians = {
        label: {key: round(float(np.median(values)), 4) for key, values in groups.items()}
        for label, groups in feature_groups.items()
    }
    return {
        "frame": frame_id,
        "role": role,
        "elapsed_ms": round(elapsed_ms, 2),
        "error": error,
        "grid_detected": bool(result.grid_detected),
        "image_width": int(result.image_width),
        "image_height": int(result.image_height),
        "detected_cells": int(result.detected_cells),
        "normal_cells": int(result.normal_cells),
        "component_count": int(result.component_count),
        "cell_width": int(result.cell_width),
        "cell_height": int(result.cell_height),
        "damage_score": round(float(result.damage_score), 4),
        "reasons": reason_counts,
        "feature_medians": feature_medians,
        "defects": defects,
    }


def eval_chunk(entries: tuple[tuple[str, str, str, str], ...], config: GridDamageAnalysisConfig) -> list[dict]:
    """Analyze one chunk. Cache stays off. Safe to run in a spawned worker."""

    binary_config = replace(config, cell_representation="binary").normalized()
    payloads = []
    for frame_id, role, mask_path, confidence_path in entries:
        started = perf_counter()
        try:
            result = analyze_grid_frame_path(
                mask_path,
                frame_id=f"{role}:{frame_id}",
                config=binary_config,
                use_cache=False,
                read_cache=False,
                write_cache=False,
                confidence_path=confidence_path or None,
            )
            payloads.append(_compact_result(frame_id, role, result, (perf_counter() - started) * 1000.0))
        except Exception as error:
            payloads.append(
                _compact_result(
                    frame_id,
                    role,
                    None,
                    (perf_counter() - started) * 1000.0,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return payloads


def _chunks(items: list, size: int) -> list[tuple]:
    return [tuple(items[offset : offset + size]) for offset in range(0, len(items), size)]


def run_jobs(jobs: list[tuple[str, str, str, str]], config: GridDamageAnalysisConfig, workers: int) -> list[dict]:
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        return eval_chunk(tuple(jobs), config)
    chunk_size = max(1, min(4, max(1, len(jobs) // (workers * 2))))
    chunks = _chunks(jobs, chunk_size)
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=configure_grid_worker_process,
        initargs=(1,),
    ) as executor:
        futures = [executor.submit(eval_chunk, chunk, config) for chunk in chunks]
        for future in as_completed(futures):
            batch = future.result()
            rows.extend(batch)
            done = len(rows)
            print(f"analyzed {done}/{len(jobs)}", flush=True)
    rows.sort(key=lambda item: (item["role"], item["frame"]))
    return rows


def _reason_totals(rows: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for reason, count in (row.get("reasons") or {}).items():
            totals[reason] = totals.get(reason, 0) + int(count)
    return dict(sorted(totals.items()))


def write_summary(output_dir: Path, rows: list[dict], config: GridDamageAnalysisConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = _reason_totals(rows)
    payload = {
        "config": config.cache_payload(),
        "frame_count": len(rows),
        "elapsed_ms": round(sum(float(row.get("elapsed_ms") or 0.0) for row in rows), 2),
        "reason_totals": totals,
        "frames": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "role",
                "elapsed_ms",
                "grid_detected",
                "detected_cells",
                "normal_cells",
                "component_count",
                "cell_width",
                "cell_height",
                "damage_score",
                "reason",
                "count",
            ]
        )
        for row in rows:
            reasons = row.get("reasons") or {}
            if not reasons:
                writer.writerow(
                    [
                        row.get("frame"),
                        row.get("role"),
                        row.get("elapsed_ms"),
                        row.get("grid_detected"),
                        row.get("detected_cells"),
                        row.get("normal_cells"),
                        row.get("component_count"),
                        row.get("cell_width"),
                        row.get("cell_height"),
                        row.get("damage_score"),
                        "",
                        0,
                    ]
                )
                continue
            for reason, count in sorted(reasons.items()):
                writer.writerow(
                    [
                        row.get("frame"),
                        row.get("role"),
                        row.get("elapsed_ms"),
                        row.get("grid_detected"),
                        row.get("detected_cells"),
                        row.get("normal_cells"),
                        row.get("component_count"),
                        row.get("cell_width"),
                        row.get("cell_height"),
                        row.get("damage_score"),
                        reason,
                        count,
                    ]
                )


def _draw_overlay(source: np.ndarray, defects: list[dict]) -> np.ndarray:
    canvas = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    for defect in defects:
        x, y, width, height = (int(value) for value in defect["bbox"][:4])
        reason = _primary_reason(defect.get("reasons") or ())
        color = REASON_COLORS_BGR.get(reason, (180, 180, 180))
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 1)
    legend_y = 24
    counts: dict[str, int] = {}
    for defect in defects:
        for reason in defect.get("reasons") or ():
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
        color = REASON_COLORS_BGR.get(reason, (180, 180, 180))
        cv2.putText(
            canvas,
            f"{reason} {count}",
            (12, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        legend_y += 24
    return canvas


def write_overlays(output_dir: Path, data_root: Path, rows: list[dict]) -> None:
    for row in rows:
        if row.get("error"):
            continue
        frame_id = str(row["frame"])
        source_path = data_root / "source_test" / FRAME_STEM.format(index=int(frame_id))
        source = load_gray(source_path)
        if source is None:
            continue
        overlay = _draw_overlay(source, list(row.get("defects") or []))
        destination = output_dir / "overlays" / str(row["role"])
        destination.mkdir(parents=True, exist_ok=True)
        _write_image(destination / f"{frame_id}.jpg", overlay)


def write_galleries(output_dir: Path, data_root: Path, rows: list[dict], top_n: int) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, str, dict]]] = {}
    for row in rows:
        for defect in row.get("defects") or []:
            for reason in defect.get("reasons") or ():
                grouped.setdefault((str(row["role"]), str(reason)), []).append(
                    (float(defect.get("score") or 0.0), str(row["frame"]), defect)
                )
    for (role, reason), items in grouped.items():
        items.sort(key=lambda item: (-item[0], item[1], item[2]["bbox"]))
        folder = output_dir / "gallery" / role / reason
        folder.mkdir(parents=True, exist_ok=True)
        for rank, (score, frame_id, defect) in enumerate(items[:top_n], start=1):
            source = load_gray(data_root / "source_test" / FRAME_STEM.format(index=int(frame_id)))
            mask_rel, _confidence_rel = ROLE_DIRS[role]
            mask = load_gray(data_root / mask_rel / FRAME_STEM.format(index=int(frame_id)))
            if source is None or mask is None:
                continue
            crop_source, crop_mask = _paired_crop(source, mask, defect["bbox"])
            pair = np.hstack([cv2.cvtColor(crop_source, cv2.COLOR_GRAY2BGR), cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)])
            _write_image(folder / f"{rank:02d}_{frame_id}_{score:.3f}.jpg", pair)


def _paired_crop(source: np.ndarray, mask: np.ndarray, bbox: list[int], pad: int = 24) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = (int(value) for value in bbox[:4])
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(source.shape[1], x + width + pad)
    y1 = min(source.shape[0], y + height + pad)
    source_crop = source[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1].copy()
    color = cv2.cvtColor(source_crop, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color, (x - x0, y - y0), (x - x0 + width, y - y0 + height), (0, 255, 255), 1)
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), mask_crop


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


def write_mosaic(output_dir: Path, data_root: Path, indexes: list[int]) -> None:
    """source | direct | direct_conf | inv | inv_conf crops where the inv mask is densest and sparsest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        name = FRAME_STEM.format(index=index)
        panels = {
            "source": load_gray(data_root / "source_test" / name),
            "direct": load_gray(data_root / ROLE_DIRS["direct"][0] / name),
            "direct_conf": load_gray(data_root / ROLE_DIRS["direct"][1] / name),
            "inv": load_gray(data_root / ROLE_DIRS["inv"][0] / name),
            "inv_conf": load_gray(data_root / ROLE_DIRS["inv"][1] / name),
        }
        if any(image is None for image in panels.values()):
            print(f"skip mosaic {name}: missing image")
            continue
        inv = panels["inv"]
        assert inv is not None
        origins = _contrast_origins(inv, side=300)
        strips = []
        for label, origin in origins:
            tiles = []
            for key in ("source", "direct", "direct_conf", "inv", "inv_conf"):
                image = panels[key]
                assert image is not None
                x, y = origin
                crop = image[y : y + 300, x : x + 300]
                tile = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                cv2.putText(tile, f"{key} {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            strips.append(np.hstack(tiles))
        mosaic = np.vstack(strips)
        _write_image(output_dir / f"mosaic_{index:04d}.jpg", mosaic)
        print(f"mosaic {index:04d} -> {output_dir / f'mosaic_{index:04d}.jpg'}")


def _contrast_origins(mask: np.ndarray, side: int = 300) -> list[tuple[str, tuple[int, int]]]:
    binary = mask >= 127
    height, width = binary.shape
    step = side
    windows = []
    for y in range(0, max(1, height - side + 1), step):
        for x in range(0, max(1, width - side + 1), step):
            window = binary[y : y + side, x : x + side]
            windows.append((int(np.count_nonzero(window)), x, y))
    if not windows:
        return [("origin", (0, 0))]
    windows.sort()
    sparse = windows[0]
    dense = windows[-1]
    mid = windows[len(windows) // 2]
    chosen = [("sparse", (sparse[1], sparse[2])), ("mid", (mid[1], mid[2])), ("dense", (dense[1], dense[2]))]
    unique = []
    seen = set()
    for label, origin in chosen:
        if origin in seen:
            continue
        seen.add(origin)
        unique.append((label, origin))
    return unique


def compare_summaries(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    before_totals = before.get("reason_totals") or {}
    after_totals = after.get("reason_totals") or {}
    reasons = sorted(set(before_totals) | set(after_totals))
    print(f"{'reason':<24} {'before':>8} {'after':>8} {'delta':>8}")
    for reason in reasons:
        left = int(before_totals.get(reason, 0))
        right = int(after_totals.get(reason, 0))
        print(f"{reason:<24} {left:8d} {right:8d} {right - left:8d}")
    before_frames = {(row.get("role"), row.get("frame")): row for row in before.get("frames") or []}
    after_frames = {(row.get("role"), row.get("frame")): row for row in after.get("frames") or []}
    print("\nper-frame deltas (only reasons that changed):")
    for key in sorted(set(before_frames) | set(after_frames)):
        left = (before_frames.get(key) or {}).get("reasons") or {}
        right = (after_frames.get(key) or {}).get("reasons") or {}
        changed = []
        for reason in sorted(set(left) | set(right)):
            delta = int(right.get(reason, 0)) - int(left.get(reason, 0))
            if delta:
                changed.append(f"{reason}:{int(left.get(reason, 0))}->{int(right.get(reason, 0))}")
        if changed:
            role, frame = key
            print(f"  {role} {frame}: {', '.join(changed)}")


def _assert_output_allowed(output_dir: Path, data_root: Path) -> Path:
    resolved = output_dir.resolve()
    repo = Path(__file__).resolve().parents[3]
    data = data_root.resolve()
    if resolved == repo or repo in resolved.parents:
        raise SystemExit(f"refusing to write inside the repo: {resolved}")
    if resolved == data or data in resolved.parents:
        raise SystemExit(f"refusing to write inside the data root: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Karakal cell-defect inspection on real frames.")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("KARAKAL_REAL_DATA") or DEFAULT_DATA_ROOT))
    parser.add_argument("--frames", default="", help="1,17,42 or 1-10. Empty uses --sample.")
    parser.add_argument("--sample", type=int, default=18, help="Deterministic frame count when --frames is empty.")
    parser.add_argument("--role", choices=("direct", "inv", "both"), default="both")
    parser.add_argument("--config", type=Path, default=None, help="JSON config/calibration exported from the app.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Default is a temp folder.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top", type=int, default=8, help="Gallery crops per reason.")
    parser.add_argument("--mosaic", action="store_true", help="Write 300x300 source/mask strips and skip detection.")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Print reason deltas between two summary.json files.")
    parser.add_argument("--no-images", action="store_true", help="Write only summary.csv/json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        compare_summaries(Path(args.compare[0]), Path(args.compare[1]))
        return 0
    data_root = args.data
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")
    indexes = frame_indexes(args.frames or None, args.sample)
    if args.out is None:
        output_dir = Path(tempfile.gettempdir()) / "karakal_grid_real_eval"
    else:
        output_dir = args.out
    output_dir = _assert_output_allowed(output_dir, data_root)
    if args.mosaic:
        write_mosaic(output_dir / "mosaic", data_root, indexes)
        return 0
    payload = {}
    if args.config is not None:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = app_config_from_payload(payload)
    roles = ("direct", "inv") if args.role == "both" else (args.role,)
    jobs = discover_jobs(data_root, indexes, roles)
    if not jobs:
        raise SystemExit("no frames matched")
    print(f"frames={indexes} roles={roles} jobs={len(jobs)} workers={args.workers} out={output_dir}")
    started = perf_counter()
    rows = run_jobs(jobs, config, max(1, int(args.workers)))
    write_summary(output_dir, rows, config)
    if not args.no_images:
        write_overlays(output_dir, data_root, rows)
        write_galleries(output_dir, data_root, rows, max(1, int(args.top)))
    totals = _reason_totals(rows)
    print(f"done in {perf_counter() - started:.1f}s")
    print(json.dumps({"reason_totals": totals, "summary": str(output_dir / "summary.json")}, ensure_ascii=False))
    for row in rows:
        reasons = row.get("reasons") or {}
        print(
            f"{row['role']} {row['frame']}: cells={row.get('detected_cells')} "
            f"normal={row.get('normal_cells')} size={row.get('cell_width')}x{row.get('cell_height')} "
            f"components={row.get('component_count')} {reasons} {row.get('elapsed_ms')}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

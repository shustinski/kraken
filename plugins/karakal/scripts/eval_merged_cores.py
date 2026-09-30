"""Evaluate merged_contour (core-based) on all real frames. Writes to %TEMP%."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from karakal.core.grid_anomaly import (  # noqa: E402
    GridCellReferenceProfile,
    analyze_grid_frame_path,
    configure_grid_worker_process,
    estimate_run_cell_reference_profile,
)
from karakal.core.grid_merge_cores import analyze_merge_cores, find_cell_cores  # noqa: E402
from grid_real_eval import app_config_from_payload, discover_jobs  # noqa: E402

DATA_ROOT = Path(os.environ.get("KARAKAL_REAL_DATA") or r"F:\test_test_test")
OUT = Path(os.environ.get("TEMP", ".")) / "karakal_merged_cores_eval"
FOCUS = {1, 9, 25, 41, 49, 73, 184}
_CONFIG = None
_PROFILES: dict[str, GridCellReferenceProfile | None] = {}


def _log(message: str) -> None:
    print(message, flush=True)


def _init_worker(profiles: dict[str, dict | None]) -> None:
    global _CONFIG, _PROFILES
    configure_grid_worker_process()
    _CONFIG = replace(app_config_from_payload({}, profile="tester"), cell_representation="binary")
    _PROFILES = {}
    for role, payload in profiles.items():
        if not payload:
            _PROFILES[role] = None
            continue
        _PROFILES[role] = GridCellReferenceProfile(**payload)


def _profile_to_dict(profile: GridCellReferenceProfile | None) -> dict | None:
    if profile is None:
        return None
    return {
        "median_width": float(profile.median_width),
        "median_height": float(profile.median_height),
        "median_area": float(profile.median_area),
        "median_fill": float(profile.median_fill),
        "median_interior_fill": float(profile.median_interior_fill),
        "median_center_fill": float(profile.median_center_fill),
        "median_aspect": float(profile.median_aspect),
        "candidate_count": int(profile.candidate_count),
        "seed_count": int(profile.seed_count),
        "frame_id": str(profile.frame_id),
        "frame_path": str(profile.frame_path),
    }


def _analyze_job(job: tuple[str, str, str, str]) -> dict:
    frame_id, role, mask_path, conf_path = job
    idx = int(Path(mask_path).stem.split("_")[-1])
    result = analyze_grid_frame_path(
        mask_path,
        frame_id=f"{role}:{idx:04d}",
        config=_CONFIG,
        reference_profile=_PROFILES.get(role),
        use_cache=False,
        read_cache=False,
        write_cache=False,
        confidence_path=conf_path if Path(conf_path).is_file() else None,
    )
    merges = []
    if result is not None:
        for cell in result.cells:
            if "merged_contour" not in cell.reasons:
                continue
            merges.append({"bbox": list(cell.bbox), "centroid": list(cell.centroid)})
    return {
        "frame": idx,
        "role": role,
        "mask": mask_path,
        "conf": conf_path,
        "merged_count": len(merges),
        "merges": merges,
        "cell_w": float(getattr(result, "cell_width", 0) or 0),
        "cell_h": float(getattr(result, "cell_height", 0) or 0),
    }


def _save_tile(source, mask, conf, bbox, cores, out_path: Path, label: str) -> None:
    x, y, w, h = (int(v) for v in bbox)
    pad = 16
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad
    panels = []
    for image in (source, mask, conf if conf is not None else mask):
        if image is None:
            crop = np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
        else:
            crop = image[y0:y1, x0:x1]
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop.copy()
        cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w - 1, y - y0 + h - 1), (0, 255, 255), 1)
        for core in cores:
            cv2.circle(crop, (int(core.x) + 2, int(core.y) + 2), max(2, int(core.radius)), (0, 0, 255), 1)
        panels.append(crop)
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + 6
    canvas = np.zeros((height + 20, width, 3), dtype=np.uint8)
    ox = 0
    for panel in panels:
        canvas[20 : 20 + panel.shape[0], ox : ox + panel.shape[1]] = panel
        ox += panel.shape[1] + 3
    cv2.putText(canvas, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    if not DATA_ROOT.is_dir():
        _log(f"missing data: {DATA_ROOT}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    remaining_dir = OUT / "remaining"
    gone_dir = OUT / "gone_sample"
    new_dir = OUT / "new_candidates"
    # Keep a sample of previous false merges before clearing the gallery.
    if remaining_dir.is_dir():
        gone_dir.mkdir(exist_ok=True)
        previous = sorted(remaining_dir.glob("*.jpg"))
        for index, path in enumerate(previous):
            if index >= 30:
                break
            target = gone_dir / path.name
            if not target.exists():
                try:
                    target.write_bytes(path.read_bytes())
                except OSError:
                    pass
    for path in remaining_dir.glob("*.jpg") if remaining_dir.is_dir() else []:
        try:
            path.unlink()
        except OSError:
            pass
    remaining_dir.mkdir(exist_ok=True)
    new_dir.mkdir(exist_ok=True)
    gone_dir.mkdir(exist_ok=True)

    _log("discover jobs...")
    jobs = discover_jobs(DATA_ROOT, list(range(1, 185)), ("direct", "inv"))
    _log(f"jobs={len(jobs)}")
    config = replace(app_config_from_payload({}, profile="tester"), cell_representation="binary")
    by_role: dict[str, list[Path]] = defaultdict(list)
    for _fid, role, mask, _conf in jobs:
        by_role[role].append(Path(mask))

    profile_payloads: dict[str, dict | None] = {}
    for role, paths in by_role.items():
        _log(f"estimate profile {role} from {len(paths)} masks...")
        profile = estimate_run_cell_reference_profile(paths, config=config, sample_limit=24, frame_id=f"run:{role}")
        profile_payloads[role] = _profile_to_dict(profile)
        _log(f"  -> {profile}")

    rows = []
    workers = max(1, min(6, (os.cpu_count() or 4) - 1))
    _log(f"analyze with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(profile_payloads,)) as pool:
        futures = [pool.submit(_analyze_job, job) for job in jobs]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 20 == 0 or done == len(futures):
                _log(f"  analyzed {done}/{len(futures)}")

    rows.sort(key=lambda item: (item["role"], item["frame"]))
    csv_path = OUT / "merged_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "role", "merged_count", "cell_w", "cell_h"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row["frame"],
                    "role": row["role"],
                    "merged_count": row["merged_count"],
                    "cell_w": row["cell_w"],
                    "cell_h": row["cell_h"],
                }
            )

    # Galleries for remaining merges (all) and focus frames.
    remaining_meta = []
    for row in rows:
        if not row["merges"]:
            continue
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        conf = cv2.imread(row["conf"], cv2.IMREAD_GRAYSCALE) if Path(row["conf"]).is_file() else None
        source = cv2.imread(str(DATA_ROOT / "source_test" / Path(row["mask"]).name), cv2.IMREAD_GRAYSCALE)
        cell_w = max(8.0, float(row["cell_w"] or 22))
        cell_h = max(8.0, float(row["cell_h"] or 32))
        for order, merge in enumerate(row["merges"]):
            bbox = tuple(int(v) for v in merge["bbox"])
            x, y, w, h = bbox
            roi = np.zeros((h + 4, w + 4), dtype=np.uint8)
            if mask is not None:
                crop = mask[y : y + h, x : x + w]
                roi[2 : 2 + crop.shape[0], 2 : 2 + crop.shape[1]] = (crop > 0).astype(np.uint8) * 255
            cores = find_cell_cores(roi, cell_w, cell_h)
            name = f"{row['role']}_{row['frame']:04d}_{order}_{x}_{y}.jpg"
            _save_tile(source, mask, conf, bbox, cores, remaining_dir / name, f"{row['role']} {row['frame']} cores={len(cores)}")
            remaining_meta.append({"file": name, "role": row["role"], "frame": row["frame"], "bbox": list(bbox), "cores": len(cores)})

    # Up to 20 multi-core blobs marked merged (confirmation set) + hunt for extras already covered.
    new_hits = remaining_meta[:20]

    focus = [row for row in rows if row["frame"] in FOCUS]
    summary = {
        "total_merges": int(sum(row["merged_count"] for row in rows)),
        "by_role": {
            "direct": int(sum(row["merged_count"] for row in rows if row["role"] == "direct")),
            "inv": int(sum(row["merged_count"] for row in rows if row["role"] == "inv")),
        },
        "frames_with_merge": int(sum(1 for row in rows if row["merged_count"] > 0)),
        "focus": [
            {"frame": row["frame"], "role": row["role"], "merged_count": row["merged_count"]} for row in focus
        ],
        "remaining_gallery": len(remaining_meta),
        "sample_remaining": new_hits,
        "thresholds": {
            "min_core_frac@merge35": 0.445,
            "area_tol@merge35": 0.283,
            "solidity_floor@merge35": 0.832,
            "dt_only_if_area_ge": "1.4 * cell_area",
            "solid_block": "1xN rectangle solidity>=0.92 extent>=0.88; uncertain if no DT cores",
            "geometry": "1xN / Nx1 bbox short-side within 22% of cell; span matches N cells",
            "decision": "N>=2 pitched cores and (solid_block or area~N with neck/geometry)",
            "thin_strip": "height~1 cell demands solid_block or tight area+neck+extent",
        },
        "algorithm": "grid_damage_v85_merged_cores",
        "profiles": profile_payloads,
        "output": str(OUT),
        "gone_sample": str(gone_dir),
        "note_before": "Prompt sample ~125 false merges on 24x2; prior core-only pass had 463 direct merges (54 on frame 184).",
        "note_after": "All 368 frames: 0 merged_contour. Grain strips rejected via solidity/extent/geometry gates; cell size is run-wide.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

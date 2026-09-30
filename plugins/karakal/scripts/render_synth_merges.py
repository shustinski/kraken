"""Render synthetic merged-cores gallery for step-0 report."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies
from karakal.core.grid_merge_cores import analyze_merge_cores, find_cell_cores, filled_contour_roi

OUT = Path(os.environ.get("TEMP", ".")) / "karakal_reports" / "synth_merges"
OUT.mkdir(parents=True, exist_ok=True)


def _solid(image, x, y, w=22, h=32):
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), 255, -1)


def _lattice(rows=5, cols=7):
    image = np.zeros((16 + rows * 48, 16 + cols * 36), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            _solid(image, 16 + col * 36, 16 + row * 48)
    return image


def _cfg():
    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=0,
        morphology_open=0,
        morphology_close=0,
        min_contour_area=40.0,
        min_cell_size=6,
        fill_sensitivity=60,
        debris_sensitivity=75,
        geometry_sensitivity=40,
        merge_sensitivity=35,
        min_grid_candidate_cells=4,
    )


def _save(name: str, image: np.ndarray, label: str) -> None:
    result = detect_grid_cell_anomalies(image, config=_cfg())
    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    panels = []
    canvas_h = image.shape[0]
    canvas_w = image.shape[1]
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for cell in merged:
        x, y, w, h = (int(v) for v in cell.bbox)
        cv2.rectangle(base, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
        # cores from contour ROI
        # find matching contour
        cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            bx, by, bw, bh = cv2.boundingRect(contour)
            if abs(bx - x) <= 2 and abs(by - y) <= 2:
                an = analyze_merge_cores(
                    contour,
                    (bx, by, bw, bh),
                    area=float(cv2.contourArea(contour)),
                    cell_width=max(8.0, float(result.cell_width or 22)),
                    cell_height=max(8.0, float(result.cell_height or 32)),
                )
                roi, ox, oy = filled_contour_roi(contour, (bx, by, bw, bh))
                for core in an.cores:
                    cv2.circle(base, (int(core.x) + ox, int(core.y) + oy), max(2, int(core.radius)), (0, 0, 255), 1)
                break
    header = np.zeros((28, canvas_w, 3), dtype=np.uint8)
    text = f"{label} merges={len(merged)}"
    cv2.putText(header, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([header, base])
    path = OUT / f"{name}.png"
    cv2.imwrite(str(path), out)
    print(path, "merges", len(merged))


def main() -> int:
    # pair with neck
    image = _lattice()
    cv2.rectangle(image, (16 + 22, 16 + 12), (16 + 36, 16 + 20), 255, -1)
    _save("01_pair_neck", image, "pair+neck")

    # triple
    image = _lattice(rows=4, cols=8)
    y = 16 + 48
    for col in range(3):
        _solid(image, 16 + col * 36, y)
        if col:
            cv2.rectangle(image, (16 + (col - 1) * 36 + 22, y + 10), (16 + col * 36, y + 18), 255, -1)
    _save("02_triple", image, "triple")

    # six chain
    image = _lattice(rows=4, cols=8)
    y2 = 16 + 48
    for col in range(6):
        _solid(image, 16 + col * 36, y2)
        if col:
            cv2.rectangle(image, (16 + (col - 1) * 36 + 22, y2 + 10), (16 + col * 36, y2 + 18), 255, -1)
    _save("03_six_chain", image, "six-chain")

    # solid block no neck
    image = _lattice()
    cv2.rectangle(image, (16, 16), (16 + 36 + 22 - 1, 16 + 32 - 1), 255, -1)
    _save("04_solid_block", image, "solid-block")

    # elongated single
    image = _lattice()
    cv2.rectangle(image, (16, 16), (16 + 40, 16 + 32), 255, -1)
    _save("05_elongated_single", image, "elongated-single")

    # grain
    image = _lattice()
    rng = np.random.default_rng(7)
    for _ in range(80):
        x = int(rng.integers(20, 260))
        y = int(rng.integers(280, 320))
        w = int(rng.integers(4, 18))
        h = int(rng.integers(4, 14))
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)
    _save("06_grain", image, "grain")
    print("OUT", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

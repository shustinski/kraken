"""Regression gate for cell-defect inspection on the real SEM set.

The images stay outside the repo. Set KARAKAL_REAL_DATA or keep them at F:\\test_test_test.
Labels are lattice cells measured from the masks, not from the detector.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from karakal.app.presenter import KarakalPresenter
from karakal.core.grid_anomaly import analyze_grid_frame_path

DATA_ROOT = Path(os.environ.get("KARAKAL_REAL_DATA") or r"F:\test_test_test")
LABELS_PATH = Path(__file__).resolve().parents[1] / "data" / "grid_real_labels.json"
ROLE_DIRS = {
    "direct": ("NN_res/cells/Direct_result_test", "NN_res/cells/Direct_Confidence_test"),
    "inv": ("NN_res/cells_inv/inv_result_test", "NN_res/cells_inv/Confidence_inv_test"),
}
FORBIDDEN_ON_NORMAL = {"broken_geometry", "merged_contour", "filled_cell", "geometry_mismatch"}

pytestmark = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="real SEM set is not mounted")


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    union = float(aw * ah + bw * bh) - intersection
    return intersection / union if union else 0.0


def _analyze(frame: int, role: str):
    name = f"TEST_DIRECT_{int(frame):04d}.jpg"
    mask_rel, confidence_rel = ROLE_DIRS[role]
    mask = DATA_ROOT / mask_rel / name
    confidence = DATA_ROOT / confidence_rel / name
    config = replace(KarakalPresenter._grid_damage_config_from_payload({}), cell_representation="binary")
    return analyze_grid_frame_path(
        mask,
        frame_id=f"{role}:{frame:04d}",
        config=config,
        use_cache=False,
        read_cache=False,
        write_cache=False,
        confidence_path=confidence if confidence.is_file() else None,
    )


def test_labeled_normal_lattice_cells_are_not_geometry_or_merge_defects() -> None:
    payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    items = [item for item in payload["items"] if item["expected"] == "normal"]
    assert len(items) >= 20
    frames = {(int(item["frame"]), str(item["role"])) for item in items}
    results = {key: _analyze(*key) for key in frames}
    misses = []
    for item in items:
        result = results[(int(item["frame"]), str(item["role"]))]
        bbox = tuple(int(value) for value in item["bbox"])
        for cell in result.per_cell_results:
            if not (set(cell.reasons) & FORBIDDEN_ON_NORMAL):
                continue
            if _iou(bbox, tuple(int(value) for value in cell.bbox)) >= 0.3:
                misses.append((item["role"], item["frame"], bbox, cell.reasons, cell.bbox))
                break
    assert not misses, misses[:8]


def test_clean_inverse_frame_has_no_false_geometry_or_merge() -> None:
    result = _analyze(90, "inv")
    reasons = {reason for cell in result.per_cell_results for reason in cell.reasons}
    assert "broken_geometry" not in reasons
    assert "merged_contour" not in reasons
    assert "filled_cell" not in reasons
    assert result.cell_width == 20
    assert result.cell_height == 33

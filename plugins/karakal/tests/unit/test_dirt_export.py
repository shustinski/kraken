"""Files of the dirty regions: JSON per frame, label image, picture and the run's CSV."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from karakal.core.dirt_export import CSV_NAME, DIRT_FOLDER, DirtExportFrame, export_dirt_frames, read_contour
from karakal.core.dirt_regions import analyze_dirt_frame

from test_dirt_regions import lattice, smear, sure


def _frame(folder, name: str, mask: np.ndarray, *, confidence: bool = True):
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / f"{name}.png"
    cv2.imencode(".png", mask)[1].tofile(str(mask_path))
    confidence_path = None
    if confidence:
        confidence_path = folder / f"{name}_confidence.png"
        cv2.imencode(".png", sure(mask))[1].tofile(str(confidence_path))
    return analyze_dirt_frame(mask_path, frame_id=name, confidence_path=confidence_path)


def test_export_writes_json_label_view_and_csv(tmp_path) -> None:
    masks = tmp_path / "Маски"
    dirty = _frame(masks, "TEST_00001", smear(smear(lattice(), 240, 350, 55, 80, 1), 520, 350, 55, 80, 2))
    clean = _frame(masks, "TEST_00002", lattice(), confidence=False)
    report = export_dirt_frames(
        [
            DirtExportFrame("TEST_00001.png", "Нули", dirty),
            DirtExportFrame("TEST_00002.png", "Нули", clean),
            DirtExportFrame("TEST_00003.png", "Нули", None),
        ],
        tmp_path / "out",
    )
    root = tmp_path / "out" / DIRT_FOLDER
    assert report.output_dir == root and not report.cancelled
    assert report.missing_results == ("Нули/TEST_00003.png",)
    assert report.regions == len(dirty.regions) >= 1

    payload = json.loads((root / "Нули" / "TEST_00001.dirt.json").read_text(encoding="utf-8"))
    assert payload["frame"] == "TEST_00001" and payload["layer"] == "Нули"
    assert (payload["width"], payload["height"]) == (dirty.width, dirty.height)
    assert payload["source"]["confidence"]
    assert payload["settings"]["merge_gap_cells"] == 3.0
    region = payload["regions"][0]
    assert set(region) >= {"id", "level", "dirtiness", "bbox", "contour", "parts", "edge", "signals"}
    assert set(region["bbox"]) == {"x", "y", "w", "h"}
    assert region["contour"] == [list(point) for point in dirty.regions[0].contour]

    clean_payload = json.loads((root / "Нули" / "TEST_00002.dirt.json").read_text(encoding="utf-8"))
    assert clean_payload["regions"] == [] and clean_payload["source"]["confidence"] is None

    label = cv2.imdecode(np.fromfile(str(root / "Нули" / "TEST_00001.dirt.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    assert label.shape == (dirty.height, dirty.width)
    assert set(np.unique(label)) == {0, *range(1, len(dirty.regions) + 1)}
    assert (root / "Нули" / "TEST_00001.dirt_view.jpg").is_file()

    with (root / CSV_NAME).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(dirty.regions)
    row = rows[0]
    assert row["frame"] == "TEST_00001" and row["layer"] == "Нули" and row["id"] == "1"
    assert [int(row[key]) for key in ("x", "y", "w", "h")] == list(dirty.regions[0].bbox)
    assert read_contour(row["contour"]) == list(dirty.regions[0].contour)


def test_export_skips_clean_frames_and_stops_on_cancel(tmp_path) -> None:
    masks = tmp_path / "Маски"
    clean = _frame(masks, "TEST_00002", lattice())
    report = export_dirt_frames([DirtExportFrame("TEST_00002.png", "Единицы", clean)], tmp_path / "out", skip_empty=True)
    assert report.skipped_empty == ("Единицы/TEST_00002.png",)
    assert not (tmp_path / "out" / DIRT_FOLDER / "Единицы").exists()

    stopped = export_dirt_frames(
        [DirtExportFrame("TEST_00002.png", "Единицы", clean)], tmp_path / "out2", cancelled=lambda: True
    )
    assert stopped.cancelled and stopped.regions == 0

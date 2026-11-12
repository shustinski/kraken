"""Scoring checks against synthetic validation frames."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
import numpy as np

from karakal.core.synthetic_scorecard import (
    FoundItem,
    Scorecard,
    from_relative_box,
    rank_auc,
    spearman,
    true_error,
)
from karakal.core.synthetic_validation import DIAG_BAD_DATA, DIAG_MODEL_ERROR, make_case


def _perfect(frame, model_id: str = "a") -> list[FoundItem]:
    return [
        FoundItem(item.codes[0], None if item.box is None else tuple(float(v) for v in item.box), item.diagnosis, model_id)
        for item in frame.items_for(model_id)
        if item.is_positive
    ]


def test_a_perfect_check_finds_everything_and_nothing_else() -> None:
    card = Scorecard()
    for index, case in enumerate(("clean", "open", "bridge", "edge_shift", "residue", "boundary_jitter")):
        frame = make_case(case, index)
        card.add_frame(f"F{index}", frame, _perfect(frame), model_id="a")
    assert card.recall("open") == card.recall("bridge") == card.recall("edge_shift") == card.recall("residue") == 1.0
    assert card.false_findings == []
    assert card.clean_frames == 2 and card.clean_frames_flagged == 0
    assert card.diagnosis_accuracy == 1.0
    assert "| open | 1 | 1 | 1.00 |" in card.to_markdown("perfect")


def test_a_silent_check_finds_nothing() -> None:
    card = Scorecard()
    frame = make_case("open", 0)
    card.add_frame("F0", frame, [], model_id="a")
    assert card.recall("open") == 0.0 and card.false_findings == []


def test_wrong_place_or_wrong_code_is_a_false_finding() -> None:
    frame = make_case("open", 0)
    (item,) = frame.items
    x, y, w, h = item.box
    far = (float((x + 200) % 400), float((y + 200) % 400), 5.0, 5.0)
    card = Scorecard()
    card.add_frame("F0", frame, [FoundItem("open", far), FoundItem("bridge", tuple(map(float, item.box)))], model_id="a")
    assert card.recall("open") == 0.0
    assert len(card.false_findings) == 2


def test_aliases_map_a_checks_own_codes() -> None:
    frame = make_case("open", 0)
    (item,) = frame.items
    card = Scorecard(aliases={"thin_metal": "open"})
    card.add_frame("F0", frame, [FoundItem("thin_metal", tuple(map(float, item.box)))], model_id="a")
    assert card.recall("open") == 1.0


def test_any_finding_on_a_clean_frame_is_false() -> None:
    card = Scorecard(any_code=True)
    card.add_frame("F0", make_case("clean", 0), [FoundItem("whatever", (1.0, 1.0, 3.0, 3.0))], model_id="a")
    assert card.clean_frames_flagged == 1 and len(card.false_findings) == 1


def test_without_codes_a_local_finding_does_not_name_a_whole_frame_problem() -> None:
    frame = make_case("blur", 0)
    local = Scorecard(any_code=True)
    local.add_frame("F0", frame, [FoundItem("x", (10.0, 10.0, 5.0, 5.0))], model_id="a")
    whole = Scorecard(any_code=True)
    whole.add_frame("F0", frame, [FoundItem("x", None)], model_id="a")
    assert local.recall("blur") == 0.0 and whole.recall("blur") == 1.0


def test_diagnosis_confusion_counts_wrong_diagnoses() -> None:
    frame = make_case("open", 0)
    (item,) = frame.items
    card = Scorecard()
    card.add_frame("F0", frame, [FoundItem("open", tuple(map(float, item.box)), DIAG_BAD_DATA)], model_id="a")
    assert card.diagnosis_pairs == {(DIAG_MODEL_ERROR, DIAG_BAD_DATA): 1}
    assert card.diagnosis_accuracy == 0.0


def test_relative_boxes_become_frame_pixels() -> None:
    assert from_relative_box((0.25, 0.5, 0.1, 0.2), (200, 400)) == (100.0, 100.0, 40.0, 40.0)
    assert from_relative_box(None, (10, 10)) is None


def test_rank_auc() -> None:
    assert rank_auc([3, 4], [1, 2]) == 1.0
    assert rank_auc([1, 2], [3, 4]) == 0.0
    assert rank_auc([1, 1], [1, 1]) == 0.5
    assert rank_auc([], [1]) is None


def test_spearman() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_true_error_sees_the_planted_error() -> None:
    clean = true_error(make_case("clean", 0), "a")
    broken = true_error(make_case("open", 0), "a")
    assert clean == {"iou_error": 0.0, "component_delta": 0.0}
    assert broken["iou_error"] > 0.0 and broken["component_delta"] == 1.0


def _script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validation_synth_set.py"
    spec = importlib.util.spec_from_file_location("karakal_validation_synth_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_set_folders_open_as_karakal_layers(tmp_path: Path) -> None:
    module = _script()
    folder, frames = module.write_style(
        tmp_path,
        "vias",
        per_case=1,
        clean=2,
        model_count=2,
        behaviour="calibrated",
        encoding="probability",
        shared=False,
        reaction="ignores",
    )
    answers = json.loads((folder / "answers.json").read_text(encoding="utf-8"))
    assert answers["geometry"] == "point" and len(answers["frames"]) == len(frames)
    for key in frames:
        for sub in ("photos", "truth"):
            assert (folder / sub / f"{key}.png").is_file()
        for model_id in ("a", "b"):
            assert (folder / f"model_{model_id}" / f"{key}.png").is_file()
            assert (folder / f"model_{model_id}" / f"{key}_confidence.png").is_file()
    # The masks on disk are the masks of the answers.
    key = next(iter(frames))
    import cv2

    on_disk = cv2.imread(str(folder / "model_a" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(on_disk, frames[key].masks["a"])


def test_check_measures_todays_methods(tmp_path: Path) -> None:
    module = _script()
    folder, frames = module.write_style(
        tmp_path,
        "manhattan",
        per_case=1,
        clean=3,
        model_count=2,
        behaviour="calibrated",
        encoding="probability",
        shared=False,
        reaction="ignores",
    )
    lines = module.check_style(folder, frames, 2, 1)
    text = "\n".join(lines)
    for title in ("Model comparison", "Confidence check", "Single-result risk"):
        assert title in text
    # Comparing two models sees a missing conductor of one of them.
    row = next(line for line in lines if line.startswith("| missing_conductor |") and "AUC" not in line)
    assert row.split("|")[4].strip() == "1.00"

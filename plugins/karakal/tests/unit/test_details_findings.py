"""Everything the checks find on a frame, in the frame window: marks, heat, photo maps, the way through."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np
from PyQt6.QtCore import QPointF

from karakal.core.domain import BuildOptions, BuildResult, FrameAnalysisSummary, FrameRecord, GeometryMode, ModelSpec
from karakal.core.findings import DIAG_BAD_DATA, DIAG_MODEL_ERROR, DIAG_UNKNOWN, make_finding
from karakal.core.frame_details import load_frame_detail_findings
from karakal.core.synthetic_validation import ModelError, ModelPlan, make_validation_frame
from karakal.ui.details_dialog import (
    FINDINGS_HEAT_KIND,
    FINDINGS_MARKS_KIND,
    PHOTO_MAP_KINDS,
    ExtendFrameDetailsDialog,
)
from karakal.ui.finding_marks import add_finding_items, finding_heat, label_places, merge_findings, place_labels


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


def _frame_run(tmp_path: Path, *, summary: FrameAnalysisSummary | None = None, with_photo: bool = True):
    frame = make_validation_frame(
        seed=3,
        models=(ModelPlan("a", (ModelError("open"), ModelError("extra_blob"))), ModelPlan("b")),
        data_defect="residue",
    )
    record = FrameRecord(
        "F_00",
        "F_00",
        original_path=_write(tmp_path / "photos" / "F_00.png", frame.photo) if with_photo else None,
        model_mask_paths={m: _write(tmp_path / m / "F_00.png", frame.masks[m]) for m in ("a", "b")},
        model_prob_paths={m: _write(tmp_path / m / "F_00_confidence.png", frame.confidences[m]) for m in ("a", "b")},
        summary=summary,
    )
    specs = tuple(ModelSpec(m, f"Model {m}", tmp_path / m, prob_folder=tmp_path / m) for m in ("a", "b"))
    run = BuildResult(records=(record,), model_specs=specs, options=BuildOptions(geometry_mode=GeometryMode.MASK), scores_computed=True)
    return frame, record, run


def test_the_frame_findings_are_gathered_and_computed_when_missing(tmp_path: Path) -> None:
    _frame, record, run = _frame_run(tmp_path)
    payload = load_frame_detail_findings(record, run)
    found = payload["findings"]
    assert set(payload["computed"]) == {"topology", "photo_check"}
    kinds = {(item.detector, item.kind, item.model_id) for item in found}
    assert ("topology", "open", "a") in kinds and ("topology", "extra_object", "a") in kinds
    assert ("photo_check", "open", "a") in kinds
    assert any(item.kind == "bad_photo" for item in found)
    assert payload["photo_quality"].ndim == 2 and payload["photo_metal"].shape == (512, 512)
    assert set(payload["model_ids"]) == {"a", "b"}


def test_findings_the_run_kept_are_not_computed_again(tmp_path: Path) -> None:
    kept = make_finding("open", "topology", 0.9, (0.1, 0.1, 0.05, 0.05), diagnosis=DIAG_MODEL_ERROR, model_id="a")
    summary = FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, findings=(kept,))
    _frame, record, run = _frame_run(tmp_path, summary=summary)
    payload = load_frame_detail_findings(record, run)
    assert payload["computed"] == ["photo_check"]
    assert [item for item in payload["findings"] if item.detector == "topology"] == [kept]


def test_without_a_photo_there_are_no_photo_maps(tmp_path: Path) -> None:
    _frame, record, run = _frame_run(tmp_path, with_photo=False)
    payload = load_frame_detail_findings(record, run)
    assert payload["computed"] == ["topology"] and "photo_quality" not in payload


def test_heat_is_the_worst_severity_and_leaves_the_whole_frame_out() -> None:
    findings = (
        make_finding("open", "topology", 0.9, (0.0, 0.0, 0.5, 0.5)),
        make_finding("neck", "topology", 0.4, (0.25, 0.25, 0.5, 0.5)),
        make_finding("boundary_shift", "topology", 1.0, None),
    )
    heat = finding_heat(findings, (100, 100))
    assert heat[10, 10] == pytest.approx(0.9) and heat[40, 40] == pytest.approx(0.9) and heat[60, 60] == pytest.approx(0.4)
    assert heat[90, 90] == 0.0


def test_mark_items_by_diagnosis(qtbot) -> None:
    from PyQt6.QtWidgets import QGraphicsScene

    scene = QGraphicsScene()
    solid = add_finding_items(scene, make_finding("open", "t", 0.9, (0.1, 0.1, 0.1, 0.1), diagnosis="model_error"), 200, 100)
    sure = add_finding_items(scene, make_finding("open", "t", 0.9, (0.1, 0.1, 0.1, 0.1), diagnosis="confident_error"), 200, 100)
    whole = add_finding_items(scene, make_finding("blur", "t", 1.0, None, diagnosis="bad_data"), 200, 100, label="blur")
    assert len(solid) == 1 and len(sure) == 9 and len(whole) == 2
    assert solid[0].rect().width() == pytest.approx(20.0) and solid[0].rect().height() == pytest.approx(10.0)


@pytest.fixture()
def dialog(tmp_path: Path, qtbot):
    frame, record, run = _frame_run(tmp_path)
    window = ExtendFrameDetailsDialog(record, run)
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window._findings_payload is not None and not window._payload_loading, timeout=60000)
    return frame, window


def test_the_window_opens_on_the_findings(dialog) -> None:
    _frame, window = dialog
    kinds = [window.result_kind_combo.itemData(index) for index in range(window.result_kind_combo.count())]
    assert kinds[:2] == [None, FINDINGS_MARKS_KIND]  # a group header, then the marks
    assert FINDINGS_HEAT_KIND in kinds and all(kind in kinds for kind in PHOTO_MAP_KINDS)
    assert window._selected_result_kind() == FINDINGS_MARKS_KIND
    assert window._finding_items, "marks on the scene"
    assert window.findings_legend_bar.isVisibleTo(window)
    assert window.findings_list.count() == len(window._findings_navigable()) > 0
    assert "open" in {kind for kind in window._findings_legend_buttons}


def test_ctrl_arrows_go_through_the_findings(dialog) -> None:
    _frame, window = dialog
    assert window._step_marks(+1)
    first = window._finding_hot
    assert first is window._findings_navigable()[0] and window.findings_position_label.text().startswith("1 /")
    assert window._step_marks(+1)
    assert window._finding_hot is not first
    assert window._step_marks(-1) and window._finding_hot is first
    assert window.findings_list.currentRow() == 0


def test_hover_tells_what_where_and_whose(dialog) -> None:
    _frame, window = dialog
    finding = next(item for item in window._findings_navigable() if item.kind == "open" and item.model_id == "a")
    rect = window.overlay_view.scene().sceneRect()
    x, y, w, h = finding.box
    point = QPointF((x + w / 2) * rect.width(), (y + h / 2) * rect.height())
    found = window._finding_at(point)
    assert found is not None and found.kind == "open"
    text = window._finding_hover_text(finding)
    assert window._finding_kind_label("open") in text and "Model a" in text


def test_a_kind_singled_out_and_one_model(dialog) -> None:
    _frame, window = dialog
    window._toggle_finding_solo("open")
    assert all(item.kind == "open" for item in window._findings_navigable())
    window._toggle_finding_solo("open")
    index = window.findings_model_combo.findData("b")
    window.findings_model_combo.setCurrentIndex(index)
    assert all(item.model_id in ("", "b") for item in window._findings_shown())


def test_heat_and_photo_layers_draw(dialog) -> None:
    _frame, window = dialog
    for kind in (FINDINGS_HEAT_KIND, *PHOTO_MAP_KINDS):
        window.result_kind_combo.setCurrentIndex(window.result_kind_combo.findData(kind))
        assert not window._findings_result_pixmap(kind).isNull(), kind
        assert window._finding_items == [], "marks only on the marks layer"
        assert not window.findings_legend_bar.isVisibleTo(window)
    window._show_findings_layer()
    assert window._selected_result_kind() == FINDINGS_MARKS_KIND and window._finding_items


def test_repeats_fold_into_the_most_telling_finding() -> None:
    by_topology = make_finding("open", "topology", 0.6, (0.10, 0.10, 0.05, 0.05), diagnosis=DIAG_UNKNOWN, model_id="a")
    by_photo = make_finding("open", "photo_check", 0.5, (0.11, 0.10, 0.05, 0.05), diagnosis=DIAG_MODEL_ERROR, model_id="a")
    other_model = make_finding("open", "topology", 0.6, (0.10, 0.10, 0.05, 0.05), diagnosis=DIAG_UNKNOWN, model_id="b")
    dirt_a = make_finding("bad_photo", "photo_check", 0.4, (0.5, 0.5, 0.02, 0.02), diagnosis=DIAG_BAD_DATA, model_id="a")
    dirt_b = make_finding("bad_photo", "photo_check", 0.5, (0.5, 0.5, 0.02, 0.02), diagnosis=DIAG_BAD_DATA, model_id="b")
    merged = merge_findings([by_topology, by_photo, other_model, dirt_a, dirt_b])
    leads = {(lead.detector, lead.kind, lead.model_id): (lead, twins) for lead, twins in merged}
    assert len(merged) == 3, "an open of model a seen twice, of model b once, one speck of dirt"
    lead, twins = leads[("photo_check", "open", "a")]
    assert lead.diagnosis == DIAG_MODEL_ERROR and lead.severity == pytest.approx(0.6) and twins == (by_topology,)
    dirt, _twins = leads[("photo_check", "bad_photo", "")]
    assert dirt.severity == pytest.approx(0.5), "a fault of the photo belongs to no model"


def test_labels_of_one_place_stack_and_crowded_ones_wait_for_zoom() -> None:
    from PyQt6.QtCore import QPointF

    near = [
        make_finding("neck", "topology", 0.6, (0.1, 0.1, 0.05, 0.05), model_id="a"),
        make_finding("bulge", "topology", 0.5, (0.1, 0.1, 0.05, 0.04), model_id="b"),
        make_finding("blur", "photo_check", 1.0, None),
        make_finding("not_aligned", "photo_check", 1.0, None),
    ]
    places = label_places(near)
    assert [len(place) for place in places] == [2, 2]
    stacks = [(QPointF(10.0, 10.0), (80.0, 30.0)), (QPointF(12.0, 12.0), (80.0, 30.0))]
    assert place_labels(stacks, 1.0) == [0], "the second would cover the first"
    assert place_labels(stacks, 50.0) == [0, 1], "zoomed in, both fit"


def test_labels_do_not_cover_each_other_in_the_window(dialog) -> None:
    _frame, window = dialog
    plates = [item for item in window._finding_label_items if item.parentItem() is None]
    assert plates
    view = window.overlay_view
    rects = [view.mapFromScene(item.scenePos()) for item in plates]
    boxes = [(point.x(), point.y(), item.rect().width(), item.rect().height()) for point, item in zip(rects, plates)]
    for index, (x, y, w, h) in enumerate(boxes):
        for x2, y2, w2, h2 in boxes[index + 1 :]:
            assert x + w <= x2 + 1 or x2 + w2 <= x + 1 or y + h <= y2 + 1 or y2 + h2 <= y + 1

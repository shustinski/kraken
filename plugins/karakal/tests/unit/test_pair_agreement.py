"""Pairs of one cell class: a cell only one network found is a place to check, dirt never is."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from kraken_core.analysis_protocol import AnalysisProfileKind
from PyQt6.QtCore import QSettings

from karakal.core import grid_anomaly, pair_agreement
from karakal.core.cell_components import extract_components
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.normal_bank import build_normal_bank
from karakal.core.pair_agreement import (
    AGREEMENT_COVER_FEATURE,
    AGREEMENT_REASON,
    AgreementThresholds,
    agreement_features,
    agreement_side,
    analyze_network_disagreement,
    analyze_network_disagreement_chunk,
    filter_network_disagreement,
)
from karakal.core.pair_conflict import (
    PAIR_CONFLICT_LAYER,
    agreement_key,
    is_agreement_key,
    pair_key,
    parse_pair_key,
)

pytestmark = pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for cell components")

SHAPE = (200, 300)
CELL_W, CELL_H = 14, 40


def _boxes() -> list[tuple[int, int, int, int]]:
    return [(20 + 24 * col, 20 + 56 * row, CELL_W, CELL_H) for row in range(3) for col in range(10)]


BOXES = _boxes()


def _frame(boxes=BOXES) -> np.ndarray:
    image = np.zeros(SHAPE, dtype=np.uint8)
    for x, y, width, height in boxes:
        image[y : y + height, x : x + width] = 255
    return image


@pytest.fixture(scope="module")
def bank():
    built = build_normal_bank([extract_components(_frame()) for _ in range(4)])
    assert built.ready
    return built


def _blob(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    grid_y, grid_x = np.ogrid[: image.shape[0], : image.shape[1]]
    out = image.copy()
    out[(grid_x - x) ** 2 + (grid_y - y) ** 2 <= radius**2] = 255
    return out


def test_agreement_keys_are_pair_keys_of_their_own_kind() -> None:
    key = agreement_key("night_a1", "night_b2")
    assert parse_pair_key(key) == ("night_a1", "night_b2")
    assert is_agreement_key(key)
    assert not is_agreement_key(pair_key("night_a1", "night_b2"))
    assert not is_agreement_key("night_a1")
    assert not is_agreement_key("same::only")


def test_the_same_cells_are_no_disagreement(bank) -> None:
    result = analyze_network_disagreement(_frame(), _frame(), bank, bank)
    assert not result.per_cell_results


def test_contours_drawn_differently_are_no_disagreement(bank) -> None:
    other = _frame([(x + 2, y + 1, width + 1, height - 2) for x, y, width, height in BOXES])
    assert not analyze_network_disagreement(_frame(), other, bank, bank).per_cell_results


def test_a_cell_only_one_network_found_is_reported_on_its_side(bank) -> None:
    missing = BOXES[12]
    fewer = _frame([box for box in BOXES if box != missing])

    (cell,) = analyze_network_disagreement(_frame(), fewer, bank, bank).per_cell_results
    assert cell.reasons == (AGREEMENT_REASON,)
    assert cell.bbox == missing
    assert agreement_side(cell) == 0
    assert agreement_features(cell)[AGREEMENT_COVER_FEATURE] == 0.0

    (cell,) = analyze_network_disagreement(fewer, _frame(), bank, bank).per_cell_results
    assert cell.bbox == missing
    assert agreement_side(cell) == 1


def test_dirt_of_one_network_is_no_disagreement(bank) -> None:
    # A round blob of a cell's area, a smaller one and a long streak: none of them is a cell.
    dirty = _blob(_blob(_frame(), 280, 100, 13), 280, 170, 6)
    dirty[6:12, 20:270] = 255
    assert not analyze_network_disagreement(_frame(), dirty, bank, bank).per_cell_results
    assert not analyze_network_disagreement(dirty, _frame(), bank, bank).per_cell_results


def test_a_network_that_painted_nothing_misses_every_cell(bank) -> None:
    result = analyze_network_disagreement(_frame(), np.zeros(SHAPE, dtype=np.uint8), bank, bank)
    assert len(result.per_cell_results) == len(BOXES)
    assert {agreement_side(cell) for cell in result.per_cell_results} == {0}


def test_frames_without_uncovered_pieces_never_reach_the_bank(bank, monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("the bank is only for the pieces the other network did not cover")

    monkeypatch.setattr(pair_agreement, "extract_components", fail)
    assert not analyze_network_disagreement(_frame(), _frame(), bank, bank).per_cell_results


def test_the_cover_limit_hides_cells_without_a_new_run(bank) -> None:
    x, y, width, height = BOXES[3]
    half = _frame([box for box in BOXES if box != BOXES[3]] + [(x, y, width, int(height * 0.4))])
    result = analyze_network_disagreement(_frame(), half, bank, bank)
    (cell,) = result.per_cell_results
    assert 0.3 < agreement_features(cell)[AGREEMENT_COVER_FEATURE] < 0.5

    assert not filter_network_disagreement(result, AgreementThresholds(max_cover=0.30)).per_cell_results
    kept = filter_network_disagreement(result, AgreementThresholds(max_cover=0.55), labels=("only in A", "only in B"))
    (cell,) = kept.per_cell_results
    assert cell.feature_cluster_label == "only in A"
    assert kept.broken_cells == 1


def _save(folder: Path, name: str, image: np.ndarray) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    assert _grayscale_array_to_qimage(image).save(str(path))
    return path


def test_the_chunk_gives_the_layer_of_a_pair_and_reports_bad_frames(tmp_path, bank) -> None:
    first = _save(tmp_path / "a", "F.png", _frame())
    second = _save(tmp_path / "b", "F.png", _frame(BOXES[1:]))
    payloads, errors = analyze_network_disagreement_chunk(
        (("F.png", str(first), str(second)), ("gone.png", str(first), str(tmp_path / "gone.png"))),
        first_bank=bank,
        second_bank=bank,
        use_cache=False,
    )
    assert len(payloads["F.png"][PAIR_CONFLICT_LAYER].per_cell_results) == 1
    assert errors == {"gone.png": "decode_error"}


def _layers(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "src"
    for index in range(1, 7):
        name = f"F_{index:04d}.png"
        _save(source, name, np.zeros(SHAPE, dtype=np.uint8))
        _save(tmp_path / "ones_a", name, _frame())
        _save(tmp_path / "ones_b", name, _frame(BOXES[1:] if index == 1 else BOXES))
        _save(tmp_path / "zeros", name, _frame())
    return tmp_path / "ones_a", tmp_path / "ones_b", tmp_path / "zeros", source


def _widget(tmp_path: Path, qtbot):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


def _loaded(tmp_path: Path, qtbot, classes: tuple[str, ...]):
    from karakal.ui.ui_constants import FOLDER_ORIGINAL_ROLE

    *folders, source = _layers(tmp_path)
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    presenter._on_analysis_profile_changed(AnalysisProfileKind.GRID_DEFECTS.value)
    items = [presenter._append_folder_item(folder, checked=True) for folder in folders[: len(classes)]]
    for item in items:
        item.setData(FOLDER_ORIGINAL_ROLE, str(source))
    presenter._refresh_folder_rows()
    for item, cell_class in zip(items, classes):
        presenter._set_folder_item_cell_class(item, cell_class)
    return widget, presenter, items


def test_two_layers_of_one_class_make_a_pair_and_three_layers_make_both_kinds(tmp_path, qtbot) -> None:
    widget, presenter, items = _loaded(tmp_path, qtbot, ("ones", "ones", "zeros"))
    ids = [presenter._model_id_for_folder_item(item) for item in items]

    (same,) = presenter._grid_agreement_pairs()
    assert (same.first_id, same.second_id, same.same_class) == (ids[0], ids[1], "ones")
    assert same.key == agreement_key(ids[0], ids[1])
    assert len(presenter._grid_conflict_pairs()) == 2
    assert len(presenter._grid_pair_buttons) == 3

    # The kind of a pair is on its cell, in the list of matrices and in the name.
    assert presenter._grid_pair_buttons[same.key].text() == "≠"
    conflict = presenter._grid_conflict_pairs()[0]
    assert presenter._grid_pair_buttons[conflict.key].text() == "∩"
    assert widget.grid_matrix_layer_combo.findData(same.key) >= 0
    assert "≠" in presenter._grid_pair_label(same) and presenter._t("pairs.class.ones") in presenter._grid_pair_label(same)
    assert "∩" in presenter._grid_pair_label(conflict)


def test_the_pair_matrix_opens_with_two_layers_and_chooses_before_the_run(tmp_path, qtbot) -> None:
    widget, presenter, _items = _loaded(tmp_path, qtbot, ("ones", "ones", "zeros"))
    assert widget.pair_matrix_group.isChecked()

    pairs = presenter._grid_layer_pairs()
    assert presenter._grid_pairs_to_compute() == pairs, "every pair is ticked at first"
    assert not presenter._grid_pair_buttons[pairs[0].key].isEnabled(), "nothing is computed by a click before the run"

    presenter._on_grid_pair_compute_toggled(pairs[0].key, False)
    assert pairs[0] not in presenter._grid_pairs_to_compute()
    assert pairs[0].key in presenter._grid_inspection_config_payload()["skipped_pairs"]

    presenter._on_grid_pair_compute_toggled(pairs[0].key, True)
    assert presenter._grid_pairs_to_compute() == pairs


def test_only_ticked_pairs_are_computed_and_a_pair_shows_its_findings(tmp_path, qtbot) -> None:
    from karakal.core.analytics import collect_frame_records
    from karakal.core.domain import BuildOptions

    widget, presenter, _items = _loaded(tmp_path, qtbot, ("ones", "ones", "zeros"))
    (same,) = presenter._grid_agreement_pairs()
    skipped = presenter._grid_conflict_pairs()[0]
    presenter._on_grid_pair_compute_toggled(skipped.key, False)

    presenter._on_build_finished(collect_frame_records(presenter._checked_model_specs(), BuildOptions(union_layer_frames=True)))
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    state = presenter._current_tab_state()
    if not state.grid_inspection_payloads_by_pair:
        presenter._start_compute_grid_inspection(state)
        qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)

    assert same.key in state.grid_inspection_payloads_by_pair
    assert skipped.key not in state.grid_inspection_payloads_by_pair

    presenter._select_grid_pair(same.key)
    assert presenter._grid_pair_conflict_frame_count(state, same) == 1
    assert presenter._grid_pair_buttons[same.key].text() == "≠ 1"
    shown = state.grid_inspection_payload_by_key
    (cell,) = shown["F_0001.png"].per_cell_results
    assert cell.feature_cluster_label == presenter._t("pairs.only_in", name=same.first_name)
    entries = list(presenter._iter_grid_inspection_error_payloads(state))
    assert [entry["error_type"] for entry in entries] == [AGREEMENT_REASON]

    # A skipped pair is not computed by a click either.
    presenter._select_grid_pair(skipped.key)
    assert presenter._worker_thread is None

    # The limit works on display: no new run.
    presenter._select_grid_pair(same.key)
    widget.grid_agreement_cover_spin.setValue(5)
    assert presenter._worker_thread is None
    assert presenter._grid_pair_conflict_frame_count(state, same) == 1


def test_a_pair_without_a_bank_of_normal_cells_is_not_computed_and_says_so(tmp_path, qtbot) -> None:
    from karakal.core.workers import MultiModelGridInspectionWorker

    key = agreement_key("a", "b")
    worker = MultiModelGridInspectionWorker((), (), conflict_pairs=((key, "a", "b"),))
    with qtbot.waitSignal(worker.analysisErrors) as signal:
        assert worker._pair_analysis(key) is None
    assert signal.args == [{key: "no_normal_bank"}]

    _widget_, presenter, _items = _loaded(tmp_path, qtbot, ("ones", "ones"))
    (pair,) = presenter._grid_agreement_pairs()
    presenter._on_grid_pair_errors({pair.key: "no_normal_bank"})
    button = presenter._grid_pair_buttons[pair.key]
    assert button.text() == "≠ !"
    assert button.toolTip() == presenter._t("pairs.no_bank", first=pair.first_name, second=pair.second_name)


def test_the_conductor_zone_switch_starts_off_and_old_settings_do_not_turn_it_on(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    box = widget.grid_error_type_checks["conductor_zone"]
    assert not box.isChecked()
    payload = presenter._grid_inspection_config_payload()
    assert payload["conductor_zone_enabled"] is False
    assert "conductor_zone" not in payload["enabled_error_types"]

    # Older versions always stored the zone among the types.
    presenter._set_grid_inspection_config_controls({"enabled_error_types": ["small_artifact", "conductor_zone"]})
    assert not box.isChecked()

    presenter._set_grid_inspection_config_controls(
        {"enabled_error_types": ["small_artifact", "conductor_zone"], "conductor_zone_enabled": True}
    )
    assert box.isChecked()
    assert presenter._grid_inspection_config_payload()["conductor_zone_enabled"] is True


def test_the_pair_filters_follow_the_kinds_of_ticked_pairs(tmp_path, qtbot) -> None:
    widget, presenter, items = _loaded(tmp_path, qtbot, ("ones", "ones", "zeros"))
    checks = widget.grid_error_type_checks
    conflict, network_only = checks["class_conflict"], checks[AGREEMENT_REASON]

    def on(checkbox) -> bool:
        return checkbox.isEnabledTo(checkbox.parentWidget())

    # Mixed layers: both kinds of pairs, both filters.
    assert on(conflict) and on(network_only)
    # Only the pair of one class ticked: only its filter.
    for pair in presenter._grid_conflict_pairs():
        presenter._on_grid_pair_compute_toggled(pair.key, False)
    assert on(network_only) and not on(conflict)
    assert conflict.toolTip() == presenter._t("grid_error.class_conflict_unavailable")
    # The operator's choice of an unavailable filter still holds for the export and the next run.
    assert "class_conflict" in presenter._selected_grid_error_types()
    # Only ones against zeros: only 1∩0.
    for pair in presenter._grid_conflict_pairs():
        presenter._on_grid_pair_compute_toggled(pair.key, True)
    (same,) = presenter._grid_agreement_pairs()
    presenter._on_grid_pair_compute_toggled(same.key, False)
    assert on(conflict) and not on(network_only)


def test_the_network_only_filter_hides_the_pair_findings_and_is_kept(tmp_path, qtbot) -> None:
    from karakal.core.analytics import collect_frame_records
    from karakal.core.domain import BuildOptions

    widget, presenter, _items = _loaded(tmp_path, qtbot, ("ones", "ones"))
    (same,) = presenter._grid_agreement_pairs()
    presenter._on_build_finished(collect_frame_records(presenter._checked_model_specs(), BuildOptions(union_layer_frames=True)))
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    state = presenter._current_tab_state()
    if not state.grid_inspection_payloads_by_pair:
        presenter._start_compute_grid_inspection(state)
        qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    presenter._select_grid_pair(same.key)
    assert list(presenter._iter_grid_inspection_error_payloads(state))
    checkbox = widget.grid_error_type_checks[AGREEMENT_REASON]
    checkbox.setChecked(False)
    presenter._apply_grid_view_settings()
    assert not list(presenter._iter_grid_inspection_error_payloads(state)), "hidden at once"
    assert presenter._worker_thread is None and not presenter._grid_matrix_needs_rerun(state)
    # The pair matrix still counts what the pair found.
    assert presenter._grid_pair_conflict_frame_count(state, same) == 1
    assert presenter._grid_inspection_config_payload()["network_only_enabled"] is False
    checkbox.setChecked(True)
    presenter._apply_grid_view_settings()
    assert list(presenter._iter_grid_inspection_error_payloads(state))

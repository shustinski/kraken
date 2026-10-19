"""Bank of normal cells from operator masks: every drawn cell is normal; without markup, as before."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

from karakal.core.cell_components import extract_components
from karakal.core.cell_error_analysis import analyze_cell_errors, markup_self_check
from karakal.core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION, estimate_run_normal_profile
from karakal.core.normal_bank import (
    SOURCE_REFERENCE,
    SOURCE_SESSION,
    STATUS_INSUFFICIENT,
    NormalBank,
    NormalBankConfig,
    build_markup_bank,
    build_normal_bank,
    calibrate_markup_tolerance,
)
from karakal.core.synthetic_masks import SyntheticLayout, normal_frame, synthetic_frame


def _normal_masks(count: int, start: int = 0) -> list[np.ndarray]:
    return [normal_frame(SyntheticLayout(), seed=seed).mask for seed in range(start, start + count)]


def _odd_drawn_mask() -> np.ndarray:
    # Cells an operator drew with bites and cut corners: in a markup they are normal by definition.
    return synthetic_frame({"bite": 3, "cut_corner": 3}, seed=21).mask


def _write(folder: Path, masks: list[np.ndarray]) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, mask in enumerate(masks):
        path = folder / f"FRAME_{index:04d}.png"
        assert cv2.imwrite(str(path), mask)
        paths.append(str(path))
    return paths


def test_every_drawn_cell_is_normal_with_the_markup_bank() -> None:
    masks = _normal_masks(2) + [_odd_drawn_mask()]
    bank = build_markup_bank([extract_components(mask) for mask in masks])
    assert bank.ready and bank.source == SOURCE_REFERENCE
    assert len(bank.exemplars) == 0 and len(bank.operator_exemplars) == bank.candidates

    assert markup_self_check([(f"m{index}", mask) for index, mask in enumerate(masks)], bank) == []
    # A bank of the network's cells calls the same drawn bites broken.
    run_bank = build_normal_bank([extract_components(mask) for mask in _normal_masks(4, start=10)])
    assert analyze_cell_errors(_odd_drawn_mask(), run_bank).class_counts().get("bad_geometry", 0) >= 3


def test_cells_cut_by_the_frame_edge_are_not_examples() -> None:
    sets = [extract_components(mask) for mask in _normal_masks(2)]
    margin = NormalBankConfig().markup_edge_margin_px

    def inner(item, shape) -> bool:
        x, y, w, h = item.bbox
        rows, cols = shape
        return not item.touches_border and x >= margin and y >= margin and x + w <= cols - margin and y + h <= rows - margin

    count = sum(1 for components in sets for item in components.components if inner(item, components.labels.shape))
    assert any(item.touches_border for components in sets for item in components.components)
    assert build_markup_bank(sets).candidates == count


def test_a_drawn_cell_a_few_pixels_from_the_edge_is_not_an_example() -> None:
    # Operators outline a cell cut by the frame with a small gap: it does not touch the edge.
    masks = _normal_masks(2)
    cut = np.zeros_like(masks[0])
    cut[2:20, 300:318] = 255  # half a cell, 2 px below the top edge
    bank = build_markup_bank([extract_components(mask) for mask in [*masks, cut]])
    assert bank.operator_height_range[0] > 20
    assert bank.candidates == build_markup_bank([extract_components(mask) for mask in masks]).candidates


def test_drawn_size_limits_leave_room_for_how_the_network_paints_them() -> None:
    bank = build_markup_bank([extract_components(mask) for mask in _normal_masks(2)])
    (w_lo, w_hi), (h_lo, h_hi) = bank.side_limits(100)
    assert w_hi >= bank.operator_width_range[1] + 2 and w_lo <= bank.operator_width_range[0] - 2
    assert h_hi >= bank.operator_height_range[1] + 2 and h_lo <= bank.operator_height_range[0] - 2


def _network_style(mask: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    # Rounder corners than any drawn rectangle, the way the network paints cells.
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.where(soft >= 127.5, 255, 0).astype(np.uint8)


def test_markup_bank_takes_the_network_handwriting_out() -> None:
    bank = build_markup_bank([extract_components(mask) for mask in _normal_masks(3)])
    network = [extract_components(_network_style(mask)) for mask in _normal_masks(4, start=40)]
    calibrated = calibrate_markup_tolerance(bank, network)

    # The drawn cells stay the only examples; the network only measured its handwriting.
    assert np.array_equal(calibrated.operator_exemplars, bank.operator_exemplars)
    assert len(calibrated.exemplars) == 0
    assert calibrated.style_offset > 0.0 and calibrated.tolerance_cells >= 50
    assert calibrated.drawn_norm == bank.distance_norm
    assert calibrated.distance_norm >= bank.distance_norm

    probe = _network_style(_normal_masks(1, start=77)[0])
    assert analyze_cell_errors(probe, bank).class_counts().get("bad_geometry", 0) > 10
    assert analyze_cell_errors(probe, calibrated).class_counts().get("bad_geometry", 0) == 0
    # A bite keeps its weight: cut corners of network-style cells are still broken.
    cells = [item for item in extract_components(probe).components if not item.touches_border][:6]
    bitten = probe.copy()
    for item in cells:
        x, y, w, h = item.bbox
        bitten[y : y + h // 3, x : x + w // 2] = 0
    assert analyze_cell_errors(bitten, calibrated).class_counts().get("bad_geometry", 0) == len(cells)


def test_handwriting_needs_enough_cells_and_never_lowers_the_unit() -> None:
    bank = build_markup_bank([extract_components(mask) for mask in _normal_masks(3)])
    assert calibrate_markup_tolerance(bank, []).bank_id == bank.bank_id
    same_style = calibrate_markup_tolerance(bank, [extract_components(mask) for mask in _normal_masks(4, start=40)])
    assert same_style.distance_norm >= bank.distance_norm
    restored = NormalBank.from_payload(same_style.to_payload())
    assert restored.bank_id == same_style.bank_id
    assert (restored.style_offset, restored.tolerance_cells) == (same_style.style_offset, same_style.tolerance_cells)


def test_a_drawn_size_is_inside_the_size_limits() -> None:
    masks = _normal_masks(2)
    wide = np.zeros_like(masks[0])
    wide[200:240, 300:330] = 255  # 30 px wide, the usual cell is 18
    masks.append(wide)
    bank = build_markup_bank([extract_components(mask) for mask in masks])
    (_w_lo, w_hi), _height = bank.side_limits(100)
    assert w_hi >= 30
    run_bank = build_normal_bank([extract_components(mask) for mask in _normal_masks(4)])
    assert run_bank.side_limits(100)[0][1] < 30


def test_payload_keeps_operator_cells_and_a_run_bank_keeps_its_old_keys() -> None:
    bank = build_markup_bank([extract_components(mask) for mask in _normal_masks(2)])
    restored = NormalBank.from_payload(bank.to_payload())
    assert restored.bank_id == bank.bank_id
    assert restored.operator_exemplars.shape == bank.operator_exemplars.shape
    assert restored.operator_width_range == bank.operator_width_range

    run_payload = build_normal_bank([extract_components(mask) for mask in _normal_masks(3)]).to_payload()
    assert "operator_exemplars" not in run_payload and "operator_width_range" not in run_payload


def test_markup_comes_first_and_too_little_markup_falls_back_to_the_run(tmp_path) -> None:
    run = _write(tmp_path / "run", _normal_masks(4, start=30))
    markup = _write(tmp_path / "markup", _normal_masks(2) + [_odd_drawn_mask()])

    profile = estimate_run_normal_profile(run, markup_paths=markup)
    assert profile.normal_bank.source == SOURCE_REFERENCE
    # The run's sample measured the network's handwriting around the drawn cells.
    assert profile.normal_bank.tolerance_cells > 0
    assert profile.markup_report["used"] and profile.markup_report["frames"] == 3
    assert profile.markup_report["flagged"] == []

    without = estimate_run_normal_profile(run)
    assert without.markup_report is None and without.normal_bank.source == SOURCE_SESSION

    lonely = np.zeros((120, 120), dtype=np.uint8)
    lonely[40:80, 50:68] = 255  # one drawn cell is not a bank
    few = _write(tmp_path / "few", [lonely])
    fallback = estimate_run_normal_profile(run, markup_paths=few)
    assert not fallback.markup_report["used"]
    assert fallback.markup_report["reason"] == "too_few_markup_cells"
    # The run's bank, exactly as without markup.
    assert fallback.normal_bank.bank_id == without.normal_bank.bank_id
    assert build_markup_bank([extract_components(lonely)]).status == STATUS_INSUFFICIENT


def test_layer_markup_folder_reaches_the_bank_request_and_is_kept(tmp_path, qtbot) -> None:
    from karakal.app.main_window import KarakalWidget
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE, FOLDER_MARKUP_ROLE

    run = _write(tmp_path / "masks", _normal_masks(3))
    markup_folder = tmp_path / "markup"
    markup = _write(markup_folder, _normal_masks(2, start=50))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    item = presenter._append_folder_item(tmp_path / "masks", checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    specs = presenter._checked_model_specs()
    model_id = str(specs[0].model_id)
    records = tuple(
        FrameRecord(Path(path).stem, Path(path).name, model_mask_paths={model_id: path}) for path in run
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    state = presenter._current_tab_state()

    request = presenter._estimate_run_cell_reference_profile_for_state(state, None, model_id=model_id, records=records)
    assert request.markup_paths == ()

    # A finished run, then the layer gets a markup folder: the result is stale.
    state.grid_inspection_results_ready = True
    state.grid_inspection_algorithm_version = GRID_DAMAGE_ALGORITHM_VERSION
    state.grid_inspection_config_payload = presenter._grid_inspection_config_payload()
    state.grid_inspection_markup_signature = presenter._grid_markup_signature()
    assert not presenter._grid_matrix_needs_rerun(state)
    presenter._set_folder_item_markup(item, str(markup_folder))
    assert presenter._grid_matrix_needs_rerun(state)

    request = presenter._estimate_run_cell_reference_profile_for_state(state, None, model_id=model_id, records=records)
    assert list(request.markup_paths) == sorted(markup)
    assert len(request.mask_paths) == 3  # the run goes along for the fallback

    payload = presenter._build_folder_manager_payload()
    assert payload["folders"][0]["markup_path"] == str(markup_folder)
    assert item.data(FOLDER_MARKUP_ROLE) == str(markup_folder)

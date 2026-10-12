"""Cells the operator adds to the bank by hand: their shape (and size, when asked) becomes normal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

from karakal.core.cell_components import extract_components
from karakal.core.cell_error_analysis import CellErrorConfig, analyze_cell_errors, bank_verdict
from karakal.core.grid_anomaly import estimate_run_normal_profile, with_manual_cells
from karakal.core.normal_bank import (
    STATUS_INSUFFICIENT,
    ManualCell,
    NormalBank,
    apply_manual_cells,
    build_normal_bank,
    exemplar_view,
    shape_descriptor,
)
from karakal.core.synthetic_masks import SyntheticLayout, normal_frame
from karakal.ui.cell_bank_dialog import ORIGIN_MANUAL, CellBankDialog, CellQuery
from karakal.ui.i18n import Translator


def _masks(count: int, start: int = 0) -> list[np.ndarray]:
    return [normal_frame(SyntheticLayout(), seed=seed).mask for seed in range(start, start + count)]


def _bitten(corner: int = 10) -> np.ndarray:
    """One cell of the usual size (18 x 40) with a cut corner."""

    image = np.zeros((120, 120), dtype=np.uint8)
    image[30:70, 40:58] = 255
    image[30 : 30 + corner, 40 : 40 + corner][np.tri(corner, dtype=bool)[::-1]] = 0
    return image


def _wide() -> np.ndarray:
    image = np.zeros((120, 120), dtype=np.uint8)
    image[30:70, 30:62] = 255  # 32 px wide, the usual cell is 18
    return image


def _manual(mask: np.ndarray, *, size_normal: bool = False, cell_id: str = "c1", flagged: str = "") -> ManualCell:
    components = extract_components(mask)
    item = components.components[0]
    return ManualCell(
        cell_id=cell_id,
        view=exemplar_view(item, components.labels, "f.png"),
        descriptor=shape_descriptor(item, components.labels),
        size_normal=size_normal,
        flagged=flagged,
    )


def _run_bank() -> NormalBank:
    return build_normal_bank([extract_components(mask) for mask in _masks(4)])


def test_a_cell_added_by_hand_and_its_look_alikes_stop_being_marked() -> None:
    bank = _run_bank()
    assert analyze_cell_errors(_bitten(10), bank).class_counts().get("bad_geometry") == 1
    with_cell = apply_manual_cells(bank, [_manual(_bitten(10))])
    assert with_cell.manual_ids == ("c1",) and len(with_cell.operator_exemplars) == 1
    assert analyze_cell_errors(_bitten(10), with_cell).class_counts() == {}
    # A slightly different bite is near the added cell, so it is normal too.
    assert analyze_cell_errors(_bitten(9), with_cell).class_counts() == {}
    assert with_cell.bank_id != bank.bank_id


def test_size_becomes_normal_only_when_asked() -> None:
    bank = _run_bank()
    config = CellErrorConfig()
    cell = _manual(_wide())
    assert "size" in bank_verdict(bank, cell.view.width, cell.view.height, cell.descriptor, config)
    shape_only = apply_manual_cells(bank, [cell])
    assert bank_verdict(shape_only, cell.view.width, cell.view.height, cell.descriptor, config) == "size"
    sized = apply_manual_cells(bank, [_manual(_wide(), size_normal=True)])
    assert bank_verdict(sized, cell.view.width, cell.view.height, cell.descriptor, config) == ""
    assert analyze_cell_errors(_wide(), sized).class_counts() == {}


def test_payload_round_trip_and_a_bank_without_cells_stays_as_it_is() -> None:
    cell = _manual(_bitten(), size_normal=True, flagged="shape")
    restored = ManualCell.from_payload(cell.to_payload())
    assert restored is not None and restored.cell_id == cell.cell_id and restored.size_normal
    assert restored.flagged == "shape" and restored.view.mask_bits == cell.view.mask_bits
    assert np.allclose(restored.descriptor, cell.descriptor, atol=1e-2)
    assert ManualCell.from_payload({"id": "x"}) is None

    weak = NormalBank(source="session", status=STATUS_INSUFFICIENT)
    assert apply_manual_cells(weak, [cell]) is weak


def test_profiles_keep_the_bank_without_hand_cells(tmp_path) -> None:
    folder = tmp_path / "run"
    folder.mkdir()
    paths = []
    for index, mask in enumerate(_masks(3)):
        path = folder / f"F_{index}.png"
        cv2.imwrite(str(path), mask)
        paths.append(str(path))
    plain = estimate_run_normal_profile(paths)
    cell = _manual(_bitten())
    profile = estimate_run_normal_profile(paths, manual_cells=[cell])
    assert profile.base_bank.bank_id == plain.normal_bank.bank_id
    assert profile.normal_bank.manual_ids == ("c1",)
    # Removing the last cell gives the plain bank back without reading the masks again.
    assert with_manual_cells(profile, []).normal_bank.bank_id == plain.normal_bank.bank_id
    assert with_manual_cells(plain, [cell]).normal_bank.bank_id == profile.normal_bank.bank_id


def _query(mask: np.ndarray) -> CellQuery:
    components = extract_components(mask)
    item = components.components[0]
    return CellQuery("f.png", exemplar_view(item, components.labels, "f.png"), shape_descriptor(item, components.labels))


class _Profile:
    def __init__(self, bank: NormalBank) -> None:
        self.normal_bank = bank
        self.frame_id = "run:x"
        self.markup_report = None


def test_window_adds_and_removes_hand_cells(qtbot) -> None:
    bank = _run_bank()
    dialog = CellBankDialog(Translator("en").tr, layer_name="L", geometry_sensitivity=40, shape_limit=CellErrorConfig().effective_shape_limit())
    qtbot.addWidget(dialog)
    dialog.set_profile(_Profile(bank))
    dialog.set_query(_query(_bitten()))
    assert "beyond the limit" in dialog.query_label.text() and dialog.add_button.isEnabled()

    added = []
    dialog.addRequested.connect(lambda query, size_normal, flagged: added.append((query, size_normal, flagged)))
    dialog.size_normal_check.setChecked(True)
    dialog.add_button.click()
    assert len(added) == 1 and added[0][1] is True and added[0][2] == "shape"

    cell = _manual(_bitten(), flagged="shape")
    dialog.set_manual_cells([cell])
    dialog.set_profile(_Profile(apply_manual_cells(bank, [cell])))
    assert "within the limit" in dialog.query_label.text()
    assert not dialog.add_button.isEnabled()  # the same cell is in the bank already
    dialog.show_combo.setCurrentIndex(dialog.show_combo.findData("manual"))
    assert dialog.gallery_model.rowCount() == 1
    assert dialog.gallery_model.entries[0].origin == ORIGIN_MANUAL

    removed = []
    dialog.removeRequested.connect(removed.append)
    dialog.gallery_view.setCurrentIndex(dialog.gallery_model.index(0))
    assert dialog.remove_button.isEnabled()
    dialog.remove_button.click()
    assert removed == ["c1"]


def test_presenter_keeps_hand_cells_with_the_layer(tmp_path, qtbot) -> None:
    from karakal.app.main_window import KarakalWidget
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE, FOLDER_MANUAL_CELLS_ROLE

    folder = tmp_path / "masks"
    folder.mkdir()
    masks = _masks(3)
    masks[0] = masks[0].copy()
    masks[0][100:200, 560:600] = 0
    bitten = _bitten()
    masks[0][300:420, 500:620] = bitten  # a cell with a cut corner at (540..557, 330..369)
    run = []
    for index, mask in enumerate(masks):
        path = folder / f"F_{index}.png"
        cv2.imwrite(str(path), mask)
        run.append(str(path))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    item = presenter._append_folder_item(folder, checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    specs = presenter._checked_model_specs()
    model_id = str(specs[0].model_id)
    records = tuple(FrameRecord(Path(path).stem, Path(path).name, model_mask_paths={model_id: path}) for path in run)
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    state = presenter._current_tab_state()
    state.grid_inspection_results_ready = True
    state.grid_inspection_algorithm_version = GRID_DAMAGE_ALGORITHM_VERSION
    state.grid_inspection_config_payload = presenter._grid_inspection_config_payload()
    state.grid_inspection_markup_signature = presenter._grid_markup_signature()

    class FrameWindow:
        _record = records[0]

    # Right click on the bitten cell: «Normal cell — add to the bank».
    presenter._on_cell_bank_context(FrameWindow(), (548, 350), "add")
    stored = list(item.data(FOLDER_MANUAL_CELLS_ROLE))
    assert len(stored) == 1 and stored[0]["size_normal"] is False and stored[0]["flagged"] == "shape"
    assert presenter._grid_matrix_needs_rerun(state)
    profile = state.grid_inspection_run_profiles[model_id]
    assert len(profile.normal_bank.manual_ids) == 1
    request = presenter._estimate_run_cell_reference_profile_for_state(state, None, model_id=model_id, records=records)
    assert [cell.cell_id for cell in request.manual_cells] == [stored[0]["id"]]
    assert presenter._build_folder_manager_payload()["folders"][0]["manual_cells"] == stored
    dialog = presenter._cell_bank_dialogs[model_id]
    # The window opens on that cell: it is in the bank now (texts follow the app language).
    assert dialog._query is not None and not dialog.add_button.isEnabled()

    presenter._remove_manual_cell(model_id, stored[0]["id"])
    assert list(item.data(FOLDER_MANUAL_CELLS_ROLE)) == []
    assert state.grid_inspection_run_profiles[model_id].normal_bank.manual_ids == ()
    assert not presenter._grid_matrix_needs_rerun(state)
    dialog.close()

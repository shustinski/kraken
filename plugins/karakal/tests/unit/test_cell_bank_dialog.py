"""Bank window: what the bank is made of, its examples, and a frame cell put next to it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

from karakal.core.cell_components import extract_components
from karakal.core.cell_error_analysis import CellErrorConfig
from karakal.core.grid_anomaly import estimate_run_normal_profile
from karakal.core.normal_bank import (
    ORIGIN_OPERATOR,
    ORIGIN_RUN,
    build_markup_bank,
    build_normal_bank,
    cell_at,
    exemplar_unusualness,
    exemplar_view,
    nearest_exemplars,
    shape_descriptor,
)
from karakal.core.synthetic_masks import SyntheticLayout, normal_frame, synthetic_frame
from karakal.ui.cell_bank_dialog import CellBankDialog, CellQuery
from karakal.ui.i18n import Translator


def _masks(count: int, start: int = 0) -> list[np.ndarray]:
    return [normal_frame(SyntheticLayout(), seed=seed).mask for seed in range(start, start + count)]


def _write(folder: Path, masks: list[np.ndarray]) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, mask in enumerate(masks):
        path = folder / f"FRAME_{index:04d}.png"
        assert cv2.imwrite(str(path), mask)
        paths.append(str(path))
    return paths


def test_banks_keep_a_picture_of_every_example() -> None:
    sets = [extract_components(mask) for mask in _masks(3)]
    run = build_normal_bank(sets, frame_names=["a.png", "b.png", "c.png"])
    assert len(run.exemplar_views) == len(run.exemplars) > 0
    assert {view.frame for view in run.exemplar_views} <= {"a.png", "b.png", "c.png"}
    view = run.exemplar_views[0]
    assert view.mask().shape == (view.height, view.width) and view.mask().sum() == view.area
    assert len(run.normal_sizes) >= len(run.exemplars)

    markup = build_markup_bank(sets)
    assert len(markup.operator_views) == len(markup.operator_exemplars) == markup.candidates
    unusual = exemplar_unusualness(markup)
    assert unusual[ORIGIN_OPERATOR].shape == (len(markup.operator_exemplars),)
    assert unusual[ORIGIN_RUN].shape == (0,)


def test_nearest_examples_and_the_cell_under_a_box() -> None:
    mask = synthetic_frame({"cut_corner": 1}, seed=3).mask
    components = extract_components(mask)
    item = next(item for item in components.components if not item.touches_border)
    x, y, w, h = item.bbox
    assert cell_at(components, (x, y, w, h)).label == item.label
    # A box slightly off the cell still finds it by its pixels.
    assert cell_at(components, (x + w - 3, y, 6, h)).label == item.label
    assert cell_at(components, (0, 0, 1, 1)) is None or mask[0, 0] > 0

    bank = build_markup_bank([extract_components(m) for m in _masks(2)])
    found = nearest_exemplars(bank, shape_descriptor(item, components.labels), count=5)
    assert len(found) == 5 and all(origin == ORIGIN_OPERATOR for origin, _index, _d in found)
    assert [distance for _o, _i, distance in found] == sorted(distance for _o, _i, distance in found)


def _query_of(mask: np.ndarray) -> CellQuery:
    components = extract_components(mask)
    item = next(item for item in components.components if not item.touches_border)
    return CellQuery("frame.png", exemplar_view(item, components.labels, "frame.png"), shape_descriptor(item, components.labels))


def test_window_shows_the_bank_and_a_cell_next_to_it(tmp_path, qtbot) -> None:
    run = _write(tmp_path / "run", _masks(3, start=20))
    markup = _write(tmp_path / "markup", _masks(2))
    profile = estimate_run_normal_profile(run, markup_paths=markup)
    t = Translator("en").tr
    dialog = CellBankDialog(t, layer_name="L", geometry_sensitivity=40, shape_limit=CellErrorConfig().effective_shape_limit())
    qtbot.addWidget(dialog)
    dialog.set_profile(profile)

    bank = profile.normal_bank
    assert dialog.gallery_model.rowCount() == len(bank.operator_views)
    assert "markup" in dialog.status_label.text()
    assert "1.0" not in dialog.width_title.text() and "px" in dialog.width_title.text()
    texts = [dialog.summary_form.itemAt(row, dialog.summary_form.ItemRole.FieldRole) for row in range(dialog.summary_form.rowCount())]
    assert any(item is not None and "every drawn cell is normal" in item.widget().text() for item in texts)

    dialog.set_query(_query_of(_masks(1, start=40)[0]))
    assert dialog.query_group.isVisibleTo(dialog)
    assert "within the limit" in dialog.query_label.text()
    assert dialog.nearest_model.rowCount() == 8

    broken = np.zeros((120, 120), dtype=np.uint8)
    broken[30:70, 40:58] = 255
    broken[30:50, 40:50] = 0  # a large bite out of the corner
    dialog.set_query(_query_of(broken))
    assert "beyond the limit" in dialog.query_label.text()


def test_window_builds_the_bank_in_a_thread(tmp_path, qtbot) -> None:
    run = _write(tmp_path / "run", _masks(3))
    built = []
    dialog = CellBankDialog(Translator("en").tr, layer_name="L", geometry_sensitivity=40, shape_limit=2.0)
    qtbot.addWidget(dialog)
    dialog.build(lambda: estimate_run_normal_profile(run), on_built=built.append)
    assert "Building" in dialog.status_label.text()
    qtbot.waitUntil(lambda: dialog.gallery_model.rowCount() > 0, timeout=60000)
    assert len(built) == 1 and "frames of the run" in dialog.status_label.text()


def test_layer_settings_open_the_bank_and_keep_it_for_the_layer(tmp_path, qtbot) -> None:
    from karakal.app.main_window import KarakalWidget
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE

    run = _write(tmp_path / "masks", _masks(3))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    item = presenter._append_folder_item(tmp_path / "masks", checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    specs = presenter._checked_model_specs()
    model_id = str(specs[0].model_id)
    records = tuple(FrameRecord(Path(path).stem, Path(path).name, model_mask_paths={model_id: path}) for path in run)
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    state = presenter._current_tab_state()

    presenter._open_cell_bank(model_id)
    dialog = presenter._cell_bank_dialogs[model_id]  # deletes itself on close
    qtbot.waitUntil(lambda: model_id in state.grid_inspection_run_profiles, timeout=60000)
    qtbot.waitUntil(lambda: dialog.gallery_model.rowCount() > 0, timeout=10000)

    # A second open shows the same window; the cell of a frame box is found in the layer's mask.
    presenter._open_cell_bank(model_id)
    assert presenter._cell_bank_dialogs[model_id] is dialog
    components = extract_components(cv2.imread(run[0], cv2.IMREAD_GRAYSCALE))
    cell = next(item for item in components.components if not item.touches_border)
    query = presenter._cell_bank_query(run[0], cell.bbox)
    assert query is not None and query.view.area == cell.area

    class FrameWindow:  # what the frame window gives: its frame and the defect in the yellow frame
        _record = records[0]

        @staticmethod
        def selected_grid_defect_bbox():
            return cell.bbox

    presenter._open_cell_bank_from_details(FrameWindow())
    assert dialog._query is not None and dialog._query.view.area == cell.area
    assert dialog.query_group.isVisibleTo(dialog)
    dialog.close()

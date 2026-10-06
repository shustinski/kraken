"""The operator's minimum debris size: main-window control, analysis config, saved setting, frame preview."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget
from karakal.app.presenter import KarakalPresenter
from karakal.core.grid_anomaly import CONTOUR_AREA_FEATURE, DEBRIS_MIN_AREA_PX
from karakal.ui.debris_min_area_control import DebrisMinAreaControl
from karakal.ui.i18n import Translator


def _widget(tmp_path, qtbot) -> KarakalWidget:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    return widget


def test_default_size_keeps_previous_results(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    assert widget.grid_debris_min_area_control.value() == int(DEBRIS_MIN_AREA_PX)
    payload = widget._presenter._grid_inspection_config_payload()
    assert KarakalPresenter._grid_damage_config_from_payload(payload).debris_min_area_px == int(DEBRIS_MIN_AREA_PX)


def test_main_window_value_reaches_analysis_and_is_saved(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    widget.grid_debris_min_area_control.set_value(80)
    widget.grid_debris_min_area_control.valueChanged.emit(80)

    payload = widget._presenter._grid_inspection_config_payload()
    assert payload["debris_min_area_px"] == 80
    assert KarakalPresenter._grid_damage_config_from_payload(payload).debris_min_area_px == 80

    restored = _widget(tmp_path, qtbot)
    assert restored.grid_debris_min_area_control.value() == 80
    assert restored._presenter._grid_inspection_config_payload()["debris_min_area_px"] == 80


def test_frame_window_value_moves_main_window_control(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    details_control = DebrisMinAreaControl(Translator("ru").tr)
    qtbot.addWidget(details_control)
    presenter._details_dialogs.append(SimpleNamespace(grid_debris_min_area_control=details_control))

    presenter._on_grid_debris_min_area_changed(45, details_control)

    assert widget.grid_debris_min_area_control.value() == 45
    assert presenter._grid_inspection_config_payload()["debris_min_area_px"] == 45
    presenter._details_dialogs.clear()


def test_control_keeps_slider_and_box_in_step(qtbot) -> None:
    control = DebrisMinAreaControl(Translator("ru").tr)
    qtbot.addWidget(control)
    seen: list[int] = []
    control.valueChanged.connect(seen.append)
    control._spin.setValue(120)
    assert control._slider.value() == 120
    control._slider.setValue(30)
    assert control.value() == 30
    assert seen == [120, 30]
    control.set_value(7)
    assert seen == [120, 30]


def test_frame_preview_redecide_drops_debris_below_size(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)

    @dataclass(frozen=True)
    class _Cell:
        reasons: tuple = ()
        status: str = "normal"
        score: float = 0.0
        feature_snapshot: tuple = ()

    @dataclass(frozen=True)
    class _Result:
        per_cell_results: tuple = field(default_factory=tuple)

    small, large = (
        _Cell(
            reasons=("small_artifact",),
            status="broken",
            score=0.9,
            feature_snapshot=(("debris_score", 0.97), (CONTOUR_AREA_FEATURE, area)),
        )
        for area in (20.0, 90.0)
    )
    tuning = widget._presenter._grid_tuning_values()
    _result, cells = widget._presenter._redecide_prepared_frame(
        _Result((small, large)), tuning, {}, debris_min_area=50.0
    )
    assert [tuple(item.reasons) for item in cells] == [(), ("small_artifact",)]


def _frame_dialog(qtbot, cells):
    from dataclasses import replace

    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.details_dialog import ExtendFrameDetailsDialog
    from karakal.ui.i18n import set_current_language

    set_current_language("ru")
    record = FrameRecord("frame-1", "frame_0001.jpg")
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        allowed_result_kinds=("grid_cell_defects",),
    )
    qtbot.addWidget(dialog)
    result = replace(dialog._empty_grid_cell_analysis_result(), image_width=400, image_height=400)
    dialog.apply_grid_inspection_preview(replace(result, per_cell_results=tuple(cells)))
    dialog._grid_scene_scale = lambda: (1.0, 1.0)
    # A record without models has no layer list; the frame window shows only cell defects.
    dialog._selected_result_kind = lambda: "grid_cell_defects"
    return dialog


def _cell(bbox, reasons, area=None):
    from karakal.core.grid_anomaly import GridCellAnalysisResult

    snapshot = () if area is None else ((CONTOUR_AREA_FEATURE, float(area)),)
    return GridCellAnalysisResult(
        row=0,
        col=0,
        bbox=bbox,
        centroid=(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
        contour_id=1,
        status="broken" if reasons else "normal",
        score=0.9 if reasons else 0.0,
        reasons=reasons,
        feature_snapshot=snapshot,
    )


def test_hover_over_debris_shows_its_size_and_the_limit(qtbot, monkeypatch) -> None:
    from PyQt6.QtCore import QPoint, QPointF

    from karakal.ui import details_dialog as module

    debris = _cell((100, 100, 6, 6), ("small_artifact",), area=37.4)
    geometry = _cell((200, 200, 30, 30), ("broken_geometry",), area=800)
    dialog = _frame_dialog(qtbot, (debris, geometry))
    dialog.grid_debris_min_area_control.set_value(24)

    shown: list[str] = []
    monkeypatch.setattr(module.QToolTip, "showText", lambda _pos, text, *_args: shown.append(text))
    hidden: list[bool] = []
    monkeypatch.setattr(module.QToolTip, "hideText", lambda: hidden.append(True))

    dialog._on_overlay_hover(QPointF(103, 103), QPoint(0, 0))
    assert shown == ["Мусор: 37 px (мин. размер 24 px)"]

    # Other defect types and empty space hide the hint.
    dialog._on_overlay_hover(QPointF(210, 210), QPoint(0, 0))
    dialog._on_overlay_hover(QPointF(350, 50), QPoint(0, 0))
    assert shown == ["Мусор: 37 px (мин. размер 24 px)"]
    assert len(hidden) == 2


def test_hover_skips_hidden_debris_and_results_without_area(qtbot) -> None:
    from PyQt6.QtCore import QPointF

    debris = _cell((100, 100, 6, 6), ("small_artifact",), area=37)
    dialog = _frame_dialog(qtbot, (debris,))
    dialog.grid_error_type_checks["small_artifact"].setChecked(False)
    assert dialog._debris_cell_at(QPointF(103, 103)) is None

    old = _cell((100, 100, 6, 6), ("small_artifact",))
    assert _frame_dialog(qtbot, (old,))._debris_hover_text(old) == ""

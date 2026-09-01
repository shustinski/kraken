from __future__ import annotations

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QPushButton, QSlider
from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisScaleMode

from karakal.app.main_window import KarakalWidget
from karakal.app.presenter import KarakalPresenter
from karakal.core.domain import BuildOptions, BuildResult, FrameIdentity, FrameRecord
from karakal.plugin.matrix_adapter import KarakalMatrixDataSource, project_build_result
from karakal.plugin.result_adapter import build_analysis_result_manifest
from karakal.ui.details_dialog import ExtendFrameDetailsDialog
from karakal.ui.i18n import Translator
from karakal.ui.matrix_view import MatrixLayoutConfig, MatrixLegendWidget, MatrixListWidget
from karakal.ui.ui_constants import (
    DEFAULT_GRADIENT_NAME,
    GRID_INSPECTION_DAMAGE_METRIC_KEY,
    GRID_INSPECTION_FIXED_TUNING,
)


def test_quick_setup_is_visible_and_advanced_controls_start_collapsed(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)

    profile_buttons = [
        button
        for button in widget.analysis_setup_panel.findChildren(QPushButton)
        if bool(button.property("analysisProfile"))
    ]

    assert widget.analysis_setup_panel.isVisibleTo(widget)
    assert len(profile_buttons) == 3
    assert widget.folders_info_label.isVisibleTo(widget)
    assert not widget.pair_matrix_group.isChecked()
    assert widget.pair_matrix_body.isHidden()
    assert Translator().tr("pairs.summary", count=0) in widget.pair_matrix_group.title()
    assert not widget.analysis_settings_group.isChecked()
    assert widget.matrix_gradient_combo.count() >= 4
    assert widget.matrix_gradient_combo.currentData() == DEFAULT_GRADIENT_NAME


def test_pair_matrix_panel_expansion_is_persisted(tmp_path, qtbot) -> None:
    settings_path = tmp_path / "karakal.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)

    widget.pair_matrix_group.setChecked(True)

    assert not widget.pair_matrix_body.isHidden()
    assert widget._presenter._build_build_settings_payload()["pair_panel_expanded"] is True

    widget.close()
    restored_widget = KarakalWidget(
        settings=QSettings(str(settings_path), QSettings.Format.IniFormat)
    )
    qtbot.addWidget(restored_widget)

    assert restored_widget.pair_matrix_group.isChecked()
    assert not restored_widget.pair_matrix_body.isHidden()


def test_gradient_selection_updates_matrix_views(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    gradient_name = "red_white_blue"

    widget.matrix_gradient_combo.setCurrentIndex(
        widget.matrix_gradient_combo.findData(gradient_name)
    )

    assert widget._presenter._build_build_settings_payload()["gradient_name"] == gradient_name
    assert all(
        view.color_scale_info().gradient_name == gradient_name
        for view in widget.grid_inspection_matrix_views.values()
    )


def test_grid_analysis_tuning_is_fixed_and_not_exposed_to_operator(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)

    assert not widget._grid_inspection_tuning_group.findChildren(QSlider)
    assert not hasattr(widget, "grid_strictness_slider")

    payload = widget._presenter._grid_inspection_config_payload()
    assert {key: payload[key] for key, _value in GRID_INSPECTION_FIXED_TUNING} == dict(
        GRID_INSPECTION_FIXED_TUNING
    )

    legacy_payload = {key: 0 for key, _value in GRID_INSPECTION_FIXED_TUNING}
    fixed_payload = dict(GRID_INSPECTION_FIXED_TUNING)
    assert KarakalPresenter._grid_damage_config_from_payload(
        legacy_payload
    ) == KarakalPresenter._grid_damage_config_from_payload(fixed_payload)


def test_matrix_legend_exposes_distribution_and_raw_range(qtbot) -> None:
    view = MatrixListWidget()
    legend = MatrixLegendWidget()
    qtbot.addWidget(view)
    qtbot.addWidget(legend)
    view.colorScaleChanged.connect(legend.set_scale_info)
    view.set_layout_config(MatrixLayoutConfig(mode="indexed_grid", total_frames=3, frames_per_row=3))
    view.set_records(
        [
            FrameRecord("1", "1", score=0.46, absolute_score=91.0, score_percentile=10.0, score_ready=True),
            FrameRecord("2", "2", score=0.50, absolute_score=92.0, score_percentile=50.0, score_ready=True),
            FrameRecord("3", "3", score=0.54, absolute_score=93.0, score_percentile=90.0, score_ready=True),
        ]
    )

    info = view.color_scale_info()

    assert info.sample_count == 3
    assert info.raw_low == 91.0
    assert info.raw_high == 93.0
    assert "P5" in legend.stats_label.text()


def test_frame_details_separates_overview_from_layer_controls(qtbot) -> None:
    record = FrameRecord("frame-1", "Frame 1")
    result = BuildResult(records=(record,), options=BuildOptions())
    dialog = ExtendFrameDetailsDialog(record, result)
    qtbot.addWidget(dialog)
    translator = Translator()

    assert dialog.details_control_tabs.count() == 2
    assert dialog.details_control_tabs.tabText(0) == translator.tr("details.tab.overview")
    assert dialog.details_control_tabs.tabText(1) == translator.tr("details.tab.layers")


def test_zero_grid_damage_uses_best_score_color(qtbot) -> None:
    record = FrameRecord("frame-1", "Frame 1")
    result = BuildResult(records=(record,), options=BuildOptions())
    dialog = ExtendFrameDetailsDialog(
        record,
        result,
        allowed_result_kinds=("grid_cell_defects",),
    )
    qtbot.addWidget(dialog)
    dialog._refresh_result_kind_options("grid_cell_defects")

    assert dialog._comparison_score_metric_key() == GRID_INSPECTION_DAMAGE_METRIC_KEY
    assert "background-color: #1f5f3b" in dialog._comparison_score_style(0.0)
    assert "background-color: #8c2f39" in dialog._comparison_score_style(1.0)
    assert dialog._comparison_score_text(0.0) == f"{dialog._t('score.level.good')} 0.0000"


def test_shared_matrix_adapter_preserves_coordinates_and_heatmap_metadata() -> None:
    result = BuildResult(
        records=(
            FrameRecord(
                "frame-1",
                "Frame 1",
                identity=FrameIdentity(frame_id=1, tile_x=4, tile_y=2),
                score=0.75,
                absolute_score=75.0,
                score_percentile=80.0,
                score_ready=True,
            ),
        ),
        options=BuildOptions(),
        scores_computed=True,
        best_match_key="frame-1",
    )

    projection = project_build_result(
        result,
        metric_key="overall_polygon_score",
        score_view_mode="relative",
    )
    source = KarakalMatrixDataSource(projection)
    item = projection.items[0]

    assert (item.x, item.y) == (5, 3)
    assert item.metadata["heatmap_value"] == 0.75
    assert item.metadata["reference"] is True
    assert source.session.width == 5
    assert source.session.height == 3

    manifest = build_analysis_result_manifest(
        job_id="job-1",
        project_id="project-1",
        profile=AnalysisProfileKind.MODEL_COMPARISON,
        build_result=result,
        metric_key="overall_polygon_score",
        scale_mode=AnalysisScaleMode.ABSOLUTE,
    )

    assert manifest.frames[0].frame_id == "frame-1"
    assert manifest.frames[0].metrics[0].raw_value == 75.0
    assert manifest.scales[0].low == 0.0
    assert manifest.scales[0].high == 1.0

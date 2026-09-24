from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QPushButton, QSlider
from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisScaleMode

from karakal.app.main_window import KarakalWidget
from karakal.app.presenter import KarakalPresenter
from karakal.core.domain import BuildOptions, BuildResult, FrameAnalysisSummary, FrameIdentity, FrameRecord
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.single_result_risk import RiskReason, SingleResultRiskSummary
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
    assert len(profile_buttons) == 4
    assert not hasattr(widget, "folders_info_label")
    assert not hasattr(widget.analysis_setup_panel, "intro_label")
    assert not widget.pair_matrix_group.isChecked()
    assert widget.pair_matrix_body.isHidden()
    assert Translator().tr("pairs.summary", count=0) in widget.pair_matrix_group.title()
    assert not widget.analysis_settings_group.isChecked()
    assert widget.matrix_gradient_combo.count() >= 4
    assert widget.matrix_gradient_combo.currentData() == DEFAULT_GRADIENT_NAME


def test_single_result_profile_shows_only_risk_controls(tmp_path, qtbot) -> None:
    settings_path = tmp_path / "karakal.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)

    widget._presenter._on_analysis_profile_changed(AnalysisProfileKind.SINGLE_RESULT_RISK.value)

    assert widget.matrix_score_view_combo.currentData() == "absolute"
    assert widget.metric_combo.currentData() == "single_result_risk_score"
    assert widget.matrix_score_view_combo.findData("percentile") >= 0
    assert not widget.pair_matrix_group.isVisibleTo(widget)
    assert widget._matrix_comparison_target_row.isHidden()
    assert widget._matrix_geometry_row.isHidden()
    assert widget._matrix_polygon_compare_profile_row.isHidden()
    assert not widget._matrix_single_result_sensitivity_row.isHidden()

    widget.single_result_sensitivity_combo.setCurrentIndex(
        widget.single_result_sensitivity_combo.findData("strict")
    )
    widget._presenter._persist_state()
    restored = KarakalWidget(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(restored)

    assert restored._presenter._analysis_profile == AnalysisProfileKind.SINGLE_RESULT_RISK
    assert restored.single_result_sensitivity_combo.currentData() == "strict"
    assert restored.matrix_score_view_combo.currentData() == "absolute"


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


def test_grid_analysis_tuning_sliders_start_at_balanced_preset(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)

    assert not widget._grid_inspection_tuning_group.findChildren(QSlider)
    assert not hasattr(widget, "grid_tuning_sliders")
    assert widget.grid_tuning_preset_combo.currentData() == "balanced"
    assert "unified" in widget.grid_inspection_matrix_views

    payload = widget._presenter._grid_inspection_config_payload()
    assert {key: payload[key] for key, _value in GRID_INSPECTION_FIXED_TUNING} == dict(
        GRID_INSPECTION_FIXED_TUNING
    )
    assert payload["requested_layers"] == ["confidence", "binary", "comparison"]
    assert payload["display_layer"] == "unified"

    soft = KarakalPresenter._grid_damage_config_from_payload({"fill_sensitivity": 0, "merge_sensitivity": 0})
    strict = KarakalPresenter._grid_damage_config_from_payload({"fill_sensitivity": 100, "merge_sensitivity": 100})
    assert soft.filled_ratio_delta > strict.filled_ratio_delta
    assert soft.merged_size_ratio > strict.merged_size_ratio


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

    view.set_score_view_mode("percentile")
    assert view.color_scale_info().score_view_mode == "percentile"
    assert view._display_score(view._records[0]) == 0.1


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

    percentile_projection = project_build_result(
        result,
        metric_key="overall_polygon_score",
        score_view_mode="percentile",
    )
    assert percentile_projection.items[0].metadata["heatmap_value"] == 0.8

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


def test_single_result_details_show_reasons_and_anomaly_layers(tmp_path, qtbot) -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(mask).save(str(mask_path))
    risk = SingleResultRiskSummary(
        total_risk=72.0,
        structure_risk=80.0,
        batch_outlier_risk=None,
        source_alignment_risk=50.0,
        source_mask_agreement_risk=None,
        confidence_risk=40.0,
        sensitivity="balanced",
        evidence=("mask", "original", "confidence"),
        reasons=(
            RiskReason("tiny_components", "high", 0.8, (0.1, 0.1, 0.2, 0.2)),
            RiskReason("unsupported_boundary", "medium", 0.6, (0.3, 0.3, 0.2, 0.2)),
            RiskReason("low_confidence", "medium", 0.5, (0.5, 0.5, 0.2, 0.2)),
        ),
        component_weights={"mask_structure_risk": 0.6, "source_alignment_risk": 0.3, "confidence_risk": 0.1},
        component_contributions={
            "mask_structure_risk": 48.0,
            "source_alignment_risk": 15.0,
            "confidence_risk": 4.0,
        },
        feature_values={"area_fraction": 0.1},
    )
    record = FrameRecord(
        "frame-1",
        "Frame 1",
        first_path=str(mask_path),
        second_path=str(mask_path),
        model_mask_paths={"model": str(mask_path)},
        score_ready=True,
        absolute_score=72.0,
        summary=FrameAnalysisSummary(0.0, 0.0, 0.8, 0.72, single_result_risk=risk),
    )
    result = BuildResult(records=(record,), options=BuildOptions(), scores_computed=True)
    dialog = ExtendFrameDetailsDialog(
        record,
        result,
        allowed_result_kinds=("risk_structure", "risk_source", "risk_confidence"),
    )
    qtbot.addWidget(dialog)
    dialog._payload["single_result_risk"] = risk
    dialog._refresh_result_kind_options("risk_structure")
    dialog._refresh_comparison_panel()

    assert dialog.result_kind_combo.count() == 3
    assert "risk: 72.0/100" in dialog.comparison_metrics_label.text()
    assert "tiny_components" in dialog.comparison_events_label.text()
    for kind in ("risk_structure", "risk_source", "risk_confidence"):
        assert not dialog._single_result_risk_pixmap(kind).isNull()


def test_grid_details_exposes_confidence_layer_when_uploaded(tmp_path, qtbot) -> None:
    confidence_path = tmp_path / "confidence.png"
    image = _grayscale_array_to_qimage(np.full((32, 32), 180, dtype=np.uint8))
    assert image.save(str(confidence_path))
    record = FrameRecord(
        "frame-1",
        "Frame 1",
        model_mask_paths={"model": str(confidence_path)},
        model_prob_paths={"model": str(confidence_path)},
    )
    result = BuildResult(records=(record,), options=BuildOptions())
    dialog = ExtendFrameDetailsDialog(
        record,
        result,
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
        grid_inspection_source_path=str(confidence_path),
    )
    qtbot.addWidget(dialog)
    dialog._payload["model_confidence_output_available"] = {"model": True}
    dialog._payload["model_output_probabilities"] = {"model": np.full((32, 32), 0.7, dtype=np.float32)}
    dialog._refresh_result_kind_options("grid_cell_defects")
    dialog._update_result_controls()

    assert dialog._grid_confidence_available()
    assert not dialog.second_source_layer_row.isHidden()
    assert dialog.second_source_layer_title.text() == dialog._t("details.grid_confidence_layer")
    assert dialog.result_layer_title.text() == dialog._t("details.grid_cell_defects")
    # Without an original photo the base already shows the model mask — no duplicate layer.
    assert dialog.first_source_layer_row.isHidden()


def test_grid_details_exposes_model_output_layer_with_original(tmp_path, qtbot) -> None:
    original_path = tmp_path / "original.png"
    mask_path = tmp_path / "mask.png"
    confidence_path = tmp_path / "confidence.png"
    assert _grayscale_array_to_qimage(np.full((32, 32), 40, dtype=np.uint8)).save(str(original_path))
    assert _grayscale_array_to_qimage(np.full((32, 32), 255, dtype=np.uint8)).save(str(mask_path))
    assert _grayscale_array_to_qimage(np.full((32, 32), 180, dtype=np.uint8)).save(str(confidence_path))
    record = FrameRecord(
        "frame-1",
        "Frame 1",
        original_path=str(original_path),
        model_mask_paths={"model": str(mask_path)},
        model_prob_paths={"model": str(confidence_path)},
    )
    result = BuildResult(records=(record,), options=BuildOptions())
    dialog = ExtendFrameDetailsDialog(
        record,
        result,
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
        grid_inspection_source_path=str(original_path),
    )
    qtbot.addWidget(dialog)
    dialog._payload["model_confidence_output_available"] = {"model": True}
    dialog._payload["model_output_probabilities"] = {"model": np.full((32, 32), 0.7, dtype=np.float32)}
    dialog._payload["model_source_grays"] = {"model": np.ones((32, 32), dtype=np.float32)}
    dialog._refresh_result_kind_options("grid_cell_defects")
    dialog._update_result_controls()
    dialog._refresh_scene(reset_view=False)

    assert dialog._grid_has_original()
    assert dialog._grid_model_output_layer_available()
    assert not dialog.first_source_layer_row.isHidden()
    assert dialog.first_source_layer_title.text() == dialog._t("details.model_output_layer")
    assert not dialog.second_source_layer_row.isHidden()
    assert not dialog.first_source_item.pixmap().isNull()
    assert dialog.first_source_item.isVisible()


def test_confidence_overlay_prefers_uploaded_jpeg_grayscale(tmp_path, qtbot) -> None:
    confidence = np.linspace(0, 255, 32 * 32, dtype=np.float32).reshape(32, 32)
    confidence_path = tmp_path / "confidence.png"
    image = _grayscale_array_to_qimage(np.clip(np.rint(confidence), 0, 255).astype(np.uint8))
    assert image.save(str(confidence_path))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(mask).save(str(mask_path))
    record = FrameRecord(
        "frame-1",
        "Frame 1",
        model_mask_paths={"model": str(mask_path)},
        model_prob_paths={"model": str(confidence_path)},
    )
    result = BuildResult(records=(record,), options=BuildOptions())
    dialog = ExtendFrameDetailsDialog(
        record,
        result,
        session_view_state={"preferred_model_id": "model"},
    )
    qtbot.addWidget(dialog)
    dialog._payload["original_gray"] = np.zeros((32, 32), dtype=np.float32)
    dialog._payload["model_masks"] = {"model": mask.astype(bool)}
    dialog._payload["model_probabilities"] = {"model": (mask.astype(np.float32) / 255.0)}
    dialog._payload["model_output_probabilities"] = {"model": (confidence / 255.0).astype(np.float32)}
    dialog._payload["model_confidence_output_available"] = {"model": True}

    pixmap = dialog._confidence_overlay_pixmap("model")
    assert not pixmap.isNull()
    assert pixmap.width() == 32 and pixmap.height() == 32
    # Must come from uploaded confidence grayscale, not the binary mask.
    source_gray = dialog._confidence_source_grayscale("model")
    assert source_gray is not None
    assert float(np.mean(source_gray)) > 10.0
    assert float(np.std(source_gray)) > 10.0

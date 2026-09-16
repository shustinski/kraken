from __future__ import annotations

import pytest

from kraken_manager.presentation.qt.analysis_ui import (
    SCENARIO_DIRECT,
    SCENARIO_SUBTRACT_C,
    SCENARIO_XOR_A,
    SCENARIO_XOR_C,
    AnalysisRunsPanel,
    AnalysisSetupDialog,
)
from kraken_manager.presentation.qt.pages import ProjectWorkspacePage


@pytest.mark.parametrize(
    ("scenario", "expected_operation", "required_sources"),
    (
        (SCENARIO_DIRECT, "source", {"A", "B"}),
        (SCENARIO_XOR_A, "xor", {"A", "B"}),
        (SCENARIO_XOR_C, "xor", {"A", "B", "C"}),
        (SCENARIO_SUBTRACT_C, "subtract", {"A", "B", "C"}),
    ),
)
def test_analysis_setup_builds_supported_recipes(qtbot, scenario, expected_operation, required_sources) -> None:
    dialog = AnalysisSetupDialog(
        (("model-a", "Model A"), ("model-b", "Model B"), ("model-c", "Model C")),
        selected_frame_count=12,
    )
    qtbot.addWidget(dialog)
    dialog.scenario.setCurrentIndex(dialog.scenario.findData(scenario))

    recipe = dialog.recipe()

    assert recipe.expression.left.operation == expected_operation
    assert recipe.expression.source_keys == required_sources
    assert dialog.start_button.isEnabled()
    assert dialog.configuration()["frame_count"] == 12


def test_analysis_setup_requires_selection_and_distinct_a_b(qtbot) -> None:
    dialog = AnalysisSetupDialog((("same", "Only model"),), selected_frame_count=0)
    qtbot.addWidget(dialog)

    assert not dialog.start_button.isEnabled()
    assert "Выборка" in dialog.validation_label.text()
    assert "разными" in dialog.validation_label.text()


def test_analysis_runs_panel_accepts_progressive_rows_and_emits_actions(qtbot) -> None:
    panel = AnalysisRunsPanel()
    qtbot.addWidget(panel)
    panel.set_runs(
        (
            {
                "run_id": "run-1",
                "state": "running",
                "progress": "1/2",
                "models": "A, B, C",
                "recipe": "XOR(A,B) ↔ C",
                "created_at": "now",
            },
        )
    )
    panel.set_results(({"frame_id": "frame-1", "x": 1, "y": 2, "status": "ready", "raw_value": 0.5},))
    panel.run_table.selectRow(0)
    panel.result_table.selectRow(0)

    with qtbot.waitSignal(panel.renderMapRequested) as signal:
        panel.map_button.click()

    assert signal.args == ["run-1", "frame-1"]
    assert panel.run_table.rowCount() == 1
    assert panel.result_table.rowCount() == 1


def test_project_workspace_exposes_operator_analysis_actions(qtbot) -> None:
    page = ProjectWorkspacePage()
    qtbot.addWidget(page)

    assert page.evaluate_result_button.text() == "Оценить результат"
    assert not page.analysis_runs_panel.isVisible()
    page.analysis_history_button.click()
    assert not page.analysis_runs_panel.isHidden()

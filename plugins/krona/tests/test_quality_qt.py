from pathlib import Path

import pytest

from krona.bootstrap import ApplicationContainer
from krona.presentation.qt.ui_strings import load_ui_strings
from krona.presentation.qt.window import EdifViewerWindow


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_real_viewer_displays_connectivity_uncertainty(qtbot, language):
    container = ApplicationContainer()
    strings = load_ui_strings(language)
    path = Path(__file__).resolve().parents[1] / "resources/test_edifs/explicit_cmos_inverter.edf"
    window = EdifViewerWindow(
        container.load_scene_data,
        container.extract_logic_functions,
        initial_path=path,
        initial_language=language,
        ui_strings=strings,
    )
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()
    assert strings["logic.status.unverified"] in window.logic_status_label.text()
    assert window.logic_truth_table.rowCount() == 2
    window.close()


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_viewer_displays_conflicts_without_invented_logic(qtbot, language):
    container = ApplicationContainer()
    strings = load_ui_strings(language)
    path = Path(__file__).resolve().parents[1] / "resources/test_edifs/e129_model_logic.EDF"
    window = EdifViewerWindow(
        container.load_scene_data, container.extract_logic_functions,
        initial_path=path, initial_language=language, ui_strings=strings,
    )
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()
    assert strings["logic.status.unverified"] in window.logic_status_label.text()
    assert strings["logic.status.incomplete"].split("{errors}")[0] in window.logic_status_label.text()
    assert window.logic_truth_table.rowCount() == 0
    assert window.logic_highlight_combo.count() == 0
    assert window._logic_data["meta"]["logic_status"] == "unknown_incomplete_netlist"
    window.close()

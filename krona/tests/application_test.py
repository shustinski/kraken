from pathlib import Path

from logic_analyzer.bootstrap import logic_functions_to_dict, parse_to_dict
from logic_analyzer.application.use_cases import LoadSceneData
from logic_analyzer.infrastructure.edif_repository import EdifRepository


def test_parse_to_dict_returns_top_level_data():
    result = parse_to_dict(Path("test_edifs/e1_model.EDF"))
    assert result["design"] == "SCHEMATIC1"
    assert result["library"] == "CRPROJECT"
    assert result["cell"] == "SCHEMATIC1"
    assert result["view"] == "SCHEMATIC1_SCH"
    assert len(result["instances"]) > 0
    assert len(result["nets"]) > 0


def test_logic_functions_extraction_returns_output_functions():
    result = logic_functions_to_dict(Path("test_edifs/e1_model.EDF"))
    assert len(result["inputs"]) > 0
    assert "OUT1" in result["outputs"]
    out1 = result["outputs"]["OUT1"]
    assert out1["net"]
    assert len(out1["truth_table"]) == 2 ** len(result["inputs"])


def test_logic_function_fixture_e129_matches_expected_expression():
    result = logic_functions_to_dict(Path("test_edifs/e129_model_logic.EDF"))
    assert result["inputs"] == ["IN1", "IN2", "IN3"]
    out1 = result["outputs"]["OUT1"]
    assert all(row["value"] in {0, 1} for row in out1["truth_table"])
    assert out1["simplified_expression"] == "!(IN1 & IN3 | IN2)"


def test_input_detection_fixture_e140_detects_inputs():
    result = logic_functions_to_dict(Path("test_edifs/e140_report.EDF"))
    assert len(result["inputs"]) > 0
    assert all(name.startswith("IN") for name in result["inputs"])


def test_output_with_unknown_truth_rows_has_no_printed_function():
    result = logic_functions_to_dict(Path("test_edifs/e1_model.EDF"))
    out1 = result["outputs"]["OUT1"]
    assert any(row["value"] is None for row in out1["truth_table"])
    assert out1["sum_of_products"] is None
    assert out1["simplified_expression"] is None


def test_scene_data_contains_parser_diagnostics():
    scene_data = LoadSceneData(EdifRepository()).execute(Path("test_edifs/e1_model.EDF"))
    assert isinstance(scene_data.diagnostics, list)
    assert len(scene_data.diagnostics) > 0
    assert all(hasattr(item, "severity") and hasattr(item, "message") for item in scene_data.diagnostics)
    assert all(item.severity in {"error", "warning", "info"} for item in scene_data.diagnostics)

from pathlib import Path

from logic_analyzer.bootstrap import parse_to_dict


def test_parse_to_dict_returns_top_level_data():
    result = parse_to_dict(Path("test_edifs/e1_model.EDF"))
    assert result["design"] == "SCHEMATIC1"
    assert result["library"] == "CRPROJECT"
    assert result["cell"] == "SCHEMATIC1"
    assert result["view"] == "SCHEMATIC1_SCH"
    assert len(result["instances"]) > 0
    assert len(result["nets"]) > 0

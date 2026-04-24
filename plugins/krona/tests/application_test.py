from pathlib import Path

from krona.application.logic_functions import ExtractLogicFunctions, _Transistor
from krona.bootstrap import logic_functions_to_dict, parse_to_dict
from krona.application.use_cases import LoadSceneData
from krona.infrastructure.edif_repository import EdifRepository


def test_parse_to_dict_returns_top_level_data():
    result = parse_to_dict(Path("resources/test_edifs/e1_model.EDF"))
    assert result["design"] == "SCHEMATIC1"
    assert result["library"] == "CRPROJECT"
    assert result["cell"] == "SCHEMATIC1"
    assert result["view"] == "SCHEMATIC1_SCH"
    assert len(result["instances"]) > 0
    assert len(result["nets"]) > 0


def test_logic_functions_extraction_returns_output_functions():
    result = logic_functions_to_dict(Path("resources/test_edifs/e1_model.EDF"))
    assert len(result["inputs"]) > 0
    assert "OUT1" in result["outputs"]
    assert "sequential_elements" in result
    assert isinstance(result["sequential_elements"], list)
    assert "sequential_count" in result.get("meta", {})
    out1 = result["outputs"]["OUT1"]
    assert out1["net"]
    assert len(out1["truth_table"]) == 2 ** len(result["inputs"])


def test_logic_function_fixture_e129_matches_expected_expression():
    result = logic_functions_to_dict(Path("resources/test_edifs/e129_model_logic.EDF"))
    assert result["inputs"] == ["IN1", "IN2", "IN3"]
    out1 = result["outputs"]["OUT1"]
    assert all(row["value"] in {0, 1} for row in out1["truth_table"])
    assert out1["simplified_expression"] == "!(IN1 & IN3 | IN2)"


def test_input_detection_fixture_e140_detects_inputs():
    result = logic_functions_to_dict(Path("resources/test_edifs/e140_report.EDF"))
    assert len(result["inputs"]) > 0
    assert all(name.startswith("IN") for name in result["inputs"])


def test_output_with_unknown_truth_rows_has_no_printed_function():
    result = logic_functions_to_dict(Path("resources/test_edifs/e1_model.EDF"))
    out1 = result["outputs"]["OUT1"]
    assert any(row["value"] is None for row in out1["truth_table"])
    assert out1["sum_of_products"] is None
    assert out1["simplified_expression"] is None


def test_e1_fixture_topologically_detects_d_flip_flop():
    result = logic_functions_to_dict(Path("resources/test_edifs/e1_model.EDF"))

    sequential = result.get("sequential_elements", [])
    flip_flops = [item for item in sequential if item.get("kind") == "flip_flop"]
    assert flip_flops
    ff = flip_flops[0]
    assert ff.get("subtype") == "D"
    triggering = ff.get("triggering", {})
    assert triggering.get("mode") == "edge"
    assert triggering.get("edge") in {"rising", "falling"}
    assert triggering.get("net") == result.get("meta", {}).get("input_nets", {}).get("IN1")
    data_inputs = list(ff.get("data_inputs", []))
    assert data_inputs
    data_fn = data_inputs[0].get("function", {})
    assert data_fn.get("depends_on") == ["IN2", "IN3"]
    assert data_fn.get("mux", {}).get("kind") == "mux2"
    assert data_fn.get("mux", {}).get("select") in {"IN2", "IN3"}


def test_e2_fixture_topologically_detects_pulse_triggered_d_flip_flop():
    result = logic_functions_to_dict(Path("resources/test_edifs/e2_model.EDF"))

    sequential = result.get("sequential_elements", [])
    flip_flops = [item for item in sequential if item.get("kind") == "flip_flop"]
    assert flip_flops
    ff = flip_flops[0]
    assert ff.get("subtype") == "D"
    triggering = ff.get("triggering", {})
    assert triggering.get("mode") == "edge"
    assert triggering.get("net") == result.get("meta", {}).get("input_nets", {}).get("IN3")
    assert triggering.get("edge") == "falling"
    assert ff.get("topology", {}).get("kind") == "pulse_triggered_single_latch"


def test_scene_data_contains_parser_diagnostics():
    scene_data = LoadSceneData(EdifRepository()).execute(Path("resources/test_edifs/e1_model.EDF"))
    assert isinstance(scene_data.diagnostics, list)
    assert len(scene_data.diagnostics) > 0
    assert all(hasattr(item, "severity") and hasattr(item, "message") for item in scene_data.diagnostics)
    assert all(item.severity in {"error", "warning", "info"} for item in scene_data.diagnostics)


def _tx(name: str, kind: str, gate: str, source: str, drain: str) -> _Transistor:
    return _Transistor(instance_name=name, kind=kind, gate=gate, source=source, drain=drain)


def test_sequential_detection_topologically_identifies_latch_enable_level():
    extractor = ExtractLogicFunctions(repository=None)  # type: ignore[arg-type]
    transistors = [
        # Latch storage core: cross-coupled CMOS inverters A<->B
        _tx("P1", "pmos", "A", "VDD", "B"),
        _tx("N1", "nmos", "A", "GND", "B"),
        _tx("P2", "pmos", "B", "VDD", "A"),
        _tx("N2", "nmos", "B", "GND", "A"),
        # Control phase inverter EN -> ENB
        _tx("P3", "pmos", "EN", "VDD", "ENB"),
        _tx("N3", "nmos", "EN", "GND", "ENB"),
        # Input TG D <-> A enabled by EN (pmos driven by ENB)
        _tx("P4", "pmos", "ENB", "D", "A"),
        _tx("N4", "nmos", "EN", "D", "A"),
        # Feedback TG B <-> A enabled by ENB (pmos driven by EN)
        _tx("P5", "pmos", "EN", "B", "A"),
        _tx("N5", "nmos", "ENB", "B", "A"),
    ]
    nets = {"VDD", "GND", "A", "B", "D", "EN", "ENB"}
    nets_by_canonical = {name: [name] for name in nets}

    elements = extractor._sequential_elements_topology(
        transistors=transistors,
        power_nets={"VDD"},
        ground_nets={"GND"},
        nets_by_canonical=nets_by_canonical,
        input_net_by_var={},
        output_nets={"B"},
        output_net_by_name={},
    )

    latches = [item for item in elements if item.get("kind") == "latch"]
    assert latches
    latch = latches[0]
    assert latch["subtype"] == "D"
    assert latch["triggering"]["mode"] == "level"
    assert latch["triggering"]["level"] == "high"
    assert latch["triggering"]["net"] == "EN"
    controls = {(item["role"], item["net"]): item for item in latch["control_signals"]}
    assert ("enable", "EN") in controls
    assert controls[("enable", "EN")]["activation"] == {"mode": "level", "level": "high"}
    assert ("phase_complement", "ENB") in controls


def test_sequential_detection_topologically_identifies_async_set_reset_on_latch():
    extractor = ExtractLogicFunctions(repository=None)  # type: ignore[arg-type]
    transistors = [
        # Latch storage core: cross-coupled CMOS inverters A<->B
        _tx("P1", "pmos", "A", "VDD", "B"),
        _tx("N1", "nmos", "A", "GND", "B"),
        _tx("P2", "pmos", "B", "VDD", "A"),
        _tx("N2", "nmos", "B", "GND", "A"),
        # Control phase inverter EN -> ENB
        _tx("P3", "pmos", "EN", "VDD", "ENB"),
        _tx("N3", "nmos", "EN", "GND", "ENB"),
        # Input TG D <-> A enabled by EN (pmos driven by ENB)
        _tx("P4", "pmos", "ENB", "D", "A"),
        _tx("N4", "nmos", "EN", "D", "A"),
        # Feedback TG B <-> A enabled by ENB (pmos driven by EN)
        _tx("P5", "pmos", "EN", "B", "A"),
        _tx("N5", "nmos", "ENB", "B", "A"),
        # Async set/reset forcing B (non-inverted output)
        _tx("PSET", "pmos", "SETB", "VDD", "B"),
        _tx("NRST", "nmos", "RST", "B", "GND"),
    ]
    nets = {"VDD", "GND", "A", "B", "D", "EN", "ENB", "SETB", "RST"}
    nets_by_canonical = {name: [name] for name in nets}

    elements = extractor._sequential_elements_topology(
        transistors=transistors,
        power_nets={"VDD"},
        ground_nets={"GND"},
        nets_by_canonical=nets_by_canonical,
        input_net_by_var={"IN1": "EN", "IN2": "D", "IN3": "SETB", "IN4": "RST"},
        output_nets={"B"},
        output_net_by_name={"OUT1": "B"},
    )

    latches = [item for item in elements if item.get("kind") == "latch"]
    assert latches
    latch = latches[0]
    controls = {(str(item.get("role")), str(item.get("net"))): item for item in latch.get("control_signals", [])}
    assert ("set", "SETB") in controls
    assert controls[("set", "SETB")]["activation"] == {"mode": "level", "level": "low"}
    assert controls[("set", "SETB")].get("timing") == "asynchronous"
    assert ("reset", "RST") in controls
    assert controls[("reset", "RST")]["activation"] == {"mode": "level", "level": "high"}
    assert controls[("reset", "RST")].get("timing") == "asynchronous"


def test_sequential_detection_topologically_identifies_master_slave_flip_flop_edge():
    extractor = ExtractLogicFunctions(repository=None)  # type: ignore[arg-type]
    transistors = [
        # Global clock inverter CLK -> CLKB
        _tx("PCLK", "pmos", "CLK", "VDD", "CLKB"),
        _tx("NCLK", "nmos", "CLK", "GND", "CLKB"),
        # Master latch core (M_A <-> M_B)
        _tx("MP1", "pmos", "M_A", "VDD", "M_B"),
        _tx("MN1", "nmos", "M_A", "GND", "M_B"),
        _tx("MP2", "pmos", "M_B", "VDD", "M_A"),
        _tx("MN2", "nmos", "M_B", "GND", "M_A"),
        # Master input TG (transparent when CLKB=1 => CLK=0)
        _tx("MP3", "pmos", "CLK", "D", "M_A"),
        _tx("MN3", "nmos", "CLKB", "D", "M_A"),
        # Master feedback TG (transparent when CLK=1)
        _tx("MP4", "pmos", "CLKB", "M_B", "M_A"),
        _tx("MN4", "nmos", "CLK", "M_B", "M_A"),
        # Slave latch core (S_A <-> S_B)
        _tx("SP1", "pmos", "S_A", "VDD", "S_B"),
        _tx("SN1", "nmos", "S_A", "GND", "S_B"),
        _tx("SP2", "pmos", "S_B", "VDD", "S_A"),
        _tx("SN2", "nmos", "S_B", "GND", "S_A"),
        # Slave input TG (transparent when CLK=1)
        _tx("SP3", "pmos", "CLKB", "M_B", "S_A"),
        _tx("SN3", "nmos", "CLK", "M_B", "S_A"),
        # Slave feedback TG (transparent when CLKB=1 => CLK=0)
        _tx("SP4", "pmos", "CLK", "S_B", "S_A"),
        _tx("SN4", "nmos", "CLKB", "S_B", "S_A"),
    ]
    nets = {"VDD", "GND", "CLK", "CLKB", "D", "M_A", "M_B", "S_A", "S_B"}
    nets_by_canonical = {name: [name] for name in nets}

    elements = extractor._sequential_elements_topology(
        transistors=transistors,
        power_nets={"VDD"},
        ground_nets={"GND"},
        nets_by_canonical=nets_by_canonical,
        input_net_by_var={},
        output_nets={"S_B"},
        output_net_by_name={},
    )

    flip_flops = [item for item in elements if item.get("kind") == "flip_flop"]
    assert flip_flops
    ff = flip_flops[0]
    assert ff["subtype"] == "D"
    assert ff["triggering"]["mode"] == "edge"
    assert ff["triggering"]["edge"] == "rising"
    assert ff["triggering"]["net"] == "CLK"
    assert any(str(out.get("net")) == "S_B" for out in ff.get("outputs", []))

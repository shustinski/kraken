from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from krona.analysis.structural.model import DeviceKind
from krona.analysis.structural.parsers import EdifStructuralParser
from krona.application.logic_functions import ExtractLogicFunctions, _Transistor
from krona.bootstrap import logic_functions_to_dict, structural_analysis_to_dict
from krona.domain.connectivity import ExplicitConnectivity
from krona.domain.netlist import Connection, Instance, Net, Point, TopLevelNetlist, Wire
from krona.infrastructure.edif_repository import EdifRepository


FIXTURES = Path(__file__).resolve().parents[1] / "resources/test_edifs"


def test_coincident_or_nearby_drawing_points_do_not_merge_different_nets():
    netlist = TopLevelNetlist("d", "l", "c", "v", [Instance("U1"), Instance("U2")], [
        Net("LEFT", [Connection("A", "U1")], [Wire([Point(0, 0), Point(100, 0)])]),
        Net("RIGHT", [Connection("A", "U2")], [Wire([Point(0, 0), Point(101, 0)])]),
    ])
    connectivity = ExplicitConnectivity.from_netlist(netlist)
    aliases = ExtractLogicFunctions(None)._build_net_alias_map(netlist)
    assert aliases == {"LEFT": "LEFT", "RIGHT": "RIGHT"}
    assert connectivity.pins == {"U1": {"A": "LEFT"}, "U2": {"A": "RIGHT"}}
    assert connectivity.issues == ()


def test_explicit_joined_net_connects_endpoints_without_any_drawing():
    parsed = EdifRepository().read_top_level_netlist(FIXTURES / "explicit_cmos_inverter.edf")
    assert all(not net.wires for net in parsed.nets)
    index = ExplicitConnectivity.from_netlist(parsed)
    assert index.pins["P1"]["GATE"] == index.pins["N1"]["GATE"] == index.top_ports["A"]
    assert index.pins["P1"]["DRAIN"] == index.pins["N1"]["DRAIN"] == index.top_ports["Y"]
    assert index.directions["A"] == "input"
    assert index.directions["Y"] == "output"
    assert index.issues == ()


def test_conflicting_pin_memberships_remain_conflicts_in_both_declaration_orders(tmp_path):
    source = (FIXTURES / "explicit_cmos_inverter.edf").read_text()
    # A second explicit net assigns the same physical terminal another identity.
    for i, replacement in enumerate([
        "(net OTHER (joined (portRef GATE (instanceRef N1))))\n          (net INPUT",
        "(net INPUT",
    ]):
        text = source.replace("(net INPUT", replacement, 1)
        if i:
            text = text.replace("(net LOW", "(net OTHER (joined (portRef GATE (instanceRef N1))))\n          (net LOW", 1)
        path = tmp_path / f"conflict_{i}.edf"
        path.write_text(text)
        result = logic_functions_to_dict(path)
        assert result["meta"]["net_aliases"]["OTHER"] == "OTHER"
        assert result["meta"]["net_aliases"]["INPUT"] == "INPUT"
        issues = result["meta"]["connectivity"]["issues"]
        assert {"code": "conflicting_port_nets", "severity": "error", "instance": "N1",
                "port": "GATE", "nets": ["INPUT", "OTHER"]} in issues
        assert [row["value"] for row in result["outputs"]["Y"]["truth_table"]] == [None, None]
        circuit = EdifStructuralParser(EdifRepository()).parse(path)
        assert circuit.devices["N1"].pins == {}


def test_short_pin_aliases_cannot_hide_a_conflicting_gate():
    netlist = TopLevelNetlist("d", "l", "c", "v", [Instance("M1", cell="NMOS")], [
        Net("A", [Connection("G", "M1")]), Net("B", [Connection("GATE", "M1")]),
    ])
    index = ExplicitConnectivity.from_netlist(netlist)
    assert index.invalid_instances == frozenset({"M1"})
    assert any(issue["code"] == "conflicting_port_nets" for issue in index.issues)
    assert "GATE" not in index.pins.get("M1", {})


def test_geometry_cannot_repair_missing_explicit_gate(tmp_path):
    source = FIXTURES / "explicit_cmos_inverter.edf"
    text = source.read_text().replace("(portRef GATE (instanceRef N1))", "")
    # Add identical geometry to every net: it must not restore the removed pin.
    text = text.replace("(net INPUT", "(net GEOMETRY_ONLY (figure WIRE (path (pointList (pt 0 0) (pt 10 0))))) (net INPUT", 1)
    path = tmp_path / "missing.edf"
    path.write_text(text)
    result = logic_functions_to_dict(path)
    assert result["meta"]["unresolved_instances"] == ["N1"]
    assert result["meta"]["geometry_used_for_connectivity"] is False
    assert all(row["value"] is None for row in result["outputs"]["Y"]["truth_table"])
    circuit = EdifStructuralParser(EdifRepository()).parse(path)
    assert "GATE" not in circuit.devices["N1"].pins


def test_geometric_power_symbol_does_not_supply_an_unjoined_net():
    netlist = TopLevelNetlist("d", "l", "c", "v", [], [
        Net("SIGNAL", wires=[Wire([Point(0, 0), Point(10, 0)])]),
    ])
    page = "(page P (portImplementation VDD (connectLocation (figure X (dot (pt 0 0))))))"
    assert ExtractLogicFunctions(None)._supply_nets(page, netlist) == (set(), set())
    parser = EdifStructuralParser(EdifRepository())
    assert parser._detect_supplies(page, netlist, {(0, 0): {"SIGNAL"}}) == {"power": set(), "ground": set()}


def test_unmodeled_explicit_device_prevents_claiming_a_complete_logic_function(tmp_path):
    text = (FIXTURES / "explicit_cmos_inverter.edf").read_text()
    text = text.replace("(instance P1", "(instance X1 (viewRef SYMBOL (cellRef UNKNOWN_CELL (libraryRef PRIMITIVES)))) (instance P1", 1)
    text = text.replace("(portRef Y)", "(portRef Y) (portRef TERMINAL (instanceRef X1))", 1)
    path = tmp_path / "unsupported.edf"
    path.write_text(text)
    result = logic_functions_to_dict(path)
    assert result["meta"]["unsupported_instances"] == ["X1"]
    assert [r["value"] for r in result["outputs"]["Y"]["truth_table"]] == [None, None]


def test_ambiguous_gate_selection_never_prefers_input_or_peer_net():
    extractor = ExtractLogicFunctions(None)
    assert extractor._select_gate_net(["A", "B"], {"A"}, {}, "M1") is None
    assert extractor._select_gate_net(["INTERNAL"], {"A"}, {"M2": ["INTERNAL", "A"]}, "M1") == "INTERNAL"


def test_possible_opposite_drive_keeps_level_unknown_and_rail_alias_conflict_is_unknown():
    extractor = ExtractLogicFunctions(None)
    tx = [_Transistor("P1", "pmos", "GND", "VDD", "OUT"),
          _Transistor("N1", "nmos", "FLOATING", "GND", "OUT")]
    assert extractor._evaluate_output_state("OUT", {}, {}, {"VDD"}, {"GND"}, tx) is None
    assert extractor._evaluate_output_state("OUT", {}, {}, {"OUT"}, {"OUT"}, []) is None


def test_parser_structural_pmos_kind_and_provenance_are_preserved():
    path = FIXTURES / "explicit_cmos_inverter.edf"
    repository = EdifRepository()
    circuit = EdifStructuralParser(repository).parse(path)
    assert circuit.devices["P1"].kind is DeviceKind.PMOS
    assert circuit.devices["N1"].kind is DeviceKind.NMOS
    report = structural_analysis_to_dict(path)
    for metadata in [report["metadata"], logic_functions_to_dict(path)["meta"]]:
        assert metadata["source_sha256"] == sha256(path.read_bytes()).hexdigest()
        assert metadata["analysis_basis"] == "explicit_edif_joined"
        assert metadata["electrical_connectivity"] == "unverified"
        assert metadata["geometry_used_for_connectivity"] is False
        assert metadata["connectivity"]["top_ports"]["Y"] == "OUTPUT"
    assert not [issue for issue in repository.read_diagnostics(path) if issue.severity == "error"]


def test_structural_analysis_accepts_complete_explicit_master_slave_dff():
    report = structural_analysis_to_dict(FIXTURES / "explicit_master_slave_dff.edf")
    kinds = {structure["kind"] for structure in report["recognized_structures"]}
    assert {"master_slave_dff", "edge_triggered_dff"} <= kinds
    assert report["metadata"]["connectivity"]["issues"] == []


def test_duplicate_nets_and_missing_instance_are_diagnostics_not_connections():
    netlist = TopLevelNetlist("d", "l", "c", "v", [Instance("M1", cell="NMOS")], [
        Net("A", [Connection("GATE", "M1")]), Net("A", [Connection("DRAIN", "MISSING")]),
    ])
    index = ExplicitConnectivity.from_netlist(netlist)
    assert {i["code"] for i in index.issues} >= {"duplicate_net", "conflicting_port_nets", "unknown_instance"}
    assert "M1" in index.invalid_instances


def test_structural_analysis_rejects_source_revision_change(tmp_path):
    path = tmp_path / "mutating.edf"
    path.write_bytes((FIXTURES / "explicit_cmos_inverter.edf").read_bytes())

    class ChangingRepository(EdifRepository):
        def read_top_level_netlist(self, source):
            parsed = super().read_top_level_netlist(source)
            source.write_bytes(source.read_bytes() + b"\n")
            return parsed

    with pytest.raises(ValueError, match="source changed"):
        EdifStructuralParser(ChangingRepository()).parse(path)


def test_standalone_cli_exports_explicit_logic_and_source_revision():
    path = FIXTURES / "explicit_cmos_inverter.edf"
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "plugins/krona/src")]))
    completed = subprocess.run([sys.executable, "-m", "krona", str(path), "--logic"],
                               cwd=root, env=environment, capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(completed.stdout)
    assert [r["value"] for r in result["outputs"]["Y"]["truth_table"]] == [1, 0]
    assert result["meta"]["source_sha256"] == sha256(path.read_bytes()).hexdigest()

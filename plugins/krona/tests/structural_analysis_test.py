from __future__ import annotations

from krona.analysis.structural.engine import StructuralSequentialAnalyzer
from krona.analysis.structural.model import CircuitDevice, CircuitModel, CircuitNet, DeviceDomain, DeviceKind


def _net(name: str) -> CircuitNet:
    return CircuitNet(name=name, is_power=name == "VDD", is_ground=name == "GND")


def test_structural_analyzer_detects_gated_d_latch_from_switch_level_scc():
    nets = {name: _net(name) for name in ["VDD", "GND", "A", "B", "D", "EN", "ENB"]}
    devices = {
        "P1": CircuitDevice("P1", DeviceKind.PMOS, DeviceDomain.SWITCH, pins={"GATE": "A", "SOURCE": "VDD", "DRAIN": "B"}),
        "N1": CircuitDevice("N1", DeviceKind.NMOS, DeviceDomain.SWITCH, pins={"GATE": "A", "SOURCE": "GND", "DRAIN": "B"}),
        "P2": CircuitDevice("P2", DeviceKind.PMOS, DeviceDomain.SWITCH, pins={"GATE": "B", "SOURCE": "VDD", "DRAIN": "A"}),
        "N2": CircuitDevice("N2", DeviceKind.NMOS, DeviceDomain.SWITCH, pins={"GATE": "B", "SOURCE": "GND", "DRAIN": "A"}),
        "P3": CircuitDevice("P3", DeviceKind.PMOS, DeviceDomain.SWITCH, pins={"GATE": "EN", "SOURCE": "VDD", "DRAIN": "ENB"}),
        "N3": CircuitDevice("N3", DeviceKind.NMOS, DeviceDomain.SWITCH, pins={"GATE": "EN", "SOURCE": "GND", "DRAIN": "ENB"}),
        "P4": CircuitDevice("P4", DeviceKind.PMOS, DeviceDomain.SWITCH, pins={"GATE": "ENB", "SOURCE": "D", "DRAIN": "A"}),
        "N4": CircuitDevice("N4", DeviceKind.NMOS, DeviceDomain.SWITCH, pins={"GATE": "EN", "SOURCE": "D", "DRAIN": "A"}),
        "P5": CircuitDevice("P5", DeviceKind.PMOS, DeviceDomain.SWITCH, pins={"GATE": "EN", "SOURCE": "B", "DRAIN": "A"}),
        "N5": CircuitDevice("N5", DeviceKind.NMOS, DeviceDomain.SWITCH, pins={"GATE": "ENB", "SOURCE": "B", "DRAIN": "A"}),
    }
    circuit = CircuitModel("synthetic", "latch", nets, devices, tuple(devices), {}, {})

    report = StructuralSequentialAnalyzer().analyze(circuit)
    kinds = [item.kind.value for item in report.recognized_structures]

    assert "gated_d_latch" in kinds
    assert report.graph.storage_scc_count >= 1


def test_structural_analyzer_detects_master_slave_and_edge_dff():
    names = ["VDD", "GND", "CLK", "CLKB", "D", "M_A", "M_B", "S_A", "S_B"]
    nets = {name: _net(name) for name in names}
    items = [
        ("PCLK", "PMOS", "CLK", "VDD", "CLKB"),
        ("NCLK", "NMOS", "CLK", "GND", "CLKB"),
        ("MP1", "PMOS", "M_A", "VDD", "M_B"),
        ("MN1", "NMOS", "M_A", "GND", "M_B"),
        ("MP2", "PMOS", "M_B", "VDD", "M_A"),
        ("MN2", "NMOS", "M_B", "GND", "M_A"),
        ("MP3", "PMOS", "CLK", "D", "M_A"),
        ("MN3", "NMOS", "CLKB", "D", "M_A"),
        ("MP4", "PMOS", "CLKB", "M_B", "M_A"),
        ("MN4", "NMOS", "CLK", "M_B", "M_A"),
        ("SP1", "PMOS", "S_A", "VDD", "S_B"),
        ("SN1", "NMOS", "S_A", "GND", "S_B"),
        ("SP2", "PMOS", "S_B", "VDD", "S_A"),
        ("SN2", "NMOS", "S_B", "GND", "S_A"),
        ("SP3", "PMOS", "CLKB", "M_B", "S_A"),
        ("SN3", "NMOS", "CLK", "M_B", "S_A"),
        ("SP4", "PMOS", "CLK", "S_B", "S_A"),
        ("SN4", "NMOS", "CLKB", "S_B", "S_A"),
    ]
    devices = {
        name: CircuitDevice(
            name,
            DeviceKind[kind],
            DeviceDomain.SWITCH,
            pins={"GATE": gate, "SOURCE": source, "DRAIN": drain},
        )
        for name, kind, gate, source, drain in items
    }
    circuit = CircuitModel("synthetic", "msff", nets, devices, tuple(devices), {}, {})

    report = StructuralSequentialAnalyzer().analyze(circuit)
    kinds = {item.kind.value for item in report.recognized_structures}

    assert "master_slave_dff" in kinds
    assert "edge_triggered_dff" in kinds


def test_structural_analyzer_decodes_tff_from_dff_xor_feedback():
    nets = {name: _net(name) for name in ["CLK", "Q", "D", "T"]}
    devices = {
        "XOR1": CircuitDevice(
            "XOR1",
            DeviceKind.XOR,
            DeviceDomain.GATE,
            pins={"A": "Q", "B": "T", "Y": "D"},
        ),
        "FF1": CircuitDevice(
            "FF1",
            DeviceKind.FF_D,
            DeviceDomain.SEQUENTIAL_MACRO,
            pins={"D": "D", "Q": "Q", "CLK": "CLK"},
        ),
    }
    circuit = CircuitModel("synthetic", "tff", nets, devices, tuple(devices), {}, {})

    report = StructuralSequentialAnalyzer().analyze(circuit)
    kinds = [item.kind.value for item in report.recognized_structures]

    assert "d_flip_flop" in kinds
    assert "t_flip_flop" in kinds
    assert "sat_extension_plan" in report.as_dict()


def test_structural_analyzer_marks_synchronous_reset_feature_for_macro_ff():
    nets = {name: _net(name) for name in ["CLK", "Q", "D", "SCLR"]}
    devices = {
        "FF1": CircuitDevice(
            "FF1",
            DeviceKind.FF_D,
            DeviceDomain.SEQUENTIAL_MACRO,
            pins={"D": "D", "Q": "Q", "CLK": "CLK", "SCLR": "SCLR"},
        ),
    }
    circuit = CircuitModel("synthetic", "ff_sync_rst", nets, devices, tuple(devices), {}, {})

    report = StructuralSequentialAnalyzer().analyze(circuit)
    dffs = [item for item in report.recognized_structures if item.kind.value == "d_flip_flop"]

    assert dffs
    assert any("sync_reset" in [feature.value for feature in item.features] for item in dffs)
    assert any(
        control.get("role") == "reset" and control.get("timing") == "synchronous"
        for item in dffs
        for control in item.controls
    )

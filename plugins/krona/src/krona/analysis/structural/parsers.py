from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from krona.domain.connectivity import ExplicitConnectivity
from pathlib import Path
import re
from typing import Protocol

from krona.application.ports import NetlistRepository
from krona.domain.netlist import TopLevelNetlist

from .model import CircuitDevice, CircuitModel, CircuitNet, DeviceDomain, DeviceKind


class NetlistParser(Protocol):
    def parse(self, source: str | Path) -> CircuitModel:
        ...


class SpiceStructuralParser:
    """
    Parser-layer placeholder for future switch-level SPICE support.

    The structural analysis engine is parser-agnostic; only this adapter is pending.
    """

    def parse(self, source: str | Path) -> CircuitModel:
        raise NotImplementedError("SPICE parser is not implemented yet. Use EDIF parser adapter.")


class VerilogStructuralParser:
    """
    Parser-layer placeholder for future gate-level Verilog support.
    """

    def parse(self, source: str | Path) -> CircuitModel:
        raise NotImplementedError("Verilog parser is not implemented yet. Use EDIF parser adapter.")


@dataclass(frozen=True)
class _DeviceSignature:
    kind: DeviceKind
    domain: DeviceDomain
    attributes: dict[str, str]


class EdifStructuralParser:
    """
    Converts existing EDIF top-level netlist representation into an analysis-oriented
    circuit model with pin-role inference and basic device classification.
    """

    def __init__(self, repository: NetlistRepository):
        self._repository = repository

    def parse(self, source: str | Path) -> CircuitModel:
        path = Path(source)
        source_digest = sha256(path.read_bytes()).hexdigest()
        netlist = self._repository.read_top_level_netlist(path)
        connectivity = ExplicitConnectivity.from_netlist(netlist)
        top_ports = {port: [net] for port, net in connectivity.top_ports.items()}
        supply_sets = self._detect_supplies("", netlist, {})
        pins_by_instance = self._pins_from_net_connections(netlist)

        devices: dict[str, CircuitDevice] = {}
        device_order: list[str] = []
        for instance in netlist.instances:
            explicit_pins = dict(pins_by_instance.get(instance.name, {}))
            merged_pins = explicit_pins
            signature = self._classify_device(instance.name, instance.cell, instance.view, merged_pins)

            device = CircuitDevice(
                name=instance.name,
                kind=signature.kind,
                domain=signature.domain,
                cell_name=instance.cell,
                view_name=instance.view,
                library_name=instance.library,
                pins=merged_pins,
                attributes=signature.attributes,
            )
            devices[instance.name] = device
            device_order.append(instance.name)

        nets: dict[str, CircuitNet] = {}
        for net in netlist.nets:
            port_name = self._find_top_port_name_for_net(net.name, top_ports)
            direction = connectivity.directions.get(port_name) if port_name else None
            nets[net.name] = CircuitNet(
                name=net.name,
                is_power=net.name in supply_sets["power"],
                is_ground=net.name in supply_sets["ground"],
                top_port_name=port_name,
                port_direction=direction,
            )

        if sha256(path.read_bytes()).hexdigest() != source_digest:
            raise ValueError("EDIF source changed during structural analysis")
        return CircuitModel(
            source_format="edif",
            design_name=netlist.design,
            nets=nets,
            devices=devices,
            device_order=tuple(device_order),
            top_ports={port: nets_for_port[0] for port, nets_for_port in top_ports.items() if nets_for_port},
            metadata={
                "analysis_basis": "explicit_edif_joined",
                "electrical_connectivity": "unverified",
                "source_artifact": str(path.resolve()),
                "source_sha256": source_digest,
                "connectivity": connectivity.provenance(),
                "geometry_used_for_connectivity": False,
                "library": netlist.library,
                "cell": netlist.cell,
                "view": netlist.view,
                "device_count": len(devices),
                "net_count": len(nets),
            },
        )

    @staticmethod
    def _merge_pins(explicit_pins: dict[str, str], inferred_pins: dict[str, str]) -> dict[str, str]:
        return dict(explicit_pins)

    @staticmethod
    def _pins_from_net_connections(netlist: TopLevelNetlist) -> dict[str, dict[str, str]]:
        connectivity = ExplicitConnectivity.from_netlist(netlist)
        return {name: pins for name, pins in connectivity.pins.items() if name not in connectivity.invalid_instances}

    @staticmethod
    def _build_net_geometry(
        netlist: TopLevelNetlist,
    ) -> tuple[dict[tuple[int, int], set[str]], dict[str, list[tuple[int, int, int, int]]]]:
        point_to_nets: dict[tuple[int, int], set[str]] = {}
        segments_by_net: dict[str, list[tuple[int, int, int, int]]] = {}
        for net in netlist.nets:
            segments: list[tuple[int, int, int, int]] = []
            for wire in net.wires:
                for point in wire.points:
                    point_to_nets.setdefault((point.x, point.y), set()).add(net.name)
                for start, end in zip(wire.points, wire.points[1:]):
                    segments.append((start.x, start.y, end.x, end.y))
            segments_by_net[net.name] = segments
        return point_to_nets, segments_by_net

    @staticmethod
    def _top_port_map(top_page: str, point_to_nets: dict[tuple[int, int], set[str]]) -> dict[str, list[str]]:
        # Kept for private-call compatibility; drawing locations are never
        # an electrical port map. parse() uses explicit portRef membership.
        return {}

    @staticmethod
    def _find_top_port_name_for_net(net_name: str, top_ports: dict[str, list[str]]) -> str | None:
        for port_name, nets in top_ports.items():
            if net_name in nets:
                return port_name
        return None

    @staticmethod
    def _port_direction_guess(port_name: str | None) -> str | None:
        if not port_name:
            return None
        upper = port_name.upper()
        if re.fullmatch(r"IN\d+", upper) or upper.startswith("IN_"):
            return "input"
        if re.fullmatch(r"OUT\d+", upper) or upper.startswith("OUT_"):
            return "output"
        if upper in {"CLK", "CLOCK"} or "CLK" in upper:
            return "input"
        if any(token in upper for token in ["RST", "RESET", "SET", "EN"]):
            return "input"
        return None

    def _detect_supplies(
        self,
        top_page: str,
        netlist: TopLevelNetlist,
        point_to_nets: dict[tuple[int, int], set[str]],
    ) -> dict[str, set[str]]:
        connectivity = ExplicitConnectivity.from_netlist(netlist)
        power_names = {"1", "VDD", "VCC", "+5", "5PLUS", "POWER"}
        ground_names = {"0", "&0", "GND", "VSS"}
        power = {net.name for net in netlist.nets if net.name.upper() in power_names}
        ground = {net.name for net in netlist.nets if net.name.upper() in ground_names}
        power.update(net for port, net in connectivity.top_ports.items() if port in power_names)
        ground.update(net for port, net in connectivity.top_ports.items() if port in ground_names)
        return {"power": power, "ground": ground}

    @staticmethod
    def _orientation_apply(orientation: str, tx: int, ty: int, x: int, y: int) -> tuple[int, int]:
        matrix = {
            "R0": (1, 0, 0, 1),
            "R90": (0, -1, 1, 0),
            "R180": (-1, 0, 0, -1),
            "R270": (0, 1, -1, 0),
            "MX": (1, 0, 0, -1),
            "MY": (-1, 0, 0, 1),
            "MXR90": (0, 1, 1, 0),
            "MYR90": (0, -1, -1, 0),
        }.get(orientation, (1, 0, 0, 1))
        a, b, c, d = matrix
        return a * x + b * y + tx, c * x + d * y + ty

    @staticmethod
    def _nets_at_point(
        x: int,
        y: int,
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
    ) -> set[str]:
        # EDIF explicit wire points are safer than segment-crossing inference:
        # many schematic renderers draw crossing wires without junction dots.
        return set(point_to_nets.get((x, y), set()))

    def _infer_pins_from_symbol(
        self,
        instance_block: str | None,
        view_index: dict[tuple[str, str, str], str],
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
    ) -> dict[str, str]:
        # Symbol geometry cannot establish a missing electrical connection.
        return {}

    @classmethod
    def _classify_device(
        cls,
        instance_name: str,
        cell_name: str | None,
        view_name: str | None,
        pins: dict[str, str],
    ) -> _DeviceSignature:
        payload = " ".join(filter(None, [instance_name, cell_name, view_name])).upper()
        pin_names = set(pins.keys())

        # Switch-level primitives and pass devices.
        if "NMOS" in payload and "PMOS" in payload:
            return _DeviceSignature(DeviceKind.UNKNOWN, DeviceDomain.UNKNOWN, {"conflicting_channel_names": "1"})
        if "NMOS" in payload:
            attrs = {"channel": "n"}
            if "PASS" in payload or "TRAN" in payload:
                return _DeviceSignature(DeviceKind.PASS_NMOS, DeviceDomain.SWITCH, attrs)
            return _DeviceSignature(DeviceKind.NMOS, DeviceDomain.SWITCH, attrs)
        if "PMOS" in payload:
            attrs = {"channel": "p"}
            if "PASS" in payload or "TRAN" in payload:
                return _DeviceSignature(DeviceKind.PASS_PMOS, DeviceDomain.SWITCH, attrs)
            return _DeviceSignature(DeviceKind.PMOS, DeviceDomain.SWITCH, attrs)
        if any(token in payload for token in ["TGATE", "TRANSMISSION", "PASSGATE", "TG_"]):
            return _DeviceSignature(DeviceKind.TRANSMISSION_GATE, DeviceDomain.SWITCH, {})

        # Sequential macros (gate-level libraries).
        if any(token in payload for token in ["DFF", "FD", "FLIPFLOP"]):
            return _DeviceSignature(DeviceKind.FF_D, DeviceDomain.SEQUENTIAL_MACRO, {})
        if any(token in payload for token in ["JKFF", "FJK", "JK_FLIP"]):
            return _DeviceSignature(DeviceKind.FF_JK, DeviceDomain.SEQUENTIAL_MACRO, {})
        if re.search(r"(?<![A-Z])TFF|_TFF|FLIPFLOP_T", payload):
            return _DeviceSignature(DeviceKind.FF_T, DeviceDomain.SEQUENTIAL_MACRO, {})
        if any(token in payload for token in ["LATCH", "DLAT", "LD"]):
            return _DeviceSignature(DeviceKind.LATCH_D, DeviceDomain.SEQUENTIAL_MACRO, {})
        if "RS" in payload and "LATCH" in payload:
            return _DeviceSignature(DeviceKind.LATCH_RS, DeviceDomain.SEQUENTIAL_MACRO, {})
        if "COUNTER" in payload or re.search(r"\bCNT\d*\b", payload):
            return _DeviceSignature(DeviceKind.COUNTER, DeviceDomain.SEQUENTIAL_MACRO, {})

        # Logic-level gates.
        if any(token in payload for token in ["MUX", "MX2", "MUX2"]):
            return _DeviceSignature(DeviceKind.MUX2, DeviceDomain.GATE, {})
        if any(token in payload for token in ["XNOR", "XNR"]):
            return _DeviceSignature(DeviceKind.XNOR, DeviceDomain.GATE, {})
        if any(token in payload for token in ["XOR", "XOR2"]):
            return _DeviceSignature(DeviceKind.XOR, DeviceDomain.GATE, {})
        if re.search(r"\bINV\b", payload) or any(token in payload for token in ["NOT", "INVERTER"]):
            return _DeviceSignature(DeviceKind.INV, DeviceDomain.GATE, {})
        if re.search(r"\bBUF\b", payload) or "BUFFER" in payload:
            return _DeviceSignature(DeviceKind.BUF, DeviceDomain.GATE, {})
        if "NAND" in payload:
            return _DeviceSignature(DeviceKind.NAND, DeviceDomain.GATE, {})
        if "NOR" in payload:
            return _DeviceSignature(DeviceKind.NOR, DeviceDomain.GATE, {})
        if re.search(r"\bAND\b", payload):
            return _DeviceSignature(DeviceKind.AND, DeviceDomain.GATE, {})
        if re.search(r"\bOR\b", payload):
            return _DeviceSignature(DeviceKind.OR, DeviceDomain.GATE, {})

        # Fallback pin-shape heuristics when cell naming is not descriptive.
        if {"D", "Q"} <= pin_names and any(name in pin_names for name in {"CLK", "CLOCK", "CP", "CK"}):
            return _DeviceSignature(DeviceKind.FF_D, DeviceDomain.SEQUENTIAL_MACRO, {"inferred_from_pins": "1"})
        if {"D", "Q"} <= pin_names and any(name in pin_names for name in {"EN", "G", "E"}):
            return _DeviceSignature(DeviceKind.LATCH_D, DeviceDomain.SEQUENTIAL_MACRO, {"inferred_from_pins": "1"})
        if {"S", "R", "Q"} <= pin_names or {"SET", "RESET", "Q"} <= pin_names:
            return _DeviceSignature(DeviceKind.LATCH_RS, DeviceDomain.SEQUENTIAL_MACRO, {"inferred_from_pins": "1"})
        if any(name in pin_names for name in {"A", "B", "Y", "Z"}) and len(pin_names) <= 5:
            return _DeviceSignature(DeviceKind.UNKNOWN, DeviceDomain.GATE, {"inferred_from_pins": "1"})

        return _DeviceSignature(DeviceKind.UNKNOWN, DeviceDomain.UNKNOWN, {})

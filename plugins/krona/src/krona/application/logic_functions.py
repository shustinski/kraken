from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

from krona.application.ports import NetlistRepository
from krona.domain.netlist import Instance, TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser


@dataclass(frozen=True)
class _Transistor:
    instance_name: str
    kind: str  # "nmos" or "pmos"
    gate: str
    source: str
    drain: str


class ExtractLogicFunctions:
    def __init__(self, repository: NetlistRepository):
        self._repository = repository

    def execute(self, path: str | Path) -> dict:
        source_path = Path(path)
        netlist = self._repository.read_top_level_netlist(source_path)
        top_page = self._repository.read_top_page_block(source_path)
        view_index = self._repository.read_view_index(source_path)
        alias_map = self._build_net_alias_map(netlist)

        input_net_by_var = self._canonicalize_dict_values(self._input_nets(top_page, netlist), alias_map)
        output_net_by_name = self._canonicalize_dict_values(self._output_nets(top_page, netlist), alias_map)
        power_nets_raw, ground_nets_raw = self._supply_nets(top_page, netlist)
        power_nets = self._canonicalize_set(power_nets_raw, alias_map)
        ground_nets = self._canonicalize_set(ground_nets_raw, alias_map)
        point_to_nets, segments_by_net = self._build_net_geometry(netlist)
        transistors = self._transistors(
            netlist=netlist,
            input_net_by_var=input_net_by_var,
            power_nets=power_nets,
            ground_nets=ground_nets,
            output_net_by_name=output_net_by_name,
            top_page=top_page,
            view_index=view_index,
            point_to_nets=point_to_nets,
            segments_by_net=segments_by_net,
            alias_map=alias_map,
        )

        inputs = sorted(input_net_by_var.keys(), key=self._natural_key)
        nets_by_canonical: dict[str, list[str]] = {}
        for net in netlist.nets:
            canonical = alias_map.get(net.name, net.name)
            nets_by_canonical.setdefault(canonical, []).append(net.name)
        sequential_elements = self._sequential_elements(
            netlist=netlist,
            alias_map=alias_map,
            top_page=top_page,
            view_index=view_index,
            point_to_nets=point_to_nets,
            segments_by_net=segments_by_net,
            nets_by_canonical=nets_by_canonical,
            transistors=transistors,
            power_nets=power_nets,
            ground_nets=ground_nets,
            input_net_by_var=input_net_by_var,
            output_net_by_name=output_net_by_name,
        )
        outputs: dict[str, dict] = {}
        for output_name, output_net in sorted(output_net_by_name.items(), key=lambda item: self._natural_key(item[0])):
            rows = []
            for bits in product([0, 1], repeat=len(inputs)):
                assignment = dict(zip(inputs, bits))
                state = self._evaluate_output_state(
                    output_net=output_net,
                    assignment=assignment,
                    input_net_by_var=input_net_by_var,
                    power_nets=power_nets,
                    ground_nets=ground_nets,
                    transistors=transistors,
                )
                rows.append({"inputs": assignment, "value": state})
            has_unknown = any(row.get("value") is None for row in rows)
            path_nets, path_instances = self._logic_path_for_output(
                output_net=output_net,
                transistors=transistors,
                nets_by_canonical=nets_by_canonical,
            )
            outputs[output_name] = {
                "net": output_net,
                "truth_table": rows,
                "sum_of_products": None if has_unknown else self._sum_of_products(inputs, rows),
                "simplified_expression": None if has_unknown else self._simplified_expression(inputs, rows),
                "logic_path": {
                    "nets": path_nets,
                    "instances": path_instances,
                },
            }

        return {
            "inputs": inputs,
            "outputs": outputs,
            "sequential_elements": sequential_elements,
            "meta": {
                "input_nets": input_net_by_var,
                "power_nets": sorted(power_nets, key=self._natural_key),
                "ground_nets": sorted(ground_nets, key=self._natural_key),
                "transistor_count": len(transistors),
                "sequential_count": len(sequential_elements),
                "flip_flop_count": sum(1 for item in sequential_elements if item.get("kind") == "flip_flop"),
                "latch_count": sum(1 for item in sequential_elements if item.get("kind") == "latch"),
            },
        }

    @staticmethod
    def _natural_key(value: str) -> tuple:
        return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))

    @staticmethod
    def _instance_kind(instance: Instance) -> str | None:
        payload = " ".join(filter(None, [instance.name, instance.cell, instance.view, instance.library])).upper()
        if "NMOS" in payload:
            return "nmos"
        if "PMOS" in payload:
            return "pmos"
        return None

    def _build_net_alias_map(self, netlist: TopLevelNetlist) -> dict[str, str]:
        parent: dict[str, str] = {net.name: net.name for net in netlist.nets}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                return
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

        # Nets that share the same explicit wire point represent one electrical node.
        point_to_nets: dict[tuple[int, int], set[str]] = {}
        for net in netlist.nets:
            for wire in net.wires:
                for point in wire.points:
                    point_to_nets.setdefault((point.x, point.y), set()).add(net.name)
        for nets in point_to_nets.values():
            names = list(nets)
            for idx in range(1, len(names)):
                union(names[0], names[idx])

        # If same instance pin appears in multiple net names, treat them as aliases.
        pin_to_nets: dict[tuple[str, str], set[str]] = {}
        for net in netlist.nets:
            for connection in net.connections:
                if not connection.instance:
                    continue
                key = (connection.instance, connection.port.upper())
                pin_to_nets.setdefault(key, set()).add(net.name)
        for nets in pin_to_nets.values():
            names = list(nets)
            for idx in range(1, len(names)):
                union(names[0], names[idx])

        return {name: find(name) for name in parent}

    @staticmethod
    def _canonicalize_dict_values(values: dict[str, str], alias_map: dict[str, str]) -> dict[str, str]:
        return {key: alias_map.get(value, value) for key, value in values.items()}

    @staticmethod
    def _canonicalize_set(values: set[str], alias_map: dict[str, str]) -> set[str]:
        return {alias_map.get(value, value) for value in values}

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

    def _input_nets(self, top_page: str, netlist: TopLevelNetlist) -> dict[str, str]:
        result: dict[str, str] = {}

        # 1) Explicit stimulus instances (INSINx) used by some generated EDF files.
        for net in netlist.nets:
            for connection in net.connections:
                if not connection.instance:
                    continue
                match = re.fullmatch(r"INSIN(\d+)", connection.instance.upper())
                if not match:
                    continue
                if connection.port.upper() not in {"VPULSECR", "IN", "INPUT"}:
                    continue
                result[f"IN{match.group(1)}"] = net.name

        # 2) Top-level module ports (INx) represented via portImplementation blocks.
        point_to_nets: dict[tuple[int, int], set[str]] = {}
        for net in netlist.nets:
            for wire in net.wires:
                for point in wire.points:
                    point_to_nets.setdefault((point.x, point.y), set()).add(net.name)

        for port_impl in EdifTextParser.find_direct_classes(top_page, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if name_block:
                port_name = EdifTextParser.parse_header_name(name_block, "name")
            else:
                port_name = EdifTextParser.parse_header_name(port_impl, "portImplementation")
            upper_name = port_name.upper()
            if not re.fullmatch(r"IN\d+", upper_name):
                continue

            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            points = EdifTextParser.parse_points(connect_location) if connect_location else []
            if not points:
                continue
            nets = point_to_nets.get((points[0].x, points[0].y))
            if not nets:
                continue
            result[upper_name] = sorted(nets, key=self._natural_key)[0]
        return result

    def _output_nets(self, top_page: str, netlist: TopLevelNetlist) -> dict[str, str]:
        point_to_nets: dict[tuple[int, int], set[str]] = {}
        for net in netlist.nets:
            for wire in net.wires:
                for point in wire.points:
                    key = (point.x, point.y)
                    point_to_nets.setdefault(key, set()).add(net.name)

        outputs: dict[str, str] = {}
        for port_impl in EdifTextParser.find_direct_classes(top_page, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if name_block:
                port_name = EdifTextParser.parse_header_name(name_block, "name")
            else:
                port_name = EdifTextParser.parse_header_name(port_impl, "portImplementation")
            if not re.fullmatch(r"OUT\d+", port_name.upper()):
                continue
            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            if not connect_location:
                continue
            points = EdifTextParser.parse_points(connect_location)
            if not points:
                continue
            nets = point_to_nets.get((points[0].x, points[0].y))
            if not nets:
                continue
            outputs[port_name] = sorted(nets, key=self._natural_key)[0]
        return outputs

    def _build_net_geometry(self, netlist: TopLevelNetlist) -> tuple[dict[tuple[int, int], set[str]], dict[str, list[tuple[int, int, int, int]]]]:
        point_to_nets: dict[tuple[int, int], set[str]] = {}
        segments_by_net: dict[str, list[tuple[int, int, int, int]]] = {}
        for net in netlist.nets:
            segments: list[tuple[int, int, int, int]] = []
            for wire in net.wires:
                if not wire.points:
                    continue
                for point in wire.points:
                    point_to_nets.setdefault((point.x, point.y), set()).add(net.name)
                for start, end in zip(wire.points, wire.points[1:]):
                    segments.append((start.x, start.y, end.x, end.y))
            segments_by_net[net.name] = segments
        return point_to_nets, segments_by_net

    @staticmethod
    def _point_on_segment(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        if x1 == x2:
            if px != x1:
                return False
            return min(y1, y2) <= py <= max(y1, y2)
        if y1 == y2:
            if py != y1:
                return False
            return min(x1, x2) <= px <= max(x1, x2)
        # Fallback for non-orthogonal segments.
        dx1 = px - x1
        dy1 = py - y1
        dx2 = x2 - x1
        dy2 = y2 - y1
        cross = dx1 * dy2 - dy1 * dx2
        if cross != 0:
            return False
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

    def _nets_at_point(
        self,
        x: int,
        y: int,
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
    ) -> set[str]:
        # Use explicit EDIF wire points only. Segment-intersection inference can
        # incorrectly join crossing wires that are not electrically connected.
        return set(point_to_nets.get((x, y), set()))

    def _infer_pins_from_symbol(
        self,
        instance_block: str,
        view_index: dict[tuple[str, str, str], str],
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
    ) -> dict[str, list[str]]:
        view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
        cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
        lib_ref = EdifTextParser.first_any_or_none(instance_block, "(libraryRef ")
        if not view_ref or not cell_ref or not lib_ref:
            return {}
        view_name = EdifTextParser.parse_header_name(view_ref, "viewRef")
        cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef")
        lib_name = EdifTextParser.parse_header_name(lib_ref, "libraryRef")
        view_block = view_index.get((lib_name, cell_name, view_name))
        if not view_block:
            return {}
        symbol_block = EdifTextParser.first_any_or_none(view_block, "(symbol")
        if not symbol_block:
            return {}

        transform_block = EdifTextParser.first_direct_or_none(instance_block, "(transform")
        tx, ty = 0, 0
        orientation = "R0"
        if transform_block:
            origin_block = EdifTextParser.first_any_or_none(transform_block, "(origin")
            points = EdifTextParser.parse_points(origin_block) if origin_block else []
            if points:
                tx, ty = points[0].x, points[0].y
            orientation_block = EdifTextParser.first_any_or_none(transform_block, "(orientation")
            if orientation_block:
                orientation = EdifTextParser.parse_header_name(orientation_block, "orientation")

        inferred: dict[str, list[str]] = {}
        for port_impl in EdifTextParser.find_direct_classes(symbol_block, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if not name_block:
                continue
            pin_name = EdifTextParser.parse_header_name(name_block, "name").upper()
            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            points = EdifTextParser.parse_points(connect_location) if connect_location else []
            if not points:
                continue
            wx, wy = self._orientation_apply(orientation, tx, ty, points[0].x, points[0].y)
            nets = sorted(self._nets_at_point(wx, wy, point_to_nets, segments_by_net), key=self._natural_key)
            if nets:
                inferred[pin_name] = nets
        return inferred

    def _supply_nets(self, top_page: str, netlist: TopLevelNetlist) -> tuple[set[str], set[str]]:
        power_nets: set[str] = set()
        ground_nets: set[str] = set()
        instance_by_name = {instance.name: instance for instance in netlist.instances}

        for net in netlist.nets:
            upper_net = net.name.upper()
            if upper_net in {"0", "GND", "VSS"} or "GND" in upper_net:
                ground_nets.add(net.name)
            if "VDD" in upper_net or "VCC" in upper_net:
                power_nets.add(net.name)
            for connection in net.connections:
                if not connection.instance:
                    continue
                instance = instance_by_name.get(connection.instance)
                if instance is None:
                    continue
                signature = " ".join(filter(None, [instance.name, instance.cell, instance.view])).upper()
                if any(token in signature for token in ["VDC", "+5", "5PLUS", "VDD", "VCC"]):
                    power_nets.add(net.name)
                if any(token in signature for token in ["GND", "VSS"]):
                    ground_nets.add(net.name)

        point_to_nets: dict[tuple[int, int], set[str]] = {}
        for net in netlist.nets:
            for wire in net.wires:
                for point in wire.points:
                    key = (point.x, point.y)
                    point_to_nets.setdefault(key, set()).add(net.name)

        for port_impl in EdifTextParser.find_direct_classes(top_page, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if name_block:
                port_name = EdifTextParser.parse_header_name(name_block, "name")
            else:
                port_name = EdifTextParser.parse_header_name(port_impl, "portImplementation")
            upper_name = port_name.upper()
            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            points = EdifTextParser.parse_points(connect_location) if connect_location else []
            if not points:
                continue
            nets = point_to_nets.get((points[0].x, points[0].y), set())
            if not nets:
                continue
            if upper_name in {"0", "&0", "GND"}:
                ground_nets.update(nets)
            if any(token in upper_name for token in ["+5", "VDD", "VCC", "5PLUS", "POWER"]):
                power_nets.update(nets)
        return power_nets, ground_nets

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            if item not in out:
                out.append(item)
        return out

    def _select_gate_net(
        self,
        gate_candidates: list[str],
        input_nets: set[str],
        instance_gate_candidates: dict[str, list[str]],
        instance_name: str,
    ) -> str | None:
        if not gate_candidates:
            return None
        for candidate in gate_candidates:
            if candidate in input_nets:
                return candidate
        # If a transistor has only one gate net and it is shared with another transistor
        # that also has an input-connected gate candidate, reuse that known input gate.
        if len(gate_candidates) == 1:
            shared = gate_candidates[0]
            for peer_name, peer_candidates in instance_gate_candidates.items():
                if peer_name == instance_name or shared not in peer_candidates:
                    continue
                for peer_candidate in peer_candidates:
                    if peer_candidate in input_nets:
                        return peer_candidate
        # Normalize to nearest known input net when generator produces near-number aliases
        # (e.g. N00726 instead of N00727).
        numeric_inputs: list[tuple[str, int]] = []
        for input_net in input_nets:
            match = re.fullmatch(r"[A-Za-z_]*([0-9]+)", input_net)
            if match:
                numeric_inputs.append((input_net, int(match.group(1))))
        if numeric_inputs:
            best_candidate = gate_candidates[0]
            best_distance = None
            for candidate in gate_candidates:
                match = re.fullmatch(r"[A-Za-z_]*([0-9]+)", candidate)
                if not match:
                    continue
                candidate_value = int(match.group(1))
                for input_net, input_value in numeric_inputs:
                    distance = abs(candidate_value - input_value)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_candidate = input_net
            if best_distance is not None and best_distance <= 1:
                return best_candidate
        return gate_candidates[0]

    def _select_terminal_pair(
        self,
        kind: str,
        terminal_candidates: list[str],
        power_nets: set[str],
        ground_nets: set[str],
        net_connection_degree: dict[str, int],
        pmos_power_adjacent_nets: set[str],
        nmos_ground_adjacent_nets: set[str],
        output_nets: set[str],
    ) -> tuple[str, str] | None:
        terminal_candidates = self._unique(terminal_candidates)
        options: list[tuple[str, str]] = []
        for idx, left in enumerate(terminal_candidates):
            for right in terminal_candidates[idx + 1:]:
                if left != right:
                    options.append((left, right))
        if not options:
            return None

        def score(option: tuple[str, str]) -> tuple[int, int, int]:
            left, right = option
            supply_score = 0
            if kind == "pmos":
                if left in power_nets or right in power_nets:
                    supply_score += 4
                if left in pmos_power_adjacent_nets or right in pmos_power_adjacent_nets:
                    supply_score += 2
            else:
                if left in ground_nets or right in ground_nets:
                    supply_score += 4
                if left in nmos_ground_adjacent_nets or right in nmos_ground_adjacent_nets:
                    supply_score += 2
            output_score = 2 if (left in output_nets or right in output_nets) else 0
            degree_score = net_connection_degree.get(left, 0) + net_connection_degree.get(right, 0)
            return supply_score, output_score, degree_score

        options.sort(key=lambda option: score(option), reverse=True)
        return options[0]

    def _transistors(
        self,
        netlist: TopLevelNetlist,
        input_net_by_var: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        output_net_by_name: dict[str, str],
        top_page: str,
        view_index: dict[tuple[str, str, str], str],
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
        alias_map: dict[str, str],
    ) -> list[_Transistor]:
        pins_by_instance: dict[str, dict[str, list[str]]] = {}
        net_connection_degree: dict[str, int] = {}
        for net in netlist.nets:
            canonical_name = alias_map.get(net.name, net.name)
            net_connection_degree[canonical_name] = max(net_connection_degree.get(canonical_name, 0), len(net.connections))
            for connection in net.connections:
                if not connection.instance:
                    continue
                ports = pins_by_instance.setdefault(connection.instance, {})
                ports.setdefault(connection.port.upper(), []).append(canonical_name)

        input_nets = set(input_net_by_var.values())
        output_nets = set(output_net_by_name.values())
        instance_gate_candidates: dict[str, list[str]] = {}
        for instance_name, ports in pins_by_instance.items():
            gate_candidates = self._unique((ports.get("GATE") or []) + (ports.get("G") or []))
            if gate_candidates:
                instance_gate_candidates[instance_name] = gate_candidates

        # Precompute adjacency hints for ambiguous terminal choices.
        pmos_power_adjacent_nets: set[str] = set()
        nmos_ground_adjacent_nets: set[str] = set()
        for instance in netlist.instances:
            kind = self._instance_kind(instance)
            if kind is None:
                continue
            ports = pins_by_instance.get(instance.name, {})
            source_candidates = self._unique((ports.get("SOURCE") or []) + (ports.get("S") or []))
            drain_candidates = self._unique((ports.get("DRAIN") or []) + (ports.get("D") or []))
            terminal_candidates = self._unique(source_candidates + drain_candidates)
            if kind == "pmos":
                if any(net in power_nets for net in terminal_candidates):
                    pmos_power_adjacent_nets.update(net for net in terminal_candidates if net not in power_nets)
            else:
                if any(net in ground_nets for net in terminal_candidates):
                    nmos_ground_adjacent_nets.update(net for net in terminal_candidates if net not in ground_nets)

        result: list[_Transistor] = []
        instance_blocks = {
            EdifTextParser.parse_header_name(block, "instance"): block
            for block in EdifTextParser.find_direct_classes(top_page, "(instance ")
        }
        for instance in netlist.instances:
            kind = self._instance_kind(instance)
            if kind is None:
                continue
            pins = {key: list(values) for key, values in pins_by_instance.get(instance.name, {}).items()}
            if instance.name in instance_blocks:
                inferred = self._infer_pins_from_symbol(
                    instance_block=instance_blocks[instance.name],
                    view_index=view_index,
                    point_to_nets=point_to_nets,
                    segments_by_net=segments_by_net,
                )
                for key, nets in inferred.items():
                    canonical_nets = [alias_map.get(net_name, net_name) for net_name in nets]
                    merged = self._unique((pins.get(key) or []) + canonical_nets)
                    pins[key] = merged
            gate_candidates = self._unique((pins.get("GATE") or []) + (pins.get("G") or []))
            source_candidates = self._unique((pins.get("SOURCE") or []) + (pins.get("S") or []))
            drain_candidates = self._unique((pins.get("DRAIN") or []) + (pins.get("D") or []))
            terminal_candidates = self._unique(source_candidates + drain_candidates)
            gate = self._select_gate_net(
                gate_candidates=gate_candidates,
                input_nets=input_nets,
                instance_gate_candidates=instance_gate_candidates,
                instance_name=instance.name,
            )
            terminals = self._select_terminal_pair(
                kind=kind,
                terminal_candidates=terminal_candidates,
                power_nets=power_nets,
                ground_nets=ground_nets,
                net_connection_degree=net_connection_degree,
                pmos_power_adjacent_nets=pmos_power_adjacent_nets,
                nmos_ground_adjacent_nets=nmos_ground_adjacent_nets,
                output_nets=output_nets,
            )
            if not gate or terminals is None:
                continue
            result.append(_Transistor(instance_name=instance.name, kind=kind, gate=gate, source=terminals[0], drain=terminals[1]))
        return result

    def _degenerate_transistor_markers(
        self,
        netlist: TopLevelNetlist,
        input_net_by_var: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        top_page: str,
        view_index: dict[tuple[str, str, str], str],
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
        alias_map: dict[str, str],
    ) -> list[dict]:
        pins_by_instance: dict[str, dict[str, list[str]]] = {}
        for net in netlist.nets:
            canonical_name = alias_map.get(net.name, net.name)
            for connection in net.connections:
                if not connection.instance:
                    continue
                ports = pins_by_instance.setdefault(connection.instance, {})
                ports.setdefault(connection.port.upper(), []).append(canonical_name)

        input_nets = set(input_net_by_var.values())
        instance_gate_candidates: dict[str, list[str]] = {}
        for instance_name, ports in pins_by_instance.items():
            gate_candidates = self._unique((ports.get("GATE") or []) + (ports.get("G") or []))
            if gate_candidates:
                instance_gate_candidates[instance_name] = gate_candidates

        instance_blocks = {
            EdifTextParser.parse_header_name(block, "instance"): block
            for block in EdifTextParser.find_direct_classes(top_page, "(instance ")
        }

        markers: list[dict] = []
        for instance in netlist.instances:
            kind = self._instance_kind(instance)
            if kind is None:
                continue

            pins = {key: list(values) for key, values in pins_by_instance.get(instance.name, {}).items()}
            if instance.name in instance_blocks:
                inferred = self._infer_pins_from_symbol(
                    instance_block=instance_blocks[instance.name],
                    view_index=view_index,
                    point_to_nets=point_to_nets,
                    segments_by_net=segments_by_net,
                )
                for key, nets in inferred.items():
                    canonical_nets = [alias_map.get(net_name, net_name) for net_name in nets]
                    pins[key] = self._unique((pins.get(key) or []) + canonical_nets)

            gate_candidates = self._unique((pins.get("GATE") or []) + (pins.get("G") or []))
            source_candidates = self._unique((pins.get("SOURCE") or []) + (pins.get("S") or []))
            drain_candidates = self._unique((pins.get("DRAIN") or []) + (pins.get("D") or []))
            terminal_candidates = self._unique(source_candidates + drain_candidates)
            gate = self._select_gate_net(
                gate_candidates=gate_candidates,
                input_nets=input_nets,
                instance_gate_candidates=instance_gate_candidates,
                instance_name=instance.name,
            )
            if not gate or not terminal_candidates:
                continue

            terminals_set = set(terminal_candidates)
            style: str | None = None
            node: str | None = None
            if len(terminals_set) == 1:
                style = "sd_shorted"
                node = next(iter(terminals_set))
            elif terminals_set & power_nets and terminals_set & ground_nets:
                style = "rail_bridge"
            if style is None:
                continue

            markers.append(
                {
                    "instance": instance.name,
                    "kind": kind,
                    "gate": gate,
                    "terminals": sorted(terminals_set, key=self._natural_key),
                    "node": node,
                    "style": style,
                }
            )

        return sorted(
            markers,
            key=lambda item: (
                self._natural_key(str(item.get("gate", ""))),
                self._natural_key(str(item.get("node", ""))),
                self._natural_key(str(item.get("instance", ""))),
            ),
        )

    def _sequential_elements(
        self,
        netlist: TopLevelNetlist,
        alias_map: dict[str, str],
        top_page: str,
        view_index: dict[tuple[str, str, str], str],
        point_to_nets: dict[tuple[int, int], set[str]],
        segments_by_net: dict[str, list[tuple[int, int, int, int]]],
        nets_by_canonical: dict[str, list[str]],
        transistors: list[_Transistor] | None = None,
        power_nets: set[str] | None = None,
        ground_nets: set[str] | None = None,
        input_net_by_var: dict[str, str] | None = None,
        output_net_by_name: dict[str, str] | None = None,
    ) -> list[dict]:
        _ = (netlist, alias_map, top_page, view_index, point_to_nets, segments_by_net)
        if transistors is None or power_nets is None or ground_nets is None:
            return []
        degenerate_transistor_markers = self._degenerate_transistor_markers(
            netlist=netlist,
            input_net_by_var=input_net_by_var or {},
            power_nets=power_nets,
            ground_nets=ground_nets,
            top_page=top_page,
            view_index=view_index,
            point_to_nets=point_to_nets,
            segments_by_net=segments_by_net,
            alias_map=alias_map,
        )
        return self._sequential_elements_topology(
            transistors=transistors,
            power_nets=power_nets,
            ground_nets=ground_nets,
            nets_by_canonical=nets_by_canonical,
            input_net_by_var=input_net_by_var or {},
            output_nets=set((output_net_by_name or {}).values()),
            output_net_by_name=output_net_by_name or {},
            degenerate_transistor_markers=degenerate_transistor_markers,
        )

    def _sequential_elements_topology(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
        input_net_by_var: dict[str, str],
        output_nets: set[str],
        output_net_by_name: dict[str, str],
        degenerate_transistor_markers: list[dict] | None = None,
    ) -> list[dict]:
        if not transistors:
            return []

        inverter_map = self._detect_cmos_inverters(transistors, power_nets, ground_nets)
        if not inverter_map:
            return []
        switches = self._detect_controlled_switches(transistors, power_nets, ground_nets)
        tri_state_inverters = self._detect_tristate_cmos_inverters(
            transistors=transistors,
            power_nets=power_nets,
            ground_nets=ground_nets,
            inverter_map=inverter_map,
        )
        if not switches and not tri_state_inverters:
            return []

        latch_candidates = self._detect_topological_latch_candidates(
            transistors=transistors,
            inverter_map=inverter_map,
            switches=switches,
            power_nets=power_nets,
            ground_nets=ground_nets,
            nets_by_canonical=nets_by_canonical,
            output_nets=output_nets,
        )
        hybrid_latch_candidates = self._detect_topological_hybrid_latch_candidates(
            transistors=transistors,
            inverter_map=inverter_map,
            tri_state_inverters=tri_state_inverters,
            switches=switches,
            power_nets=power_nets,
            ground_nets=ground_nets,
            nets_by_canonical=nets_by_canonical,
            output_nets=output_nets,
        )
        if hybrid_latch_candidates:
            latch_candidates.extend(hybrid_latch_candidates)
        if not latch_candidates:
            return []

        ff_elements, used_latch_indexes = self._merge_topological_latches_into_flip_flops(
            latches=latch_candidates,
            inverter_map=inverter_map,
            nets_by_canonical=nets_by_canonical,
            output_nets=output_nets,
        )
        pulse_ff_elements, pulse_used_latch_indexes = self._promote_pulse_triggered_latches_to_flip_flops(
            latches=latch_candidates,
            used_latch_indexes=used_latch_indexes,
            degenerate_transistor_markers=degenerate_transistor_markers or [],
            power_nets=power_nets,
            ground_nets=ground_nets,
            nets_by_canonical=nets_by_canonical,
        )
        if pulse_ff_elements:
            ff_elements.extend(pulse_ff_elements)
            used_latch_indexes = set(used_latch_indexes) | set(pulse_used_latch_indexes)

        elements: list[dict] = list(ff_elements)
        latch_counter = 1
        for idx, candidate in enumerate(latch_candidates):
            if idx in used_latch_indexes:
                continue
            element = dict(candidate)
            element["instance"] = f"LATCH{latch_counter}"
            latch_counter += 1
            elements.append(element)

        self._annotate_sequential_elements(
            elements=elements,
            transistors=transistors,
            power_nets=power_nets,
            ground_nets=ground_nets,
            inverter_map=inverter_map,
            input_net_by_var=input_net_by_var,
            output_net_by_name=output_net_by_name,
        )
        public_elements = [{key: value for key, value in element.items() if not key.startswith("_")} for element in elements]
        for idx, element in enumerate(public_elements, start=1):
            if not element.get("instance"):
                prefix = "FF" if element.get("kind") == "flip_flop" else "LATCH"
                element["instance"] = f"{prefix}{idx}"

        return sorted(public_elements, key=lambda item: self._natural_key(str(item.get("instance", ""))))

    def _two_transistor_paths_to_supply(
        self,
        transistors: list[_Transistor],
        kind: str,
        start_net: str,
        target_nets: set[str],
    ) -> list[dict]:
        items = [(idx, transistor) for idx, transistor in enumerate(transistors) if transistor.kind == kind]
        paths: list[dict] = []
        for first_idx, first in items:
            if start_net not in {first.source, first.drain}:
                continue
            mid = first.drain if first.source == start_net else first.source
            if mid in target_nets:
                continue
            for second_idx, second in items:
                if second_idx == first_idx:
                    continue
                if mid not in {second.source, second.drain}:
                    continue
                other = second.drain if second.source == mid else second.source
                if other not in target_nets:
                    continue
                paths.append(
                    {
                        "transistor_indexes": [first_idx, second_idx],
                        "gates": [first.gate, second.gate],
                        "instances": [first.instance_name, second.instance_name],
                        "mid_net": mid,
                    }
                )
        return paths

    @staticmethod
    def _constant_gate_conduction_state(
        transistor_kind: str,
        gate_net: str,
        power_nets: set[str],
        ground_nets: set[str],
    ) -> bool | None:
        if gate_net in power_nets:
            return transistor_kind == "nmos"
        if gate_net in ground_nets:
            return transistor_kind == "pmos"
        return None

    def _detect_cmos_inverters(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> dict[tuple[str, str], dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for idx, transistor in enumerate(transistors):
            terminals = [transistor.source, transistor.drain]
            if transistor.kind == "pmos":
                if (terminals[0] in power_nets) == (terminals[1] in power_nets):
                    continue
                output_net = terminals[1] if terminals[0] in power_nets else terminals[0]
            else:
                if (terminals[0] in ground_nets) == (terminals[1] in ground_nets):
                    continue
                output_net = terminals[1] if terminals[0] in ground_nets else terminals[0]
            key = (transistor.gate, output_net)
            bucket = grouped.setdefault(
                key,
                {
                    "input": transistor.gate,
                    "output": output_net,
                    "pmos_indexes": [],
                    "nmos_indexes": [],
                    "instances": [],
                },
            )
            bucket[f"{transistor.kind}_indexes"].append(idx)
            bucket["instances"].append(transistor.instance_name)

        out: dict[tuple[str, str], dict] = {}
        for key, bucket in grouped.items():
            if not bucket["pmos_indexes"] or not bucket["nmos_indexes"]:
                continue
            out[key] = {
                "input": bucket["input"],
                "output": bucket["output"],
                "pmos_indexes": list(bucket["pmos_indexes"]),
                "nmos_indexes": list(bucket["nmos_indexes"]),
                "transistor_indexes": list(bucket["pmos_indexes"]) + list(bucket["nmos_indexes"]),
                "instances": self._unique(list(bucket["instances"])),
            }
        return out

    def _detect_controlled_switches(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for idx, transistor in enumerate(transistors):
            terminals = self._unique([transistor.source, transistor.drain])
            if len(terminals) != 2:
                continue
            if any(net in power_nets or net in ground_nets for net in terminals):
                continue
            left, right = sorted(terminals, key=self._natural_key)
            bucket = grouped.setdefault(
                (left, right),
                {
                    "nets": (left, right),
                    "nmos": [],
                    "pmos": [],
                },
            )
            bucket[transistor.kind].append(
                {
                    "index": idx,
                    "gate": transistor.gate,
                    "instance": transistor.instance_name,
                }
            )

        switches: list[dict] = []
        for bucket in grouped.values():
            nmos_items = list(bucket.get("nmos", []))
            pmos_items = list(bucket.get("pmos", []))
            if not nmos_items and not pmos_items:
                continue

            nmos_gates = sorted(self._unique([item["gate"] for item in nmos_items]), key=self._natural_key)
            pmos_gates = sorted(self._unique([item["gate"] for item in pmos_items]), key=self._natural_key)
            nmos_gate_states = {
                net_name: self._constant_gate_conduction_state("nmos", net_name, power_nets, ground_nets)
                for net_name in nmos_gates
            }
            pmos_gate_states = {
                net_name: self._constant_gate_conduction_state("pmos", net_name, power_nets, ground_nets)
                for net_name in pmos_gates
            }
            controllable_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is None]
            controllable_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is None]
            constant_on_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is True]
            constant_on_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is True]
            constant_off_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is False]
            constant_off_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is False]
            controls: list[dict] = []
            for net_name in nmos_gates:
                constant_state = nmos_gate_states.get(net_name)
                if constant_state is None:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "nmos",
                            "activation": {"mode": "level", "level": "high"},
                        }
                    )
                else:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "nmos",
                            "activation": {"mode": "constant", "state": "on" if constant_state else "off"},
                        }
                    )
            for net_name in pmos_gates:
                constant_state = pmos_gate_states.get(net_name)
                if constant_state is None:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "pmos",
                            "activation": {"mode": "level", "level": "low"},
                        }
                    )
                else:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "pmos",
                            "activation": {"mode": "constant", "state": "on" if constant_state else "off"},
                        }
                    )

            primary_phase: dict | None = None
            if len(controllable_nmos_gates) == 1:
                primary_phase = {"net": controllable_nmos_gates[0], "level": "high", "kind": "nmos"}
            elif len(controllable_nmos_gates) == 0 and len(controllable_pmos_gates) == 1:
                primary_phase = {"net": controllable_pmos_gates[0], "level": "low", "kind": "pmos"}

            switches.append(
                {
                    "nets": bucket["nets"],
                    "instances": sorted(
                        self._unique([item["instance"] for item in nmos_items + pmos_items]),
                        key=self._natural_key,
                    ),
                    "transistor_indexes": [item["index"] for item in nmos_items + pmos_items],
                    "nmos_gates": nmos_gates,
                    "pmos_gates": pmos_gates,
                    "controllable_nmos_gates": controllable_nmos_gates,
                    "controllable_pmos_gates": controllable_pmos_gates,
                    "constant_on_nmos_gates": constant_on_nmos_gates,
                    "constant_on_pmos_gates": constant_on_pmos_gates,
                    "constant_off_nmos_gates": constant_off_nmos_gates,
                    "constant_off_pmos_gates": constant_off_pmos_gates,
                    "controls": controls,
                    "primary_phase": primary_phase,
                    "kind": (
                        "transmission_gate"
                        if nmos_items and pmos_items
                        else ("pass_nmos" if nmos_items else "pass_pmos")
                    ),
                }
            )

        return sorted(switches, key=lambda item: (self._natural_key(item["nets"][0]), self._natural_key(item["nets"][1])))

    def _detect_tristate_cmos_inverters(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
    ) -> list[dict]:
        records: list[dict] = []
        seen: set[tuple[str, str, str, str, tuple[int, ...]]] = set()

        for (static_input, static_output), static_inv in inverter_map.items():
            n_paths = self._two_transistor_paths_to_supply(
                transistors=transistors,
                kind="nmos",
                start_net=static_input,
                target_nets=ground_nets,
            )
            p_paths = self._two_transistor_paths_to_supply(
                transistors=transistors,
                kind="pmos",
                start_net=static_input,
                target_nets=power_nets,
            )
            if not n_paths or not p_paths:
                continue

            for n_path in n_paths:
                if n_path["gates"].count(static_output) != 1:
                    continue
                n_phase_gates = [gate for gate in n_path["gates"] if gate != static_output]
                if len(n_phase_gates) != 1:
                    continue
                n_phase_net = n_phase_gates[0]
                n_extra_state = self._constant_gate_conduction_state("nmos", n_phase_net, power_nets, ground_nets)
                if n_extra_state is False:
                    continue
                for p_path in p_paths:
                    if p_path["gates"].count(static_output) != 1:
                        continue
                    p_phase_gates = [gate for gate in p_path["gates"] if gate != static_output]
                    if len(p_phase_gates) != 1:
                        continue
                    p_phase_net = p_phase_gates[0]
                    p_extra_state = self._constant_gate_conduction_state("pmos", p_phase_net, power_nets, ground_nets)
                    if p_extra_state is False:
                        continue

                    phase: dict | None = None
                    phase_complement: dict | None = None
                    topology_style = "tristate"
                    if n_extra_state is None and p_extra_state is None:
                        if n_phase_net == p_phase_net:
                            phase = {"net": n_phase_net, "level": "high"}
                            phase_complement = {"net": p_phase_net, "level": "low"}
                            topology_style = "tristate_same_clock"
                        elif self._nets_are_inverted(n_phase_net, p_phase_net, inverter_map):
                            phase = {"net": n_phase_net, "level": "high"}
                            phase_complement = {"net": p_phase_net, "level": "low"}
                        else:
                            continue
                    elif n_extra_state is None and p_extra_state is True:
                        phase = {"net": n_phase_net, "level": "high"}
                        phase_complement = None
                        topology_style = "pseudo_tristate_n_phase"
                    elif n_extra_state is True and p_extra_state is None:
                        phase = {"net": p_phase_net, "level": "low"}
                        phase_complement = None
                        topology_style = "pseudo_tristate_p_phase"
                    else:
                        continue
                    if phase is None:
                        continue

                    transistor_indexes = sorted(
                        self._unique(n_path["transistor_indexes"] + p_path["transistor_indexes"])
                    )
                    key = (
                        static_output,
                        static_input,
                        str(phase.get("net") or ""),
                        str(phase.get("level") or ""),
                        tuple(transistor_indexes),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    instances = self._unique(n_path["instances"] + p_path["instances"])
                    records.append(
                        {
                            "input": static_output,
                            "output": static_input,
                            "phase": phase,
                            "phase_complement": phase_complement,
                            "topology_style": topology_style,
                            "static_inverter": static_inv,
                            "nmos_path": n_path,
                            "pmos_path": p_path,
                            "n_path_extra_gate": {
                                "net": n_phase_net,
                                "constant_state": None if n_extra_state is None else ("on" if n_extra_state else "off"),
                            },
                            "p_path_extra_gate": {
                                "net": p_phase_net,
                                "constant_state": None if p_extra_state is None else ("on" if p_extra_state else "off"),
                            },
                            "transistor_indexes": transistor_indexes,
                            "instances": instances,
                        }
                    )

        return sorted(
            records,
            key=lambda item: (
                self._natural_key(str(item.get("output", ""))),
                self._natural_key(str(item.get("input", ""))),
                self._natural_key(str(item.get("phase", {}).get("net", ""))),
            ),
        )

    @staticmethod
    def _phase_relation(
        left: dict | None,
        right: dict | None,
        inverter_map: dict[tuple[str, str], dict],
    ) -> str | None:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None
        left_net = str(left.get("net") or "")
        right_net = str(right.get("net") or "")
        left_level = str(left.get("level") or "")
        right_level = str(right.get("level") or "")
        if left_level not in {"high", "low"} or right_level not in {"high", "low"}:
            return None
        if not left_net or not right_net:
            return None
        if left_net == right_net:
            return "same" if left_level == right_level else "opposite"
        inverted = (left_net, right_net) in inverter_map or (right_net, left_net) in inverter_map
        if not inverted:
            return None
        return "opposite" if left_level == right_level else "same"

    def _detect_topological_hybrid_latch_candidates(
        self,
        transistors: list[_Transistor],
        inverter_map: dict[tuple[str, str], dict],
        tri_state_inverters: list[dict],
        switches: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> list[dict]:
        all_hybrid_core_nets: set[str] = set()
        for tri in tri_state_inverters:
            all_hybrid_core_nets.add(str(tri.get("input", "")))
            all_hybrid_core_nets.add(str(tri.get("output", "")))

        terminal_degree = self._terminal_degree(transistors)
        candidates: list[dict] = []
        seen_cores: set[tuple[str, str, str, str]] = set()

        for tri in tri_state_inverters:
            core_input = str(tri.get("input") or "")
            core_output = str(tri.get("output") or "")
            if not core_input or not core_output or core_input == core_output:
                continue

            static_inv = tri.get("static_inverter") or {}
            regen_phase = tri.get("phase") or {}
            regen_phase_complement = tri.get("phase_complement") or {}
            if not isinstance(regen_phase, dict):
                continue
            core_key = tuple(sorted((core_input, core_output), key=self._natural_key))
            dedup_key = (
                core_key[0],
                core_key[1],
                str(regen_phase.get("net") or ""),
                str(regen_phase.get("level") or ""),
            )
            if dedup_key in seen_cores:
                continue
            seen_cores.add(dedup_key)

            tri_tx_indexes = set(int(idx) for idx in tri.get("transistor_indexes", []))
            transparency_switches: list[dict] = []
            for switch in switches:
                switch_phase = switch.get("primary_phase")
                if self._phase_relation(switch_phase, regen_phase, inverter_map) != "opposite":
                    continue
                if tri_tx_indexes & set(int(idx) for idx in switch.get("transistor_indexes", [])):
                    continue
                switch_nets = set(switch.get("nets", ()))
                if not switch_nets & {core_input, core_output}:
                    continue
                external_nets = [
                    net_name
                    for net_name in switch.get("nets", ())
                    if net_name not in {core_input, core_output}
                    and net_name not in power_nets
                    and net_name not in ground_nets
                ]
                if not external_nets:
                    continue
                transparency_switches.append(switch)

            if not transparency_switches:
                continue

            def transparency_score(switch: dict) -> tuple[int, int, int, int]:
                ext_nets = [
                    net_name
                    for net_name in switch.get("nets", ())
                    if net_name not in {core_input, core_output}
                    and net_name not in power_nets
                    and net_name not in ground_nets
                ]
                noncore_ext = [net_name for net_name in ext_nets if net_name not in all_hybrid_core_nets]
                touches_output_net = any(net_name in output_nets for net_name in switch.get("nets", ()))
                return (
                    1 if noncore_ext else 0,
                    1 if switch.get("kind") == "transmission_gate" else 0,
                    1 if touches_output_net else 0,
                    len(ext_nets),
                )

            transparency_switches.sort(key=transparency_score, reverse=True)
            primary_transparency_switch = transparency_switches[0]
            enable_phase = dict(primary_transparency_switch.get("primary_phase") or {})
            if not enable_phase:
                continue

            data_nets: list[str] = []
            for switch in transparency_switches:
                for net_name in switch.get("nets", ()):
                    if net_name in {core_input, core_output} or net_name in power_nets or net_name in ground_nets:
                        continue
                    if net_name not in data_nets:
                        data_nets.append(net_name)
            if not data_nets:
                continue
            data_nets.sort(key=self._natural_key)

            core_nets = {core_input, core_output}
            involved_instances = self._unique(
                list(static_inv.get("instances", []))
                + list(tri.get("instances", []))
                + [
                    instance_name
                    for switch in transparency_switches
                    for instance_name in switch.get("instances", [])
                ]
            )
            involved_nets: set[str] = set(core_nets)
            involved_nets.update(data_nets)
            if str(enable_phase.get("net") or ""):
                involved_nets.add(str(enable_phase["net"]))
            if str(regen_phase.get("net") or ""):
                involved_nets.add(str(regen_phase["net"]))
            if str(regen_phase_complement.get("net") or ""):
                involved_nets.add(str(regen_phase_complement["net"]))
            for switch in transparency_switches:
                involved_nets.update(net_name for net_name in switch.get("nets", ()) if isinstance(net_name, str))
                for control in switch.get("controls", []):
                    control_net = str(control.get("net") or "")
                    if control_net:
                        involved_nets.add(control_net)

            def core_output_score(net_name: str) -> tuple[int, int, int]:
                touches_output = 1 if net_name in output_nets else 0
                touches_switch = 0
                for switch in transparency_switches:
                    if net_name in set(switch.get("nets", ())):
                        touches_switch += 1
                return (touches_output, touches_switch, terminal_degree.get(net_name, 0))

            ordered_core = sorted([core_output, core_input], key=lambda net_name: core_output_score(net_name), reverse=True)
            primary_output_net = ordered_core[0]
            secondary_output_net = ordered_core[1]
            output_entries = [
                {"port": None, "net": primary_output_net, "polarity": "non_inverted"},
                {"port": None, "net": secondary_output_net, "polarity": "inverted"},
            ]
            data_entries = [{"port": None, "net": net_name} for net_name in data_nets]

            control_signals: list[dict] = [
                {
                    "role": "enable",
                    "port": None,
                    "net": str(enable_phase.get("net") or ""),
                    "activation": {"mode": "level", "level": str(enable_phase.get("level") or "high")},
                },
                {
                    "role": "hold_phase",
                    "port": None,
                    "net": str(regen_phase.get("net") or ""),
                    "activation": {"mode": "level", "level": str(regen_phase.get("level") or "high")},
                },
            ]
            if str(regen_phase_complement.get("net") or ""):
                control_signals.append(
                    {
                        "role": "phase_complement",
                        "port": None,
                        "net": str(regen_phase_complement.get("net") or ""),
                        "activation": {"mode": "level", "level": str(regen_phase_complement.get("level") or "low")},
                    }
                )
            control_signals = self._unique_seq_entries(control_signals)

            control_nets = [item["net"] for item in control_signals if item.get("net")]
            output_nets_local = [item["net"] for item in output_entries if item.get("net")]
            data_nets_local = [item["net"] for item in data_entries if item.get("net")]
            involved_tx_indexes = sorted(
                self._unique(
                    list(static_inv.get("transistor_indexes", []))
                    + list(tri.get("transistor_indexes", []))
                    + [
                        idx
                        for switch in transparency_switches
                        for idx in switch.get("transistor_indexes", [])
                    ]
                )
            )
            all_nets_canonical = sorted(self._unique(list(involved_nets)), key=self._natural_key)
            expanded_all_nets = sorted(self._expanded_aliases(all_nets_canonical, nets_by_canonical), key=self._natural_key)
            expanded_control_nets = sorted(self._expanded_aliases(control_nets, nets_by_canonical), key=self._natural_key)
            expanded_output_nets = sorted(self._expanded_aliases(output_nets_local, nets_by_canonical), key=self._natural_key)
            expanded_data_nets = sorted(self._expanded_aliases(data_nets_local, nets_by_canonical), key=self._natural_key)

            candidates.append(
                {
                    "_core_nets": sorted(core_nets, key=self._natural_key),
                    "_enable": {
                        "net": str(enable_phase.get("net") or ""),
                        "level": str(enable_phase.get("level") or "high"),
                        "complement_nets": [
                            net_name
                            for net_name in [
                                str(regen_phase.get("net") or ""),
                                str(regen_phase_complement.get("net") or ""),
                            ]
                            if net_name and net_name != str(enable_phase.get("net") or "")
                        ],
                    },
                    "_data_nets": data_nets_local,
                    "_output_nets": output_nets_local,
                    "_all_instances": sorted(self._unique(involved_instances), key=self._natural_key),
                    "_all_nets_canonical": all_nets_canonical,
                    "_transistor_indexes": involved_tx_indexes,
                    "instance": None,
                    "cell": None,
                    "view": None,
                    "library": None,
                    "designator": None,
                    "kind": "latch",
                    "subtype": "D",
                    "outputs": output_entries,
                    "data_inputs": data_entries,
                    "control_signals": control_signals,
                    "triggering": {
                        "mode": "level",
                        "level": str(enable_phase.get("level") or "high"),
                        "port": None,
                        "net": str(enable_phase.get("net") or ""),
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": sorted(self._unique(involved_instances), key=self._natural_key),
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": sorted(self._unique(involved_instances), key=self._natural_key),
                        "control_nets": expanded_control_nets,
                        "data_nets": expanded_data_nets,
                        "output_nets": expanded_output_nets,
                    },
                }
            )

        return candidates

    @staticmethod
    def _nets_are_inverted(left: str, right: str, inverter_map: dict[tuple[str, str], dict]) -> bool:
        return (left, right) in inverter_map or (right, left) in inverter_map

    @staticmethod
    def _inverter_instances_between(left: str, right: str, inverter_map: dict[tuple[str, str], dict]) -> list[str]:
        instances: list[str] = []
        for key in ((left, right), (right, left)):
            record = inverter_map.get(key)
            if record:
                instances.extend(record.get("instances", []))
        out: list[str] = []
        for item in instances:
            if item not in out:
                out.append(item)
        return out

    def _terminal_degree(self, transistors: list[_Transistor]) -> dict[str, int]:
        degree: dict[str, int] = {}
        for transistor in transistors:
            for net in self._unique([transistor.source, transistor.drain]):
                degree[net] = degree.get(net, 0) + 1
        return degree

    def _detect_topological_latch_candidates(
        self,
        transistors: list[_Transistor],
        inverter_map: dict[tuple[str, str], dict],
        switches: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> list[dict]:
        terminal_degree = self._terminal_degree(transistors)
        cross_coupled_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for in_net, out_net in inverter_map:
            if in_net == out_net or (out_net, in_net) not in inverter_map:
                continue
            pair = tuple(sorted((in_net, out_net), key=self._natural_key))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            cross_coupled_pairs.append(pair)
        cross_coupled_pairs.sort(key=lambda pair: (self._natural_key(pair[0]), self._natural_key(pair[1])))
        all_core_nets: set[str] = set()
        for pair in cross_coupled_pairs:
            all_core_nets.update(pair)

        latch_candidates: list[dict] = []
        for pair in cross_coupled_pairs:
            core_nets = set(pair)
            core_inverters = [inverter_map[(pair[0], pair[1])], inverter_map[(pair[1], pair[0])]]
            core_instances = self._unique(core_inverters[0]["instances"] + core_inverters[1]["instances"])

            attached_switches: list[dict] = []
            control_votes: dict[tuple[str, str], dict] = {}
            involved_instances = list(core_instances)
            involved_nets: set[str] = set(core_nets)

            for switch in switches:
                switch_nets = set(switch["nets"])
                touch_core = len(switch_nets & core_nets)
                if touch_core == 0:
                    continue
                attached_switches.append(switch)
                for instance_name in switch.get("instances", []):
                    if instance_name not in involved_instances:
                        involved_instances.append(instance_name)
                involved_nets.update(switch_nets)
                primary_phase = switch.get("primary_phase")
                if not isinstance(primary_phase, dict):
                    continue
                key = (str(primary_phase.get("net")), str(primary_phase.get("level")))
                if not key[0] or key[1] not in {"high", "low"}:
                    continue
                external_nets = [
                    net_name
                    for net_name in switch["nets"]
                    if net_name not in core_nets and net_name not in power_nets and net_name not in ground_nets
                ]
                noncore_external_nets = [net_name for net_name in external_nets if net_name not in all_core_nets]
                interstage_nets = [net_name for net_name in external_nets if net_name in all_core_nets]
                vote = control_votes.setdefault(
                    key,
                    {
                        "net": key[0],
                        "level": key[1],
                        "phase_kind": str(primary_phase.get("kind") or ""),
                        "switches": [],
                        "switch_count": 0,
                        "external_switch_count": 0,
                        "noncore_external_switch_count": 0,
                        "interstage_switch_count": 0,
                        "feedback_switch_count": 0,
                        "complement_nets": set(),
                    },
                )
                vote["switches"].append(switch)
                vote["switch_count"] += 1
                if touch_core == 1 and external_nets:
                    vote["external_switch_count"] += 1
                if touch_core == 1 and noncore_external_nets:
                    vote["noncore_external_switch_count"] += 1
                if touch_core == 1 and interstage_nets:
                    vote["interstage_switch_count"] += 1
                if touch_core == 2:
                    vote["feedback_switch_count"] += 1
                if primary_phase.get("kind") == "nmos":
                    opposite_gates = switch.get("pmos_gates", [])
                else:
                    opposite_gates = switch.get("nmos_gates", [])
                for gate_net in opposite_gates:
                    if self._nets_are_inverted(key[0], gate_net, inverter_map):
                        vote["complement_nets"].add(gate_net)
                        for instance_name in self._inverter_instances_between(key[0], gate_net, inverter_map):
                            if instance_name not in involved_instances:
                                involved_instances.append(instance_name)
                        involved_nets.add(gate_net)

            if not attached_switches or not control_votes:
                continue

            ranked_votes = sorted(
                control_votes.values(),
                key=lambda item: (
                    -int(item["noncore_external_switch_count"]),
                    -int(item["external_switch_count"]),
                    -int(item["interstage_switch_count"]),
                    -int(item["switch_count"]),
                    -int(item["feedback_switch_count"]),
                    0 if item.get("phase_kind") == "nmos" else 1,
                    self._natural_key(str(item.get("net", ""))),
                    str(item.get("level", "")),
                ),
            )
            primary_vote = ranked_votes[0]
            if int(primary_vote.get("external_switch_count", 0)) <= 0:
                continue

            primary_enable_net = str(primary_vote["net"])
            primary_enable_level = str(primary_vote["level"])
            complement_nets = sorted(primary_vote.get("complement_nets", set()), key=self._natural_key)
            involved_nets.add(primary_enable_net)
            involved_nets.update(complement_nets)

            data_nets: list[str] = []
            preferred_data_nets: list[str] = []
            fallback_interstage_data_nets: list[str] = []
            for switch in primary_vote.get("switches", []):
                switch_nets = set(switch["nets"])
                if len(switch_nets & core_nets) != 1:
                    continue
                for net_name in switch["nets"]:
                    if net_name in core_nets or net_name in power_nets or net_name in ground_nets:
                        continue
                    if net_name in all_core_nets:
                        if net_name not in fallback_interstage_data_nets:
                            fallback_interstage_data_nets.append(net_name)
                    else:
                        if net_name not in preferred_data_nets:
                            preferred_data_nets.append(net_name)
            data_nets = preferred_data_nets or fallback_interstage_data_nets
            data_nets.sort(key=self._natural_key)

            if not data_nets:
                continue

            def core_net_score(net_name: str) -> tuple[int, int, int]:
                external_touch = 0
                for switch in attached_switches:
                    switch_set = set(switch["nets"])
                    if net_name not in switch_set:
                        continue
                    if len(switch_set & core_nets) == 1 and any(other not in core_nets for other in switch["nets"]):
                        external_touch += 1
                return (
                    1 if net_name in output_nets else 0,
                    external_touch,
                    terminal_degree.get(net_name, 0),
                )

            ordered_core_nets = sorted(pair, key=lambda net_name: (core_net_score(net_name),), reverse=True)
            if len(ordered_core_nets) < 2:
                ordered_core_nets = list(pair)
            primary_output_net = ordered_core_nets[0]
            secondary_output_net = ordered_core_nets[1]

            control_signals: list[dict] = [
                {
                    "role": "enable",
                    "port": None,
                    "net": primary_enable_net,
                    "activation": {"mode": "level", "level": primary_enable_level},
                }
            ]
            for comp_net in complement_nets:
                control_signals.append(
                    {
                        "role": "phase_complement",
                        "port": None,
                        "net": comp_net,
                        "activation": {
                            "mode": "level",
                            "level": "low" if primary_enable_level == "high" else "high",
                        },
                    }
                )
            for vote in ranked_votes[1:]:
                net_name = str(vote.get("net", ""))
                level = str(vote.get("level", ""))
                if not net_name or level not in {"high", "low"}:
                    continue
                role = "phase" if self._nets_are_inverted(primary_enable_net, net_name, inverter_map) else "control"
                control_signals.append(
                    {
                        "role": role,
                        "port": None,
                        "net": net_name,
                        "activation": {"mode": "level", "level": level},
                    }
                )
                involved_nets.add(net_name)
            control_signals = self._unique_seq_entries(control_signals)

            output_entries = [
                {"port": None, "net": primary_output_net, "polarity": "non_inverted"},
                {"port": None, "net": secondary_output_net, "polarity": "inverted"},
            ]
            data_entries = [{"port": None, "net": net_name} for net_name in data_nets]

            control_nets = [item["net"] for item in control_signals if item.get("net")]
            output_nets_local = [item["net"] for item in output_entries if item.get("net")]
            data_nets_local = [item["net"] for item in data_entries if item.get("net")]
            involved_tx_indexes = sorted(
                self._unique(
                    core_inverters[0].get("transistor_indexes", [])
                    + core_inverters[1].get("transistor_indexes", [])
                    + [idx for switch in attached_switches for idx in switch.get("transistor_indexes", [])]
                )
            )

            expanded_all_nets = sorted(
                self._expanded_aliases(sorted(involved_nets, key=self._natural_key), nets_by_canonical),
                key=self._natural_key,
            )
            expanded_control_nets = sorted(self._expanded_aliases(control_nets, nets_by_canonical), key=self._natural_key)
            expanded_output_nets = sorted(self._expanded_aliases(output_nets_local, nets_by_canonical), key=self._natural_key)
            expanded_data_nets = sorted(self._expanded_aliases(data_nets_local, nets_by_canonical), key=self._natural_key)

            latch_candidates.append(
                {
                    "_core_nets": sorted(core_nets, key=self._natural_key),
                    "_enable": {
                        "net": primary_enable_net,
                        "level": primary_enable_level,
                        "complement_nets": complement_nets,
                    },
                    "_data_nets": data_nets_local,
                    "_output_nets": output_nets_local,
                    "_all_instances": sorted(involved_instances, key=self._natural_key),
                    "_all_nets_canonical": sorted(involved_nets, key=self._natural_key),
                    "_transistor_indexes": involved_tx_indexes,
                    "instance": None,
                    "cell": None,
                    "view": None,
                    "library": None,
                    "designator": None,
                    "kind": "latch",
                    "subtype": "D",
                    "outputs": output_entries,
                    "data_inputs": data_entries,
                    "control_signals": control_signals,
                    "triggering": {
                        "mode": "level",
                        "level": primary_enable_level,
                        "port": None,
                        "net": primary_enable_net,
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": sorted(involved_instances, key=self._natural_key),
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": sorted(involved_instances, key=self._natural_key),
                        "control_nets": expanded_control_nets,
                        "data_nets": expanded_data_nets,
                        "output_nets": expanded_output_nets,
                    },
                }
            )

        return latch_candidates

    @staticmethod
    def _controls_are_opposite_phase(
        left: dict,
        right: dict,
        inverter_map: dict[tuple[str, str], dict],
    ) -> bool:
        left_net = str(left.get("net", ""))
        right_net = str(right.get("net", ""))
        left_level = str(left.get("level", ""))
        right_level = str(right.get("level", ""))
        if not left_net or not right_net or left_level not in {"high", "low"} or right_level not in {"high", "low"}:
            return False
        if left_net == right_net:
            return left_level != right_level
        inverted = (left_net, right_net) in inverter_map or (right_net, left_net) in inverter_map
        return inverted and left_level == right_level

    @staticmethod
    def _nets_share_or_invert_path(
        source_nets: set[str],
        target_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
    ) -> int:
        if source_nets & target_nets:
            return 3
        for source in source_nets:
            for target in target_nets:
                if (source, target) in inverter_map or (target, source) in inverter_map:
                    return 2
        return 0

    def _merge_topological_latches_into_flip_flops(
        self,
        latches: list[dict],
        inverter_map: dict[tuple[str, str], dict],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> tuple[list[dict], set[int]]:
        pair_options: list[dict] = []
        for left_idx in range(len(latches)):
            left = latches[left_idx]
            left_enable = left.get("_enable") or {}
            for right_idx in range(left_idx + 1, len(latches)):
                right = latches[right_idx]
                right_enable = right.get("_enable") or {}
                if not self._controls_are_opposite_phase(left_enable, right_enable, inverter_map):
                    continue

                left_to_right = self._nets_share_or_invert_path(
                    set(left.get("_output_nets", [])),
                    set(right.get("_data_nets", [])),
                    inverter_map,
                )
                right_to_left = self._nets_share_or_invert_path(
                    set(right.get("_output_nets", [])),
                    set(left.get("_data_nets", [])),
                    inverter_map,
                )
                if left_to_right == 0 and right_to_left == 0:
                    continue
                if left_to_right > right_to_left:
                    master_idx, slave_idx, direction_score = left_idx, right_idx, left_to_right
                elif right_to_left > left_to_right:
                    master_idx, slave_idx, direction_score = right_idx, left_idx, right_to_left
                else:
                    left_hits_output = bool(set(left.get("_output_nets", [])) & output_nets)
                    right_hits_output = bool(set(right.get("_output_nets", [])) & output_nets)
                    if right_hits_output and not left_hits_output:
                        master_idx, slave_idx, direction_score = left_idx, right_idx, left_to_right
                    elif left_hits_output and not right_hits_output:
                        master_idx, slave_idx, direction_score = right_idx, left_idx, right_to_left
                    else:
                        continue

                slave = latches[slave_idx]
                slave_enable = slave.get("_enable") or {}
                edge = "rising" if slave_enable.get("level") == "high" else "falling"
                pair_options.append(
                    {
                        "master_idx": master_idx,
                        "slave_idx": slave_idx,
                        "score": (
                            direction_score,
                            1 if set(slave.get("_output_nets", [])) & output_nets else 0,
                        ),
                        "edge": edge,
                    }
                )

        pair_options.sort(key=lambda item: item["score"], reverse=True)
        used: set[int] = set()
        ff_elements: list[dict] = []
        ff_counter = 1

        for option in pair_options:
            master_idx = int(option["master_idx"])
            slave_idx = int(option["slave_idx"])
            if master_idx in used or slave_idx in used:
                continue
            master = latches[master_idx]
            slave = latches[slave_idx]
            master_enable = master.get("_enable") or {}
            slave_enable = slave.get("_enable") or {}
            if not self._controls_are_opposite_phase(master_enable, slave_enable, inverter_map):
                continue

            clock_net = str(slave_enable.get("net") or master_enable.get("net") or "")
            if not clock_net:
                continue
            edge = str(option["edge"])

            union_instances = sorted(
                self._unique(list(master.get("_all_instances", [])) + list(slave.get("_all_instances", []))),
                key=self._natural_key,
            )
            union_nets_canonical = sorted(
                self._unique(list(master.get("_all_nets_canonical", [])) + list(slave.get("_all_nets_canonical", []))),
                key=self._natural_key,
            )
            union_transistor_indexes = sorted(
                self._unique(list(master.get("_transistor_indexes", [])) + list(slave.get("_transistor_indexes", [])))
            )
            expanded_all_nets = sorted(self._expanded_aliases(union_nets_canonical, nets_by_canonical), key=self._natural_key)

            ff_control_signals = [
                {
                    "role": "clock",
                    "port": None,
                    "net": clock_net,
                    "activation": {"mode": "edge", "edge": edge},
                },
                {
                    "role": "master_phase",
                    "port": None,
                    "net": str(master_enable.get("net") or ""),
                    "activation": {"mode": "level", "level": str(master_enable.get("level") or "high")},
                },
                {
                    "role": "slave_phase",
                    "port": None,
                    "net": str(slave_enable.get("net") or ""),
                    "activation": {"mode": "level", "level": str(slave_enable.get("level") or "high")},
                },
            ]

            # Keep additional non-enable controls from both latches.
            for source_label, latch in (("master", master), ("slave", slave)):
                for control in latch.get("control_signals", []):
                    role = str(control.get("role") or "")
                    if role == "enable":
                        continue
                    net_name = str(control.get("net") or "")
                    if not net_name:
                        continue
                    ff_control_signals.append(
                        {
                            "role": f"{source_label}_{role}",
                            "port": None,
                            "net": net_name,
                            "activation": dict(control.get("activation") or {}),
                        }
                    )
            ff_control_signals = self._unique_seq_entries(ff_control_signals)

            outputs = [dict(item) for item in (slave.get("outputs") or [])]
            data_inputs = [dict(item) for item in (master.get("data_inputs") or [])]

            ff_output_nets = [str(item.get("net")) for item in outputs if item.get("net")]
            ff_data_nets = [str(item.get("net")) for item in data_inputs if item.get("net")]
            ff_control_nets = [str(item.get("net")) for item in ff_control_signals if item.get("net")]

            ff_elements.append(
                {
                    "_core_nets": sorted(
                        self._unique(list(master.get("_core_nets", [])) + list(slave.get("_core_nets", []))),
                        key=self._natural_key,
                    ),
                    "_all_nets_canonical": union_nets_canonical,
                    "_transistor_indexes": union_transistor_indexes,
                    "instance": f"FF{ff_counter}",
                    "cell": None,
                    "view": None,
                    "library": None,
                    "designator": None,
                    "kind": "flip_flop",
                    "subtype": "D",
                    "outputs": outputs,
                    "data_inputs": data_inputs,
                    "control_signals": ff_control_signals,
                    "triggering": {
                        "mode": "edge",
                        "edge": edge,
                        "port": None,
                        "net": clock_net,
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": union_instances,
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": union_instances,
                        "control_nets": sorted(self._expanded_aliases(ff_control_nets, nets_by_canonical), key=self._natural_key),
                        "data_nets": sorted(self._expanded_aliases(ff_data_nets, nets_by_canonical), key=self._natural_key),
                        "output_nets": sorted(self._expanded_aliases(ff_output_nets, nets_by_canonical), key=self._natural_key),
                    },
                }
            )
            ff_counter += 1
            used.add(master_idx)
            used.add(slave_idx)

        return ff_elements, used

    def _promote_pulse_triggered_latches_to_flip_flops(
        self,
        latches: list[dict],
        used_latch_indexes: set[int],
        degenerate_transistor_markers: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
    ) -> tuple[list[dict], set[int]]:
        if not latches or not degenerate_transistor_markers:
            return [], set()

        markers_by_gate: dict[str, list[dict]] = {}
        for marker in degenerate_transistor_markers:
            gate_net = str(marker.get("gate") or "")
            if not gate_net:
                continue
            markers_by_gate.setdefault(gate_net, []).append(marker)

        promoted: list[dict] = []
        used: set[int] = set()
        ff_counter = 1
        for idx, latch in enumerate(latches):
            if idx in used_latch_indexes:
                continue
            if str(latch.get("kind") or "") != "latch" or str(latch.get("subtype") or "") != "D":
                continue

            enable = latch.get("_enable") or {}
            enable_net = str(enable.get("net") or "")
            enable_level = str(enable.get("level") or "")
            if not enable_net or enable_level not in {"high", "low"}:
                continue

            control_signals = [dict(item) for item in (latch.get("control_signals") or []) if isinstance(item, dict)]
            roles = {str(item.get("role") or ""): item for item in control_signals}
            enable_control = roles.get("enable")
            hold_control = roles.get("hold_phase")
            if not isinstance(enable_control, dict) or not isinstance(hold_control, dict):
                continue
            if str(enable_control.get("net") or "") != enable_net:
                continue
            if str(hold_control.get("net") or "") != enable_net:
                continue
            hold_activation = hold_control.get("activation") if isinstance(hold_control.get("activation"), dict) else {}
            hold_level = str(hold_activation.get("level") or "")
            if hold_level not in {"high", "low"} or hold_level == enable_level:
                continue

            markers = list(markers_by_gate.get(enable_net, []))
            if len(markers) < 2:
                continue
            has_rail_bridge = any(str(marker.get("style") or "") == "rail_bridge" for marker in markers)
            if not has_rail_bridge:
                continue

            local_shorted_markers = [
                marker
                for marker in markers
                if str(marker.get("style") or "") == "sd_shorted"
                and str(marker.get("node") or "")
                and str(marker.get("node") or "") not in power_nets
                and str(marker.get("node") or "") not in ground_nets
            ]
            if not local_shorted_markers:
                continue
            marker_kinds = {str(marker.get("kind") or "") for marker in markers if marker.get("kind")}
            if len(marker_kinds) < 2:
                continue

            edge = "rising" if enable_level == "high" else "falling"
            ff_control_signals = [
                {
                    "role": "clock",
                    "port": None,
                    "net": enable_net,
                    "activation": {"mode": "edge", "edge": edge},
                }
            ]
            ff_control_signals.extend(control_signals)
            ff_control_signals = self._unique_seq_entries(ff_control_signals)

            ff_element = {key: value for key, value in latch.items()}
            ff_element["instance"] = f"PULSE_FF{ff_counter}"
            ff_element["kind"] = "flip_flop"
            ff_element["subtype"] = "D"
            ff_element["control_signals"] = ff_control_signals
            ff_element["triggering"] = {
                "mode": "edge",
                "edge": edge,
                "port": None,
                "net": enable_net,
            }
            ff_element["topology"] = {
                "kind": "pulse_triggered_single_latch",
                "clock_net": enable_net,
                "pulse_level": enable_level,
                "evidence": {
                    "degenerate_mos_count": len(markers),
                    "local_clocked_mos_caps": [
                        {
                            "instance": str(marker.get("instance") or ""),
                            "kind": str(marker.get("kind") or ""),
                            "node": str(marker.get("node") or ""),
                        }
                        for marker in local_shorted_markers
                    ],
                    "rail_bridge_instances": sorted(
                        [
                            str(marker.get("instance") or "")
                            for marker in markers
                            if str(marker.get("style") or "") == "rail_bridge"
                        ],
                        key=self._natural_key,
                    ),
                },
            }

            marker_instances = [str(marker.get("instance") or "") for marker in markers if marker.get("instance")]
            marker_nets_canonical: list[str] = []
            for marker in markers:
                gate_net = str(marker.get("gate") or "")
                if gate_net:
                    marker_nets_canonical.append(gate_net)
                for net_name in marker.get("terminals", []) or []:
                    if isinstance(net_name, str) and net_name:
                        marker_nets_canonical.append(net_name)
            marker_nets_canonical = sorted(self._unique(marker_nets_canonical), key=self._natural_key)
            expanded_marker_nets = sorted(self._expanded_aliases(marker_nets_canonical, nets_by_canonical), key=self._natural_key)

            ff_element["_all_instances"] = sorted(
                self._unique([str(name) for name in (ff_element.get("_all_instances") or [])] + marker_instances),
                key=self._natural_key,
            )
            ff_element["_all_nets_canonical"] = sorted(
                self._unique([str(net) for net in (ff_element.get("_all_nets_canonical") or [])] + marker_nets_canonical),
                key=self._natural_key,
            )

            logic_path = ff_element.get("logic_path")
            if isinstance(logic_path, dict):
                logic_path["instances"] = sorted(
                    self._unique([str(name) for name in (logic_path.get("instances") or [])] + marker_instances),
                    key=self._natural_key,
                )
                logic_path["nets"] = sorted(
                    self._unique([str(net) for net in (logic_path.get("nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )

            highlight = ff_element.get("highlight")
            if isinstance(highlight, dict):
                highlight["instances"] = sorted(
                    self._unique([str(name) for name in (highlight.get("instances") or [])] + marker_instances),
                    key=self._natural_key,
                )
                highlight["nets"] = sorted(
                    self._unique([str(net) for net in (highlight.get("nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )
                highlight["control_nets"] = sorted(
                    self._unique([str(net) for net in (highlight.get("control_nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )

            promoted.append(ff_element)
            used.add(idx)
            ff_counter += 1

        return promoted, used

    def _annotate_sequential_elements(
        self,
        elements: list[dict],
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
        input_net_by_var: dict[str, str],
        output_net_by_name: dict[str, str],
    ) -> None:
        if not elements:
            return

        inputs = sorted(input_net_by_var.keys(), key=self._natural_key)
        input_names_by_net: dict[str, list[str]] = {}
        for name, net_name in input_net_by_var.items():
            input_names_by_net.setdefault(net_name, []).append(name)
        output_names_by_net: dict[str, list[str]] = {}
        for name, net_name in output_net_by_name.items():
            output_names_by_net.setdefault(net_name, []).append(name)

        function_cache: dict[str, dict] = {}
        for element in elements:
            async_controls = self._detect_async_force_controls_for_element(
                element=element,
                transistors=transistors,
                power_nets=power_nets,
                ground_nets=ground_nets,
            )
            if async_controls:
                existing = list(element.get("control_signals", []))
                element["control_signals"] = self._unique_seq_entries(existing + async_controls)
                control_nets = [str(item.get("net")) for item in element["control_signals"] if item.get("net")]
                highlight = element.get("highlight")
                if isinstance(highlight, dict):
                    current_control_nets = [str(net) for net in highlight.get("control_nets", [])]
                    merged_control = self._unique(current_control_nets + control_nets)
                    highlight["control_nets"] = merged_control
                    current_nets = [str(net) for net in highlight.get("nets", [])]
                    highlight["nets"] = self._unique(current_nets + merged_control)
                logic_path = element.get("logic_path")
                if isinstance(logic_path, dict):
                    current_nets = [str(net) for net in logic_path.get("nets", [])]
                    logic_path["nets"] = self._unique(current_nets + control_nets)

            for output_entry in element.get("outputs", []) or []:
                net_name = str(output_entry.get("net") or "")
                output_entry["labels"] = sorted(self._unique(output_names_by_net.get(net_name, [])), key=self._natural_key)
                output_entry["function"] = {
                    "kind": "state",
                    "role": "Qn" if output_entry.get("polarity") == "inverted" else "Q",
                }

            for data_entry in element.get("data_inputs", []) or []:
                net_name = str(data_entry.get("net") or "")
                data_entry["labels"] = sorted(self._unique(input_names_by_net.get(net_name, [])), key=self._natural_key)
                data_entry["function"] = self._analyze_combinational_net_function(
                    target_net=net_name,
                    inputs=inputs,
                    input_net_by_var=input_net_by_var,
                    output_net_by_name=output_net_by_name,
                    power_nets=power_nets,
                    ground_nets=ground_nets,
                    transistors=transistors,
                    cache=function_cache,
                    allow_mux=True,
                )

            for control_entry in element.get("control_signals", []) or []:
                net_name = str(control_entry.get("net") or "")
                control_entry["labels"] = sorted(self._unique(input_names_by_net.get(net_name, [])), key=self._natural_key)
                control_entry["function"] = self._analyze_combinational_net_function(
                    target_net=net_name,
                    inputs=inputs,
                    input_net_by_var=input_net_by_var,
                    output_net_by_name=output_net_by_name,
                    power_nets=power_nets,
                    ground_nets=ground_nets,
                    transistors=transistors,
                    cache=function_cache,
                    allow_mux=False,
                )
            triggering = element.get("triggering")
            if isinstance(triggering, dict):
                trigger_net = str(triggering.get("net") or "")
                if trigger_net:
                    trigger_labels = sorted(self._unique(input_names_by_net.get(trigger_net, [])), key=self._natural_key)
                    if trigger_labels:
                        triggering["labels"] = trigger_labels
                    trigger_function = self._analyze_combinational_net_function(
                        target_net=trigger_net,
                        inputs=inputs,
                        input_net_by_var=input_net_by_var,
                        output_net_by_name=output_net_by_name,
                        power_nets=power_nets,
                        ground_nets=ground_nets,
                        transistors=transistors,
                        cache=function_cache,
                        allow_mux=False,
                    )
                    triggering["function"] = trigger_function

            data_inputs = list(element.get("data_inputs", []) or [])
            if len(data_inputs) == 1:
                data_function = data_inputs[0].get("function")
                if isinstance(data_function, dict):
                    element["data_function"] = data_function
                    mux_info = data_function.get("mux")
                    if isinstance(mux_info, dict):
                        element["data_path"] = {"kind": "mux2", **mux_info}

            io_map_inputs: list[dict] = []
            for data_entry in element.get("data_inputs", []) or []:
                io_map_inputs.append(
                    {
                        "role": "data",
                        "net": data_entry.get("net"),
                        "labels": data_entry.get("labels", []),
                        "function": data_entry.get("function"),
                    }
                )
            for control_entry in element.get("control_signals", []) or []:
                io_map_inputs.append(
                    {
                        "role": control_entry.get("role"),
                        "net": control_entry.get("net"),
                        "labels": control_entry.get("labels", []),
                        "activation": control_entry.get("activation"),
                        "timing": control_entry.get("timing"),
                        "function": control_entry.get("function"),
                    }
                )
            io_map_outputs: list[dict] = []
            for output_entry in element.get("outputs", []) or []:
                io_map_outputs.append(
                    {
                        "role": "Qn" if output_entry.get("polarity") == "inverted" else "Q",
                        "net": output_entry.get("net"),
                        "labels": output_entry.get("labels", []),
                        "function": output_entry.get("function"),
                    }
                )
            element["io_map"] = {"inputs": io_map_inputs, "outputs": io_map_outputs}

    def _detect_async_force_controls_for_element(
        self,
        element: dict,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> list[dict]:
        core_nets = set(str(net) for net in (element.get("_core_nets") or []))
        if not core_nets:
            core_nets = {str(item.get("net")) for item in (element.get("outputs") or []) if item.get("net")}
        core_nets.discard("")
        if not core_nets:
            return []

        recognized_tx_indexes = {int(idx) for idx in (element.get("_transistor_indexes") or []) if isinstance(idx, int) or str(idx).isdigit()}
        known_control_nets = {str(item.get("net")) for item in (element.get("control_signals") or []) if item.get("net")}
        known_control_nets.discard("")
        data_nets = {str(item.get("net")) for item in (element.get("data_inputs") or []) if item.get("net")}
        data_nets.discard("")

        output_polarity_by_net: dict[str, str] = {}
        for output_item in element.get("outputs", []) or []:
            net_name = str(output_item.get("net") or "")
            polarity = str(output_item.get("polarity") or "non_inverted")
            if net_name:
                output_polarity_by_net[net_name] = polarity
        if not output_polarity_by_net:
            return []

        candidates: list[dict] = []

        def add_force_candidate(
            role: str,
            gate_net: str,
            active_level: str,
            core_net: str,
            path_depth: int,
        ) -> None:
            if not role or not gate_net:
                return
            candidates.append(
                {
                    "role": role,
                    "port": None,
                    "net": gate_net,
                    "activation": {"mode": "level", "level": active_level},
                    "timing": "asynchronous",
                    "topology": {
                        "kind": "force_path",
                        "path_depth": path_depth,
                        "target_storage_net": core_net,
                    },
                }
            )

        for idx, transistor in enumerate(transistors):
            if idx in recognized_tx_indexes:
                continue
            terminals = {transistor.source, transistor.drain}
            touched_core = list(terminals & core_nets)
            if len(touched_core) != 1:
                continue
            core_net = touched_core[0]
            if transistor.kind == "pmos":
                if not (terminals & power_nets):
                    continue
                forced_value = 1
                active_level = "low"
            else:
                if not (terminals & ground_nets):
                    continue
                forced_value = 0
                active_level = "high"
            gate_net = transistor.gate
            if gate_net in core_nets or gate_net in data_nets:
                continue
            role = self._force_role_from_output_polarity(core_net, forced_value, output_polarity_by_net)
            if role is None:
                continue
            add_force_candidate(role=role, gate_net=gate_net, active_level=active_level, core_net=core_net, path_depth=1)

        for core_net in sorted(core_nets, key=self._natural_key):
            for kind, targets, forced_value, active_level in (
                ("pmos", power_nets, 1, "low"),
                ("nmos", ground_nets, 0, "high"),
            ):
                for path in self._two_transistor_paths_to_supply(
                    transistors=transistors,
                    kind=kind,
                    start_net=core_net,
                    target_nets=targets,
                ):
                    path_indexes = {int(idx) for idx in path.get("transistor_indexes", [])}
                    if path_indexes and path_indexes.issubset(recognized_tx_indexes):
                        continue
                    external_gates = []
                    for gate_net in path.get("gates", []):
                        if gate_net in core_nets or gate_net in data_nets or gate_net in known_control_nets:
                            continue
                        if gate_net not in external_gates:
                            external_gates.append(gate_net)
                    if len(external_gates) != 1:
                        continue
                    role = self._force_role_from_output_polarity(core_net, forced_value, output_polarity_by_net)
                    if role is None:
                        continue
                    add_force_candidate(
                        role=role,
                        gate_net=external_gates[0],
                        active_level=active_level,
                        core_net=core_net,
                        path_depth=2,
                    )

        # Remove duplicates and avoid shadowing the regular phase controls with same net/level.
        deduped = self._unique_seq_entries(candidates)
        final_controls: list[dict] = []
        for candidate in deduped:
            candidate_net = str(candidate.get("net") or "")
            candidate_level = str((candidate.get("activation") or {}).get("level") or "")
            role = str(candidate.get("role") or "")
            if role not in {"set", "reset"}:
                continue
            conflict = False
            for existing in element.get("control_signals", []) or []:
                existing_net = str(existing.get("net") or "")
                existing_level = str((existing.get("activation") or {}).get("level") or "")
                existing_role = str(existing.get("role") or "")
                if candidate_net == existing_net and candidate_level == existing_level and existing_role in {"enable", "phase", "master_phase", "slave_phase", "hold_phase", "phase_complement", "master_hold_phase", "slave_hold_phase", "clock"}:
                    conflict = True
                    break
            if not conflict:
                final_controls.append(candidate)
        return final_controls

    @staticmethod
    def _force_role_from_output_polarity(core_net: str, forced_value: int, output_polarity_by_net: dict[str, str]) -> str | None:
        polarity = output_polarity_by_net.get(core_net)
        if polarity is None:
            return None
        if polarity == "non_inverted":
            return "set" if forced_value == 1 else "reset"
        return "reset" if forced_value == 1 else "set"

    def _analyze_combinational_net_function(
        self,
        target_net: str,
        inputs: list[str],
        input_net_by_var: dict[str, str],
        output_net_by_name: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        transistors: list[_Transistor],
        cache: dict[str, dict],
        allow_mux: bool = True,
    ) -> dict:
        target_net = str(target_net or "")
        if not target_net:
            return {"kind": "net_function", "expression": None}
        cache_key = f"{target_net}|mux={1 if allow_mux else 0}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        input_names = sorted([name for name, net_name in input_net_by_var.items() if net_name == target_net], key=self._natural_key)
        output_names = sorted([name for name, net_name in output_net_by_name.items() if net_name == target_net], key=self._natural_key)
        if input_names:
            expression = " | ".join(input_names) if len(input_names) > 1 else input_names[0]
            result = {
                "kind": "net_function",
                "expression": expression,
                "simplified_expression": expression,
                "sum_of_products": expression,
                "depends_on": list(input_names),
                "mux": None,
                "source_ports": {
                    "inputs": input_names,
                    "outputs": output_names,
                },
                "is_primary_input": True,
            }
            cache[cache_key] = result
            return result

        if len(inputs) > 12:
            result = {
                "kind": "net_function",
                "expression": None,
                "simplified_expression": None,
                "sum_of_products": None,
                "depends_on": [],
                "mux": None,
                "source_ports": {
                    "inputs": input_names,
                    "outputs": output_names,
                },
                "reason": "too_many_primary_inputs",
            }
            cache[cache_key] = result
            return result

        rows: list[dict] = []
        for bits in product([0, 1], repeat=len(inputs)):
            assignment = dict(zip(inputs, bits))
            value = self._evaluate_output_state(
                output_net=target_net,
                assignment=assignment,
                input_net_by_var=input_net_by_var,
                power_nets=power_nets,
                ground_nets=ground_nets,
                transistors=transistors,
            )
            rows.append({"inputs": assignment, "value": value})

        has_unknown = any(row.get("value") is None for row in rows)
        simplified = None if has_unknown else self._simplified_expression(inputs, rows)
        sop = None if has_unknown else self._sum_of_products(inputs, rows)
        mux = None if (has_unknown or not allow_mux) else self._detect_mux2_from_truth_table(inputs, rows)
        expression = mux.get("expression") if isinstance(mux, dict) and mux.get("expression") else simplified

        depends_on: list[str] = []
        if not has_unknown:
            for var in inputs:
                other_vars = [name for name in inputs if name != var]
                slices: dict[tuple[tuple[str, int], ...], dict[int, int | None]] = {}
                for row in rows:
                    key = tuple((name, int(row["inputs"].get(name, 0))) for name in other_vars)
                    slices.setdefault(key, {})[int(row["inputs"].get(var, 0))] = row.get("value")
                if any(values.get(0) != values.get(1) for values in slices.values() if 0 in values and 1 in values):
                    depends_on.append(var)

        result = {
            "kind": "net_function",
            "expression": expression,
            "simplified_expression": simplified,
            "sum_of_products": sop,
            "depends_on": depends_on,
            "mux": mux,
            "source_ports": {
                "inputs": input_names,
                "outputs": output_names,
            },
            "is_primary_input": False,
        }
        cache[cache_key] = result
        return result

    def _detect_mux2_from_truth_table(self, inputs: list[str], truth_table: list[dict]) -> dict | None:
        if not inputs or any(row.get("value") is None for row in truth_table):
            return None

        best: dict | None = None
        best_score: tuple[int, int] | None = None
        for select in inputs:
            others = [name for name in inputs if name != select]
            rows_by_other: dict[tuple[tuple[str, int], ...], dict[int, int]] = {}
            valid = True
            for row in truth_table:
                value = row.get("value")
                if value not in {0, 1}:
                    valid = False
                    break
                inputs_map = row.get("inputs", {})
                if select not in inputs_map:
                    valid = False
                    break
                key = tuple((name, int(inputs_map.get(name, 0))) for name in others)
                branch = int(inputs_map.get(select, 0))
                bucket = rows_by_other.setdefault(key, {})
                bucket[branch] = int(value)
            if not valid or not rows_by_other:
                continue
            if any(0 not in bucket or 1 not in bucket for bucket in rows_by_other.values()):
                continue
            if all(bucket[0] == bucket[1] for bucket in rows_by_other.values()):
                continue

            rows0 = []
            rows1 = []
            for key, bucket in rows_by_other.items():
                assignment = {name: bit for name, bit in key}
                rows0.append({"inputs": assignment, "value": bucket[0]})
                rows1.append({"inputs": assignment, "value": bucket[1]})
            rows0.sort(key=lambda row: tuple(row["inputs"].get(name, 0) for name in others))
            rows1.sort(key=lambda row: tuple(row["inputs"].get(name, 0) for name in others))
            expr0 = self._simplified_expression(others, rows0) if others or rows0 else None
            expr1 = self._simplified_expression(others, rows1) if others or rows1 else None
            if expr0 is None or expr1 is None or expr0 == expr1:
                continue

            mux_expr = f"mux({select}, {expr0}, {expr1})"
            score = (int(select.startswith("IN")), -(len(mux_expr)))
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "kind": "mux2",
                    "select": select,
                    "when0": expr0,
                    "when1": expr1,
                    "expression": mux_expr,
                }
        return best

    @staticmethod
    def _expanded_aliases(canonical_nets: list[str], nets_by_canonical: dict[str, list[str]]) -> list[str]:
        expanded: set[str] = set()
        for net in canonical_nets:
            expanded.update(nets_by_canonical.get(net, [net]))
        return sorted(expanded)

    @staticmethod
    def _unique_seq_entries(entries: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, str, str, str]] = set()
        for entry in entries:
            activation = entry.get("activation") if isinstance(entry.get("activation"), dict) else {}
            activation_mode = str(activation.get("mode", ""))
            activation_value = str(activation.get("edge", activation.get("level", "")))
            key = (
                str(entry.get("role", entry.get("port", ""))),
                str(entry.get("port", "")),
                str(entry.get("net", "")),
                f"{activation_mode}:{activation_value}",
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
        return out

    def _logic_path_for_output(
        self,
        output_net: str,
        transistors: list[_Transistor],
        nets_by_canonical: dict[str, list[str]],
    ) -> tuple[list[str], list[str]]:
        conduction_frontier = {output_net}
        dependency_frontier = {output_net}
        visited_dependency_nets: set[str] = set()
        used_instances: set[str] = set()
        used_transistors: set[int] = set()

        while dependency_frontier:
            current_target_nets = dependency_frontier - visited_dependency_nets
            if not current_target_nets:
                break
            visited_dependency_nets.update(current_target_nets)
            local_changed = True
            while local_changed:
                local_changed = False
                for idx, transistor in enumerate(transistors):
                    if idx in used_transistors:
                        continue
                    if {transistor.source, transistor.drain}.isdisjoint(current_target_nets | conduction_frontier):
                        continue
                    used_transistors.add(idx)
                    used_instances.add(transistor.instance_name)
                    before = len(conduction_frontier)
                    conduction_frontier.update({transistor.source, transistor.drain})
                    if len(conduction_frontier) != before:
                        local_changed = True

            next_dependencies: set[str] = set()
            for idx in used_transistors:
                gate_net = transistors[idx].gate
                if gate_net not in visited_dependency_nets:
                    next_dependencies.add(gate_net)
            dependency_frontier = next_dependencies

        expanded_nets: set[str] = set()
        for canonical in conduction_frontier | visited_dependency_nets:
            expanded_nets.update(nets_by_canonical.get(canonical, [canonical]))

        return (
            sorted(expanded_nets, key=self._natural_key),
            sorted(used_instances, key=self._natural_key),
        )

    @staticmethod
    def _build_active_graph(transistors: list[_Transistor], net_values: dict[str, bool | None]) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        def connect(a: str, b: str) -> None:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for transistor in transistors:
            gate_value = net_values.get(transistor.gate)
            if gate_value is None:
                continue
            is_on = gate_value if transistor.kind == "nmos" else (not gate_value)
            if is_on:
                connect(transistor.source, transistor.drain)
        return graph

    @staticmethod
    def _build_possible_graph(transistors: list[_Transistor], net_values: dict[str, bool | None]) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        def connect(a: str, b: str) -> None:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for transistor in transistors:
            gate_value = net_values.get(transistor.gate)
            if gate_value is None:
                # Unknown gate can still create a possible conduction path.
                connect(transistor.source, transistor.drain)
                continue
            is_on = gate_value if transistor.kind == "nmos" else (not gate_value)
            if is_on:
                connect(transistor.source, transistor.drain)
        return graph

    @staticmethod
    def _is_reachable(graph: dict[str, set[str]], start: str, targets: set[str]) -> bool:
        if start in targets:
            return True
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in graph.get(current, set()):
                if nxt in seen:
                    continue
                if nxt in targets:
                    return True
                seen.add(nxt)
                stack.append(nxt)
        return False

    def _evaluate_output_state(
        self,
        output_net: str,
        assignment: dict[str, int],
        input_net_by_var: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        transistors: list[_Transistor],
    ) -> int | None:
        net_values: dict[str, bool | None] = {}
        for variable, bit in assignment.items():
            input_net = input_net_by_var[variable]
            net_values[input_net] = bool(bit)
        for net in power_nets:
            net_values[net] = True
        for net in ground_nets:
            net_values[net] = False

        # Fixed-point propagation of known levels through currently-on transistor network.
        changed = True
        while changed:
            changed = False
            graph = self._build_active_graph(transistors, net_values)
            seen: set[str] = set()
            for start in graph:
                if start in seen:
                    continue
                stack = [start]
                component: list[str] = []
                seen.add(start)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for nxt in graph.get(node, set()):
                        if nxt in seen:
                            continue
                        seen.add(nxt)
                        stack.append(nxt)
                has_high = any(net_values.get(node) is True for node in component)
                has_low = any(net_values.get(node) is False for node in component)
                if has_high and has_low:
                    continue
                if has_high ^ has_low:
                    target = True if has_high else False
                    for node in component:
                        if net_values.get(node) is None:
                            net_values[node] = target
                            changed = True

        output_value = net_values.get(output_net)
        if output_value is True:
            return 1
        if output_value is False:
            return 0

        graph = self._build_active_graph(transistors, net_values)
        pull_down = self._is_reachable(graph, output_net, ground_nets)
        pull_up = self._is_reachable(graph, output_net, power_nets)
        if pull_down and not pull_up:
            return 0
        if pull_up and not pull_down:
            return 1

        # Fallback for partially-connected or under-specified EDF netlists:
        # infer from possible (not guaranteed) paths.
        possible_graph = self._build_possible_graph(transistors, net_values)
        possible_pull_down = self._is_reachable(possible_graph, output_net, ground_nets)
        possible_pull_up = self._is_reachable(possible_graph, output_net, power_nets)
        if possible_pull_up and not possible_pull_down:
            return 1
        if possible_pull_down and not possible_pull_up:
            return 0
        return None

    @staticmethod
    def _sum_of_products(inputs: list[str], truth_table: list[dict]) -> str:
        ones = [row for row in truth_table if row["value"] == 1]
        if not ones:
            return "0"
        if len(ones) == len(truth_table):
            return "1"
        terms: list[str] = []
        for row in ones:
            literals = []
            for variable in inputs:
                value = row["inputs"][variable]
                literals.append(variable if value else f"!{variable}")
            terms.append(" & ".join(literals))
        return " | ".join(terms)

    @classmethod
    def _simplified_expression(cls, inputs: list[str], truth_table: list[dict]) -> str:
        one_terms: list[str] = []
        zero_terms: list[str] = []
        dc_terms: list[str] = []
        for row in truth_table:
            bits = "".join("1" if row["inputs"].get(name) == 1 else "0" for name in inputs)
            value = row.get("value")
            if value == 1:
                one_terms.append(bits)
            elif value == 0:
                zero_terms.append(bits)
            elif value is None:
                dc_terms.append(bits)

        if not one_terms:
            return "0"
        if len(one_terms) == len(truth_table):
            return "1"

        parity_expr = cls._detect_parity_expression(inputs, truth_table)
        if parity_expr is not None:
            return parity_expr

        direct_expr = cls._simplify_from_on_set(inputs, truth_table, one_terms, dc_terms)
        inverted_expr = cls._invert_candidate(
            cls._simplify_from_on_set(inputs, truth_table, zero_terms, dc_terms)
        )
        return cls._pick_better_expression(direct_expr, inverted_expr)

    @classmethod
    def _simplify_from_on_set(
        cls,
        inputs: list[str],
        truth_table: list[dict],
        one_terms: list[str],
        dc_terms: list[str],
    ) -> str:
        if not one_terms:
            return "0"
        prime_implicants = cls._prime_implicants(one_terms, dc_terms)
        if not prime_implicants:
            return cls._sum_of_products(inputs, truth_table)

        selected = cls._select_cover(one_terms, prime_implicants)
        if not selected:
            return cls._sum_of_products(inputs, truth_table)
        literal_terms = [cls._implicant_to_literals(term, inputs) for term in selected]
        return cls._render_expression(literal_terms)

    @staticmethod
    def _invert_candidate(expr: str) -> str:
        if expr == "0":
            return "1"
        if expr == "1":
            return "0"
        if expr.startswith("!(") and expr.endswith(")"):
            return expr[2:-1]
        if " " in expr or "|" in expr or "&" in expr or "^" in expr:
            return f"!({expr})"
        return f"!{expr}"

    @staticmethod
    def _pick_better_expression(direct_expr: str, inverted_expr: str) -> str:
        def score(expr: str) -> tuple[int, int]:
            op_count = expr.count("&") + expr.count("|") + expr.count("^")
            return op_count, len(expr)

        return inverted_expr if score(inverted_expr) < score(direct_expr) else direct_expr

    @classmethod
    def _detect_parity_expression(cls, inputs: list[str], truth_table: list[dict]) -> str | None:
        if not inputs:
            return None
        if any(row.get("value") is None for row in truth_table):
            return None

        var_indexes = list(range(len(inputs)))
        best_expr: str | None = None
        best_score: tuple[int, int] | None = None

        # Search F = parity(selected vars) xor const.
        for mask in range(1, 1 << len(inputs)):
            selected = [idx for idx in var_indexes if mask & (1 << idx)]
            for const in (0, 1):
                matches = True
                for row in truth_table:
                    inputs_map = row.get("inputs", {})
                    parity = const
                    for idx in selected:
                        parity ^= 1 if inputs_map.get(inputs[idx]) == 1 else 0
                    if parity != row.get("value"):
                        matches = False
                        break
                if not matches:
                    continue

                base = " ^ ".join(inputs[idx] for idx in selected)
                expr = f"!({base})" if const == 1 and len(selected) > 1 else (f"!{base}" if const == 1 else base)
                score = (expr.count("^"), len(expr))
                if best_score is None or score < best_score:
                    best_score = score
                    best_expr = expr

        return best_expr

    @staticmethod
    def _combine_implicants(a: str, b: str) -> str | None:
        diff = 0
        out: list[str] = []
        for ca, cb in zip(a, b):
            if ca == cb:
                out.append(ca)
            elif ca != "-" and cb != "-":
                diff += 1
                out.append("-")
            else:
                return None
            if diff > 1:
                return None
        return "".join(out) if diff == 1 else None

    @classmethod
    def _prime_implicants(cls, one_terms: list[str], dc_terms: list[str]) -> list[str]:
        current = sorted(set(one_terms + dc_terms))
        prime_implicants: set[str] = set()

        while current:
            used: set[str] = set()
            next_terms: set[str] = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    combined = cls._combine_implicants(current[i], current[j])
                    if combined is not None:
                        used.add(current[i])
                        used.add(current[j])
                        next_terms.add(combined)
            for term in current:
                if term not in used:
                    prime_implicants.add(term)
            current = sorted(next_terms)
        return sorted(prime_implicants)

    @staticmethod
    def _implicant_covers(implicant: str, minterm: str) -> bool:
        for ci, cm in zip(implicant, minterm):
            if ci != "-" and ci != cm:
                return False
        return True

    @classmethod
    def _select_cover(cls, one_terms: list[str], prime_implicants: list[str]) -> list[str]:
        uncovered = set(one_terms)
        selected: list[str] = []

        coverage: dict[str, set[str]] = {
            implicant: {m for m in one_terms if cls._implicant_covers(implicant, m)} for implicant in prime_implicants
        }

        # Essential prime implicants
        for minterm in one_terms:
            covering = [implicant for implicant, covered in coverage.items() if minterm in covered]
            if len(covering) == 1 and covering[0] not in selected:
                selected.append(covering[0])
                uncovered -= coverage[covering[0]]

        # Greedy completion by best coverage, then by simplicity.
        while uncovered:
            best = None
            best_score: tuple[int, int, int, str] | None = None
            for implicant, covered in coverage.items():
                gain = len(covered & uncovered)
                if gain == 0:
                    continue
                score = (gain, implicant.count("-"), -implicant.count("1"), implicant)
                if best_score is None or score > best_score:
                    best = implicant
                    best_score = score
            if best is None:
                break
            if best not in selected:
                selected.append(best)
            uncovered -= coverage[best]

        return selected if not uncovered else []

    @staticmethod
    def _implicant_to_expr(implicant: str, inputs: list[str]) -> str:
        literals: list[str] = []
        for bit, variable in zip(implicant, inputs):
            if bit == "-":
                continue
            literals.append(variable if bit == "1" else f"!{variable}")
        if not literals:
            return "1"
        return " & ".join(literals)

    @staticmethod
    def _implicant_to_literals(implicant: str, inputs: list[str]) -> frozenset[tuple[str, bool]]:
        literals: set[tuple[str, bool]] = set()
        for bit, variable in zip(implicant, inputs):
            if bit == "-":
                continue
            is_negated = bit == "0"
            literals.add((variable, is_negated))
        return frozenset(literals)

    @staticmethod
    def _render_literal(literal: tuple[str, bool]) -> str:
        variable, is_negated = literal
        return f"!{variable}" if is_negated else variable

    @classmethod
    def _render_and_term(cls, term: frozenset[tuple[str, bool]]) -> str:
        if not term:
            return "1"
        ordered = sorted(term, key=lambda item: item[0])
        return " & ".join(cls._render_literal(literal) for literal in ordered)

    @staticmethod
    def _absorb_terms(terms: list[frozenset[tuple[str, bool]]]) -> list[frozenset[tuple[str, bool]]]:
        unique = sorted(set(terms), key=lambda term: (len(term), sorted(term)))
        out: list[frozenset[tuple[str, bool]]] = []
        for term in unique:
            if any(other.issubset(term) for other in unique if other is not term):
                continue
            out.append(term)
        return out

    @classmethod
    def _detect_xor_or_xnor(cls, terms: list[frozenset[tuple[str, bool]]]) -> str | None:
        if len(terms) != 2:
            return None
        t1, t2 = terms
        vars1 = {name for name, _ in t1}
        vars2 = {name for name, _ in t2}
        if vars1 != vars2 or len(vars1) != 2:
            return None

        ordered_vars = sorted(vars1)
        signs1: dict[str, bool] = {name: neg for name, neg in t1}
        signs2: dict[str, bool] = {name: neg for name, neg in t2}
        if any(signs1[var] == signs2[var] for var in ordered_vars):
            return None

        v1, v2 = ordered_vars
        xor_expr = f"{v1} ^ {v2}"
        is_xnor = signs1[v1] == signs1[v2]
        return f"!({xor_expr})" if is_xnor else xor_expr

    @classmethod
    def _render_expression(cls, terms: list[frozenset[tuple[str, bool]]]) -> str:
        terms = cls._absorb_terms(terms)
        if not terms:
            return "0"
        if any(len(term) == 0 for term in terms):
            return "1"
        if len(terms) == 1:
            return cls._render_and_term(terms[0])

        common = set(terms[0])
        for term in terms[1:]:
            common &= set(term)
        if common:
            common_term = frozenset(common)
            reduced_terms = [frozenset(set(term) - common) for term in terms]
            inner = cls._render_expression(reduced_terms)
            outer = cls._render_and_term(common_term)
            if inner == "1":
                return outer
            return f"{outer} & ({inner})"

        xor_or_xnor = cls._detect_xor_or_xnor(terms)
        if xor_or_xnor is not None:
            return xor_or_xnor

        rendered_terms = sorted(cls._render_and_term(term) for term in terms)
        if len(rendered_terms) == 1:
            return rendered_terms[0]
        return " | ".join(rendered_terms)

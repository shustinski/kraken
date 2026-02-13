from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

from logic_analyzer.application.ports import NetlistRepository
from logic_analyzer.domain.netlist import Instance, TopLevelNetlist
from logic_analyzer.infrastructure.edif_parser import EdifTextParser


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
            "meta": {
                "input_nets": input_net_by_var,
                "power_nets": sorted(power_nets, key=self._natural_key),
                "ground_nets": sorted(ground_nets, key=self._natural_key),
                "transistor_count": len(transistors),
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

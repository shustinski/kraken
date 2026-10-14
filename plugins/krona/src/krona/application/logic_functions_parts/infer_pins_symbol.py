"""logic_functions: infer pins symbol."""
from __future__ import annotations
import re
from krona.domain.netlist import TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser


class InferPinsSymbolMixin:
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

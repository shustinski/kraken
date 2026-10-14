"""logic_functions: build net alias map."""
from __future__ import annotations
import re
from krona.domain.netlist import Instance, TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser


class BuildNetAliasMapMixin:
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

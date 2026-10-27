"""logic_functions: build net alias map."""
from __future__ import annotations
import re
from krona.domain.netlist import Instance, TopLevelNetlist


class BuildNetAliasMapMixin:
    @staticmethod
    def _natural_key(value: str) -> tuple:
        return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))

    @staticmethod
    def _instance_kind(instance: Instance) -> str | None:
        payload = " ".join(filter(None, [instance.name, instance.cell, instance.view, instance.library])).upper()
        if "NMOS" in payload and "PMOS" not in payload:
            return "nmos"
        if "PMOS" in payload and "NMOS" not in payload:
            return "pmos"
        return None

    def _build_net_alias_map(self, netlist: TopLevelNetlist) -> dict[str, str]:
        # Each declared net remains its own identity. Repeated pin membership
        # is reported as a conflict by ExplicitConnectivity, never an alias.
        return {net.name: net.name for net in netlist.nets}

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
        from krona.domain.connectivity import ExplicitConnectivity

        connectivity = ExplicitConnectivity.from_netlist(netlist)
        result = connectivity.ports_with_direction("input")
        for port, net in connectivity.top_ports.items():
            if port not in connectivity.directions and re.fullmatch(r"IN\d+", port):
                result[port] = net
        # Explicit stimulus terminal membership is retained for legacy EDIF
        # generators. Merely drawing an input symbol at a point adds no link.
        for instance, pins in connectivity.pins.items():
            match = re.fullmatch(r"INSIN(\d+)", instance.upper())
            if not match or instance in connectivity.invalid_instances:
                continue
            names = {pins[p] for p in ("VPULSECR", "IN", "INPUT") if p in pins}
            if len(names) == 1:
                result.setdefault(f"IN{match.group(1)}", next(iter(names)))
        return result

    def _output_nets(self, top_page: str, netlist: TopLevelNetlist) -> dict[str, str]:
        from krona.domain.connectivity import ExplicitConnectivity

        connectivity = ExplicitConnectivity.from_netlist(netlist)
        outputs = connectivity.ports_with_direction("output")
        for port, net in connectivity.top_ports.items():
            if port not in connectivity.directions and re.fullmatch(r"OUT\d+", port):
                outputs[port] = net
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

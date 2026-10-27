"""logic_functions: infer pins symbol."""
from __future__ import annotations
from krona.domain.netlist import TopLevelNetlist


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
        # Symbol locations are display information. They cannot fill an
        # electrical terminal missing from joined/portRef declarations.
        return {}

    def _supply_nets(self, top_page: str, netlist: TopLevelNetlist) -> tuple[set[str], set[str]]:
        from krona.domain.connectivity import ExplicitConnectivity

        connectivity = ExplicitConnectivity.from_netlist(netlist)
        power_names = {"1", "VDD", "VCC", "+5", "5PLUS", "POWER"}
        ground_names = {"0", "&0", "GND", "VSS"}
        power = {net.name for net in netlist.nets if net.name.upper() in power_names}
        ground = {net.name for net in netlist.nets if net.name.upper() in ground_names}
        power.update(net for port, net in connectivity.top_ports.items() if port in power_names)
        ground.update(net for port, net in connectivity.top_ports.items() if port in ground_names)
        return power, ground

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
        # No input preference, peer-net guess, numeric proximity, or
        # declaration-order tie-break can resolve conflicting pin membership.
        candidates = set(gate_candidates)
        return next(iter(candidates)) if len(candidates) == 1 else None

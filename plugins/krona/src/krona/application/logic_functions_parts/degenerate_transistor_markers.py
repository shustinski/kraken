"""logic_functions: degenerate transistor markers."""
from __future__ import annotations
from krona.domain.netlist import TopLevelNetlist


class DegenerateTransistorMarkersMixin:
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
        transistors = self._transistors(
            netlist, input_net_by_var, power_nets, ground_nets, {}, top_page,
            view_index, point_to_nets, segments_by_net, alias_map,
        )
        markers = []
        for transistor in transistors:
            terminals = {transistor.source, transistor.drain}
            style = "sd_shorted" if len(terminals) == 1 else None
            if terminals & power_nets and terminals & ground_nets:
                style = "rail_bridge"
            if style:
                markers.append({"instance": transistor.instance_name, "kind": transistor.kind,
                                "gate": transistor.gate, "terminals": sorted(terminals),
                                "node": next(iter(terminals)) if len(terminals) == 1 else None,
                                "style": style})
        return markers

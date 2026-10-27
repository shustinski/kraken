"""Read complete unambiguous transistor terminals from explicit EDIF nets."""
from __future__ import annotations

from krona.domain.connectivity import ExplicitConnectivity
from krona.domain.netlist import TopLevelNetlist
from krona.application.logic_functions_parts.transistor import _Transistor


class TransistorsMixin:
    def _transistors(
        self, netlist: TopLevelNetlist, input_net_by_var: dict[str, str],
        power_nets: set[str], ground_nets: set[str], output_net_by_name: dict[str, str],
        top_page: str, view_index: dict, point_to_nets: dict, segments_by_net: dict, alias_map: dict,
    ) -> list[_Transistor]:
        connectivity = ExplicitConnectivity.from_netlist(netlist)
        result = []
        for instance in netlist.instances:
            kind = self._instance_kind(instance)
            if kind is None or instance.name in connectivity.invalid_instances:
                continue
            pins = connectivity.pins.get(instance.name, {})

            def terminal(*roles):
                names = {pins[role] for role in roles if role in pins}
                return next(iter(names)) if len(names) == 1 else None

            gate, source, drain = terminal("GATE", "G"), terminal("SOURCE", "S"), terminal("DRAIN", "D")
            if gate is None or source is None or drain is None:
                continue
            result.append(_Transistor(instance.name, kind, gate, source, drain))
        return result

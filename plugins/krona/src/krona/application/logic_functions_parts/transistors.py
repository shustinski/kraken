"""logic_functions: transistors."""
from __future__ import annotations
from krona.domain.netlist import TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser

from krona.application.logic_functions_parts.transistor import _Transistor


class TransistorsMixin:
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

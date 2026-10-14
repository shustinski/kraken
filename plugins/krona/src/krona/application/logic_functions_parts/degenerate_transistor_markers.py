"""logic_functions: degenerate transistor markers."""
from __future__ import annotations
from krona.domain.netlist import TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser


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

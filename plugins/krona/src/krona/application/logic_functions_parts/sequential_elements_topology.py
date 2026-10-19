"""logic_functions: sequential elements topology."""
from __future__ import annotations
from typing import TYPE_CHECKING
from krona.domain.netlist import TopLevelNetlist


class SequentialElementsTopologyMixin:
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

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor

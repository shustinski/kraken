"""logic_functions: execute."""
from __future__ import annotations
from itertools import product
from pathlib import Path


class ExecuteMixin:
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
        sequential_elements = self._sequential_elements(
            netlist=netlist,
            alias_map=alias_map,
            top_page=top_page,
            view_index=view_index,
            point_to_nets=point_to_nets,
            segments_by_net=segments_by_net,
            nets_by_canonical=nets_by_canonical,
            transistors=transistors,
            power_nets=power_nets,
            ground_nets=ground_nets,
            input_net_by_var=input_net_by_var,
            output_net_by_name=output_net_by_name,
        )
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
            "sequential_elements": sequential_elements,
            "meta": {
                "input_nets": input_net_by_var,
                "power_nets": sorted(power_nets, key=self._natural_key),
                "ground_nets": sorted(ground_nets, key=self._natural_key),
                "transistor_count": len(transistors),
                "sequential_count": len(sequential_elements),
                "flip_flop_count": sum(1 for item in sequential_elements if item.get("kind") == "flip_flop"),
                "latch_count": sum(1 for item in sequential_elements if item.get("kind") == "latch"),
            },
        }

"""logic_functions: evaluate output state."""
from __future__ import annotations


class EvaluateOutputStateMixin:
    def _evaluate_output_state(
        self,
        output_net: str,
        assignment: dict[str, int],
        input_net_by_var: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        transistors: list[_Transistor],
    ) -> int | None:
        net_values: dict[str, bool | None] = {}
        for variable, bit in assignment.items():
            input_net = input_net_by_var[variable]
            net_values[input_net] = bool(bit)
        for net in power_nets:
            net_values[net] = True
        for net in ground_nets:
            net_values[net] = False

        # Fixed-point propagation of known levels through currently-on transistor network.
        changed = True
        while changed:
            changed = False
            graph = self._build_active_graph(transistors, net_values)
            seen: set[str] = set()
            for start in graph:
                if start in seen:
                    continue
                stack = [start]
                component: list[str] = []
                seen.add(start)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for nxt in graph.get(node, set()):
                        if nxt in seen:
                            continue
                        seen.add(nxt)
                        stack.append(nxt)
                has_high = any(net_values.get(node) is True for node in component)
                has_low = any(net_values.get(node) is False for node in component)
                if has_high and has_low:
                    continue
                if has_high ^ has_low:
                    target = True if has_high else False
                    for node in component:
                        if net_values.get(node) is None:
                            net_values[node] = target
                            changed = True

        output_value = net_values.get(output_net)
        if output_value is True:
            return 1
        if output_value is False:
            return 0

        graph = self._build_active_graph(transistors, net_values)
        pull_down = self._is_reachable(graph, output_net, ground_nets)
        pull_up = self._is_reachable(graph, output_net, power_nets)
        if pull_down and not pull_up:
            return 0
        if pull_up and not pull_down:
            return 1

        # Fallback for partially-connected or under-specified EDF netlists:
        # infer from possible (not guaranteed) paths.
        possible_graph = self._build_possible_graph(transistors, net_values)
        possible_pull_down = self._is_reachable(possible_graph, output_net, ground_nets)
        possible_pull_up = self._is_reachable(possible_graph, output_net, power_nets)
        if possible_pull_up and not possible_pull_down:
            return 1
        if possible_pull_down and not possible_pull_up:
            return 0
        return None

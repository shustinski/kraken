"""logic_functions: evaluate output state."""
from __future__ import annotations
from typing import TYPE_CHECKING


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
        driven_levels: dict[str, set[bool]] = {}

        def drive(net, level):
            driven_levels.setdefault(net, set()).add(level)
            net_values[net] = level

        for variable, bit in assignment.items():
            input_net = input_net_by_var[variable]
            drive(input_net, bool(bit))
        for net in power_nets:
            drive(net, True)
        for net in ground_nets:
            drive(net, False)
        if any(len(levels) > 1 for levels in driven_levels.values()):
            return None

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

        graph = self._build_active_graph(transistors, net_values)
        # A previously propagated level must not hide a newly activated short.
        component = {output_net}
        pending = [output_net]
        while pending:
            node = pending.pop()
            for neighbor in graph.get(node, set()):
                if neighbor not in component:
                    component.add(neighbor)
                    pending.append(neighbor)
        levels = {net_values[node] for node in component if net_values.get(node) is not None}
        if len(levels) > 1:
            return None
        output_value = net_values.get(output_net)
        if output_value is not None:
            # A known pull-up cannot prove HIGH when a floating gate may
            # enable a competing pull-down (and conversely for LOW).
            possible = self._build_possible_graph(transistors, net_values)
            opposite_drivers = {net for net, levels in driven_levels.items() if (not output_value) in levels}
            if self._is_reachable(possible, output_net, opposite_drivers):
                return None
            return int(output_value)

        pull_down = self._is_reachable(graph, output_net, ground_nets)
        pull_up = self._is_reachable(graph, output_net, power_nets)
        if pull_down and not pull_up:
            return 0
        if pull_up and not pull_down:
            return 1

        # A possible path through an unknown gate does not prove conduction.
        return None

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor

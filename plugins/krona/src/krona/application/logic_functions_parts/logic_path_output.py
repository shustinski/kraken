"""logic_functions: logic path output."""
from __future__ import annotations
from typing import TYPE_CHECKING


class LogicPathOutputMixin:
    @staticmethod
    def _expanded_aliases(canonical_nets: list[str], nets_by_canonical: dict[str, list[str]]) -> list[str]:
        expanded: set[str] = set()
        for net in canonical_nets:
            expanded.update(nets_by_canonical.get(net, [net]))
        return sorted(expanded)

    @staticmethod
    def _unique_seq_entries(entries: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, str, str, str]] = set()
        for entry in entries:
            activation = entry.get("activation") if isinstance(entry.get("activation"), dict) else {}
            activation_mode = str(activation.get("mode", ""))
            activation_value = str(activation.get("edge", activation.get("level", "")))
            key = (
                str(entry.get("role", entry.get("port", ""))),
                str(entry.get("port", "")),
                str(entry.get("net", "")),
                f"{activation_mode}:{activation_value}",
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
        return out

    def _logic_path_for_output(
        self,
        output_net: str,
        transistors: list[_Transistor],
        nets_by_canonical: dict[str, list[str]],
    ) -> tuple[list[str], list[str]]:
        conduction_frontier = {output_net}
        dependency_frontier = {output_net}
        visited_dependency_nets: set[str] = set()
        used_instances: set[str] = set()
        used_transistors: set[int] = set()

        while dependency_frontier:
            current_target_nets = dependency_frontier - visited_dependency_nets
            if not current_target_nets:
                break
            visited_dependency_nets.update(current_target_nets)
            local_changed = True
            while local_changed:
                local_changed = False
                for idx, transistor in enumerate(transistors):
                    if idx in used_transistors:
                        continue
                    if {transistor.source, transistor.drain}.isdisjoint(current_target_nets | conduction_frontier):
                        continue
                    used_transistors.add(idx)
                    used_instances.add(transistor.instance_name)
                    before = len(conduction_frontier)
                    conduction_frontier.update({transistor.source, transistor.drain})
                    if len(conduction_frontier) != before:
                        local_changed = True

            next_dependencies: set[str] = set()
            for idx in used_transistors:
                gate_net = transistors[idx].gate
                if gate_net not in visited_dependency_nets:
                    next_dependencies.add(gate_net)
            dependency_frontier = next_dependencies

        expanded_nets: set[str] = set()
        for canonical in conduction_frontier | visited_dependency_nets:
            expanded_nets.update(nets_by_canonical.get(canonical, [canonical]))

        return (
            sorted(expanded_nets, key=self._natural_key),
            sorted(used_instances, key=self._natural_key),
        )

    @staticmethod
    def _build_active_graph(transistors: list[_Transistor], net_values: dict[str, bool | None]) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        def connect(a: str, b: str) -> None:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for transistor in transistors:
            gate_value = net_values.get(transistor.gate)
            if gate_value is None:
                continue
            is_on = gate_value if transistor.kind == "nmos" else (not gate_value)
            if is_on:
                connect(transistor.source, transistor.drain)
        return graph

    @staticmethod
    def _build_possible_graph(transistors: list[_Transistor], net_values: dict[str, bool | None]) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        def connect(a: str, b: str) -> None:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for transistor in transistors:
            gate_value = net_values.get(transistor.gate)
            if gate_value is None:
                # Unknown gate can still create a possible conduction path.
                connect(transistor.source, transistor.drain)
                continue
            is_on = gate_value if transistor.kind == "nmos" else (not gate_value)
            if is_on:
                connect(transistor.source, transistor.drain)
        return graph

    @staticmethod
    def _is_reachable(graph: dict[str, set[str]], start: str, targets: set[str]) -> bool:
        if start in targets:
            return True
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in graph.get(current, set()):
                if nxt in seen:
                    continue
                if nxt in targets:
                    return True
                seen.add(nxt)
                stack.append(nxt)
        return False

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor

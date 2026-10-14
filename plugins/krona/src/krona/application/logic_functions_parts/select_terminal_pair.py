"""logic_functions: select terminal pair."""
from __future__ import annotations


class SelectTerminalPairMixin:
    def _select_terminal_pair(
        self,
        kind: str,
        terminal_candidates: list[str],
        power_nets: set[str],
        ground_nets: set[str],
        net_connection_degree: dict[str, int],
        pmos_power_adjacent_nets: set[str],
        nmos_ground_adjacent_nets: set[str],
        output_nets: set[str],
    ) -> tuple[str, str] | None:
        terminal_candidates = self._unique(terminal_candidates)
        options: list[tuple[str, str]] = []
        for idx, left in enumerate(terminal_candidates):
            for right in terminal_candidates[idx + 1:]:
                if left != right:
                    options.append((left, right))
        if not options:
            return None

        def score(option: tuple[str, str]) -> tuple[int, int, int]:
            left, right = option
            supply_score = 0
            if kind == "pmos":
                if left in power_nets or right in power_nets:
                    supply_score += 4
                if left in pmos_power_adjacent_nets or right in pmos_power_adjacent_nets:
                    supply_score += 2
            else:
                if left in ground_nets or right in ground_nets:
                    supply_score += 4
                if left in nmos_ground_adjacent_nets or right in nmos_ground_adjacent_nets:
                    supply_score += 2
            output_score = 2 if (left in output_nets or right in output_nets) else 0
            degree_score = net_connection_degree.get(left, 0) + net_connection_degree.get(right, 0)
            return supply_score, output_score, degree_score

        options.sort(key=lambda option: score(option), reverse=True)
        return options[0]

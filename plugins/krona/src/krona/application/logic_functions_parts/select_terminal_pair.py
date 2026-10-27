"""Compatibility helper: do not rank ambiguous terminal candidates."""
from __future__ import annotations


class SelectTerminalPairMixin:
    def _select_terminal_pair(
        self, kind, terminal_candidates, power_nets, ground_nets, net_connection_degree,
        pmos_power_adjacent_nets, nmos_ground_adjacent_nets, output_nets,
    ):
        names = self._unique(terminal_candidates)
        return tuple(names) if len(names) == 2 else None

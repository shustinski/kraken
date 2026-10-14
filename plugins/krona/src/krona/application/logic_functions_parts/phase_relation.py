"""logic_functions: phase relation."""
from __future__ import annotations


class PhaseRelationMixin:
    @staticmethod
    def _phase_relation(
        left: dict | None,
        right: dict | None,
        inverter_map: dict[tuple[str, str], dict],
    ) -> str | None:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None
        left_net = str(left.get("net") or "")
        right_net = str(right.get("net") or "")
        left_level = str(left.get("level") or "")
        right_level = str(right.get("level") or "")
        if left_level not in {"high", "low"} or right_level not in {"high", "low"}:
            return None
        if not left_net or not right_net:
            return None
        if left_net == right_net:
            return "same" if left_level == right_level else "opposite"
        inverted = (left_net, right_net) in inverter_map or (right_net, left_net) in inverter_map
        if not inverted:
            return None
        return "opposite" if left_level == right_level else "same"

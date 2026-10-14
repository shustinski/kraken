"""logic_functions: controls are opposite phase."""
from __future__ import annotations


class ControlsAreOppositePhaseMixin:
    @staticmethod
    def _controls_are_opposite_phase(
        left: dict,
        right: dict,
        inverter_map: dict[tuple[str, str], dict],
    ) -> bool:
        left_net = str(left.get("net", ""))
        right_net = str(right.get("net", ""))
        left_level = str(left.get("level", ""))
        right_level = str(right.get("level", ""))
        if not left_net or not right_net or left_level not in {"high", "low"} or right_level not in {"high", "low"}:
            return False
        if left_net == right_net:
            return left_level != right_level
        inverted = (left_net, right_net) in inverter_map or (right_net, left_net) in inverter_map
        return inverted and left_level == right_level

    @staticmethod
    def _nets_share_or_invert_path(
        source_nets: set[str],
        target_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
    ) -> int:
        if source_nets & target_nets:
            return 3
        for source in source_nets:
            for target in target_nets:
                if (source, target) in inverter_map or (target, source) in inverter_map:
                    return 2
        return 0

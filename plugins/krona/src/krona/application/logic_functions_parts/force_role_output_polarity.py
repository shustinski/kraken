"""logic_functions: force role output polarity."""
from __future__ import annotations


class ForceRoleOutputPolarityMixin:
    @staticmethod
    def _force_role_from_output_polarity(core_net: str, forced_value: int, output_polarity_by_net: dict[str, str]) -> str | None:
        polarity = output_polarity_by_net.get(core_net)
        if polarity is None:
            return None
        if polarity == "non_inverted":
            return "set" if forced_value == 1 else "reset"
        return "reset" if forced_value == 1 else "set"

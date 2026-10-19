"""logic_functions: inverter instances between."""
from __future__ import annotations
from typing import TYPE_CHECKING


class InverterInstancesBetweenMixin:
    @staticmethod
    def _nets_are_inverted(left: str, right: str, inverter_map: dict[tuple[str, str], dict]) -> bool:
        return (left, right) in inverter_map or (right, left) in inverter_map

    @staticmethod
    def _inverter_instances_between(left: str, right: str, inverter_map: dict[tuple[str, str], dict]) -> list[str]:
        instances: list[str] = []
        for key in ((left, right), (right, left)):
            record = inverter_map.get(key)
            if record:
                instances.extend(record.get("instances", []))
        out: list[str] = []
        for item in instances:
            if item not in out:
                out.append(item)
        return out

    def _terminal_degree(self, transistors: list[_Transistor]) -> dict[str, int]:
        degree: dict[str, int] = {}
        for transistor in transistors:
            for net in self._unique([transistor.source, transistor.drain]):
                degree[net] = degree.get(net, 0) + 1
        return degree

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor

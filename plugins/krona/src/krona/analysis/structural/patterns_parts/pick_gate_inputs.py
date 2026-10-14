"""patterns: pick gate inputs."""
from __future__ import annotations
from krona.analysis.structural.model import (
    CircuitDevice,
)


class PickGateInputsMixin:
    @staticmethod
    def _looks_like_clock(net: str) -> bool:
        upper = net.upper()
        return "CLK" in upper or "CLOCK" in upper or upper in {"CK", "CP"}

    @staticmethod
    def _pick_pin(device: CircuitDevice, *aliases: str) -> str | None:
        normalized = {name.upper(): net for name, net in device.pins.items()}
        for alias in aliases:
            value = normalized.get(alias.upper())
            if value:
                return value
        return None

    def _pick_gate_output(self, device: CircuitDevice) -> str | None:
        return self._pick_pin(device, "Y", "Z", "Q", "OUT", "O")

    def _pick_gate_inputs(self, device: CircuitDevice) -> list[str]:
        out = self._pick_gate_output(device)
        result: list[str] = []
        for pin, net in device.pins.items():
            if net == out:
                continue
            if pin.upper() in {"VDD", "VSS", "GND", "VCC", "VPWR", "VGND"}:
                continue
            result.append(net)
        seen: set[str] = set()
        unique: list[str] = []
        for net in result:
            if net in seen:
                continue
            seen.add(net)
            unique.append(net)
        return unique

    @staticmethod
    def _dedupe_dicts(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for item in items:
            key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

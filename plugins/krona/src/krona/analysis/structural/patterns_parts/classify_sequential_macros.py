"""patterns: classify sequential macros."""
from __future__ import annotations
from krona.analysis.structural.model import (
    ActivationMode,
    ActivationSpec,
    CompositeFeature,
    LogicLevel,
    RecognizedCellKind,
    RecognizedStructure,
)


class ClassifySequentialMacrosMixin:
    def _classify_sequential_macros(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        out: list[RecognizedStructure] = []
        clock_gated_nets = {item.get("gated_clock") for item in patterns.clock_gating_features}
        toggle_features = {str(item.get("ff_device")): item for item in patterns.xor_toggle_features}

        for device in self._circuit.devices.values():
            if device.domain.name != "SEQUENTIAL_MACRO":
                continue
            q = self._pin(device, "Q", "QN", "QB", "OUT", "Z")
            d = self._pin(device, "D", "DATA", "IN")
            clk = self._pin(device, "CLK", "CLOCK", "CK", "CP")
            rst = self._pin(device, "RST", "RESET", "CLR", "RN", "CLRB", "RESETB")
            set_ = self._pin(device, "SET", "PRE", "SN", "PREB", "SETB")
            sync_rst = self._pin(device, "SRST", "SRESET", "SCLR", "SCLRN", "SRN")
            sync_set = self._pin(device, "SSET", "SPRE", "SPRESET", "SPREN", "SSETN")
            en = self._pin(device, "EN", "E", "G", "CE")
            j = self._pin(device, "J")
            k = self._pin(device, "K")
            t = self._pin(device, "T")

            kind = RecognizedCellKind.UNKNOWN_STORAGE
            base_conf = 0.75
            if device.kind.name == "LATCH_D":
                kind = RecognizedCellKind.GATED_D_LATCH
            elif device.kind.name == "LATCH_RS":
                kind = RecognizedCellKind.RS_LATCH
            elif device.kind.name == "FF_D":
                kind = RecognizedCellKind.D_FLIP_FLOP
            elif device.kind.name == "FF_JK":
                kind = RecognizedCellKind.JK_FLIP_FLOP
            elif device.kind.name == "FF_T":
                kind = RecognizedCellKind.T_FLIP_FLOP
            elif device.kind.name == "COUNTER":
                kind = RecognizedCellKind.COUNTER_CELL

            controls: list[dict] = []
            features: list[CompositeFeature] = []
            inputs = [net for net in [d, j, k, t] if net]
            outputs = [q] if q else []
            clocks = [clk] if clk else []

            if clk:
                controls.append({"role": "clock", "net": clk, "activation": {"mode": "edge_or_level_unknown"}})
                if clk in clock_gated_nets:
                    features.append(CompositeFeature.CLOCK_GATING)
            if en:
                controls.append(
                    {
                        "role": "enable",
                        "net": en,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(en) else LogicLevel.HIGH,
                        ).as_dict(),
                    }
                )
                features.append(CompositeFeature.CLOCK_ENABLE)
            if rst:
                controls.append(
                    {
                        "role": "reset",
                        "net": rst,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(rst) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "asynchronous",
                    }
                )
                features.append(CompositeFeature.ASYNC_RESET)
            if set_:
                controls.append(
                    {
                        "role": "set",
                        "net": set_,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(set_) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "asynchronous",
                    }
                )
                features.append(CompositeFeature.ASYNC_SET)
            if sync_rst:
                controls.append(
                    {
                        "role": "reset",
                        "net": sync_rst,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(sync_rst) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "synchronous",
                    }
                )
                features.append(CompositeFeature.SYNC_RESET)
            if sync_set:
                controls.append(
                    {
                        "role": "set",
                        "net": sync_set,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(sync_set) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "synchronous",
                    }
                )
                features.append(CompositeFeature.SYNC_SET)

            evidence = {
                "macro_device": device.name,
                "device_kind": device.kind.value,
                "pins": dict(device.pins),
            }
            if device.name in toggle_features and kind in {RecognizedCellKind.D_FLIP_FLOP, RecognizedCellKind.EDGE_TRIGGERED_DFF}:
                features.append(CompositeFeature.XOR_TOGGLE_FEEDBACK)
                evidence["xor_toggle_feedback"] = toggle_features[device.name]

            out.append(
                RecognizedStructure(
                    kind=kind,
                    confidence=base_conf,
                    storage_nodes=outputs or ([q] if q else []),
                    outputs=outputs,
                    inputs=inputs,
                    clocks=clocks,
                    controls=controls,
                    features=self._unique_features(features),
                    evidence=evidence,
                    components=[device.name],
                )
            )
        return out

"""patterns: detect jk feedback equation."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Iterable
from krona.analysis.structural.model import (
    CircuitDevice,
    CompositeFeature,
    RecognizedStructure,
)


class DetectJkFeedbackEquationMixin:
    def _detect_jk_feedback_equation(self, item: RecognizedStructure) -> dict | None:
        d_net = None
        if isinstance(item.evidence, dict):
            pins = item.evidence.get("pins", {})
            if isinstance(pins, dict):
                d_net = pins.get("D") or pins.get("DATA") or pins.get("IN")
        q_nets = set(item.outputs or item.storage_nodes)
        if not d_net:
            return None
        incoming = self._graphs.influence.incoming.get(d_net, [])
        feedback_sources = [edge.source for edge in incoming if edge.source in q_nets]
        external_sources = [edge.source for edge in incoming if edge.source not in q_nets]
        if not feedback_sources or len(set(external_sources)) < 2:
            return None
        has_inv = any(edge.inversion == 1 for edge in incoming)
        has_noninv = any(edge.inversion == 0 for edge in incoming)
        if not (has_inv and has_noninv):
            return None
        return {
            "d_net": d_net,
            "feedback_sources": sorted(set(feedback_sources)),
            "inputs": sorted(set(external_sources))[:2],
            "method": "mixed_polarity_feedback_cone_heuristic",
        }

    def _controls_from_async(self, controls: list[AsyncControlCandidate]) -> list[dict]:
        return [
            {
                "role": control.role,
                "net": control.signal,
                "activation": {"mode": "level", "level": control.active_level.value},
                "timing": "asynchronous",
            }
            for control in controls
        ]

    def _phase_profile_to_dict(self, profile: StoragePhaseProfile) -> dict:
        return {
            "storage_nodes": list(profile.storage_nodes),
            "data_sources": list(profile.data_sources),
            "feedback_sources": list(profile.feedback_sources),
            "control_signals": list(profile.control_signals),
            "transparency": profile.transparency,
            "hold_checks": profile.hold_checks,
        }

    def _core_component_devices(self, core: BistableCoreCandidate) -> list[str]:
        core_set = set(core.nodes)
        components: set[str] = set()
        for edge in self._graphs.influence.edges:
            if edge.source in core_set or edge.target in core_set:
                components.update(edge.through_devices)
        for edge in self._graphs.conduction.edges:
            if edge.source in core_set or edge.target in core_set:
                components.update(edge.through_devices)
        return sorted(components)

    def _influence_path_exists(self, source: str, target: str) -> bool:
        return self._phase.influence_reachable(source, target, assignment={}, allow_unknown_controls=True)

    @staticmethod
    def _pin(device: CircuitDevice, *aliases: str) -> str | None:
        normalized = {name.upper(): net for name, net in device.pins.items()}
        for alias in aliases:
            value = normalized.get(alias.upper())
            if value:
                return value
        return None

    @staticmethod
    def _is_active_low_name(net: str) -> bool:
        upper = net.upper()
        return upper.endswith("B") or upper.endswith("_N") or upper.endswith("N") or upper.endswith("BAR")

    @classmethod
    def _is_likely_complement_name_pair(cls, a: str, b: str) -> bool:
        return cls._base_clock_name(a, b) is not None

    @staticmethod
    def _base_clock_name(a: str, b: str) -> str | None:
        au = a.upper()
        bu = b.upper()
        candidates = []
        for name in (au, bu):
            if name.endswith("B"):
                candidates.append(name[:-1])
            if name.endswith("_N"):
                candidates.append(name[:-2])
            if name.endswith("N"):
                candidates.append(name[:-1])
            if name.endswith("BAR"):
                candidates.append(name[:-3])
        for base in candidates:
            if not base:
                continue
            variants = {base + "B", base + "_N", base + "N", base + "BAR"}
            if au == base and bu in variants:
                return a
            if bu == base and au in variants:
                return b
        return None

    def _infer_edge_polarity(self, master: RecognizedStructure, slave: RecognizedStructure, base_clock: str) -> str:
        master_level = self._dominant_clock_level(master, base_clock)
        slave_level = self._dominant_clock_level(slave, base_clock)
        if master_level == 0 and slave_level == 1:
            return "rising"
        if master_level == 1 and slave_level == 0:
            return "falling"
        return "unknown"

    @staticmethod
    def _dominant_clock_level(item: RecognizedStructure, base_clock: str) -> int | None:
        evidence = item.evidence if isinstance(item.evidence, dict) else {}
        phase_analysis = evidence.get("phase_analysis", [])
        counts = {0: 0, 1: 0}
        if not isinstance(phase_analysis, list):
            return None
        for profile in phase_analysis:
            if not isinstance(profile, dict):
                continue
            transparency = profile.get("transparency", {})
            if not isinstance(transparency, dict):
                continue
            for phase in transparency.values():
                if not isinstance(phase, dict) or not phase.get("transparent"):
                    continue
                assignment = phase.get("assignment", {})
                if isinstance(assignment, dict) and base_clock in assignment:
                    counts[int(assignment[base_clock])] += 1
        if counts[0] == counts[1] == 0:
            return None
        return 1 if counts[1] > counts[0] else 0

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @staticmethod
    def _unique_features(features: Iterable[CompositeFeature]) -> list[CompositeFeature]:
        out: list[CompositeFeature] = []
        seen: set[CompositeFeature] = set()
        for feature in features:
            if feature in seen:
                continue
            seen.add(feature)
            out.append(feature)
        return out

    @staticmethod
    def _dedupe_recognized(items: list[RecognizedStructure]) -> list[RecognizedStructure]:
        by_key: dict[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], RecognizedStructure] = {}
        for item in items:
            key = (
                item.kind.value,
                tuple(sorted(item.storage_nodes)),
                tuple(sorted(item.outputs)),
                tuple(sorted(item.components)),
            )
            prev = by_key.get(key)
            if prev is None or item.confidence > prev.confidence:
                by_key[key] = item
        return sorted(
            by_key.values(),
            key=lambda item: (-item.confidence, item.kind.value, ",".join(sorted(item.storage_nodes))),
        )

if TYPE_CHECKING:
    from krona.analysis.structural.patterns_parts.storage_phase_profile import AsyncControlCandidate, BistableCoreCandidate, StoragePhaseProfile

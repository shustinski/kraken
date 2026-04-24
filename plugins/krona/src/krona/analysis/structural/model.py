from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeviceDomain(str, Enum):
    SWITCH = "switch"
    GATE = "gate"
    SEQUENTIAL_MACRO = "sequential_macro"
    ANALOG = "analog"
    UNKNOWN = "unknown"


class DeviceKind(str, Enum):
    NMOS = "nmos"
    PMOS = "pmos"
    PASS_NMOS = "pass_nmos"
    PASS_PMOS = "pass_pmos"
    TRANSMISSION_GATE = "transmission_gate"
    INV = "inv"
    BUF = "buf"
    NAND = "nand"
    NOR = "nor"
    AND = "and"
    OR = "or"
    XOR = "xor"
    XNOR = "xnor"
    MUX2 = "mux2"
    LATCH_D = "latch_d"
    LATCH_RS = "latch_rs"
    FF_D = "ff_d"
    FF_JK = "ff_jk"
    FF_T = "ff_t"
    COUNTER = "counter"
    UNKNOWN = "unknown"


class NodeKind(str, Enum):
    NET = "net"
    DEVICE = "device"
    STORAGE = "storage"


class EdgeSemantic(str, Enum):
    INFLUENCE = "influence"
    CONDUCTION = "conduction"
    FEEDBACK = "feedback"
    CLOCKING = "clocking"
    ASYNC_CONTROL = "async_control"


class ActivationMode(str, Enum):
    LEVEL = "level"
    EDGE = "edge"
    UNKNOWN = "unknown"


class LogicLevel(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class EdgePolarity(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    UNKNOWN = "unknown"


class RecognizedCellKind(str, Enum):
    RS_LATCH = "rs_latch"
    GATED_D_LATCH = "gated_d_latch"
    D_FLIP_FLOP = "d_flip_flop"
    MASTER_SLAVE_DFF = "master_slave_dff"
    EDGE_TRIGGERED_DFF = "edge_triggered_dff"
    JK_FLIP_FLOP = "jk_flip_flop"
    T_FLIP_FLOP = "t_flip_flop"
    COUNTER_CELL = "counter_cell"
    DYNAMIC_LATCH = "dynamic_latch"
    UNKNOWN_STORAGE = "unknown_storage"


class CompositeFeature(str, Enum):
    ASYNC_RESET = "async_reset"
    ASYNC_SET = "async_set"
    SYNC_RESET = "sync_reset"
    SYNC_SET = "sync_set"
    CLOCK_ENABLE = "clock_enable"
    CLOCK_GATING = "clock_gating"
    INPUT_MUX = "input_mux"
    FEEDBACK_MUX = "feedback_mux"
    CROSS_COUPLED_INVERTERS = "cross_coupled_inverters"
    KEEPER = "keeper"
    DYNAMIC_PRECHARGE_EVALUATE = "dynamic_precharge_evaluate"
    XOR_TOGGLE_FEEDBACK = "xor_toggle_feedback"
    TOGGLE_CHAIN = "toggle_chain"


@dataclass(frozen=True)
class ActivationSpec:
    mode: ActivationMode = ActivationMode.UNKNOWN
    level: LogicLevel = LogicLevel.UNKNOWN
    edge: EdgePolarity = EdgePolarity.UNKNOWN

    def as_dict(self) -> dict[str, str]:
        payload = {"mode": self.mode.value}
        if self.mode == ActivationMode.LEVEL:
            payload["level"] = self.level.value
        elif self.mode == ActivationMode.EDGE:
            payload["edge"] = self.edge.value
        return payload


@dataclass(frozen=True)
class ConditionLiteral:
    signal: str
    required_level: bool
    source_device: str | None = None
    negated: bool = False

    def evaluate(self, assignment: dict[str, bool]) -> bool | None:
        value = assignment.get(self.signal)
        if value is None:
            return None
        return value == self.required_level

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "required_level": int(self.required_level),
            "source_device": self.source_device,
            "negated": self.negated,
        }


@dataclass(frozen=True)
class ConditionalEdge:
    source: str
    target: str
    semantic: EdgeSemantic
    inversion: int | None = None
    controls: tuple[ConditionLiteral, ...] = ()
    through_devices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_enabled(self, assignment: dict[str, bool]) -> bool | None:
        unknown_seen = False
        for literal in self.controls:
            decision = literal.evaluate(assignment)
            if decision is None:
                unknown_seen = True
                continue
            if decision is False:
                return False
        if unknown_seen:
            return None
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "semantic": self.semantic.value,
            "inversion": self.inversion,
            "controls": [item.as_dict() for item in self.controls],
            "through_devices": list(self.through_devices),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CircuitNet:
    name: str
    is_power: bool = False
    is_ground: bool = False
    top_port_name: str | None = None
    port_direction: str | None = None


@dataclass(frozen=True)
class CircuitDevice:
    name: str
    kind: DeviceKind
    domain: DeviceDomain
    cell_name: str | None = None
    view_name: str | None = None
    library_name: str | None = None
    pins: dict[str, str] = field(default_factory=dict)
    pin_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def pin(self, *names: str) -> str | None:
        normalized = {key.upper(): value for key, value in self.pins.items()}
        for name in names:
            value = normalized.get(name.upper())
            if value:
                return value
        return None


@dataclass(frozen=True)
class CircuitModel:
    source_format: str
    design_name: str | None
    nets: dict[str, CircuitNet]
    devices: dict[str, CircuitDevice]
    device_order: tuple[str, ...]
    top_ports: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def power_nets(self) -> set[str]:
        return {name for name, net in self.nets.items() if net.is_power}

    def ground_nets(self) -> set[str]:
        return {name for name, net in self.nets.items() if net.is_ground}


@dataclass(frozen=True)
class BistableCoreEvidence:
    scc_nodes: tuple[str, ...]
    positive_feedback: bool
    stable_states: list[dict[str, int]]
    cross_coupled_inverters: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ClockChainEvidence:
    clocks: list[str]
    inverted_pairs: list[tuple[str, str]]
    switch_controlled_edges: list[dict[str, Any]]
    transparency_by_phase: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MuxEvidence:
    storage_node: str
    select_net: str | None
    data_path_sources: list[str]
    feedback_path_sources: list[str]
    edges: list[dict[str, Any]]


@dataclass(frozen=True)
class ResetSetEvidence:
    signal: str
    role: str
    active_level: LogicLevel
    asynchronous: bool
    target_nodes: list[str]
    path_edges: list[dict[str, Any]]


@dataclass(frozen=True)
class RecognizedStructure:
    kind: RecognizedCellKind
    confidence: float
    storage_nodes: list[str]
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    clocks: list[str] = field(default_factory=list)
    controls: list[dict[str, Any]] = field(default_factory=list)
    features: list[CompositeFeature] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    components: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": round(self.confidence, 3),
            "storage_nodes": list(self.storage_nodes),
            "outputs": list(self.outputs),
            "inputs": list(self.inputs),
            "clocks": list(self.clocks),
            "controls": list(self.controls),
            "features": [feature.value for feature in self.features],
            "evidence": dict(self.evidence),
            "components": list(self.components),
        }


@dataclass(frozen=True)
class GraphSummary:
    influence_node_count: int
    influence_edge_count: int
    conditional_edge_count: int
    scc_count: int
    storage_scc_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "influence_node_count": self.influence_node_count,
            "influence_edge_count": self.influence_edge_count,
            "conditional_edge_count": self.conditional_edge_count,
            "scc_count": self.scc_count,
            "storage_scc_count": self.storage_scc_count,
        }


@dataclass(frozen=True)
class StructuralAnalysisReport:
    design_name: str | None
    source_format: str
    graph: GraphSummary
    recognized_structures: list[RecognizedStructure]
    clocks: dict[str, Any]
    features: dict[str, Any]
    sccs: list[dict[str, Any]]
    diagnostics: list[str] = field(default_factory=list)
    method_limitations: list[str] = field(default_factory=list)
    sat_extension_plan: list[str] = field(default_factory=list)
    dynamic_support_plan: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "design_name": self.design_name,
            "source_format": self.source_format,
            "graph": self.graph.as_dict(),
            "recognized_structures": [item.as_dict() for item in self.recognized_structures],
            "clocks": dict(self.clocks),
            "features": dict(self.features),
            "sccs": list(self.sccs),
            "diagnostics": list(self.diagnostics),
            "method_limitations": list(self.method_limitations),
            "sat_extension_plan": list(self.sat_extension_plan),
            "dynamic_support_plan": list(self.dynamic_support_plan),
            "metadata": dict(self.metadata),
        }

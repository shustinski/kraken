from __future__ import annotations

from dataclasses import dataclass

from .model import (
    CircuitDevice,
    CircuitModel,
    ConditionalEdge,
    ConditionLiteral,
    DeviceDomain,
    DeviceKind,
    EdgeSemantic,
)


@dataclass(frozen=True)
class InfluenceGraph:
    nodes: tuple[str, ...]
    edges: tuple[ConditionalEdge, ...]
    adjacency: dict[str, set[str]]
    incoming: dict[str, list[ConditionalEdge]]
    outgoing: dict[str, list[ConditionalEdge]]


@dataclass(frozen=True)
class ConditionalConductionGraph:
    nodes: tuple[str, ...]
    edges: tuple[ConditionalEdge, ...]
    adjacency: dict[str, list[ConditionalEdge]]

    def enabled_neighbors(
        self,
        node: str,
        assignment: dict[str, bool],
        *,
        allow_unknown_controls: bool = False,
    ) -> set[str]:
        neighbors: set[str] = set()
        for edge in self.adjacency.get(node, []):
            state = edge.is_enabled(assignment)
            if state is True or (allow_unknown_controls and state is None):
                neighbors.add(edge.target)
        return neighbors


@dataclass(frozen=True)
class GraphBundle:
    influence: InfluenceGraph
    conduction: ConditionalConductionGraph
    all_edges: tuple[ConditionalEdge, ...]
    device_to_edges: dict[str, list[ConditionalEdge]]
    diagnostics: tuple[str, ...] = ()


class GraphBuilder:
    """
    Builds signal-influence graph for switch-level and gate-level devices.

    Influence edges are conditional where a data dependency only exists in a phase
    (e.g. transmission gate transparent only when EN=1 and ENB=0).
    """

    _POWER_PIN_ALIASES = ("VDD", "VCC", "VPWR", "POWER")
    _GROUND_PIN_ALIASES = ("VSS", "GND", "VGND")

    def build(self, circuit: CircuitModel) -> GraphBundle:
        influence_edges: list[ConditionalEdge] = []
        conduction_edges: list[ConditionalEdge] = []
        device_to_edges: dict[str, list[ConditionalEdge]] = {}
        diagnostics: list[str] = []

        transistor_devices: list[CircuitDevice] = []
        for device_name in circuit.device_order:
            device = circuit.devices[device_name]
            if device.kind in {DeviceKind.NMOS, DeviceKind.PMOS, DeviceKind.PASS_NMOS, DeviceKind.PASS_PMOS}:
                transistor_devices.append(device)
            dev_infl, dev_cond, dev_diag = self._edges_for_device(device, circuit)
            influence_edges.extend(dev_infl)
            conduction_edges.extend(dev_cond)
            diagnostics.extend(dev_diag)
            if dev_infl or dev_cond:
                device_to_edges.setdefault(device.name, []).extend([*dev_infl, *dev_cond])

        # Synthesize inverter/buffer influence edges from transistor pull-up/pull-down stacks
        # to enable SCC polarity analysis on transistor-level netlists.
        cmos_macro_edges, cmos_diags = self._synthesize_cmos_gate_edges(transistor_devices, circuit)
        influence_edges.extend(cmos_macro_edges)
        diagnostics.extend(cmos_diags)
        for edge in cmos_macro_edges:
            for dev in edge.through_devices:
                device_to_edges.setdefault(dev, []).append(edge)

        influence = self._build_influence_graph(circuit, influence_edges)
        conduction = self._build_conduction_graph(circuit, conduction_edges)
        all_edges = tuple([*influence_edges, *conduction_edges])
        return GraphBundle(
            influence=influence,
            conduction=conduction,
            all_edges=all_edges,
            device_to_edges=device_to_edges,
            diagnostics=tuple(diagnostics),
        )

    def _edges_for_device(
        self,
        device: CircuitDevice,
        circuit: CircuitModel,
    ) -> tuple[list[ConditionalEdge], list[ConditionalEdge], list[str]]:
        if device.domain == DeviceDomain.SWITCH:
            return self._switch_device_edges(device, circuit)
        if device.domain in {DeviceDomain.GATE, DeviceDomain.SEQUENTIAL_MACRO}:
            return self._logic_device_edges(device, circuit)
        return [], [], []

    def _switch_device_edges(
        self,
        device: CircuitDevice,
        circuit: CircuitModel,
    ) -> tuple[list[ConditionalEdge], list[ConditionalEdge], list[str]]:
        influence: list[ConditionalEdge] = []
        conduction: list[ConditionalEdge] = []
        diagnostics: list[str] = []

        if device.kind in {DeviceKind.NMOS, DeviceKind.PMOS, DeviceKind.PASS_NMOS, DeviceKind.PASS_PMOS}:
            gate = self._pin(device, "G", "GATE", "CTRL", "CONTROL")
            src = self._pin(device, "S", "SOURCE", "A", "P1", "1")
            drn = self._pin(device, "D", "DRAIN", "B", "P2", "2")
            if not src or not drn:
                # Some EDIF instances expose only one terminal because of missing joined links.
                # Pin reconstruction fills many of these, but keep a diagnostic for coverage tracking.
                diagnostics.append(f"Device {device.name}: incomplete switch terminals for {device.kind.value}")
                return influence, conduction, diagnostics
            if not gate:
                diagnostics.append(f"Device {device.name}: missing gate/control pin for {device.kind.value}")
                return influence, conduction, diagnostics
            active_high = device.kind in {DeviceKind.NMOS, DeviceKind.PASS_NMOS}
            controls = (ConditionLiteral(signal=gate, required_level=active_high, source_device=device.name),)
            for a, b in ((src, drn), (drn, src)):
                conduction.append(
                    ConditionalEdge(
                        source=a,
                        target=b,
                        semantic=EdgeSemantic.CONDUCTION,
                        inversion=0,
                        controls=controls,
                        through_devices=(device.name,),
                        metadata={"device_kind": device.kind.value},
                    )
                )
            influence.append(
                ConditionalEdge(
                    source=gate,
                    target=src,
                    semantic=EdgeSemantic.CLOCKING if self._looks_like_clock(gate) else EdgeSemantic.INFLUENCE,
                    inversion=None,
                    through_devices=(device.name,),
                    metadata={"controls_conduction": True, "paired_terminal": drn},
                )
            )
            influence.append(
                ConditionalEdge(
                    source=gate,
                    target=drn,
                    semantic=EdgeSemantic.CLOCKING if self._looks_like_clock(gate) else EdgeSemantic.INFLUENCE,
                    inversion=None,
                    through_devices=(device.name,),
                    metadata={"controls_conduction": True, "paired_terminal": src},
                )
            )
            return influence, conduction, diagnostics

        if device.kind == DeviceKind.TRANSMISSION_GATE:
            a = self._pin(device, "A", "IN", "SOURCE", "P1")
            b = self._pin(device, "B", "OUT", "DRAIN", "P2", "Z")
            if not a or not b:
                diagnostics.append(f"Device {device.name}: missing TG data pins")
                return influence, conduction, diagnostics

            ctrl, ctrlb = self._control_pair_for_tgate(device)
            controls: list[ConditionLiteral] = []
            if ctrl:
                controls.append(ConditionLiteral(signal=ctrl, required_level=True, source_device=device.name))
            if ctrlb:
                controls.append(ConditionLiteral(signal=ctrlb, required_level=False, source_device=device.name))
            if not controls:
                en = self._pin(device, "EN", "G", "CTRL")
                if en:
                    controls.append(ConditionLiteral(signal=en, required_level=True, source_device=device.name))

            for x, y in ((a, b), (b, a)):
                conduction.append(
                    ConditionalEdge(
                        source=x,
                        target=y,
                        semantic=EdgeSemantic.CONDUCTION,
                        inversion=0,
                        controls=tuple(controls),
                        through_devices=(device.name,),
                        metadata={"device_kind": device.kind.value},
                    )
                )
            for literal in controls:
                for node in (a, b):
                    influence.append(
                        ConditionalEdge(
                            source=literal.signal,
                            target=node,
                            semantic=EdgeSemantic.CLOCKING if self._looks_like_clock(literal.signal) else EdgeSemantic.INFLUENCE,
                            inversion=None,
                            through_devices=(device.name,),
                            metadata={"controls_conduction": True},
                        )
                    )
            return influence, conduction, diagnostics

        return influence, conduction, diagnostics

    def _logic_device_edges(
        self,
        device: CircuitDevice,
        circuit: CircuitModel,
    ) -> tuple[list[ConditionalEdge], list[ConditionalEdge], list[str]]:
        influence: list[ConditionalEdge] = []
        conduction: list[ConditionalEdge] = []
        diagnostics: list[str] = []

        if device.domain == DeviceDomain.GATE:
            out_pin = self._gate_output_pin(device)
            if not out_pin:
                diagnostics.append(f"Device {device.name}: no output pin for logic gate {device.kind.value}")
                return influence, conduction, diagnostics
            input_pins = self._gate_input_pins(device, exclude={out_pin})

            if device.kind == DeviceKind.MUX2:
                sel = self._pin(device, "S", "SEL", "SELECT")
                d0 = self._pin(device, "A", "I0", "D0", "IN0")
                d1 = self._pin(device, "B", "I1", "D1", "IN1")
                if d0 and sel:
                    influence.append(
                        ConditionalEdge(
                            source=d0,
                            target=out_pin,
                            semantic=EdgeSemantic.INFLUENCE,
                            inversion=0,
                            controls=(ConditionLiteral(signal=sel, required_level=False, source_device=device.name),),
                            through_devices=(device.name,),
                            metadata={"gate_kind": "mux2", "path": "d0"},
                        )
                    )
                if d1 and sel:
                    influence.append(
                        ConditionalEdge(
                            source=d1,
                            target=out_pin,
                            semantic=EdgeSemantic.INFLUENCE,
                            inversion=0,
                            controls=(ConditionLiteral(signal=sel, required_level=True, source_device=device.name),),
                            through_devices=(device.name,),
                            metadata={"gate_kind": "mux2", "path": "d1"},
                        )
                    )
                if sel:
                    influence.append(
                        ConditionalEdge(
                            source=sel,
                            target=out_pin,
                            semantic=EdgeSemantic.INFLUENCE,
                            inversion=None,
                            through_devices=(device.name,),
                            metadata={"gate_kind": "mux2", "role": "select"},
                        )
                    )
                return influence, conduction, diagnostics

            inversion_map: dict[DeviceKind, int | None] = {
                DeviceKind.INV: 1,
                DeviceKind.BUF: 0,
                DeviceKind.NAND: 1,
                DeviceKind.NOR: 1,
                DeviceKind.AND: 0,
                DeviceKind.OR: 0,
                DeviceKind.XOR: None,
                DeviceKind.XNOR: None,
                DeviceKind.UNKNOWN: None,
            }
            inversion = inversion_map.get(device.kind, None)
            for pin_net in input_pins:
                influence.append(
                    ConditionalEdge(
                        source=pin_net,
                        target=out_pin,
                        semantic=EdgeSemantic.INFLUENCE,
                        inversion=inversion,
                        through_devices=(device.name,),
                        metadata={"gate_kind": device.kind.value},
                    )
                )
            return influence, conduction, diagnostics

        if device.domain == DeviceDomain.SEQUENTIAL_MACRO:
            q = self._pin(device, "Q", "QN", "QB", "OUT", "Z")
            d = self._pin(device, "D", "DATA", "IN")
            clk = self._pin(device, "CLK", "CLOCK", "CK", "CP")
            en = self._pin(device, "EN", "E", "G", "GATE", "CE")
            rst = self._pin(device, "RST", "RESET", "CLR", "RN", "RESETB", "CLRB")
            set_ = self._pin(device, "SET", "PRE", "SN", "PREB", "SETB")

            if q and d:
                controls: list[ConditionLiteral] = []
                if en:
                    en_active_high = not self._is_active_low_name("EN", en)
                    controls.append(ConditionLiteral(signal=en, required_level=en_active_high, source_device=device.name))
                if device.kind == DeviceKind.LATCH_D and clk:
                    clk_active_high = not self._is_active_low_name("CLK", clk)
                    controls.append(
                        ConditionLiteral(signal=clk, required_level=clk_active_high, source_device=device.name)
                    )
                influence.append(
                    ConditionalEdge(
                        source=d,
                        target=q,
                        semantic=EdgeSemantic.INFLUENCE,
                        inversion=0,
                        controls=tuple(controls),
                        through_devices=(device.name,),
                        metadata={"macro_kind": device.kind.value, "role": "data_path"},
                    )
                )
            if clk and q:
                influence.append(
                    ConditionalEdge(
                        source=clk,
                        target=q,
                        semantic=EdgeSemantic.CLOCKING,
                        inversion=None,
                        through_devices=(device.name,),
                        metadata={"macro_kind": device.kind.value, "role": "clock"},
                    )
                )
            for role, control_net in (("reset", rst), ("set", set_)):
                if not control_net or not q:
                    continue
                influence.append(
                    ConditionalEdge(
                        source=control_net,
                        target=q,
                        semantic=EdgeSemantic.ASYNC_CONTROL,
                        inversion=None,
                        through_devices=(device.name,),
                        metadata={
                            "macro_kind": device.kind.value,
                            "role": role,
                            "active_low_name": self._is_active_low_name(role.upper(), control_net),
                        },
                    )
                )
            return influence, conduction, diagnostics

        return influence, conduction, diagnostics

    def _synthesize_cmos_gate_edges(
        self,
        transistor_devices: list[CircuitDevice],
        circuit: CircuitModel,
    ) -> tuple[list[ConditionalEdge], list[str]]:
        diagnostics: list[str] = []
        edges: list[ConditionalEdge] = []
        power_nets = circuit.power_nets()
        ground_nets = circuit.ground_nets()

        # Group "simple inverter" candidates by output and gate:
        # PMOS: power -> out, gate=inp; NMOS: out -> ground, gate=inp
        pmos_by_out_in: dict[tuple[str, str], list[CircuitDevice]] = {}
        nmos_by_out_in: dict[tuple[str, str], list[CircuitDevice]] = {}
        for device in transistor_devices:
            gate = self._pin(device, "G", "GATE")
            src = self._pin(device, "S", "SOURCE")
            drn = self._pin(device, "D", "DRAIN")
            if not gate or not src or not drn:
                continue
            if device.kind == DeviceKind.PMOS:
                if src in power_nets:
                    pmos_by_out_in.setdefault((drn, gate), []).append(device)
                elif drn in power_nets:
                    pmos_by_out_in.setdefault((src, gate), []).append(device)
            elif device.kind == DeviceKind.NMOS:
                if src in ground_nets:
                    nmos_by_out_in.setdefault((drn, gate), []).append(device)
                elif drn in ground_nets:
                    nmos_by_out_in.setdefault((src, gate), []).append(device)

        for key, p_list in pmos_by_out_in.items():
            out_net, in_net = key
            n_list = nmos_by_out_in.get(key, [])
            if not n_list:
                continue
            through_devices = tuple(sorted({dev.name for dev in [*p_list, *n_list]}))
            edges.append(
                ConditionalEdge(
                    source=in_net,
                    target=out_net,
                    semantic=EdgeSemantic.INFLUENCE,
                    inversion=1,
                    through_devices=through_devices,
                    metadata={"synthetic": "cmos_inverter"},
                )
            )
        return edges, diagnostics

    def _build_influence_graph(self, circuit: CircuitModel, edges: list[ConditionalEdge]) -> InfluenceGraph:
        nodes = set(circuit.nets.keys())
        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        incoming: dict[str, list[ConditionalEdge]] = {node: [] for node in nodes}
        outgoing: dict[str, list[ConditionalEdge]] = {node: [] for node in nodes}
        for edge in edges:
            nodes.add(edge.source)
            nodes.add(edge.target)
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set())
            incoming.setdefault(edge.target, []).append(edge)
            incoming.setdefault(edge.source, [])
            outgoing.setdefault(edge.source, []).append(edge)
            outgoing.setdefault(edge.target, [])
        return InfluenceGraph(
            nodes=tuple(sorted(nodes)),
            edges=tuple(edges),
            adjacency=adjacency,
            incoming=incoming,
            outgoing=outgoing,
        )

    def _build_conduction_graph(self, circuit: CircuitModel, edges: list[ConditionalEdge]) -> ConditionalConductionGraph:
        nodes = set(circuit.nets.keys())
        adjacency: dict[str, list[ConditionalEdge]] = {node: [] for node in nodes}
        for edge in edges:
            nodes.add(edge.source)
            nodes.add(edge.target)
            adjacency.setdefault(edge.source, []).append(edge)
            adjacency.setdefault(edge.target, [])
        return ConditionalConductionGraph(nodes=tuple(sorted(nodes)), edges=tuple(edges), adjacency=adjacency)

    @staticmethod
    def _pin(device: CircuitDevice, *aliases: str) -> str | None:
        normalized = {name.upper(): net for name, net in device.pins.items()}
        for alias in aliases:
            value = normalized.get(alias.upper())
            if value:
                return value
        return None

    def _control_pair_for_tgate(self, device: CircuitDevice) -> tuple[str | None, str | None]:
        ctrl = self._pin(device, "C", "CTRL", "EN", "G")
        ctrlb = self._pin(device, "CB", "CTRLB", "ENB", "GN", "GB")
        if ctrl and ctrlb:
            return ctrl, ctrlb
        # Detect from pin names like CLK/CLKB.
        if not ctrl:
            ctrl = self._pin(device, "CLK")
        if not ctrlb:
            ctrlb = self._pin(device, "CLKB", "CLKB_", "CLKN")
        return ctrl, ctrlb

    def _gate_output_pin(self, device: CircuitDevice) -> str | None:
        for alias in ("Y", "Z", "Q", "OUT", "O"):
            value = self._pin(device, alias)
            if value:
                return value
        # Heuristic fallback: pin names commonly used for outputs in gate libraries.
        for pin_name, net_name in device.pins.items():
            upper = pin_name.upper()
            if upper.startswith("Q") or upper in {"ZN", "ZN0"}:
                return net_name
        return None

    def _gate_input_pins(self, device: CircuitDevice, *, exclude: set[str]) -> list[str]:
        inputs: list[str] = []
        for pin_name, net_name in device.pins.items():
            if net_name in exclude:
                continue
            upper = pin_name.upper()
            if upper in {"VDD", "VCC", "VSS", "GND", "VPWR", "VGND"}:
                continue
            if upper in {"Y", "Z", "Q", "QN", "QB", "OUT", "O"}:
                continue
            inputs.append(net_name)
        # Stable order for reporting / deterministic SCC traversal.
        seen: set[str] = set()
        ordered: list[str] = []
        for net in inputs:
            if net in seen:
                continue
            seen.add(net)
            ordered.append(net)
        return ordered

    @staticmethod
    def _looks_like_clock(net_name: str) -> bool:
        upper = net_name.upper()
        return "CLK" in upper or "CLOCK" in upper or upper in {"CK", "CP"}

    @staticmethod
    def _is_active_low_name(pin_role: str, net_name: str) -> bool:
        upper = net_name.upper()
        return upper.endswith("B") or upper.endswith("_N") or upper.endswith("N") or "/" in upper

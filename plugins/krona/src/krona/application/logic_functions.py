from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
from krona.application.ports import NetlistRepository
from krona.domain.netlist import Instance, TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser

from .logic_functions_parts.construction import ConstructionMixin
from .logic_functions_parts.execute import ExecuteMixin
from .logic_functions_parts.build_net_alias_map import BuildNetAliasMapMixin
from .logic_functions_parts.infer_pins_symbol import InferPinsSymbolMixin
from .logic_functions_parts.select_terminal_pair import SelectTerminalPairMixin
from .logic_functions_parts.transistors import TransistorsMixin
from .logic_functions_parts.degenerate_transistor_markers import DegenerateTransistorMarkersMixin
from .logic_functions_parts.sequential_elements_topology import SequentialElementsTopologyMixin
from .logic_functions_parts.detect_cmos_inverters import DetectCmosInvertersMixin
from .logic_functions_parts.detect_controlled_switches import DetectControlledSwitchesMixin
from .logic_functions_parts.detect_tristate_cmos_inverters import DetectTristateCmosInvertersMixin
from .logic_functions_parts.phase_relation import PhaseRelationMixin
from .logic_functions_parts.detect_topological_hybrid_latch import DetectTopologicalHybridLatchMixin
from .logic_functions_parts.inverter_instances_between import InverterInstancesBetweenMixin
from .logic_functions_parts.detect_topological_latch_candidates import DetectTopologicalLatchCandidatesMixin
from .logic_functions_parts.controls_are_opposite_phase import ControlsAreOppositePhaseMixin
from .logic_functions_parts.merge_topological_latches_into import MergeTopologicalLatchesIntoMixin
from .logic_functions_parts.promote_pulse_triggered_latches import PromotePulseTriggeredLatchesMixin
from .logic_functions_parts.annotate_sequential_elements import AnnotateSequentialElementsMixin
from .logic_functions_parts.detect_async_force_controls import DetectAsyncForceControlsMixin
from .logic_functions_parts.force_role_output_polarity import ForceRoleOutputPolarityMixin
from .logic_functions_parts.analyze_combinational_net_function import AnalyzeCombinationalNetFunctionMixin
from .logic_functions_parts.detect_mux2_truth_table import DetectMux2TruthTableMixin
from .logic_functions_parts.logic_path_output import LogicPathOutputMixin
from .logic_functions_parts.evaluate_output_state import EvaluateOutputStateMixin
from .logic_functions_parts.detect_parity_expression import DetectParityExpressionMixin
from .logic_functions_parts.select_cover import SelectCoverMixin
from .logic_functions_parts.transistor import _Transistor

class ExtractLogicFunctions(
    ConstructionMixin,
    ExecuteMixin,
    BuildNetAliasMapMixin,
    InferPinsSymbolMixin,
    SelectTerminalPairMixin,
    TransistorsMixin,
    DegenerateTransistorMarkersMixin,
    SequentialElementsTopologyMixin,
    DetectCmosInvertersMixin,
    DetectControlledSwitchesMixin,
    DetectTristateCmosInvertersMixin,
    PhaseRelationMixin,
    DetectTopologicalHybridLatchMixin,
    InverterInstancesBetweenMixin,
    DetectTopologicalLatchCandidatesMixin,
    ControlsAreOppositePhaseMixin,
    MergeTopologicalLatchesIntoMixin,
    PromotePulseTriggeredLatchesMixin,
    AnnotateSequentialElementsMixin,
    DetectAsyncForceControlsMixin,
    ForceRoleOutputPolarityMixin,
    AnalyzeCombinationalNetFunctionMixin,
    DetectMux2TruthTableMixin,
    LogicPathOutputMixin,
    EvaluateOutputStateMixin,
    DetectParityExpressionMixin,
    SelectCoverMixin,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _Transistor,
)

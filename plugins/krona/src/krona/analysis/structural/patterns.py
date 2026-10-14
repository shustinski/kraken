from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable
from .graphs import GraphBundle
from .model import (
    ActivationMode,
    ActivationSpec,
    BistableCoreEvidence,
    CircuitDevice,
    CircuitModel,
    CompositeFeature,
    ConditionalEdge,
    EdgeSemantic,
    LogicLevel,
    RecognizedCellKind,
    RecognizedStructure,
)
from .phases import ClockChainAnalysis, PhaseAnalyzer
from .scc import TarjanSCCDetector

from .patterns_parts.run import RunMixin
from .patterns_parts.detect_bistable_cores import DetectBistableCoresMixin
from .patterns_parts.detect_async_controls import DetectAsyncControlsMixin
from .patterns_parts.detect_dynamic_precharge_evaluate import DetectDynamicPrechargeEvaluateMixin
from .patterns_parts.detect_toggle_chains import DetectToggleChainsMixin
from .patterns_parts.pick_gate_inputs import PickGateInputsMixin
from .patterns_parts.classify import ClassifyMixin
from .patterns_parts.classify_sequential_macros import ClassifySequentialMacrosMixin
from .patterns_parts.classify_bistable_cores import ClassifyBistableCoresMixin
from .patterns_parts.analyze_core_phases import AnalyzeCorePhasesMixin
from .patterns_parts.compose_master_slave_ffs import ComposeMasterSlaveFfsMixin
from .patterns_parts.upgrade_jk_t_counter import UpgradeJkTCounterMixin
from .patterns_parts.detect_jk_feedback_equation import DetectJkFeedbackEquationMixin
from .patterns_parts.storage_phase_profile import (
    BistableCoreCandidate,
    InputMuxCandidate,
    AsyncControlCandidate,
    StoragePhaseProfile,
    PatternDatabase,
)

class PatternMatcher(
    RunMixin,
    DetectBistableCoresMixin,
    DetectAsyncControlsMixin,
    DetectDynamicPrechargeEvaluateMixin,
    DetectToggleChainsMixin,
    PickGateInputsMixin,
):
    """
    Finds structural patterns and low-level evidence needed by semantic classification.
    """

class SemanticClassifier(
    ClassifyMixin,
    ClassifySequentialMacrosMixin,
    ClassifyBistableCoresMixin,
    AnalyzeCorePhasesMixin,
    ComposeMasterSlaveFfsMixin,
    UpgradeJkTCounterMixin,
    DetectJkFeedbackEquationMixin,
):
    """
    Converts structural evidence into semantic storage-cell classes.
    """

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    BistableCoreCandidate,
    InputMuxCandidate,
    AsyncControlCandidate,
    StoragePhaseProfile,
    PatternDatabase,
)

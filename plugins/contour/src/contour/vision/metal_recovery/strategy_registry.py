"""Declarative registry for production conductor-segmentation strategies.

The registry is deliberately independent from Qt.  Backend validation, the
settings serializer, presets and the UI all consume the same parameter
metadata, so defaults and ranges cannot silently diverge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

ParameterKind = Literal["bool", "int", "float", "choice"]
ParameterValue = bool | int | float | str


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label: str
    kind: ParameterKind
    default: ParameterValue
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    tooltip: str = ""
    advanced: bool = False
    units: str = ""

    def normalize(self, value: Any) -> ParameterValue:
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.kind == "choice":
            text = str(value)
            allowed = {choice_value for choice_value, _label in self.choices}
            return text if text in allowed else self.default
        if self.kind == "int":
            parsed: float = float(int(value))
        else:
            parsed = float(value)
        if self.minimum is not None:
            parsed = max(float(self.minimum), parsed)
        if self.maximum is not None:
            parsed = min(float(self.maximum), parsed)
        return int(parsed) if self.kind == "int" else float(parsed)


@dataclass(frozen=True, slots=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    description: str
    backend_path: str | None = None
    parameters: tuple[ParameterSpec, ...] = ()
    partition_based: bool = False
    preserves_instances: bool = False

    def load_backend(self) -> Callable[..., Any] | None:
        if self.backend_path is None:
            return None
        module_name, attribute = self.backend_path.split(":", 1)
        return getattr(import_module(module_name), attribute)


def _p(
    key: str,
    label: str,
    kind: ParameterKind,
    default: ParameterValue,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    *,
    choices: tuple[tuple[str, str], ...] = (),
    tooltip: str,
    advanced: bool = False,
    units: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        key=key,
        label=label,
        kind=kind,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        choices=choices,
        tooltip=tooltip,
        advanced=advanced,
        units=units,
    )


_CLASSIFICATION_PARAMETERS = (
    _p(
        "core_evidence_weight",
        "Core metal evidence weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.1,
        tooltip="Higher values favor regions containing reliable conductor-core seeds. Too high a value can retain isolated bright texture.",
    ),
    _p(
        "substrate_evidence_weight",
        "Substrate evidence weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.1,
        tooltip="Higher values reject regions supported by substrate or groove evidence. Too high a value can lose dark conductor interiors.",
    ),
    _p(
        "intensity_evidence_weight",
        "Intensity evidence weight",
        "float",
        0.45,
        0.0,
        8.0,
        0.05,
        tooltip="Weights robust region intensity without assuming that every metal pixel is bright. Too high a value makes illumination dominate structure.",
    ),
    _p(
        "local_contrast_evidence_weight",
        "Local contrast evidence weight",
        "float",
        0.8,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values favor regions that differ from their local background. Too high a value may fragment textured plates.",
    ),
    _p(
        "boundary_side_evidence_weight",
        "Boundary-side evidence weight",
        "float",
        0.6,
        0.0,
        8.0,
        0.05,
        tooltip="Weights the intensity transition across a region boundary. Too high a value can reject border-touching conductors.",
    ),
    _p(
        "minimum_metal_confidence",
        "Minimum metal confidence",
        "float",
        0.6,
        0.0,
        1.0,
        0.01,
        tooltip="Minimum normalized material score classified as metal. Higher values reduce false metal but increase misses.",
    ),
    _p(
        "minimum_background_confidence",
        "Minimum background confidence",
        "float",
        0.45,
        0.0,
        1.0,
        0.01,
        tooltip="Scores below this level are background. Keep it no greater than the metal threshold to leave an explicit ambiguous interval.",
    ),
    _p(
        "ambiguous_region_policy",
        "Ambiguous region policy",
        "choice",
        "background",
        choices=(("metal", "Metal"), ("background", "Background"), ("preserve", "Preserve seed evidence")),
        tooltip="Controls regions between the background and metal confidence thresholds. Preserve uses high-confidence core evidence only.",
    ),
)

_SIGNED_GRAPH_PARAMETERS = (
    _p(
        "graph_domain",
        "Graph domain",
        "choice",
        "atomic_regions",
        choices=(("pixels", "Pixels"), ("atomic_regions", "Atomic regions")),
        tooltip="Pixels preserve every gap and are limited to 1,000,000 nodes for deterministic in-process execution. Atomic regions provide a sparse full-resolution graph without resizing the image.",
    ),
    _p(
        "connectivity",
        "Connectivity",
        "choice",
        "4",
        choices=(("4", "4"), ("8", "8")),
        tooltip="Eight-neighbour graphs include diagonals and join diagonal evidence; four-neighbour graphs preserve corner separation.",
    ),
    _p(
        "intensity_attraction_weight",
        "Intensity attraction weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values merge regions of similar intensity. Too high a value can bridge a weak physical boundary.",
    ),
    _p(
        "local_contrast_attraction_weight",
        "Local contrast attraction weight",
        "float",
        0.55,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values join regions with similar local-background contrast. Too high a value may merge parallel traces under uniform illumination.",
    ),
    _p(
        "orientation_attraction_weight",
        "Orientation attraction weight",
        "float",
        0.5,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values join locally orientation-compatible regions. Too high a value may connect separate aligned traces.",
    ),
    _p(
        "core_attraction_weight",
        "Core/interior attraction weight",
        "float",
        0.8,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values preserve continuity between conductor-core regions. Too high a value can absorb nearby bright substrate.",
    ),
    _p(
        "boundary_repulsion_weight",
        "Boundary repulsion weight",
        "float",
        1.25,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values discourage joining across structural boundaries. Too high a value splits a conductor along internal SEM texture.",
    ),
    _p(
        "rim_repulsion_weight",
        "Rim repulsion weight",
        "float",
        0.8,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values treat persistent bright rims as separation evidence. Too high a value can outline rather than fill broad conductors.",
    ),
    _p(
        "oriented_boundary_repulsion_weight",
        "Orientation-coherent boundary weight",
        "float",
        0.7,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values strengthen long coherent boundaries and suppress isolated texture. Too high a value can over-split directional patterns.",
    ),
    _p(
        "affinity_normalization",
        "Affinity normalization",
        "choice",
        "weighted_mean",
        choices=(("weighted_mean", "Weighted mean"), ("weighted_sum", "Weighted sum")),
        tooltip="Weighted mean keeps thresholds stable when weights change; weighted sum exposes their absolute scale.",
    ),
    _p(
        "affinity_temperature",
        "Affinity temperature",
        "float",
        1.0,
        0.05,
        10.0,
        0.05,
        tooltip="Lower values make affinities more decisive; higher values compress confidence toward neutral.",
    ),
    _p(
        "minimum_attractive_confidence",
        "Minimum attractive confidence",
        "float",
        0.52,
        0.0,
        1.0,
        0.01,
        tooltip="Edges below this attraction confidence are not attractive. Higher values reduce merging and may increase splits.",
    ),
    _p(
        "minimum_repulsive_confidence",
        "Minimum repulsive confidence",
        "float",
        0.52,
        0.0,
        1.0,
        0.01,
        tooltip="Edges below this repulsion confidence are not repulsive. Higher values ignore weak boundaries and may increase merges.",
    ),
    _p(
        "atomic_segmentation_method",
        "Atomic segmentation method",
        "choice",
        "oriented_watershed",
        choices=(("oriented_watershed", "Oriented watershed"), ("regular_grid", "Regular grid")),
        tooltip="Oriented watershed follows local contour orientation; regular grid is a deterministic diagnostic partition.",
    ),
    _p(
        "atomic_region_scale",
        "Atomic region scale",
        "int",
        16,
        2,
        64,
        1,
        tooltip="Approximate marker spacing for the full-resolution oriented watershed. Larger values reduce graph size; boundaries are still evaluated at native resolution, but very large values can under-represent tiny isolated regions.",
        units="px",
    ),
    _p(
        "minimum_atomic_region_area",
        "Minimum atomic region area",
        "int",
        6,
        1,
        4096,
        1,
        tooltip="Atomic regions smaller than this are absorbed into a neighbour. Too high a value removes small SEM structures.",
        units="px²",
    ),
)

_OWT_PARAMETERS = (
    _p(
        "contour_source",
        "Contour source",
        "choice",
        "combined",
        choices=(
            ("structural_gradient", "Structural gradient"),
            ("oriented_gradient", "Oriented gradient"),
            ("combined", "Combined"),
        ),
        tooltip="Selects the contour evidence used by oriented watershed. Combined balances persistent structure and local oriented contrast.",
    ),
    _p(
        "orientation_bins",
        "Orientation bins",
        "int",
        8,
        8,
        8,
        1,
        tooltip="The Berkeley OWT implementation uses its original fixed bank of eight contour orientations.",
    ),
    _p(
        "orientation_smoothing_sigma",
        "Orientation smoothing sigma",
        "float",
        2.0,
        0.1,
        16.0,
        0.1,
        tooltip="Higher values stabilize contour direction over longer distances; too high a value blurs bends and short gaps.",
        units="px",
    ),
    _p(
        "contour_smoothing_sigma",
        "Contour smoothing sigma",
        "float",
        1.0,
        0.1,
        8.0,
        0.1,
        tooltip="Higher values suppress texture before watershed; too high a value displaces narrow boundaries.",
        units="px",
    ),
    _p(
        "minimum_contour_strength",
        "Minimum contour strength",
        "float",
        0.12,
        0.0,
        1.0,
        0.01,
        tooltip="Weak oriented responses below this confidence are suppressed. High values open real but faint boundaries.",
    ),
    _p(
        "watershed_minima_suppression",
        "Watershed minima suppression",
        "float",
        0.06,
        0.0,
        1.0,
        0.01,
        tooltip="Higher values suppress shallow minima and produce fewer initial basins; too high a value undersegments.",
    ),
    _p(
        "minimum_initial_basin_area",
        "Minimum initial basin area",
        "int",
        6,
        1,
        4096,
        1,
        tooltip="Removes tiny watershed minima caused by SEM texture. Too high a value removes legitimate narrow regions.",
        units="px²",
    ),
    _p(
        "hierarchy_level",
        "Hierarchy level",
        "float",
        0.2,
        0.0,
        1.0,
        0.01,
        tooltip="Fine at 0 and coarse at 1. Higher values merge across progressively stronger UCM boundaries.",
    ),
    _p(
        "boundary_aggregation",
        "Boundary aggregation",
        "choice",
        "mean",
        choices=(("mean", "BSR dynamic mean"),),
        tooltip="The original Berkeley UCM recomputes the mean shared-boundary strength after every region merge.",
    ),
    _p(
        "contour_continuity_weight",
        "Contour continuity weight",
        "float",
        0.65,
        0.0,
        4.0,
        0.05,
        tooltip="Higher values promote boundaries that persist along their tangent. Too high a value discounts short physical boundaries.",
    ),
    _p(
        "minimum_output_region_area",
        "Minimum output region area",
        "int",
        20,
        1,
        65536,
        1,
        tooltip="Final hierarchy regions smaller than this are joined to their least-cost neighbour. Too high a value removes small conductors.",
        units="px²",
    ),
    *_CLASSIFICATION_PARAMETERS,
)

_MSP_PARAMETERS = (
    _p(
        "gradient_field_sensitivity",
        "Gradient field sensitivity",
        "float",
        1.0,
        0.25,
        4.0,
        0.05,
        tooltip="Higher values recover weaker locally bright ribbons between opposite gradient rims; lower values keep only stronger gradient-supported conductors.",
    ),
    _p(
        "paired_rim_recovery_enabled",
        "Paired-rim ribbon recovery",
        "bool",
        True,
        tooltip="Recovers narrow conductors that are locally bright between opposite gradient rims but remain below the global intensity threshold.",
        advanced=True,
    ),
    _p(
        "connectivity",
        "Local connectivity",
        "choice",
        "8",
        choices=(("4", "4"), ("8", "8")),
        tooltip="Eight-neighbour continuity keeps diagonal separator pixels connected; four-neighbour mode preserves diagonal object contact.",
    ),
    _p(
        "separator_unary_weight",
        "Separator unary weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Weights direct evidence that a pixel is a separator. Too high a value produces broad separator bands.",
    ),
    _p(
        "region_unary_weight",
        "Region unary weight",
        "float",
        0.55,
        0.0,
        8.0,
        0.05,
        tooltip="Weights evidence that pixels belong to object/background regions. Too high a value closes weak separators.",
    ),
    _p(
        "boundary_separator_weight",
        "Boundary separator weight",
        "float",
        1.2,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values select strong structural boundaries as explicit separator nodes. Too high a value follows internal texture.",
    ),
    _p(
        "intensity_affinity_weight",
        "Intensity affinity weight",
        "float",
        0.6,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values keep similar neighbouring pixels in one region. Too high a value crosses faint boundaries.",
    ),
    _p(
        "gradient_repulsion_weight",
        "Gradient repulsion weight",
        "float",
        0.9,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values favor separators on strong local transitions. Too high a value fragments textured plates.",
    ),
    _p(
        "orientation_consistency_weight",
        "Orientation consistency weight",
        "float",
        0.7,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values prefer directionally coherent separators and reject isolated gradient noise.",
    ),
    _p(
        "separator_continuity_weight",
        "Separator continuity weight",
        "float",
        0.8,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values extend supported separator chains across short weak spans. Too high a value invents long walls.",
    ),
    _p(
        "minimum_separator_confidence",
        "Minimum separator confidence",
        "float",
        0.56,
        0.0,
        1.0,
        0.01,
        tooltip="Minimum optimized separator score. Higher values reduce false splits but leave more boundary gaps.",
    ),
    _p(
        "minimum_separator_length",
        "Minimum separator length",
        "int",
        4,
        1,
        1024,
        1,
        tooltip="Separator components shorter than this are removed. Too high a value removes short but important gaps.",
        units="px",
    ),
    _p(
        "minimum_region_area",
        "Minimum region area",
        "int",
        1,
        1,
        65536,
        1,
        tooltip="Regions smaller than this are absorbed after separator removal. Too high a value loses small conductors.",
        units="px²",
    ),
    _p(
        "separator_projection_enabled",
        "Project material separators",
        "bool",
        True,
        tooltip="Returns separator nodes with strong conductor-core evidence to the material region. Disable only to inspect the raw upstream partition.",
        advanced=True,
    ),
    _p(
        "paired_rim_fallback_enabled",
        "Paired-rim fallback enabled",
        "bool",
        True,
        tooltip="Uses the fast Otsu path when opposite-polarity rims prove that the conservative core detector missed a substantial conductor class.",
        advanced=True,
    ),
    _p(
        "paired_rim_evidence_threshold",
        "Paired-rim evidence threshold",
        "float",
        0.25,
        0.0,
        1.0,
        0.01,
        tooltip="Minimum paired-rim and core confidence used by the missing-core diagnostic.",
        advanced=True,
    ),
    _p(
        "paired_rim_fallback_min_core_fraction",
        "Fallback minimum core fraction",
        "float",
        0.1,
        0.0,
        1.0,
        0.01,
        tooltip="Prevents the Otsu fallback on low-core textured frames where paired edges are not reliable conductor evidence.",
        advanced=True,
    ),
    _p(
        "paired_rim_fallback_fraction",
        "Fallback missing-core fraction",
        "float",
        0.04,
        0.0,
        1.0,
        0.005,
        tooltip="Minimum image fraction supported by paired rims but absent from the original core evidence before the fast fallback is selected.",
        advanced=True,
    ),
    _p(
        "separator_projection_radius",
        "Separator projection radius",
        "int",
        1,
        0,
        8,
        1,
        tooltip="Maximum 1 px separator width restored from a neighbouring metal region. Larger values can bridge genuinely wide gaps.",
        units="px",
        advanced=True,
    ),
    _p(
        "separator_projection_min_core_evidence",
        "Projection core evidence",
        "float",
        0.25,
        0.0,
        1.0,
        0.01,
        tooltip="Minimum conductor-core evidence required before a separator pixel can become material. Lower values may fill substrate gaps.",
        advanced=True,
    ),
    _p(
        "separator_projection_core_margin",
        "Projection core margin",
        "float",
        0.25,
        0.0,
        1.0,
        0.01,
        tooltip="Required excess of conductor-core over substrate evidence. Higher values preserve more uncertain separators.",
        advanced=True,
    ),
    _p(
        "metal_merge_max_separator_confidence",
        "Metal merge separator ceiling",
        "float",
        0.7,
        0.0,
        1.0,
        0.01,
        tooltip="Two metal regions may join only across a projected separator no stronger than this value. Lower values preserve more internal region boundaries.",
        advanced=True,
    ),
    _p(
        "long_range_enabled",
        "Long-range interactions enabled",
        "bool",
        True,
        tooltip="Uses orientation-aware non-local support to continue separators across short local failures.",
    ),
    _p(
        "long_range_radius",
        "Long-range radius",
        "int",
        7,
        1,
        64,
        1,
        tooltip="Maximum radius for non-local separator support. Large values may connect unrelated parallel boundaries.",
        units="px",
    ),
    _p(
        "long_range_attraction_weight",
        "Long-range attraction weight",
        "float",
        0.35,
        0.0,
        8.0,
        0.05,
        tooltip="Supports region continuity across weak local texture. Too high a value can close true separators.",
    ),
    _p(
        "long_range_repulsion_weight",
        "Long-range repulsion weight",
        "float",
        0.65,
        0.0,
        8.0,
        0.05,
        tooltip="Supports directionally aligned separator evidence across short gaps. Too high a value creates false splits.",
    ),
    _p(
        "maximum_interaction_distance",
        "Maximum graph distance",
        "int",
        12,
        1,
        128,
        1,
        tooltip="Caps non-local interaction distance independently of image size.",
        units="px",
    ),
    _p(
        "solver",
        "Solver / heuristic",
        "choice",
        "greedy_separator_growing",
        choices=(
            ("greedy_separator_shrinking", "Upstream greedy separator shrinking"),
            ("greedy_separator_growing", "Upstream greedy separator growing"),
        ),
        tooltip="Selects an upstream native heuristic. Growing is the SEM-trace default; shrinking is available for cell-like partitions but uses substantially more full-frame memory.",
    ),
    _p(
        "solver_tile_size",
        "Native solver tile size",
        "int",
        384,
        0,
        4096,
        64,
        tooltip="Bounds native graph memory without resizing pixels. 0 builds one global graph; positive values solve overlapping full-resolution tiles.",
        units="px",
        advanced=True,
    ),
    _p(
        "solver_tile_overlap",
        "Native solver tile overlap",
        "int",
        16,
        0,
        256,
        1,
        tooltip="Context retained around every tile before its core is stitched. Use at least the maximum interaction distance to avoid graph-edge seams.",
        units="px",
        advanced=True,
    ),
    _p(
        "solver_workers",
        "Native solver workers",
        "int",
        8,
        1,
        32,
        1,
        tooltip="Number of independent full-resolution tiles solved concurrently. Higher values reduce latency but increase peak memory and CPU use.",
        advanced=True,
    ),
    _p(
        "maximum_iterations",
        "Maximum iterations",
        "int",
        0,
        0,
        10_000_000,
        1000,
        tooltip="Maximum native vertex moves per tile; 0 runs each upstream priority queue to convergence.",
        advanced=True,
    ),
    *_CLASSIFICATION_PARAMETERS,
)

_GASP_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "linkage_criterion",
        "GASP linkage criterion",
        "choice",
        "average",
        choices=(
            ("average", "Average linkage"),
            ("mutex_abs_max", "Mutex/absolute-max linkage"),
            ("sum", "Sum linkage"),
        ),
        tooltip="Average is size-stable, sum favors broad evidence, and mutex/absolute-max preserves the strongest signed conflict.",
    ),
    _p(
        "use_signed_edges",
        "Use signed edges",
        "bool",
        True,
        tooltip="When enabled, repulsive edges veto or penalize merges. Disabling them reduces GASP to attractive agglomeration.",
    ),
    _p(
        "minimum_merge_affinity",
        "Minimum merge affinity",
        "float",
        0.05,
        -1.0,
        1.0,
        0.01,
        tooltip="Clusters merge only above this signed affinity. Higher values increase region count.",
    ),
    _p(
        "maximum_repulsive_conflict",
        "Maximum repulsive conflict",
        "float",
        0.45,
        0.0,
        1.0,
        0.01,
        tooltip="Merges with stronger accumulated repulsion are rejected. Lower values preserve more boundaries.",
    ),
    _p(
        "merge_stopping_threshold",
        "Merge stopping threshold",
        "float",
        0.0,
        -1.0,
        1.0,
        0.01,
        tooltip="Stops once the best remaining linkage is at or below this value.",
    ),
    _p(
        "maximum_operations",
        "Maximum operations",
        "int",
        200000,
        1,
        5000000,
        1000,
        tooltip="Safety cap for agglomeration operations. Too low a value leaves an incomplete partition.",
        advanced=True,
    ),
    *_CLASSIFICATION_PARAMETERS,
)

_MWS_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "attractive_neighborhood_offsets",
        "Attractive neighborhood offsets",
        "choice",
        "local",
        choices=(("local", "Local graph offsets"), ("local_plus_diagonal", "Local plus diagonal")),
        tooltip="Defines which nearby relations compete as attractive edges.",
    ),
    _p(
        "mutex_neighborhood_offsets",
        "Mutex neighborhood offsets",
        "choice",
        "local_plus_long_range",
        choices=(("local", "Local"), ("local_plus_long_range", "Local plus long-range")),
        tooltip="Adds non-local mutual-exclusion evidence without changing attractive graph support.",
    ),
    _p(
        "attractive_weight_scale",
        "Attractive weight scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales attractive edge ordering. Higher values let merges act before competing mutex edges.",
    ),
    _p(
        "mutex_weight_scale",
        "Mutex weight scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales mutex edge ordering. Higher values establish exclusions earlier and reduce merges.",
    ),
    _p(
        "minimum_mutex_confidence",
        "Minimum mutex confidence",
        "float",
        0.55,
        0.0,
        1.0,
        0.01,
        tooltip="Repulsive relations below this confidence do not become mutex constraints.",
    ),
    _p(
        "long_range_mutex_distance",
        "Long-range mutex distance",
        "int",
        8,
        1,
        128,
        1,
        tooltip="Maximum sparse orientation-aware mutex distance. Too large a value can exclude unrelated structures.",
        units="px",
    ),
    _p(
        "edge_ordering",
        "Edge ordering / normalization",
        "choice",
        "descending_confidence",
        choices=(("descending_confidence", "Descending confidence"), ("signed_margin", "Signed margin")),
        tooltip="Controls the deterministic order in which attractive and mutex relations are processed.",
    ),
    *_CLASSIFICATION_PARAMETERS,
)

_MULTICUT_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "cost_transform",
        "Cost transform",
        "choice",
        "log_odds",
        choices=(("log_odds", "Log odds"), ("signed_linear", "Signed linear")),
        tooltip="Log odds emphasizes confident probabilities; signed linear keeps a bounded directly interpretable cost.",
    ),
    _p(
        "affinity_bias",
        "Probability / affinity bias",
        "float",
        0.5,
        0.01,
        0.99,
        0.01,
        tooltip="Neutral probability for the cost transform. Higher values require stronger attraction before a merge is profitable.",
    ),
    _p(
        "attraction_cost_scale",
        "Attraction cost scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales the reward for keeping attractive edges uncut.",
    ),
    _p(
        "repulsion_cost_scale",
        "Repulsion cost scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales the penalty for keeping repulsive edges inside one component.",
    ),
    _p(
        "solver",
        "Solver",
        "choice",
        "greedy_additive",
        choices=(("greedy_additive", "Greedy Additive"),),
        tooltip="Only the deterministic greedy-additive solver is shipped; unavailable exact or fusion solvers are not advertised.",
    ),
    _p(
        "initialization",
        "Initialization",
        "choice",
        "singletons",
        choices=(("singletons", "Singleton regions"), ("positive_components", "Positive-edge components")),
        tooltip="Singletons are conservative; positive components start from confident attractive connectivity.",
    ),
    _p(
        "maximum_iterations",
        "Maximum iterations",
        "int",
        200000,
        1,
        5000000,
        1000,
        tooltip="Maximum greedy contractions. Too low a value returns an unfinished partition.",
        advanced=True,
    ),
    _p(
        "time_limit_seconds",
        "Time limit",
        "float",
        30.0,
        0.1,
        3600.0,
        0.5,
        tooltip="Deterministic wall-clock safety limit. A reached limit is reported explicitly in debug metadata.",
        advanced=True,
        units="s",
    ),
    _p(
        "convergence_tolerance",
        "Convergence tolerance",
        "float",
        0.0,
        0.0,
        1.0,
        0.001,
        tooltip="Minimum positive objective gain required for another contraction.",
        advanced=True,
    ),
    *_CLASSIFICATION_PARAMETERS,
)

_LIFTED_PARAMETERS = (
    *_MULTICUT_PARAMETERS,
    _p(
        "lifted_edges_enabled",
        "Lifted edges enabled",
        "bool",
        True,
        tooltip="Adds sparse non-local relations while preserving the same local graph and cost model as Multicut.",
    ),
    _p(
        "minimum_lifted_distance",
        "Minimum lifted distance",
        "int",
        4,
        2,
        256,
        1,
        tooltip="Minimum centroid distance for a non-local relation. Smaller values duplicate local edges.",
        units="px",
    ),
    _p(
        "maximum_lifted_distance",
        "Maximum lifted distance",
        "int",
        24,
        3,
        512,
        1,
        tooltip="Maximum non-local distance. Large values increase graph density and can connect unrelated traces.",
        units="px",
    ),
    _p(
        "lifted_distance_step",
        "Lifted distance step",
        "int",
        4,
        1,
        128,
        1,
        tooltip="Sampling step for sparse lifted candidates. Smaller steps add more edges and cost memory.",
        units="px",
    ),
    _p(
        "lifted_attraction_weight",
        "Lifted attraction weight",
        "float",
        0.45,
        0.0,
        8.0,
        0.05,
        tooltip="Supports same-trace continuity across short missing evidence. Too high a value increases merges.",
    ),
    _p(
        "lifted_repulsion_weight",
        "Lifted repulsion weight",
        "float",
        0.75,
        0.0,
        8.0,
        0.05,
        tooltip="Preserves separation across short local boundary failures. Too high a value increases splits.",
    ),
    _p(
        "orientation_aligned_lifted_edges",
        "Orientation-aligned lifted edges",
        "bool",
        True,
        tooltip="Restricts lifted candidates using the local orientation field rather than hard-coded image axes.",
    ),
    _p(
        "cross_boundary_lifted_repulsion",
        "Cross-boundary lifted repulsion",
        "bool",
        True,
        tooltip="Adds repulsion when the line between regions crosses persistent boundary evidence.",
    ),
    _p(
        "same_trace_lifted_attraction",
        "Same-trace lifted attraction",
        "bool",
        True,
        tooltip="Adds attraction between orientation-compatible regions with similar material evidence.",
    ),
    _p(
        "lifted_confidence_threshold",
        "Lifted confidence threshold",
        "float",
        0.6,
        0.0,
        1.0,
        0.01,
        tooltip="Only non-local relations above this confidence are added. Higher values make the lifted graph sparser.",
    ),
    _p(
        "maximum_lifted_edges",
        "Maximum lifted edges",
        "int",
        200000,
        0,
        5000000,
        1000,
        tooltip="Hard sparse-density cap preventing quadratic graph construction.",
        advanced=True,
    ),
)


_EXISTING = (
    ("auto", "Auto (topology control)"),
    ("legacy_otsu", "Legacy Otsu"),
    ("local_adaptive", "Local Adaptive"),
    ("gradient_watershed", "Gradient Watershed"),
    ("random_walker", "Random Walker"),
    ("graph_cut", "Graph Cut"),
    ("reconstruction", "Reconstruction"),
    ("closed_boundary", "Closed Boundary"),
    ("structural_watershed", "Structural Watershed"),
)

STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    strategy_id: StrategySpec(strategy_id, display_name, "Existing production strategy.")
    for strategy_id, display_name in _EXISTING
}
STRATEGY_REGISTRY.update(
    {
        "owt_ucm": StrategySpec(
            "owt_ucm",
            "OWT-UCM",
            "Hierarchical segmentation from oriented contour strength. Best suited when physical boundaries are clear but locally incomplete.",
            "contour.vision.metal_recovery.owt_ucm:segment_owt_ucm",
            _OWT_PARAMETERS,
            True,
            True,
        ),
        "graph_multi_separator": StrategySpec(
            "graph_multi_separator",
            "Graph Multi-Separator",
            "Graph partition with explicit separator pixels and optional long-range interactions.",
            "contour.vision.metal_recovery.graph_multi_separator:segment_graph_multi_separator",
            _MSP_PARAMETERS,
            True,
            True,
        ),
        "gasp": StrategySpec(
            "gasp",
            "GASP",
            "Agglomerative clustering of a signed attraction/repulsion graph.",
            "contour.vision.metal_recovery.graph_strategies:segment_gasp",
            _GASP_PARAMETERS,
            True,
            True,
        ),
        "mutex_watershed": StrategySpec(
            "mutex_watershed",
            "Mutex Watershed",
            "Fast signed graph partition using attractive and mutually exclusive relations.",
            "contour.vision.metal_recovery.graph_strategies:segment_mutex_watershed",
            _MWS_PARAMETERS,
            True,
            True,
        ),
        "multicut": StrategySpec(
            "multicut",
            "Multicut",
            "Global graph partition minimizing attractive and repulsive edge costs.",
            "contour.vision.metal_recovery.graph_strategies:segment_multicut",
            _MULTICUT_PARAMETERS,
            True,
            True,
        ),
        "lifted_multicut": StrategySpec(
            "lifted_multicut",
            "Lifted Multicut",
            "Multicut with additional non-local relations for long-range topology constraints.",
            "contour.vision.metal_recovery.graph_strategies:segment_lifted_multicut",
            _LIFTED_PARAMETERS,
            True,
            True,
        ),
        "ic_sem_expert": StrategySpec(
            "ic_sem_expert",
            "IC-SEM Expert Knowledge",
            "Reserved extension point for documented IC-SEM design-rule priors. No unverified algorithm is implemented.",
        ),
    }
)

IMPLEMENTED_NEW_STRATEGIES = frozenset(
    {"owt_ucm", "graph_multi_separator", "gasp", "mutex_watershed", "multicut", "lifted_multicut"}
)


@dataclass(slots=True)
class MetalStrategyConfigs:
    """Validated nested strategy settings with backward-compatible defaults."""

    values: dict[str, dict[str, ParameterValue]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MetalStrategyConfigs:
        source = raw or {}
        return cls(
            values={
                strategy_id: normalize_strategy_parameters(
                    strategy_id,
                    source.get(strategy_id) if isinstance(source.get(strategy_id), Mapping) else None,
                )
                for strategy_id in sorted(IMPLEMENTED_NEW_STRATEGIES)
            }
        )

    def for_strategy(self, strategy_id: str) -> dict[str, ParameterValue]:
        return normalize_strategy_parameters(strategy_id, self.values.get(strategy_id))

    def to_dict(self) -> dict[str, dict[str, ParameterValue]]:
        return {key: dict(value) for key, value in self.values.items()}


def normalize_strategy_parameters(
    strategy_id: str,
    raw: Mapping[str, Any] | None,
) -> dict[str, ParameterValue]:
    spec = STRATEGY_REGISTRY.get(strategy_id)
    if spec is None:
        raise ValueError(f"Unknown metal segmentation strategy: {strategy_id}")
    source = raw or {}
    normalized: dict[str, ParameterValue] = {}
    for parameter in spec.parameters:
        try:
            normalized[parameter.key] = parameter.normalize(source.get(parameter.key, parameter.default))
        except (TypeError, ValueError):
            normalized[parameter.key] = parameter.default
    if strategy_id == "lifted_multicut":
        minimum = int(normalized["minimum_lifted_distance"])
        maximum = int(normalized["maximum_lifted_distance"])
        if minimum > maximum:
            maximum = minimum
            normalized["maximum_lifted_distance"] = maximum
    background = float(normalized.get("minimum_background_confidence", 0.0))
    metal = float(normalized.get("minimum_metal_confidence", 1.0))
    if background > metal:
        normalized["minimum_background_confidence"] = metal
    return normalized


def strategy_spec(strategy_id: str) -> StrategySpec:
    try:
        return STRATEGY_REGISTRY[strategy_id]
    except KeyError as exc:
        raise ValueError(f"Unknown metal segmentation strategy: {strategy_id}") from exc


def visible_strategy_specs() -> tuple[StrategySpec, ...]:
    return tuple(spec for spec in STRATEGY_REGISTRY.values() if spec.strategy_id != "ic_sem_expert")


__all__ = [
    "IMPLEMENTED_NEW_STRATEGIES",
    "STRATEGY_REGISTRY",
    "MetalStrategyConfigs",
    "ParameterSpec",
    "ParameterValue",
    "StrategySpec",
    "normalize_strategy_parameters",
    "strategy_spec",
    "visible_strategy_specs",
]

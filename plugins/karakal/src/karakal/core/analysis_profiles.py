"""User-facing analysis profiles and standalone source preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisSourceRole

from .domain import FolderSpec, ModelSpec
from .image_formats import SUPPORTED_IMAGE_EXTENSION_SET


class PreflightSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class AnalysisProfileDefinition:
    key: AnalysisProfileKind
    title_key: str
    description_key: str
    app_mode: str
    analysis_mode: str
    comparison_target: str
    minimum_models: int
    confidence_required: bool = False


ANALYSIS_PROFILES: tuple[AnalysisProfileDefinition, ...] = (
    AnalysisProfileDefinition(
        key=AnalysisProfileKind.MODEL_COMPARISON,
        title_key="profile.model_comparison.title",
        description_key="profile.model_comparison.description",
        app_mode="validation",
        analysis_mode="inter_model",
        comparison_target="outputs",
        minimum_models=2,
    ),
    AnalysisProfileDefinition(
        key=AnalysisProfileKind.CONFIDENCE_AUDIT,
        title_key="profile.confidence.title",
        description_key="profile.confidence.description",
        app_mode="validation",
        analysis_mode="model_output_confidence",
        comparison_target="confidence",
        minimum_models=1,
        confidence_required=True,
    ),
    AnalysisProfileDefinition(
        key=AnalysisProfileKind.GRID_DEFECTS,
        title_key="profile.grid_defects.title",
        description_key="profile.grid_defects.description",
        app_mode="grid_inspection",
        analysis_mode="model_output_confidence",
        comparison_target="confidence",
        minimum_models=0,
    ),
)


DEFAULT_ANALYSIS_PROFILE = AnalysisProfileKind.MODEL_COMPARISON
_PROFILE_BY_KEY = {profile.key: profile for profile in ANALYSIS_PROFILES}


def analysis_profile_definition(value: AnalysisProfileKind | str | None) -> AnalysisProfileDefinition:
    try:
        key = AnalysisProfileKind(str(value or DEFAULT_ANALYSIS_PROFILE))
    except ValueError:
        key = DEFAULT_ANALYSIS_PROFILE
    return _PROFILE_BY_KEY.get(key, _PROFILE_BY_KEY[DEFAULT_ANALYSIS_PROFILE])


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    severity: PreflightSeverity
    code: str
    message_key: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SourceRoleStatus:
    role: AnalysisSourceRole
    source_count: int
    frame_count: int
    matched_count: int
    state: str
    detail: str


@dataclass(frozen=True, slots=True)
class AnalysisPreflightReport:
    profile: AnalysisProfileKind
    roles: tuple[SourceRoleStatus, ...]
    issues: tuple[PreflightIssue, ...]
    total_frames: int
    matched_frames: int

    @property
    def can_run(self) -> bool:
        return not any(issue.severity == PreflightSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == PreflightSeverity.WARNING)


def _folder_inventory(path: Path | None) -> tuple[set[str], tuple[str, ...]]:
    if path is None:
        return set(), ()
    if not path.exists() or not path.is_dir():
        return set(), (str(path),)
    keys: set[str] = set()
    duplicates: set[str] = set()
    for candidate in path.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSION_SET:
            continue
        key = candidate.stem.casefold()
        if key in keys:
            duplicates.add(key)
        keys.add(key)
    return keys, tuple(sorted(duplicates))


def _role_status(
    role: AnalysisSourceRole,
    inventories: tuple[set[str], ...],
    baseline: set[str],
    detail: str,
) -> SourceRoleStatus:
    combined = set().union(*inventories) if inventories else set()
    matched = len(combined & baseline) if baseline else len(combined)
    if not inventories:
        state = "missing"
    elif not combined:
        state = "empty"
    elif baseline and matched < len(baseline):
        state = "partial"
    else:
        state = "ready"
    return SourceRoleStatus(
        role=role,
        source_count=len(inventories),
        frame_count=len(combined),
        matched_count=matched,
        state=state,
        detail=detail,
    )


def build_standalone_preflight(
    profile_value: AnalysisProfileKind | str | None,
    original_folder: FolderSpec | None,
    model_specs: tuple[ModelSpec, ...],
) -> AnalysisPreflightReport:
    """Validate role requirements and filename coverage before starting work."""

    profile = analysis_profile_definition(profile_value)
    issues: list[PreflightIssue] = []
    model_ids = [model.model_id for model in model_specs]
    if len(model_ids) != len(set(model_ids)):
        issues.append(PreflightIssue(PreflightSeverity.ERROR, "duplicate_model_id", "preflight.duplicate_model_id"))

    original_keys, original_duplicates = _folder_inventory(None if original_folder is None else original_folder.path)
    model_inventories: list[set[str]] = []
    confidence_inventories: list[set[str]] = []
    duplicate_details: list[str] = []
    if original_duplicates:
        duplicate_details.append(f"original: {', '.join(original_duplicates[:5])}")
    for model in model_specs:
        keys, duplicates = _folder_inventory(model.mask_folder)
        model_inventories.append(keys)
        if duplicates:
            duplicate_details.append(f"{model.display_name}: {', '.join(duplicates[:5])}")
        if model.prob_folder is not None:
            confidence_keys, confidence_duplicates = _folder_inventory(model.prob_folder)
            confidence_inventories.append(confidence_keys)
            if confidence_duplicates:
                duplicate_details.append(f"{model.display_name} confidence: {', '.join(confidence_duplicates[:5])}")

    if duplicate_details:
        issues.append(
            PreflightIssue(
                PreflightSeverity.ERROR,
                "duplicate_frame_key",
                "preflight.duplicate_frame_key",
                "; ".join(duplicate_details),
            )
        )
    if len(model_specs) < profile.minimum_models:
        issues.append(
            PreflightIssue(
                PreflightSeverity.ERROR,
                "models_required",
                "preflight.models_required",
                str(profile.minimum_models),
            )
        )
    if profile.confidence_required and not any(confidence_inventories):
        issues.append(PreflightIssue(PreflightSeverity.ERROR, "confidence_required", "preflight.confidence_required"))
    if profile.key == AnalysisProfileKind.GRID_DEFECTS and not model_inventories and not original_keys:
        issues.append(PreflightIssue(PreflightSeverity.ERROR, "grid_source_required", "preflight.grid_source_required"))

    baseline = original_keys or (model_inventories[0] if model_inventories else set())
    compared_sets = [inventory for inventory in model_inventories if inventory]
    if profile.confidence_required:
        compared_sets.extend(inventory for inventory in confidence_inventories if inventory)
    matched_keys = set(baseline)
    for inventory in compared_sets:
        matched_keys &= inventory
    if baseline and compared_sets and len(matched_keys) < len(baseline):
        issues.append(
            PreflightIssue(
                PreflightSeverity.WARNING,
                "partial_coverage",
                "preflight.partial_coverage",
                f"{len(matched_keys)}/{len(baseline)}",
            )
        )
    if not baseline:
        issues.append(PreflightIssue(PreflightSeverity.ERROR, "no_frames", "preflight.no_frames"))

    roles = (
        _role_status(
            AnalysisSourceRole.ORIGINAL,
            (original_keys,) if original_folder is not None else (),
            baseline,
            str(original_folder.path) if original_folder else "",
        ),
        _role_status(
            AnalysisSourceRole.MODEL_OUTPUT,
            tuple(model_inventories),
            baseline,
            ", ".join(model.display_name for model in model_specs),
        ),
        _role_status(
            AnalysisSourceRole.CONFIDENCE, tuple(confidence_inventories), baseline, str(len(confidence_inventories))
        ),
    )
    return AnalysisPreflightReport(
        profile=profile.key,
        roles=roles,
        issues=tuple(issues),
        total_frames=len(baseline),
        matched_frames=len(matched_keys) if baseline else 0,
    )


__all__ = [
    "ANALYSIS_PROFILES",
    "DEFAULT_ANALYSIS_PROFILE",
    "AnalysisPreflightReport",
    "AnalysisProfileDefinition",
    "PreflightIssue",
    "PreflightSeverity",
    "SourceRoleStatus",
    "analysis_profile_definition",
    "build_standalone_preflight",
]

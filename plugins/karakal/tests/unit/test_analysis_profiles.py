from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kraken_core.analysis_protocol import AnalysisParameter, AnalysisProfileKind, AnalysisScaleMode, AnalysisSourceRole

from karakal.core.analysis_profiles import PreflightSeverity, build_standalone_preflight
from karakal.core.domain import BuildOptions, FolderSpec, ModelSpec
from karakal.core.project_profile import (
    AnalysisSourceBinding,
    KarakalAnalysisProfileV1,
    SourceBindingKind,
)


def _image(folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"image")


def test_preflight_reports_partial_model_coverage(tmp_path: Path) -> None:
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    for name in ("1.png", "2.png", "3.png"):
        _image(model_a, name)
    for name in ("1.png", "2.png"):
        _image(model_b, name)

    report = build_standalone_preflight(
        AnalysisProfileKind.MODEL_COMPARISON,
        None,
        (
            ModelSpec("model_a", "Model A", model_a),
            ModelSpec("model_b", "Model B", model_b),
        ),
    )

    assert report.can_run
    assert report.total_frames == 3
    assert report.matched_frames == 2
    assert any(issue.severity == PreflightSeverity.WARNING for issue in report.issues)


def test_confidence_profile_requires_nonempty_confidence_source(tmp_path: Path) -> None:
    model = tmp_path / "model"
    confidence = tmp_path / "confidence"
    _image(model, "1.png")
    confidence.mkdir()

    report = build_standalone_preflight(
        AnalysisProfileKind.CONFIDENCE_AUDIT,
        None,
        (ModelSpec("model", "Model", model, prob_folder=confidence),),
    )

    assert not report.can_run
    assert any(issue.code == "confidence_required" for issue in report.issues)


def test_webp_is_supported_consistently_by_build_and_preflight(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _image(original, "1.webp")

    report = build_standalone_preflight(
        AnalysisProfileKind.GRID_DEFECTS,
        FolderSpec(original, "Original"),
        (),
    )

    assert ".webp" in BuildOptions().file_extensions
    assert report.can_run
    assert report.total_frames == 1


def test_grid_profile_accepts_model_output_without_original(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _image(model, "1.png")

    report = build_standalone_preflight(
        AnalysisProfileKind.GRID_DEFECTS,
        None,
        (ModelSpec("model", "Model", model),),
    )

    assert report.can_run
    assert report.total_frames == 1


def test_single_result_profile_requires_exactly_one_nonempty_model_output(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    other = tmp_path / "other"
    _image(other, "1.png")

    missing_report = build_standalone_preflight(
        AnalysisProfileKind.SINGLE_RESULT_RISK,
        None,
        (ModelSpec("empty", "Empty", empty),),
    )
    multiple_report = build_standalone_preflight(
        AnalysisProfileKind.SINGLE_RESULT_RISK,
        None,
        (ModelSpec("empty", "Empty", empty), ModelSpec("other", "Other", other)),
    )
    zero_report = build_standalone_preflight(AnalysisProfileKind.SINGLE_RESULT_RISK, None, ())

    assert not zero_report.can_run
    assert any(issue.code == "models_required" for issue in zero_report.issues)
    assert not missing_report.can_run
    assert any(issue.code == "model_output_required" for issue in missing_report.issues)
    assert not multiple_report.can_run
    assert any(issue.code == "single_model_required" for issue in multiple_report.issues)


def test_single_result_optional_partial_coverage_is_warning_only(tmp_path: Path) -> None:
    model = tmp_path / "model"
    original = tmp_path / "original"
    confidence = tmp_path / "confidence"
    for name in ("1.png", "2.png"):
        _image(model, name)
    _image(original, "1.png")
    confidence.mkdir()

    report = build_standalone_preflight(
        AnalysisProfileKind.SINGLE_RESULT_RISK,
        FolderSpec(original, "Original"),
        (ModelSpec("model", "Model", model, prob_folder=confidence),),
    )

    assert report.can_run
    assert report.total_frames == 2
    assert report.matched_frames == 2
    assert any(issue.code == "partial_coverage" for issue in report.issues)


def test_project_profile_round_trip_preserves_representation_versions() -> None:
    profile = KarakalAnalysisProfileV1(
        project_id="project-1",
        profile=AnalysisProfileKind.MODEL_COMPARISON,
        bindings=(
            AnalysisSourceBinding(
                binding_key="model-a",
                role=AnalysisSourceRole.MODEL_OUTPUT,
                kind=SourceBindingKind.REPRESENTATION,
                source_id="representation-a",
                source_version_id="version-a",
            ),
        ),
        scale_mode=AnalysisScaleMode.ABSOLUTE,
    )

    assert KarakalAnalysisProfileV1.from_payload(profile.to_payload()) == profile


def test_project_profile_v1_preserves_single_result_sensitivity() -> None:
    profile = KarakalAnalysisProfileV1(
        project_id="project-1",
        profile=AnalysisProfileKind.SINGLE_RESULT_RISK,
        bindings=(),
        parameters=(AnalysisParameter("single_result_sensitivity", "strict"),),
    )

    payload = profile.to_payload()
    restored = KarakalAnalysisProfileV1.from_payload(payload)

    assert payload["schema"] == "karakal.analysis-profile.v1"
    assert restored.parameters == (AnalysisParameter("single_result_sensitivity", "strict"),)

    percentile_profile = replace(profile, scale_mode=AnalysisScaleMode.PERCENTILE)
    assert KarakalAnalysisProfileV1.from_payload(percentile_profile.to_payload()) == percentile_profile


def test_project_profile_rejects_non_object_bindings() -> None:
    with pytest.raises(ValueError, match="bindings entries must be objects"):
        KarakalAnalysisProfileV1.from_payload(
            {
                "schema": "karakal.analysis-profile.v1",
                "profile": "model_comparison",
                "bindings": ["not-an-object"],
                "parameters": [],
                "visible_layers": [],
            }
        )

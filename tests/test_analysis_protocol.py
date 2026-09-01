from __future__ import annotations

import hashlib

import pytest

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisJobManifest,
    AnalysisParameter,
    AnalysisProfileKind,
    AnalysisResultManifest,
    AnalysisScaleMode,
    AnalysisSourceRole,
)
from kraken_core.plugins import load_plugin_catalog


def test_analysis_job_supports_multiple_named_inputs_per_frame() -> None:
    digest = hashlib.sha256(b"frame").hexdigest()
    frame = AnalysisFrameInput(
        frame_id="frame-1",
        x=3,
        y=7,
        artifacts=(
            AnalysisArtifactInput(
                binding_key="original",
                role=AnalysisSourceRole.ORIGINAL,
                artifact_id="artifact-original",
                artifact_version_id="version-original",
                relative_path="inputs/original/frame-1.png",
                media_type="image/png",
                sha256=digest,
            ),
            AnalysisArtifactInput(
                binding_key="model-a",
                role=AnalysisSourceRole.MODEL_OUTPUT,
                artifact_id="artifact-model-a",
                artifact_version_id="version-model-a",
                relative_path="inputs/model-a/frame-1.png",
                media_type="image/png",
                sha256=digest,
            ),
        ),
    )
    manifest = AnalysisJobManifest(
        job_id="job-1",
        project_id="project-1",
        profile=AnalysisProfileKind.MODEL_COMPARISON,
        frames=(frame,),
        parameters=(AnalysisParameter("metric", "dice"),),
    )

    restored = AnalysisJobManifest.from_payload(manifest.to_payload())

    assert restored == manifest
    assert [artifact.binding_key for artifact in restored.frames[0].artifacts] == ["original", "model-a"]


def test_analysis_job_rejects_duplicate_frame_bindings() -> None:
    digest = hashlib.sha256(b"frame").hexdigest()
    artifact = AnalysisArtifactInput(
        binding_key="model-a",
        role=AnalysisSourceRole.MODEL_OUTPUT,
        artifact_id="artifact-model-a",
        artifact_version_id="version-model-a",
        relative_path="inputs/model-a/frame-1.png",
        media_type="image/png",
        sha256=digest,
    )

    with pytest.raises(ValueError, match="Duplicate artifact binding"):
        AnalysisFrameInput(frame_id="frame-1", x=1, y=1, artifacts=(artifact, artifact))


def test_catalog_exposes_karakal_analysis_profiles() -> None:
    catalog = {plugin.id: plugin for plugin in load_plugin_catalog("src/kraken_hub/resources/plugins.json")}
    karakal = catalog["karakal"]

    assert karakal.analysis_protocol_version == "1.0"
    assert {capability.profile for capability in karakal.analysis_capabilities} == {
        "model_comparison",
        "confidence_audit",
        "grid_defects",
    }
    model_comparison = next(
        capability for capability in karakal.analysis_capabilities if capability.profile == "model_comparison"
    )
    assert model_comparison.modes == ("interactive", "headless")
    assert all(
        capability.modes == ("interactive",)
        for capability in karakal.analysis_capabilities
        if capability.profile != "model_comparison"
    )
    grid_capability = next(
        capability for capability in karakal.analysis_capabilities if capability.profile == "grid_defects"
    )
    assert grid_capability.required_roles == ()
    assert grid_capability.required_any_role_groups == (("original", "model_output"),)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"artifacts": ["not-an-object"]}, "frame.artifacts entries must be objects"),
        (
            {
                "schema": "kraken.analysis-job.v1",
                "protocol_version": "1.0",
                "job_id": "job-1",
                "project_id": "project-1",
                "profile": "model_comparison",
                "frames": ["not-an-object"],
                "parameters": [],
            },
            "frames entries must be objects",
        ),
    ],
)
def test_analysis_job_rejects_non_object_collection_entries(payload, message: str) -> None:
    if "artifacts" in payload:
        with pytest.raises(ValueError, match=message):
            AnalysisFrameInput.from_payload(payload)
        return

    with pytest.raises(ValueError, match=message):
        AnalysisJobManifest.from_payload(payload)


def test_analysis_result_round_trip(tmp_path) -> None:
    from kraken_core.analysis_protocol import (
        AnalysisFrameResult,
        AnalysisMetricValue,
        AnalysisOutcome,
        AnalysisScaleDefinition,
    )

    manifest = AnalysisResultManifest(
        job_id="job-1",
        project_id="project-1",
        profile=AnalysisProfileKind.CONFIDENCE_AUDIT,
        outcome=AnalysisOutcome.SUCCEEDED,
        frames=(
            AnalysisFrameResult(
                frame_id="frame-1",
                x=1,
                y=2,
                status="ready",
                metrics=(AnalysisMetricValue("confidence", 0.7, 0.7, percentile=55.0),),
            ),
        ),
        scales=(AnalysisScaleDefinition("confidence", AnalysisScaleMode.ABSOLUTE, 0.0, 1.0),),
    )
    path = tmp_path / "result.json"

    manifest.write(path)

    assert AnalysisResultManifest.read(path) == manifest

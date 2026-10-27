"""Managed source -> measured pose -> bounded rendering -> durable reopen."""
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.errors import ConcurrencyError, ConflictError
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.multilayer import ControlPoint, MultilayerModel, RegistrationFit
from kraken_manager.infrastructure.stitching import StitchRenderer, VerifiedTileSources
from test_server_workspace_inputs_parts.server_layer import _server_layer


@pytest.fixture
def geometry(tmp_path):
    server, local, session, project, layer, _, representation, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    cells = server.matrix_viewport(str(project.id), layer_id=str(layer.id),
        representation_ids=(str(representation.id),), x1=1, y1=1, x2=2, y2=2, lod=0, include_missing=False)["cells"]
    versions = [local.get_artifact_version(project.id, c["artifact_version_id"]) for c in cells]
    source_series = [local._projection(project.id).get_artifact_series(v.series_id) for v in versions]
    series = local.create_artifact_series(principal=session.principal, project_id=project.id,
        layer_id=layer.id, scope=ArtifactScope.LAYER_ATTACHMENT, name="Observed tile geometry", idempotency_key="plan-series")
    path = tmp_path / "observations.json"
    path.write_text('{"purpose":"controlled functional fixture"}', encoding="utf-8")
    plan = local.add_managed_artifact_version(principal=session.principal, project_id=project.id,
        series_id=series.id, source=path, idempotency_key="plan-source")

    def command(operation, arguments, key=None, expected=None):
        return local.mutate_multilayer_model(project_id=project.id, principal=session.principal,
            expected_revision=local.get_multilayer_model(project.id)["model"]["revision"] if expected is None else expected,
            operation=operation, arguments=arguments, idempotency_key=key or str(uuid4()))

    command("set_stack", {"layers": [{"layer_id": layer.id, "layer_type": layer.type, "z_m": 0, "thickness_m": 0}]})
    tiles = [{"frame_id": s.frame_id, "artifact_version_id": v.id, "width": 16, "height": 12,
              "pixel_size_m": [1e-6, 1e-6], "y_axis": "up"} for s, v in zip(source_series, versions)]
    tiles[1]["control_points"] = [{"source_xy": p, "target_xy": [p[0]+2, p[1]+1]}
                                 for p in [(0, 0), (10, 0), (0, 8), (10, 8)]]
    arguments = {"layer_id": layer.id, "plan_version_id": plan.id, "reference_frame_id": source_series[0].frame_id,
                 "tiles": tiles, "max_error_m": 1e-12}
    return local, session, project, layer, versions, arguments, command


def test_actual_managed_images_measured_fit_render_and_reopen(geometry, tmp_path):
    local, _, project, _, versions, args, command = geometry
    payload = command("plan_layer_stitch", args, "observed-plan")
    stitch = MultilayerModel.from_dict(payload).stitches[-1]
    fit = stitch.tiles[1].fit
    assert fit.status == "fitted"
    assert fit.affine == pytest.approx((1, 0, 2e-6, 0, 1, 1e-6), abs=1e-14)
    assert fit.max_residual_m < 1e-12
    assert fit.control_error_bound_m is None
    assert payload["assessment"]["stitches"][-1] == {"id": stitch.id, "revision": 0, "current": True,
        "geometry_status": "determined", "render_status": "not_computed"}
    paths = {}
    for version in versions:
        path = tmp_path / version.id
        path.write_bytes(local.read_project_blob(project.id, version.sha256))
        paths[str(version.id)] = path
    blocks = list(StitchRenderer(stitch).blocks(VerifiedTileSources(paths)))
    assert len(blocks) == 1
    assert blocks[0].coverage.max() == 2
    assert np.any(blocks[0].coverage == 0)
    assert np.all(blocks[0].pixels[blocks[0].coverage > 0, 0] >= 250)
    reopened = EmbeddedProjectService(local.data_dir)
    assert reopened.get_multilayer_model(project.id) == payload
    assert command("plan_layer_stitch", args, "observed-plan", expected=0) == payload
    with pytest.raises(ConcurrencyError):
        command("plan_layer_stitch", args, expected=0)
    with pytest.raises(ConflictError, match="Idempotency"):
        command("plan_layer_stitch", args | {"max_error_m": 1}, "observed-plan")
    paths[str(versions[0].id)].write_bytes(b"damaged managed download")
    with pytest.raises(DomainValidationError, match="SHA-256"):
        list(StitchRenderer(stitch).blocks(VerifiedTileSources(paths)))


def test_unresolved_fit_is_retained_and_supersedes_old_revision(geometry):
    _, _, _, _, _, args, command = geometry
    first = command("plan_layer_stitch", args)
    args["tiles"][1]["control_points"] = args["tiles"][1]["control_points"][:2]
    second = command("plan_layer_stitch", args)
    assessments = second["assessment"]["stitches"]
    assert assessments[0]["current"] is False
    assert assessments[1]["current"] is True
    assert assessments[1]["geometry_status"] == "unresolved"
    model = MultilayerModel.from_dict(second)
    assert model.stitches[-1].revision == 1
    assert model.stitches[0].tiles[1].fit == MultilayerModel.from_dict(first).stitches[0].tiles[1].fit
    with pytest.raises(DomainValidationError, match="unresolved"):
        StitchRenderer(model.stitches[-1])


def test_replacing_source_marks_plan_stale_and_foreign_frame_binding_is_rejected(geometry, tmp_path):
    local, session, project, _, versions, args, command = geometry
    command("plan_layer_stitch", args)
    changed = tmp_path / "replacement.pgm"
    changed.write_bytes(b"P5\n1 1\n255\n\x00")
    local.add_managed_artifact_version(principal=session.principal, project_id=project.id,
        series_id=versions[1].series_id, source=changed, parent_version_id=versions[1].id, idempotency_key="replace-source")
    assert local.get_multilayer_model(project.id)["assessment"]["stitches"][-1]["current"] is False
    with pytest.raises(ConflictError, match="stale"):
        command("plan_layer_stitch", args)
    args["tiles"][0]["frame_id"] = str(uuid4())
    with pytest.raises(DomainValidationError, match="managed frame"):
        command("plan_layer_stitch", args)


def test_declared_transform_has_no_residuals_and_invalid_controls_are_not_observations(geometry):
    _, _, _, _, _, args, command = geometry
    tile = args["tiles"][1]
    tile.pop("control_points")
    tile.update(declared_affine=[1, 0, 5e-6, 0, 1, 0], note="controlled stage transform for functional test")
    model = MultilayerModel.from_dict(command("plan_layer_stitch", args))
    assert model.stitches[-1].tiles[1].fit is None
    points = [{"source_xy": p, "target_xy": p} for p in [(0, 0), (20, 0), (0, 20)]]
    with pytest.raises(DomainValidationError, match="outside"):
        replace(model.stitches[-1], tiles=(model.stitches[-1].tiles[0], replace(model.stitches[-1].tiles[1],
            provenance="control_points", affine=None, control_points=tuple(ControlPoint(**p) for p in points),
            fit=RegistrationFit("rejected", "test"))))


def test_catalog_coordinates_must_match_frame_identity_and_do_not_supply_geometry(geometry):
    _, _, _, _, _, args, command = geometry
    args["tiles"][0]["catalog_xy"] = [2, 1]
    with pytest.raises(DomainValidationError, match="catalog coordinates"):
        command("plan_layer_stitch", args)

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from kraken_core.plugin_protocol import (
    FRAME_ROLE_DATASET,
    FRAME_ROLE_MODEL,
    FRAME_ROLE_SOURCE,
    PluginFrameInput,
    PluginJobManifest,
    PluginOperation,
)
from kraken_manager.infrastructure.plugin.agent_gateway import _stage_model
from kraken_server.cli import _doctor, _parser


def _frame() -> PluginFrameInput:
    return PluginFrameInput(
        frame_id="frame-1",
        x=1,
        y=1,
        artifact_version_id="version-1",
        sha256="a" * 64,
        media_type="image/png",
        relative_path="inputs/frame.png",
    )


def _job(operation: str, root: Path) -> PluginJobManifest:
    manifest = PluginJobManifest(
        job_id="job-1",
        operation=operation,
        project_id="project-1",
        layer_id="layer-1",
        actor_id="actor-1",
        target_representation_id="representation-1",
        inputs=(_frame(),),
    )
    (root / "job.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def test_gateway_stages_model_inside_the_job_workspace(tmp_path: Path) -> None:
    from kraken_agent.jobs import StagingWorkspace

    source = tmp_path / "model.onnx"
    source.write_bytes(b"model-bytes")
    digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    workspace = StagingWorkspace(tmp_path / "staging", "job-1")
    workspace.create()
    parameters = _stage_model(
        workspace,
        {"model_source_path": str(source), "model_sha256": digest},
    )
    assert parameters["model_relative_path"] == "inputs/model/model.onnx"
    assert parameters["model_sha256"] == digest
    assert (workspace.path / "inputs" / "model" / "model.onnx").read_bytes() == b"model-bytes"
    assert FRAME_ROLE_SOURCE == "source"
    assert FRAME_ROLE_MODEL == "model"


def test_contour_dataset_result_publishes_a_zip(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    from contour.kraken_bridge import ContourKrakenSession

    root = tmp_path / "job"
    outputs = root / "outputs"
    outputs.mkdir(parents=True)
    manifest = _job(PluginOperation.PREPARE_DATASET.value, root)
    (outputs / "dataset.zip").write_bytes(b"zip-bytes")
    session = ContourKrakenSession(
        manifest=manifest,
        staging_root=root,
        job_manifest_path=root / "job.json",
        result_manifest_path=root / "result.json",
        input_paths=(),
        output_directory=outputs,
    )
    result = session.build_result()
    assert result.outputs[0].media_type == "application/zip"
    assert result.outputs[0].role == FRAME_ROLE_DATASET


def test_neural_training_session_publishes_a_model(tmp_path: Path) -> None:
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    root = tmp_path / "job"
    root.mkdir()
    _job(PluginOperation.TRAIN_MODEL.value, root)
    session = NeuralImageTrainSession.load(
        job_manifest=root / "job.json",
        result_manifest=root / "result.json",
        staging_root=root,
    )
    (session.output_directory / "weights.onnx").write_bytes(b"weights")
    result = session.publish_model()
    assert result.outputs[0].role == "model"
    assert json.loads((root / "result.json").read_text(encoding="utf-8"))["job_id"] == "job-1"


def test_karakal_confidence_worker_writes_the_result_contract(tmp_path: Path) -> None:
    from karakal.worker import main

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    job = workspace / "job.json"
    _job(PluginOperation.ANALYZE_LAYER_CONFIDENCE.value, workspace)
    job.write_text((workspace / "job.json").read_text(encoding="utf-8"), encoding="utf-8")
    code = main(
        [
            "--job",
            str(workspace / "job.json"),
            "--result",
            str(workspace / "result.json"),
            "--workspace",
            str(workspace),
        ]
    )
    assert code == 0
    payload = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
    assert payload["outputs"][0]["role"] == "confidence"
    assert (workspace / "outputs" / "confidence.json").is_file()


def test_remote_project_identity_uses_the_known_catalog() -> None:
    from kraken_hub.remote_client import RemoteServerProjectService

    service = object.__new__(RemoteServerProjectService)
    service._remote_ids = {"project-1"}
    assert service.is_remote_project("project-1") is True
    assert service.is_remote_project("other") is False


def test_development_workspace_is_available(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
    from kraken_server.app import create_app
    from kraken_server.services import InMemoryServerServices

    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    admin = store.create_account("admin", "Admin", "secret")
    store.grant_global_role(admin.account_id, "server_admin")
    app = create_app(services=InMemoryServerServices(), account_store=store, project_access_mode="acl")
    client = TestClient(app)
    session = client.post("/api/v1/auth/sessions", json={"username": "admin", "password": "secret"})
    token = session.json()["access_token"]
    created = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "workspace"},
        json={"name": "Dev", "width": 1, "height": 1},
    )
    assert created.status_code == 201, created.text
    workspace = client.get(
        f"/api/v1/projects/{created.json()['project_id']}/workspace",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert workspace.status_code == 200, workspace.text
    assert Path(workspace.json()["source_project_dir"]).is_dir()


def test_doctor_reports_a_missing_configuration(tmp_path: Path, capsys) -> None:
    code = _doctor(Namespace(config=tmp_path / "missing.toml", json=True))
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["checks"][0]["check"] == "configuration"


def test_admin_parser_exposes_operator_commands() -> None:
    parser = _parser()
    migrate = parser.parse_args(["migrate", "--config", "server.toml"])
    tokens = parser.parse_args(["list-agent-tokens", "--config", "server.toml", "--json"])
    assert migrate.command == "migrate"
    assert tokens.command == "list-agent-tokens"
    assert tokens.json is True


def test_agent_lease_moves_a_queued_job_to_leased() -> None:
    pytest.importorskip("sqlalchemy")
    import sqlalchemy as sa

    from kraken_server.agent_auth import AgentIdentity
    from kraken_server.agent_gateway import PostgresAgentGateway

    engine = sa.create_engine("sqlite://")
    gateway = PostgresAgentGateway(engine, tokens=None, blobs=None)
    gateway.metadata.create_all(engine)
    now = __import__("datetime").datetime.now(__import__("datetime").UTC)
    with engine.begin() as connection:
        connection.execute(
            sa.insert(gateway.jobs).values(
                job_id="11111111-1111-1111-1111-111111111111",
                project_id="22222222-2222-2222-2222-222222222222",
                state="queued",
                payload={"capability": "frames.vectorize.v1"},
                lease_owner=None,
                lease_until=None,
                lease_attempts=0,
                last_error=None,
                created_at=now,
                updated_at=now,
            )
        )
    agent = AgentIdentity("token-1", "worker", frozenset({"frames.vectorize.v1"}))
    leased = gateway.lease(agent, seconds=60)
    assert leased is not None
    with engine.connect() as connection:
        state = connection.execute(
            sa.select(gateway.jobs.c.state).where(
                gateway.jobs.c.job_id == "11111111-1111-1111-1111-111111111111"
            )
        ).scalar_one()
    assert state == "leased"
    gateway.heartbeat("11111111-1111-1111-1111-111111111111", agent, seconds=60)

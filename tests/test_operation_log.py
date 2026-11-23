from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_server.app import create_app
from kraken_server.configuration import ServerConfig, write_config
from kraken_server.operation_log import (
    LOGGER,
    NameBook,
    Requester,
    configure_operation_log,
    describe_request,
    job_label,
    record_http_request,
    request_rank,
)
from kraken_server.services import InMemoryServerServices


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _captured(level: str):
    configure_operation_log(level)
    handler = _Capture()
    LOGGER.addHandler(handler)
    return handler


def test_ranks_follow_the_requested_events() -> None:
    assert request_rank("POST", "/api/v1/projects") == 1
    assert request_rank("POST", "/api/v1/auth/accounts") == 1
    assert request_rank("POST", "/api/v1/auth/accounts") == 1
    assert request_rank("POST", "/api/v1/projects/abc/layers") == 2
    assert request_rank("POST", "/api/v1/projects/abc/artifacts/s/uploads") == 2
    assert request_rank("POST", "/api/v1/projects/abc/artifacts/s/versions") == 2
    assert request_rank("GET", "/api/v1/health") == 3
    assert request_rank("POST", "/api/v1/projects/abc/layers/reorder") == 3


def test_none_records_nothing() -> None:
    handler = _captured("none")
    try:
        record_http_request("POST", "/api/v1/projects", 201)
        record_http_request("GET", "/api/v1/health", 200)
    finally:
        LOGGER.removeHandler(handler)
    assert handler.messages == []


def test_low_records_successful_project_and_account_creation_only() -> None:
    handler = _captured("low")
    try:
        record_http_request("POST", "/api/v1/projects", 201)
        record_http_request("POST", "/api/v1/auth/accounts", 201)
        record_http_request("POST", "/api/v1/projects", 422)
        record_http_request("POST", "/api/v1/projects/abc/layers", 201)
        record_http_request("GET", "/api/v1/projects", 200)
    finally:
        LOGGER.removeHandler(handler)
    assert handler.messages == [
        "создание проекта → 201",
        "создание учётной записи → 201",
    ]


def test_medium_adds_layers_and_uploads() -> None:
    handler = _captured("medium")
    try:
        record_http_request("POST", "/api/v1/projects", 201)
        record_http_request("POST", "/api/v1/projects/abc/layers", 201)
        record_http_request("POST", "/api/v1/projects/abc/artifacts/s/uploads/complete", 201)
        record_http_request("GET", "/api/v1/health", 200)
    finally:
        LOGGER.removeHandler(handler)
    assert [message.split(" ", 1)[0] for message in handler.messages] == [
        "создание",
        "создание",
        "загрузка",
    ]
    assert "health" not in " ".join(handler.messages)


def test_high_records_every_request_including_failures() -> None:
    handler = _captured("high")
    try:
        record_http_request("GET", "/api/v1/health", 200)
        record_http_request("POST", "/api/v1/projects", 422)
    finally:
        LOGGER.removeHandler(handler)
    assert handler.messages == [
        "запрос /health → 200",
        "создание проекта → ошибка 422",
    ]


def test_middleware_uses_the_configured_level(monkeypatch) -> None:
    monkeypatch.setenv("KRAKEN_LOG_LEVEL", "low")
    handler = _Capture()
    LOGGER.addHandler(handler)
    try:
        client = TestClient(create_app(services=InMemoryServerServices(), development=True))
        created = client.post(
            "/api/v1/projects",
            headers={"Authorization": "Bearer developer", "Idempotency-Key": "journal"},
            json={"name": "Журнал", "width": 2, "height": 2},
        )
        assert created.status_code == 201
        listed = client.get("/api/v1/projects", headers={"Authorization": "Bearer developer"})
        assert listed.status_code == 200
    finally:
        LOGGER.removeHandler(handler)
    assert any(message.endswith(" — создание проекта → 201") for message in handler.messages)
    assert not any("список проектов" in message for message in handler.messages)


def test_config_round_trip_keeps_log_level(monkeypatch, tmp_path: Path) -> None:
    config = write_config(
        tmp_path / "server.toml",
        database_url="postgresql+psycopg://kraken:secret@db/kraken",
        blob_root=tmp_path / "blobs",
        source_root=tmp_path / "source",
        derived_root=tmp_path / "derived",
        log_level="high",
    )
    loaded = ServerConfig.load(config.path)
    assert loaded.log_level == "high"
    monkeypatch.setenv("KRAKEN_LOG_LEVEL", "none")
    loaded.apply_to_environment()
    assert os.environ["KRAKEN_LOG_LEVEL"] == "high"


def test_unknown_log_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="log_level"):
        write_config(
            tmp_path / "server.toml",
            database_url="postgresql+psycopg://kraken:secret@db/kraken",
            blob_root=tmp_path / "blobs",
            source_root=tmp_path / "source",
            derived_root=tmp_path / "derived",
            log_level="verbose",
        )


def test_agent_input_download_names_project_job_and_file() -> None:
    calls: list[str] = []

    def project(project_id: str) -> str:
        calls.append(project_id)
        return "Образец 1"

    names = NameBook(
        project=project,
        job=lambda job_id: ("p-1", job_label({"kind": "analysis", "partition_index": 2, "partition_count": 8})),
        artifact=lambda project_id, version_id: "frame_r03_c12.png",
    )
    job_id = "dfdc53fb-16b7-4e03-9897-a96dc0000cac"
    version_id = "b0259d5c-8cec-440c-938d-9aaba712920a"
    for _ in range(2):
        message = describe_request(
            "GET",
            f"/api/v1/agent/jobs/{job_id}/inputs/{version_id}",
            200,
            route="/api/v1/agent/jobs/{job_id}/inputs/{version_id}",
            params={"job_id": job_id, "version_id": version_id},
            names=names,
        )
    assert message == (
        "агент скачивает входной файл: проект «Образец 1», задача «анализ, часть 3 из 8», "
        "файл «frame_r03_c12.png» → 200"
    )
    assert calls == ["p-1"]


def test_unknown_names_fall_back_to_short_identifiers() -> None:
    def broken(*_args: str) -> str:
        raise RuntimeError("database is down")

    project_id = "2fec7b57-3ade-578b-b718-b67d088cdc3e"
    message = describe_request(
        "GET",
        f"/api/v1/projects/{project_id}/analysis-runs",
        404,
        route="/api/v1/projects/{project_id}/analysis-runs",
        params={"project_id": project_id},
        names=NameBook(project=broken),
    )
    assert message == "список запусков анализа: проект «2fec7b57» → ошибка 404"


def test_middleware_names_the_project_at_high_level(monkeypatch) -> None:
    monkeypatch.setenv("KRAKEN_LOG_LEVEL", "high")
    handler = _Capture()
    LOGGER.addHandler(handler)
    try:
        client = TestClient(create_app(services=InMemoryServerServices(), development=True))
        headers = {"Authorization": "Bearer developer"}
        created = client.post(
            "/api/v1/projects",
            headers={**headers, "Idempotency-Key": "journal-names"},
            json={"name": "Журнал имён", "width": 2, "height": 2},
        )
        assert created.status_code == 201
        project_id = created.json()["project_id"]
        assert client.get(f"/api/v1/projects/{project_id}/layers", headers=headers).status_code == 200
    finally:
        LOGGER.removeHandler(handler)
    layers = [message for message in handler.messages if "список слоёв" in message]
    assert len(layers) == 1
    assert layers[0].endswith(" — список слоёв: проект «Журнал имён» → 200")
    assert not any(project_id in message for message in handler.messages)


def test_requester_is_named_before_the_action() -> None:
    names = NameBook(user=lambda principal_id: "Иван Петров", project=lambda project_id: "Образец 1")
    project_id = "2fec7b57-3ade-578b-b718-b67d088cdc3e"
    common = dict(
        route="/api/v1/projects/{project_id}/analysis-runs",
        params={"project_id": project_id},
        names=names,
    )
    path = f"/api/v1/projects/{project_id}/analysis-runs"
    assert describe_request("GET", path, 200, requester=Requester(principal_id="u-1"), **common) == (
        "Иван Петров — список запусков анализа: проект «Образец 1» → 200"
    )
    assert describe_request("GET", path, 200, requester=Requester(), **common) == (
        "список запусков анализа: проект «Образец 1» → 200"
    )


def test_agent_is_named_and_the_action_reads_naturally() -> None:
    message = describe_request(
        "POST",
        "/api/v1/agent/jobs/dfdc53fb-16b7-4e03-9897-a96dc0000cac/heartbeat",
        200,
        route="/api/v1/agent/jobs/{job_id}/heartbeat",
        params={"job_id": "dfdc53fb-16b7-4e03-9897-a96dc0000cac"},
        names=NameBook(job=lambda job_id: ("p", "сшивка"), project=lambda project_id: "Образец 1"),
        requester=Requester(agent_name="gpu-01"),
    )
    assert message == (
        "агент «gpu-01» — сообщает, что задача выполняется: проект «Образец 1», задача «сшивка» → 200"
    )

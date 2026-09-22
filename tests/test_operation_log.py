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
    configure_operation_log,
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
    assert request_rank("POST", "/api/v1/admin/accounts") == 1
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
        "создание проекта POST /api/v1/projects 201",
        "создание учётной записи POST /api/v1/auth/accounts 201",
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
        "обращение GET /api/v1/health 200",
        "создание проекта POST /api/v1/projects 422",
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
    assert any(message.startswith("создание проекта POST /api/v1/projects 201") for message in handler.messages)
    assert not any("GET /api/v1/projects" in message for message in handler.messages)


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

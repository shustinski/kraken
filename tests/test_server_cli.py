from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kraken_server import cli
from kraken_server.provisioning import (
    new_postgres_provisioning,
    provision_postgres,
    validate_postgres_identifier,
)


class _Result:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return self.value


class _Connection:
    def __init__(self, *, role_exists=False, database_exists=False):
        self.role_exists = role_exists
        self.database_exists = database_exists
        self.mutations = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, parameters=None):
        if isinstance(statement, str) and "pg_roles" in statement:
            return _Result((1,) if self.role_exists else None)
        if isinstance(statement, str) and "pg_database" in statement:
            return _Result((1,) if self.database_exists else None)
        self.mutations.append((statement, parameters))
        return _Result(None)


def _request(**overrides):
    values = {
        "host": "127.0.0.1",
        "port": 5432,
        "administrator": "postgres",
        "administrator_password": "postgres password",
        "database": "kraken_local",
        "application_user": "kraken_local_app",
        "application_password": "database password",
    }
    values.update(overrides)
    return new_postgres_provisioning(**values)


def test_postgres_provisioning_validates_names_and_encodes_supplied_url() -> None:
    request = _request(application_password="p@ss/word")

    assert request.database_url.endswith("/kraken_local")
    assert "p%40ss%2Fword" in request.database_url
    with pytest.raises(ValueError, match="Имя базы данных"):
        validate_postgres_identifier("bad-name", "Имя базы данных")


def test_postgres_provisioning_fails_before_mutation_on_conflict() -> None:
    connection = _Connection(role_exists=True)

    with pytest.raises(ValueError, match="уже существует"):
        provision_postgres(_request(), connect=lambda **_kwargs: connection)

    assert connection.mutations == []


def test_postgres_provisioning_creates_role_and_database() -> None:
    connection = _Connection()

    provision_postgres(_request(), connect=lambda **_kwargs: connection)

    assert len(connection.mutations) == 2


def test_cli_help_lists_simple_commands_arguments_and_examples(capsys) -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit) as root_help:
        parser.parse_args(["--help"])
    assert root_help.value.code == 0
    root = capsys.readouterr().out
    assert "init" in root
    assert "project-create" in root
    assert "Примеры:" in root

    with pytest.raises(SystemExit) as command_help:
        parser.parse_args(["init-local-server", "--help"])
    assert command_help.value.code == 0
    output = capsys.readouterr().out
    assert "--database-name" in output
    assert "KRAKEN_POSTGRES_ADMIN_PASSWORD" in output
    assert "SQL и psql" in output


def test_local_initialization_orchestrates_database_migrations_config_and_admin(monkeypatch, tmp_path: Path) -> None:
    calls = []
    config_path = tmp_path / "server.toml"
    args = cli._parser().parse_args(
        [
            "init-local-server",
            "--config",
            str(config_path),
            "--blob-root",
            str(tmp_path / "blobs"),
            "--non-interactive",
        ]
    )
    monkeypatch.setattr(cli, "_secret", lambda *_args, **_kwargs: "postgres password")
    monkeypatch.setattr(cli, "_password", lambda *_args, **_kwargs: "admin password")
    monkeypatch.setattr(cli, "provision_postgres", lambda request: calls.append(("provision", request)))
    monkeypatch.setattr(cli, "run_migrations", lambda url: calls.append(("migrate", url)))

    def write(path, **kwargs):
        calls.append(("config", path, kwargs))
        return SimpleNamespace(path=path, database_url=kwargs["database_url"], host="127.0.0.1", port=8080)

    monkeypatch.setattr(cli, "write_config", write)
    monkeypatch.setattr(cli, "_postgres_store", lambda _url: object())
    monkeypatch.setattr(cli, "_bootstrap", lambda *_args, **_kwargs: "account-id")

    assert cli._execute(args) == 0
    assert [call[0] for call in calls] == ["provision", "migrate", "config"]


def test_local_initialization_wizard_uses_entered_database_and_admin_credentials(monkeypatch, tmp_path: Path) -> None:
    args = cli._parser().parse_args(["init", "--config", str(tmp_path / "server.toml")])
    answers = iter(
        [
            "db.example.test",
            "5544",
            "db_creator",
            "kraken_demo",
            "kraken_runtime",
            "kraken_admin",
            "Главный администратор",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(cli, "_secret", lambda *_args, **_kwargs: "postgres password")
    monkeypatch.setattr(
        cli,
        "_password",
        lambda *_args, **kwargs: (
            "database password" if kwargs.get("environment") == "KRAKEN_DATABASE_PASSWORD" else "kraken password"
        ),
    )
    captured = {}
    monkeypatch.setattr(cli, "provision_postgres", lambda request: captured.update(request=request))
    monkeypatch.setattr(cli, "run_migrations", lambda _url: None)

    def write(path, **kwargs):
        return SimpleNamespace(path=path, database_url=kwargs["database_url"], host="127.0.0.1", port=8080)

    monkeypatch.setattr(cli, "write_config", write)
    monkeypatch.setattr(cli, "_postgres_store", lambda _url: object())

    def bootstrap(_store, username, display_name, *, password):
        captured.update(username=username, display_name=display_name, password=password)
        return "account-id"

    monkeypatch.setattr(cli, "_bootstrap", bootstrap)

    cli._execute(args)

    request = captured["request"]
    assert request.host == "db.example.test"
    assert request.port == 5544
    assert request.administrator == "db_creator"
    assert request.database == "kraken_demo"
    assert request.application_user == "kraken_runtime"
    assert request.application_password == "database password"
    assert captured["username"] == "kraken_admin"
    assert captured["display_name"] == "Главный администратор"
    assert captured["password"] == "kraken password"


def test_local_initialization_rolls_back_new_database_on_failure(monkeypatch, tmp_path: Path) -> None:
    args = cli._parser().parse_args(
        ["init-local-server", "--config", str(tmp_path / "server.toml"), "--non-interactive"]
    )
    calls = []
    monkeypatch.setattr(cli, "_secret", lambda *_args, **_kwargs: "postgres password")
    monkeypatch.setattr(cli, "_password", lambda *_args, **_kwargs: "admin password")
    monkeypatch.setattr(cli, "provision_postgres", lambda request: calls.append(("provision", request)))
    monkeypatch.setattr(cli, "remove_provisioned_postgres", lambda request: calls.append(("remove", request)))
    monkeypatch.setattr(
        cli,
        "run_migrations",
        lambda _url: (_ for _ in ()).throw(RuntimeError("migration failed")),
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        cli._execute(args)

    assert [call[0] for call in calls] == ["provision", "remove"]


def test_project_create_is_scriptable(monkeypatch, capsys) -> None:
    args = cli._parser().parse_args(["project-create", "--name", "Demo", "--width", "10", "--height", "20", "--json"])
    monkeypatch.setattr(cli, "_server_session", lambda _args: ("http://server", "token"))
    captured = {}

    def request(server, method, path, **kwargs):
        captured.update(server=server, method=method, path=path, **kwargs)
        return {"project_id": "project-id", "name": "Demo"}

    monkeypatch.setattr(cli, "_api_request", request)

    cli._create_project(args)

    assert captured["payload"]["width"] == 10
    assert captured["idempotency_key"]
    assert '"project_id": "project-id"' in capsys.readouterr().out


def test_project_cli_requires_https_outside_loopback(monkeypatch) -> None:
    args = Namespace(
        server="http://kraken.example.test",
        username="admin",
        non_interactive=False,
    )
    monkeypatch.setattr(cli, "_secret", lambda *_args, **_kwargs: "password")

    with pytest.raises(ValueError, match="HTTPS"):
        cli._server_session(args)

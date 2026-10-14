"""cli: api request."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

from kraken_admin.cli_parts.install_service import _secret

def _record_migration_check(record, database_url: str) -> None:
    try:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine

        from kraken_server.configuration import migration_root

        configuration = Config()
        configuration.set_main_option("script_location", str(migration_root()))
        heads = set(ScriptDirectory.from_config(configuration).get_heads())
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            current = set(MigrationContext.configure(connection).get_current_heads())
        record(
            "migrations",
            current == heads,
            "схема на голове миграций" if current == heads else f"текущие {sorted(current)}, голова {sorted(heads)}",
        )
    except Exception as exc:  # noqa: BLE001 - doctor reports migration inspection failures
        record("migrations", False, str(exc))

def _record_health_check(record, configuration) -> None:
    import urllib.error
    import urllib.request

    scheme = "https" if configuration.tls_cert_file is not None else "http"
    url = f"{scheme}://{configuration.host}:{configuration.port}/api/v1/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
        record("health", '"status"' in body and "ok" in body, url)
    except (OSError, urllib.error.URLError) as exc:
        record("health", False, f"{url}: {exc}")

def _record_tls_check(record, configuration) -> None:
    for name, path in (
        ("tls_cert", configuration.tls_cert_file),
        ("tls_key", configuration.tls_key_file),
    ):
        if path is None:
            continue
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                record(name, False, f"{path} не читается")
                continue
            path.read_bytes()
        except OSError as exc:
            record(name, False, str(exc))
        else:
            record(name, True, str(path))

def _record_service_check(record) -> None:
    if os.name != "nt":
        return
    result = subprocess.run(
        ["sc.exe", "query", "KrakenServer"],
        check=False,
        text=True,
        capture_output=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        record("windows_service", True, "служба не установлена")
        return
    state = next((line.strip() for line in output.splitlines() if "STATE" in line), output)
    record("windows_service", True, state)

def _service_status() -> str:
    if os.name != "nt":
        raise SystemExit("Windows Service status is available only on Windows")
    result = subprocess.run(
        ["sc.exe", "query", "KrakenServer"],
        check=False,
        text=True,
        capture_output=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return "KrakenServer is not installed"
    return output

def _uninstall_service() -> None:
    if os.name != "nt":
        raise SystemExit("Windows Service removal is available only on Windows")
    subprocess.run(["sc.exe", "stop", "KrakenServer"], check=False, text=True, capture_output=True)
    result = subprocess.run(
        ["sc.exe", "delete", "KrakenServer"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit((result.stdout + result.stderr).strip() or "Не удалось удалить службу KrakenServer")

def _api_request(
    server: str,
    method: str,
    path: str,
    *,
    token: str = "",
    payload: dict[str, object] | None = None,
    idempotency_key: str | None = None,
) -> object:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    request = urllib.request.Request(
        server.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("detail") or parsed.get("title") or detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(f"Kraken Server вернул HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к Kraken Server {server}: {exc.reason}") from exc
    return json.loads(raw) if raw else {}

def _server_session(args: argparse.Namespace) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(args.server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Адрес сервера должен быть корректным URL http:// или https://")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Сетевые подключения к Kraken Server требуют HTTPS")
    password = _secret(
        "Пароль учётной записи Kraken: ",
        environment="KRAKEN_ACCOUNT_PASSWORD",
        non_interactive=args.non_interactive,
    )
    session = _api_request(
        args.server,
        "POST",
        "/api/v1/auth/sessions",
        payload={"username": args.username, "password": password},
    )
    if not isinstance(session, dict) or not session.get("access_token"):
        raise RuntimeError("Kraken Server вернул некорректный ответ авторизации")
    return args.server, str(session["access_token"])

def _create_project(args: argparse.Namespace) -> None:
    if args.width < 1 or args.height < 1:
        raise ValueError("Ширина и высота проекта должны быть больше нуля")
    server, token = _server_session(args)
    project = _api_request(
        server,
        "POST",
        "/api/v1/projects",
        token=token,
        idempotency_key=args.idempotency_key or str(uuid4()),
        payload={
            "name": args.name,
            "width": args.width,
            "height": args.height,
            "orientation": args.orientation,
        },
    )
    if args.json:
        print(json.dumps(project, ensure_ascii=False))
    elif isinstance(project, dict):
        print(f"Проект создан: {project.get('name', args.name)}")
        print(f"Идентификатор проекта: {project.get('project_id', '')}")

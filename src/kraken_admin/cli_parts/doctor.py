"""cli: doctor."""
from __future__ import annotations
import argparse
import json
from kraken_server.configuration import (
    ServerConfig,
)

from kraken_admin.cli_parts.api_request import (
    _record_health_check,
    _record_migration_check,
    _record_service_check,
    _record_tls_check,
)

def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, object]] = []

    def record(name: str, success: bool, detail: str) -> None:
        checks.append({"check": name, "ok": success, "detail": detail})

    try:
        configuration = ServerConfig.load(args.config)
        record("configuration", True, str(configuration.path))
    except Exception as exc:  # noqa: BLE001 - diagnostic must report every configuration failure
        record("configuration", False, str(exc))
        configuration = None
    if configuration is not None:
        try:
            from sqlalchemy import create_engine, inspect, text

            engine = create_engine(configuration.database_url, pool_pre_ping=True)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                tables = set(inspect(connection).get_table_names())
            required = {"server_accounts", "domain_events", "alembic_version"}
            missing = sorted(required - tables)
            record(
                "database",
                not missing,
                "подключение и схема готовы" if not missing else f"отсутствуют таблицы: {', '.join(missing)}",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must report every database failure
            record("database", False, str(exc))
        blob_ready = configuration.blob_root.is_dir()
        record(
            "blob_storage",
            blob_ready,
            str(configuration.blob_root) if blob_ready else "каталог не существует",
        )
        for check, root in (
            ("source_root", configuration.source_root),
            ("derived_root", configuration.derived_root),
        ):
            ready = root.is_dir()
            record(check, ready, str(root) if ready else "каталог не существует")
        _record_migration_check(record, configuration.database_url)
        _record_health_check(record, configuration)
        _record_tls_check(record, configuration)
        _record_service_check(record)
        if configuration.blob_gateway_public_url is not None:
            gateway_ready = (
                configuration.blob_gateway_executable is not None
                and configuration.blob_gateway_executable.is_file()
                and configuration.blob_gateway_secret_value is not None
            )
            record(
                "blob_gateway",
                gateway_ready,
                (
                    f"{configuration.blob_gateway_public_url} ({configuration.blob_gateway_executable})"
                    if gateway_ready
                    else "исполняемый файл или защищённый секрет Blob Gateway отсутствует"
                ),
            )
    if args.json:
        print(json.dumps({"ok": all(bool(item["ok"]) for item in checks), "checks": checks}, ensure_ascii=False))
    else:
        for item in checks:
            marker = "OK" if item["ok"] else "ERROR"
            print(f"[{marker}] {item['check']}: {item['detail']}")
    return 0 if all(bool(item["ok"]) for item in checks) else 2

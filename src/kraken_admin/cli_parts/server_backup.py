"""Explicit local backup/recovery commands; database secrets stay off argv."""
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets

from kraken_admin.server_backup import (
    PostgresTools, ServerBackupError, _directory, create_server_backup,
    restore_server_backup, verify_server_backup,
)
from kraken_manager.infrastructure.filesystem._atomic import atomic_write_json, atomic_write_bytes
from kraken_server.configuration import ServerConfig, protect_secret


def _settings(config):
    return dict(
        server=dict(host=config.host, port=config.port, project_access_mode=config.project_access_mode,
                    log_level=config.log_level, tls_cert_file=str(config.tls_cert_file) if config.tls_cert_file else None,
                    tls_key_file=str(config.tls_key_file) if config.tls_key_file else None),
        database=dict(pool_size=config.db_pool_size, max_overflow=config.db_max_overflow),
        identity=dict(gitlab_issuer=config.gitlab_issuer,
                      gitlab_ca_file=str(config.gitlab_ca_file) if config.gitlab_ca_file else None),
        blob_gateway=dict(enabled=config.blob_gateway_public_url is not None, public_url=config.blob_gateway_public_url,
                          bind=config.blob_gateway_bind, executable=str(config.blob_gateway_executable) if config.blob_gateway_executable else None,
                          ticket_lifetime_seconds=config.blob_ticket_lifetime_seconds,
                          tls_cert_file=str(config.blob_gateway_tls_cert_file) if config.blob_gateway_tls_cert_file else None,
                          tls_key_file=str(config.blob_gateway_tls_key_file) if config.blob_gateway_tls_key_file else None),
        compute_admission=None if config.queue_budget is None else config.queue_budget.to_dict(),
    )


def _toml(sections):
    lines = ["# Kraken Server recovery configuration; secrets are protected separately."]
    for section, values in sections.items():
        if values is None:
            continue
        lines += ["", f"[{section}]"]
        for key, value in values.items():
            if value is not None:
                if not re.fullmatch(r"[a-z_]+", key) or type(value) not in {str, int, float, bool}:
                    raise ServerBackupError("Unsupported restored configuration value")
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_recovery_config(args, result, manifest, database_url):
    directory = Path(args.config_dir).absolute()
    parent = _directory(directory.parent)
    directory = parent / directory.name
    target = Path(args.destination).absolute()
    archive = Path(args.backup).absolute()
    if any(directory.is_relative_to(root) or root.is_relative_to(directory) for root in (target, archive)):
        raise ServerBackupError("Configuration directory must be separate from restored data and archive")
    marker = directory / "recovery.json"
    if directory.exists():
        _directory(directory)
        if not args.resume or not marker.is_file():
            raise ServerBackupError("Configuration directory exists; choose a new directory")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        config = ServerConfig.load(directory / "server.toml")
        if saved != {"backup_id": manifest["backup_id"], "destination": str(target)} or config.database_url != database_url:
            raise ServerBackupError("Existing recovery configuration does not match this restore")
        for name, path in result["storage_roots"].items():
            attribute = {"blobs": "blob_root", "source": "source_root", "derived": "derived_root"}[name]
            if Path(getattr(config, attribute)) != Path(path):
                raise ServerBackupError("Existing configuration storage roots differ")
        return config.path
    settings = json.loads(json.dumps(manifest.get("configuration") or {}))
    if not isinstance(settings.get("server"), dict):
        settings = dict(server=dict(host="127.0.0.1", port=8080, project_access_mode="acl"), database={},
                        blob_gateway=dict(enabled=False))
    if args.host is not None:
        settings["server"]["host"] = args.host
    if args.port is not None:
        settings["server"]["port"] = args.port

    def relocated(value):
        for name, root in sorted(manifest["roots"].items(), key=lambda pair: len(pair[1]["original"]), reverse=True):
            cls = PureWindowsPath if re.match(r"^(?:[A-Za-z]:|\\\\)", root["original"]) else PurePosixPath
            try:
                relative = cls(value).relative_to(cls(root["original"]))
            except ValueError:
                continue
            return str(Path(result["storage_roots"][name]).joinpath(*relative.parts))
        return value

    for section in ("server", "identity", "blob_gateway"):
        for key in ("tls_cert_file", "tls_key_file", "gitlab_ca_file"):
            if settings.get(section, {}).get(key):
                settings[section][key] = relocated(settings[section][key])
    if (args.tls_cert_file is None) != (args.tls_key_file is None):
        raise ServerBackupError("Both restored TLS files are required together")
    if args.tls_cert_file is not None:
        settings["server"].update(tls_cert_file=str(Path(args.tls_cert_file).resolve()), tls_key_file=str(Path(args.tls_key_file).resolve()))
    if args.gitlab_ca_file is not None:
        settings.setdefault("identity", {})["gitlab_ca_file"] = str(Path(args.gitlab_ca_file).resolve())
    if args.blob_gateway_executable is not None:
        settings.setdefault("blob_gateway", {})["executable"] = str(Path(args.blob_gateway_executable).resolve())
    settings.setdefault("database", {})["url_secret"] = "database-url.secret"
    settings["storage"] = {"blob_root": result["storage_roots"]["blobs"],
                           "source_root": result["storage_roots"]["source"],
                           "derived_root": result["storage_roots"]["derived"],
                           "recovery_roots": str(Path(result["blob_root"]) / ".server" / "recovery-roots.json")}
    if settings.get("blob_gateway", {}).get("enabled"):
        settings["blob_gateway"]["secret"] = "blob-gateway.secret"
    encoded = _toml(settings)
    directory.mkdir(mode=0o700, exist_ok=False)
    protect_secret(database_url, directory / "database-url.secret")
    if settings.get("blob_gateway", {}).get("enabled"):
        protect_secret(secrets.token_urlsafe(48), directory / "blob-gateway.secret")
    path = directory / "server.toml"
    atomic_write_bytes(path, encoded, overwrite=False)
    ServerConfig.load(path)  # Preserve the runtime's TLS/RBAC/config validation.
    atomic_write_json(marker, dict(backup_id=manifest["backup_id"], destination=str(target)), overwrite=False)
    return path


def execute_backup_command(args):
    tools = PostgresTools(args.pg_bin_dir, timeout=args.timeout) if args.command != "backup-verify" else None
    if args.command == "backup-server":
        config = ServerConfig.load(args.config)
        manifest = create_server_backup(config.database_url, config.blob_root, args.destination,
            source_root=config.source_root, derived_root=config.derived_root, tools=tools, configuration=_settings(config))
        result = dict(status="passed", backup_id=manifest["backup_id"], destination=str(Path(args.destination).absolute()),
                      files=len(manifest["entries"]), bytes=sum(entry["size"] for entry in manifest["entries"]),
                      tables=len(manifest["tables"]), external_references=len(manifest["external_references"]))
    elif args.command == "backup-verify":
        manifest = verify_server_backup(args.backup)
        result = dict(status="passed", backup_id=manifest["backup_id"], files=len(manifest["entries"]), tables=len(manifest["tables"]))
    else:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.database_url_env):
            raise ServerBackupError("Invalid database URL environment variable")
        database_url = os.environ.get(args.database_url_env, "").strip()
        if not database_url:
            raise ServerBackupError(f"Set {args.database_url_env} to the fresh target PostgreSQL URL")
        manifest = verify_server_backup(args.backup)
        if set(manifest["roots"]) != {"blobs", "source", "derived"}:
            raise ServerBackupError("Full server recovery requires source and derived trees as well as BlobStore")
        config_directory = Path(args.config_dir).absolute()
        _directory(config_directory.parent)
        if any(config_directory.is_relative_to(path) or path.is_relative_to(config_directory)
               for path in (Path(args.backup).absolute(), Path(args.destination).absolute())):
            raise ServerBackupError("Configuration directory must be separate from restored data and archive")
        if (args.tls_cert_file is None) != (args.tls_key_file is None):
            raise ServerBackupError("Both restored TLS files are required together")
        if config_directory.exists():
            _directory(config_directory)
            marker = config_directory / "recovery.json"
            if not args.resume or not marker.is_file():
                raise ServerBackupError("Configuration directory exists; choose a new directory")
            if json.loads(marker.read_text(encoding="utf-8")) != {
                "backup_id": manifest["backup_id"], "destination": str(Path(args.destination).absolute()),
            }:
                raise ServerBackupError("Existing recovery configuration belongs to another restore")
            current = ServerConfig.load(config_directory / "server.toml")
            if current.database_url != database_url:
                raise ServerBackupError("Existing recovery configuration targets another connection")
            if (args.host is not None and args.host != current.host) or (args.port is not None and args.port != current.port):
                raise ServerBackupError("Resume cannot change an existing configuration's listener")
        result = restore_server_backup(args.backup, database_url, args.destination, tools=tools, resume=args.resume)
        result["config"] = str(_write_recovery_config(args, result, manifest, database_url))
    print(json.dumps(result, ensure_ascii=False))
    return 0

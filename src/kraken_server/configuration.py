"""File-backed production configuration for packaged Kraken Server installs."""

from __future__ import annotations

import ctypes
import os
import secrets
import sys
import tomllib
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .blob_gateway import validate_public_gateway_url


def default_config_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
        return root / "Kraken" / "Server" / "server.toml"
    return Path("/etc/kraken/server.toml")


def default_local_config_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Kraken" / "LocalServer" / "server.toml"
    return Path.home() / ".config" / "kraken" / "local-server.toml"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(value: bytes) -> tuple[_DataBlob, object]:
    buffer = ctypes.create_string_buffer(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _windows_crypto() -> tuple[object, object]:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def protect_secret(value: str, destination: Path) -> None:
    """Persist a secret using machine-scoped DPAPI on Windows and mode 0600 elsewhere."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = value.encode("utf-8")
    if os.name == "nt":
        source, source_buffer = _blob(raw)
        output = _DataBlob()
        crypt32, kernel32 = _windows_crypto()
        if not crypt32.CryptProtectData(
            ctypes.byref(source),
            "Kraken Server database URL",
            None,
            None,
            None,
            0x4,  # CRYPTPROTECT_LOCAL_MACHINE
            ctypes.byref(output),
        ):
            raise ctypes.WinError()
        del source_buffer
        try:
            raw = ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)
        payload = b"KRAKEN-DPAPI-1\0" + raw
    else:
        payload = b"KRAKEN-PLAIN-1\0" + raw
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def unprotect_secret(source: Path) -> str:
    payload = source.read_bytes()
    dpapi_prefix = b"KRAKEN-DPAPI-1\0"
    plain_prefix = b"KRAKEN-PLAIN-1\0"
    if payload.startswith(dpapi_prefix):
        if os.name != "nt":
            raise RuntimeError("A Windows DPAPI secret cannot be read on this platform")
        encrypted, encrypted_buffer = _blob(payload[len(dpapi_prefix) :])
        output = _DataBlob()
        crypt32, kernel32 = _windows_crypto()
        if not crypt32.CryptUnprotectData(ctypes.byref(encrypted), None, None, None, None, 0, ctypes.byref(output)):
            raise ctypes.WinError()
        del encrypted_buffer
        try:
            raw = ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)
    elif payload.startswith(plain_prefix):
        raw = payload[len(plain_prefix) :]
    else:
        raise RuntimeError("Kraken Server secret file has an unsupported format")
    return raw.decode("utf-8")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    path: Path
    database_url_secret: Path
    blob_root: Path
    host: str = "127.0.0.1"
    port: int = 8080
    project_access_mode: str = "acl"
    gitlab_issuer: str | None = None
    gitlab_ca_file: Path | None = None
    tls_cert_file: Path | None = None
    tls_key_file: Path | None = None
    blob_gateway_public_url: str | None = None
    blob_gateway_bind: str | None = None
    blob_gateway_executable: Path | None = None
    blob_gateway_secret: Path | None = None
    blob_gateway_tls_cert_file: Path | None = None
    blob_gateway_tls_key_file: Path | None = None
    blob_ticket_lifetime_seconds: int = 900

    @classmethod
    def load(cls, path: Path | str) -> ServerConfig:
        config_path = Path(path).expanduser().resolve(strict=True)
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
        server = document.get("server", {})
        database = document.get("database", {})
        storage = document.get("storage", {})
        identity = document.get("identity", {})
        gateway = document.get("blob_gateway", {})

        def resolve(value: object, *, required: bool = True) -> Path | None:
            text = str(value or "").strip()
            if not text:
                if required:
                    raise ValueError("Required Kraken Server path is missing")
                return None
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        mode = str(server.get("project_access_mode", "acl")).strip().lower()
        if mode not in {"acl", "trusted_network"}:
            raise ValueError("project_access_mode must be 'acl' or 'trusted_network'")
        port = int(server.get("port", 8080))
        if not 1 <= port <= 65535:
            raise ValueError("server.port must be between 1 and 65535")
        gateway_enabled = bool(gateway.get("enabled", False))
        gateway_public_url = (
            validate_public_gateway_url(str(gateway.get("public_url", ""))) if gateway_enabled else None
        )
        gateway_tls_cert = resolve(gateway.get("tls_cert_file"), required=False) if gateway_enabled else None
        gateway_tls_key = resolve(gateway.get("tls_key_file"), required=False) if gateway_enabled else None
        if (gateway_tls_cert is None) != (gateway_tls_key is None):
            raise ValueError("blob_gateway TLS certificate and key must be configured together")
        if gateway_public_url is not None and gateway_public_url.startswith("https://") != (gateway_tls_cert is not None):
            raise ValueError("blob_gateway public URL scheme must match its TLS configuration")
        ticket_lifetime = int(gateway.get("ticket_lifetime_seconds", 900))
        if not 30 <= ticket_lifetime <= 86_400:
            raise ValueError("blob_gateway.ticket_lifetime_seconds must be between 30 and 86400")
        return cls(
            path=config_path,
            database_url_secret=resolve(database.get("url_secret")),  # type: ignore[arg-type]
            blob_root=resolve(storage.get("blob_root")),  # type: ignore[arg-type]
            host=str(server.get("host", "127.0.0.1")).strip() or "127.0.0.1",
            port=port,
            project_access_mode=mode,
            gitlab_issuer=str(identity.get("gitlab_issuer", "")).strip() or None,
            gitlab_ca_file=resolve(identity.get("gitlab_ca_file"), required=False),
            tls_cert_file=resolve(server.get("tls_cert_file"), required=False),
            tls_key_file=resolve(server.get("tls_key_file"), required=False),
            blob_gateway_public_url=gateway_public_url,
            blob_gateway_bind=(str(gateway.get("bind", "")).strip() or None) if gateway_enabled else None,
            blob_gateway_executable=(resolve(gateway.get("executable")) if gateway_enabled else None),
            blob_gateway_secret=(resolve(gateway.get("secret")) if gateway_enabled else None),
            blob_gateway_tls_cert_file=gateway_tls_cert,
            blob_gateway_tls_key_file=gateway_tls_key,
            blob_ticket_lifetime_seconds=ticket_lifetime,
        )

    @property
    def database_url(self) -> str:
        return unprotect_secret(self.database_url_secret)

    @property
    def blob_gateway_secret_value(self) -> str | None:
        return None if self.blob_gateway_secret is None else unprotect_secret(self.blob_gateway_secret)

    def apply_to_environment(self) -> None:
        os.environ["KRAKEN_SERVER_COMPOSITION"] = "kraken_server.composition:postgresql_composition"
        os.environ["KRAKEN_DATABASE_URL"] = self.database_url
        os.environ["KRAKEN_BLOB_ROOT"] = str(self.blob_root)
        os.environ["KRAKEN_PROJECT_ACCESS_MODE"] = self.project_access_mode
        if self.gitlab_issuer:
            os.environ["KRAKEN_GITLAB_ISSUER"] = self.gitlab_issuer
        if self.gitlab_ca_file:
            os.environ["KRAKEN_GITLAB_CA_FILE"] = str(self.gitlab_ca_file)
        if self.blob_gateway_public_url is not None:
            if self.blob_gateway_bind is None or self.blob_gateway_executable is None:
                raise ValueError("Enabled Blob Gateway configuration is incomplete")
            secret = self.blob_gateway_secret_value
            if secret is None:
                raise ValueError("Enabled Blob Gateway secret is missing")
            os.environ["KRAKEN_BLOB_GATEWAY_PUBLIC_URL"] = self.blob_gateway_public_url
            os.environ["KRAKEN_BLOB_GATEWAY_BIND"] = self.blob_gateway_bind
            os.environ["KRAKEN_BLOB_GATEWAY_EXECUTABLE"] = str(self.blob_gateway_executable)
            os.environ["KRAKEN_BLOB_GATEWAY_SECRET"] = secret
            os.environ["KRAKEN_BLOB_TICKET_LIFETIME"] = str(self.blob_ticket_lifetime_seconds)
            if self.blob_gateway_tls_cert_file is not None:
                os.environ["KRAKEN_BLOB_GATEWAY_TLS_CERT"] = str(self.blob_gateway_tls_cert_file)
            if self.blob_gateway_tls_key_file is not None:
                os.environ["KRAKEN_BLOB_GATEWAY_TLS_KEY"] = str(self.blob_gateway_tls_key_file)
        else:
            for name in (
                "KRAKEN_BLOB_GATEWAY_PUBLIC_URL",
                "KRAKEN_BLOB_GATEWAY_BIND",
                "KRAKEN_BLOB_GATEWAY_EXECUTABLE",
                "KRAKEN_BLOB_GATEWAY_SECRET",
                "KRAKEN_BLOB_TICKET_LIFETIME",
                "KRAKEN_BLOB_GATEWAY_TLS_CERT",
                "KRAKEN_BLOB_GATEWAY_TLS_KEY",
            ):
                os.environ.pop(name, None)


def write_config(
    path: Path,
    *,
    database_url: str,
    blob_root: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    project_access_mode: str = "acl",
    tls_cert_file: Path | None = None,
    tls_key_file: Path | None = None,
    blob_gateway_public_url: str | None = None,
    blob_gateway_bind: str = "127.0.0.1:8081",
    blob_gateway_executable: Path | None = None,
    blob_ticket_lifetime_seconds: int = 900,
) -> ServerConfig:
    import json

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret_path = path.parent / "database-url.secret"
    protect_secret(database_url, secret_path)
    gateway_enabled = blob_gateway_public_url is not None
    gateway_secret_path = path.parent / "blob-gateway.secret"
    if gateway_enabled:
        blob_gateway_public_url = validate_public_gateway_url(str(blob_gateway_public_url))
        gateway_uses_tls = blob_gateway_public_url.startswith("https://")
        if gateway_uses_tls and (tls_cert_file is None or tls_key_file is None):
            raise ValueError(
                "HTTPS Blob Gateway requires both TLS certificate and private key files"
            )
        if blob_gateway_executable is None:
            raise ValueError("Blob Gateway executable is required when the gateway is enabled")
        if not 30 <= int(blob_ticket_lifetime_seconds) <= 86_400:
            raise ValueError("Blob ticket lifetime must be between 30 and 86400 seconds")
        if gateway_secret_path.is_file():
            gateway_secret = unprotect_secret(gateway_secret_path)
        else:
            gateway_secret = secrets.token_urlsafe(48)
        protect_secret(gateway_secret, gateway_secret_path)
    blob_root = blob_root.expanduser().resolve()
    blob_root.mkdir(parents=True, exist_ok=True)
    server_lines = [
        "# Конфигурация Kraken Server.",
        "# Создана KrakenAdmin. Данные подключения к БД хранятся отдельно",
        "# и защищены Windows DPAPI; никогда не записывайте пароль в этот файл.",
        "",
        "[server]",
        "# Сетевой интерфейс Kraken Server.",
        "# 127.0.0.1 безопасен для локального теста. Сетевой интерфейс используйте только с HTTPS.",
        f"host = {json.dumps(host)}",
        "# HTTP(S)-порт Kraken Server.",
        f"port = {int(port)}",
        "# acl проверяет роли проекта. trusted_network допустим только в изолированной сети,",
        "# где каждый вошедший пользователь считается полностью доверенным.",
        f"project_access_mode = {json.dumps(project_access_mode)}",
    ]
    if tls_cert_file is not None:
        server_lines.append(f"tls_cert_file = {json.dumps(str(tls_cert_file.resolve()))}")
    if tls_key_file is not None:
        server_lines.append(f"tls_key_file = {json.dumps(str(tls_key_file.resolve()))}")
    sections = [
            *server_lines,
            "",
            "[database]",
            "# Относительный путь к защищённому DPAPI-секрету подключения PostgreSQL.",
            f"url_secret = {json.dumps(secret_path.name)}",
            "",
            "[storage]",
            "# Неизменяемые файлы сервера. Рабочие станции не открывают этот каталог напрямую.",
            f"blob_root = {json.dumps(str(blob_root))}",
            "",
    ]
    sections.extend(
        (
            "[blob_gateway]",
            "# Rust data plane streams large immutable files without routing their bytes through Python.",
            f"enabled = {'true' if gateway_enabled else 'false'}",
        )
    )
    if gateway_enabled:
        sections.extend(
            (
                f"public_url = {json.dumps(str(blob_gateway_public_url).rstrip('/'))}",
                f"bind = {json.dumps(blob_gateway_bind)}",
                f"executable = {json.dumps(str(blob_gateway_executable.resolve()))}",
                f"secret = {json.dumps(gateway_secret_path.name)}",
                f"ticket_lifetime_seconds = {int(blob_ticket_lifetime_seconds)}",
            )
        )
        if gateway_uses_tls:
            sections.append(f"tls_cert_file = {json.dumps(str(tls_cert_file.resolve()))}")
            sections.append(f"tls_key_file = {json.dumps(str(tls_key_file.resolve()))}")
    sections.append("")
    content = "\n".join(sections)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return ServerConfig.load(path)


def migration_root() -> Path:
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    candidate = bundled / "migrations"
    if candidate.is_dir():
        return candidate
    return Path(__file__).resolve().parents[2] / "migrations"


def run_migrations(database_url: str) -> None:
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError as exc:
        raise RuntimeError("Alembic is required to initialize Kraken Server") from exc
    configuration = Config()
    configuration.set_main_option("script_location", str(migration_root()))
    previous = os.environ.get("KRAKEN_DATABASE_URL")
    os.environ["KRAKEN_DATABASE_URL"] = database_url
    try:
        command.upgrade(configuration, "head")
    finally:
        if previous is None:
            os.environ.pop("KRAKEN_DATABASE_URL", None)
        else:
            os.environ["KRAKEN_DATABASE_URL"] = previous


__all__ = [
    "ServerConfig",
    "default_config_path",
    "default_local_config_path",
    "protect_secret",
    "run_migrations",
    "unprotect_secret",
    "write_config",
]

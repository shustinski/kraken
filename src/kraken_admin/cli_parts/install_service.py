"""cli: install service."""
from __future__ import annotations
import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from kraken_manager.infrastructure.auth.local import (
    Argon2PasswordHasher,
)
from kraken_server.configuration import (
    ServerConfig,
    default_config_path,
    default_local_config_path,
)

class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if (
            action.option_strings
            and action.default is not None
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in help_text
        ):
            help_text += " (по умолчанию: %(default)s)"
        return help_text

class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.pop("add_help", None)
        super().__init__(*args, add_help=False, **kwargs)
        self.add_argument("-h", "--help", action="help", help="Показать эту справку и выйти")
        self._positionals.title = "Позиционные аргументы"
        self._optionals.title = "Параметры"

def _secret(prompt: str, *, environment: str, non_interactive: bool = False) -> str:
    value = os.environ.get(environment, "")
    if value:
        return value
    if non_interactive:
        raise ValueError(f"Для неинтерактивного режима задайте переменную окружения {environment}")
    return getpass.getpass(prompt)

def _text_value(
    value: object,
    prompt: str,
    *,
    default: str,
    non_interactive: bool,
) -> str:
    supplied = str(value or "").strip()
    if supplied:
        return supplied
    if non_interactive:
        return default
    entered = input(f"{prompt} [{default}]: ").strip()
    return entered or default

def _integer_value(
    value: object,
    prompt: str,
    *,
    default: int,
    non_interactive: bool,
) -> int:
    supplied = str(value or "").strip()
    if not supplied and non_interactive:
        return default
    entered = supplied or input(f"{prompt} [{default}]: ").strip()
    try:
        return int(entered or default)
    except ValueError as exc:
        raise ValueError(f"{prompt}: требуется целое число") from exc

def _password(
    prompt: str,
    *,
    environment: str = "KRAKEN_INITIAL_ADMIN_PASSWORD",
    non_interactive: bool = False,
) -> str:
    from_environment = os.environ.get(environment, "")
    value = _secret(prompt, environment=environment, non_interactive=non_interactive)
    if from_environment:
        confirmation = value
    else:
        confirmation = getpass.getpass("Повторите пароль: ")
    if not value:
        raise ValueError("Пароль не может быть пустым")
    if value != confirmation:
        raise ValueError("Пароли не совпадают")
    return value

def _postgres_store(database_url: str) -> Any:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise SystemExit("The packaged PostgreSQL runtime is missing") from exc
    from kraken_manager.infrastructure.postgres import PostgresAccountStore

    return PostgresAccountStore(create_engine(database_url, pool_pre_ping=True), Argon2PasswordHasher())

def _database_url(args: argparse.Namespace) -> str:
    value = str(getattr(args, "database_url", "") or "").strip()
    if value:
        return value
    config = getattr(args, "config", None)
    if config is not None and Path(config).is_file():
        return ServerConfig.load(config).database_url
    return getpass.getpass("PostgreSQL URL: ").strip()

def _bootstrap(
    store: Any,
    username: str,
    display_name: str,
    *,
    password: str | None = None,
) -> str:
    if store.accounts_with_global_role("server_admin"):
        raise ValueError("Администратор Kraken Server уже существует")
    account = store.create_account(
        username,
        display_name,
        password or _password("New administrator password: "),
    )
    store.grant_global_role(account.account_id, "server_admin")
    return str(account.account_id)

def _data_root(value: object, *, flag: str, non_interactive: bool) -> Path:
    supplied = str(value or "").strip()
    if not supplied:
        if non_interactive:
            raise SystemExit(f"Укажите {flag}: каталог для папок проекта")
        supplied = input(f"{flag} (каталог): ").strip()
    if not supplied:
        raise SystemExit(f"Укажите {flag}: каталог для папок проекта")
    return Path(supplied)

def _default_blob_root() -> Path:
    return default_config_path().parent / "blobs"

def _default_local_blob_root() -> Path:
    return default_local_config_path().parent / "blobs"

def _default_server_executable() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name("KrakenServer.exe")
    return Path(sys.executable)

def _default_blob_gateway_executable() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name("KrakenBlobGateway.exe")
    return Path(__file__).resolve().parents[2] / "blob_gateway" / "target" / "release" / "kraken-blob-gateway.exe"

def _blob_gateway_public_url(args: argparse.Namespace) -> str:
    if args.blob_gateway_public_url:
        return str(args.blob_gateway_public_url)
    if args.blob_gateway_host in {"127.0.0.1", "::1", "localhost"}:
        return f"http://127.0.0.1:{args.blob_gateway_port}"
    raise SystemExit("Для сетевого Blob Gateway укажите --blob-gateway-public-url с HTTPS-адресом")

def _install_service(server_executable: Path, config: Path, *, start: bool) -> None:
    if os.name != "nt":
        raise SystemExit("Windows Service installation is available only on Windows")
    executable = server_executable.resolve(strict=True)
    configuration = config.resolve(strict=True)
    from kraken_admin.service_ownership import require_owned_service, service_image_path
    require_owned_service(executable, image_path=service_image_path())
    if executable == Path(sys.executable).resolve() and not getattr(sys, "frozen", False):
        command = f'"{executable}" -m kraken_server --service --config "{configuration}"'
    else:
        command = f'"{executable}" --service --config "{configuration}"'
    def require_matching_command(image_path: str | None) -> None:
        if image_path is not None and image_path.strip().casefold() != command.casefold():
            raise SystemExit(
                "Existing KrakenServer service has different startup arguments or --config. "
                "Explicitly reconfigure or uninstall the owned service before reinstalling."
            )

    require_matching_command(service_image_path())
    result = subprocess.run(
        [
            "sc.exe",
            "create",
            "KrakenServer",
            f"binPath= {command}",
            "start= auto",
            "DisplayName= Kraken Server",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and "1073" not in (result.stdout + result.stderr):
        raise SystemExit((result.stdout + result.stderr).strip())
    # Recheck the 1073 race before touching description/recovery settings.
    current_command = service_image_path()
    if not require_owned_service(executable, image_path=current_command):
        raise RuntimeError("KrakenServer disappeared before its installation could be verified")
    require_matching_command(current_command)
    subprocess.run(
        ["sc.exe", "description", "KrakenServer", "Kraken shared project server"],
        check=True,
    )
    subprocess.run(
        ["sc.exe", "failure", "KrakenServer", "reset= 86400", "actions= restart/5000/restart/15000/none/0"],
        check=True,
    )
    if start:
        subprocess.run(["sc.exe", "start", "KrakenServer"], check=True)

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from updater.client import (
    UpdateClientConfig,
    fetch_update_info,
    load_selected_update_channel,
    load_update_client_config,
    normalize_update_channel,
    save_selected_update_channel,
)
from updater.qt import QtUpdateController

from .core.features import display_version, tester_build
from .version import APP_VERSION, __version__

KARAKAL_UPDATE_APP_ID = "karakal"
KARAKAL_UPDATE_APP_NAME = "Karakal"
KARAKAL_UPDATE_ENV_PREFIX = "KARAKAL"
KARAKAL_UPDATE_SETTINGS_ORG = "Karakal"
KARAKAL_UPDATE_SETTINGS_APP = "Updater"
KARAKAL_UPDATE_CLIENT_FILENAME = "update_client.json"
KARAKAL_UPDATE_ROOT_ENV = "KARAKAL_UPDATE_ROOT"
KARAKAL_UPDATE_ROOT_SETTINGS_KEY = "update_root"
KARAKAL_UPDATE_ROOT_DEFAULT_SETTINGS_KEY = "update_root_default"
KARAKAL_LAST_CHECK_SETTINGS_KEY = "last_update_check"
KARAKAL_MOVED_TO_MAX_HOPS = 3
KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS = 2.0

__all__ = [
    "APP_VERSION",
    "KARAKAL_UPDATE_APP_ID",
    "KARAKAL_UPDATE_APP_NAME",
    "KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS",
    "KARAKAL_UPDATE_CLIENT_FILENAME",
    "KARAKAL_UPDATE_ENV_PREFIX",
    "KARAKAL_UPDATE_ROOT_ENV",
    "KARAKAL_UPDATE_SETTINGS_APP",
    "KARAKAL_UPDATE_SETTINGS_ORG",
    "__version__",
    "configured_karakal_update_root",
    "create_karakal_update_controller",
    "default_karakal_update_channel",
    "follow_update_root_moves",
    "karakal_changelog_path",
    "karakal_settings_ini_path",
    "karakal_update_client_config_path",
    "load_karakal_last_update_check",
    "load_karakal_update_channel",
    "load_karakal_update_client_config",
    "load_karakal_update_root",
    "repair_installer_update_root",
    "resolve_karakal_channel_manifest",
    "save_karakal_last_update_check",
    "save_karakal_update_channel",
    "save_karakal_update_root",
    "validate_karakal_update_root",
]


def karakal_resources_root() -> Path:
    if not bool(getattr(sys, "frozen", False)):
        return Path(__file__).resolve().parents[2] / "resources"

    executable_dir = Path(sys.executable).resolve().parent
    internal_dir = executable_dir / "_internal"
    if internal_dir.exists():
        return internal_dir / "resources"

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "resources"

    return executable_dir / "resources"


def karakal_update_client_config_path() -> Path:
    return karakal_resources_root() / KARAKAL_UPDATE_CLIENT_FILENAME


def karakal_changelog_path() -> Path:
    """CHANGELOG.md: bundled into resources of a build, at the plugin root in sources."""

    if bool(getattr(sys, "frozen", False)):
        return karakal_resources_root() / "CHANGELOG.md"
    return Path(__file__).resolve().parents[2] / "CHANGELOG.md"


def karakal_settings_ini_path() -> Path:
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent / "settings.ini"
    env_dir = str(os.getenv("KARAKAL_SETTINGS_DIR", "")).strip()
    if env_dir:
        return Path(env_dir) / "settings.ini"
    return Path.cwd() / "settings.ini"


def default_karakal_update_channel() -> str:
    return "beta" if tester_build() else "stable"


def _updater_qsettings():
    from PyQt6.QtCore import QSettings

    path = karakal_settings_ini_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return QSettings(str(path), QSettings.Format.IniFormat)


def _installer_written_update_root(ini_path: Path) -> str | None:
    r"""update_root as the installer wrote it, when Qt would misread it; else None.

    Installers up to 0.2.9107 wrote the folder as typed: G:\ProgramStore, in the Windows code page.
    Qt reads a backslash in settings.ini as an escape (G:rogramStore) and the text as UTF-8,
    then writes the damaged value back on the next save. Qt itself writes backslashes doubled.
    """

    try:
        data = ini_path.read_bytes()
    except OSError:
        return None
    try:
        text = data.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        text = data.decode("mbcs" if os.name == "nt" else "cp1251", errors="replace")
        utf8 = False
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            continue
        if section != "general" or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != KARAKAL_UPDATE_ROOT_SETTINGS_KEY:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        if not utf8:
            return value
        # Qt doubles every backslash it writes; a single one (G:\Store, \\PC\Share) came from the installer.
        runs = re.findall(r"\\+", value)
        if any(len(run) % 2 for run in runs):
            return value
        return None
    return None


def repair_installer_update_root() -> None:
    """Re-save an installer-written update folder through Qt before Qt damages it."""

    ini_path = karakal_settings_ini_path()
    raw = _installer_written_update_root(ini_path)
    if raw:
        save_karakal_update_root(raw.replace("\\", "/"))


def load_karakal_update_root(*, allow_default: bool = True) -> str:
    env_root = str(os.getenv(KARAKAL_UPDATE_ROOT_ENV, "")).strip()
    if env_root:
        return env_root
    try:
        repair_installer_update_root()
        settings = _updater_qsettings()
        stored = str(settings.value(KARAKAL_UPDATE_ROOT_SETTINGS_KEY, "", type=str) or "").strip()
        if stored:
            return stored
        if allow_default:
            default_root = str(
                settings.value(KARAKAL_UPDATE_ROOT_DEFAULT_SETTINGS_KEY, "", type=str) or ""
            ).strip()
            if default_root:
                return default_root
    except Exception:
        pass
    if not allow_default:
        return ""
    build_default = str(os.getenv("KARAKAL_UPDATE_ROOT_DEFAULT", "")).strip()
    return build_default or configured_karakal_update_root()


def configured_karakal_update_root() -> str:
    """``update_root`` of resources/update_client.json: the network folder baked into the build."""

    try:
        payload = json.loads(karakal_update_client_config_path().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("update_root", "") or "").strip()


def save_karakal_update_root(root: str) -> None:
    normalized = str(root or "").strip()
    settings = _updater_qsettings()
    if normalized:
        settings.setValue(KARAKAL_UPDATE_ROOT_SETTINGS_KEY, normalized)
    else:
        settings.remove(KARAKAL_UPDATE_ROOT_SETTINGS_KEY)
    settings.sync()


def load_karakal_last_update_check() -> str:
    try:
        settings = _updater_qsettings()
        return str(settings.value(KARAKAL_LAST_CHECK_SETTINGS_KEY, "", type=str) or "").strip()
    except Exception:
        return ""


def save_karakal_last_update_check(when: datetime | None = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).astimezone().strftime("%Y-%m-%d %H:%M")
    settings = _updater_qsettings()
    settings.setValue(KARAKAL_LAST_CHECK_SETTINGS_KEY, stamp)
    settings.sync()
    return stamp


def validate_karakal_update_root(root: str | Path) -> tuple[bool, str]:
    root_path = Path(str(root or "").strip())
    if not str(root_path):
        return False, "Папка обновлений не задана."
    if not root_path.exists() or not root_path.is_dir():
        return False, f"Папка недоступна: {root_path}"
    missing: list[str] = []
    for channel in ("beta", "stable"):
        manifest = root_path / channel / "version.json"
        if not manifest.is_file():
            missing.append(f"{channel}/version.json")
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return False, f"Не удалось прочитать {manifest}"
        if not isinstance(payload, dict):
            return False, f"Некорректный манифест: {manifest}"
        if payload.get("moved_to"):
            continue
        if not str(payload.get("version", "")).strip():
            return False, f"В {manifest} нет поля version"
    if missing:
        return False, "В корне должны быть каналы beta и stable с version.json. Не найдено: " + ", ".join(missing)
    return True, ""


def follow_update_root_moves(
    root: str | Path,
    *,
    channel: str,
    max_hops: int = KARAKAL_MOVED_TO_MAX_HOPS,
    persist: bool = True,
) -> str:
    current = str(root or "").strip()
    if not current:
        return ""
    seen: set[str] = set()
    requested = normalize_update_channel(channel)
    for _ in range(max(1, int(max_hops))):
        key = os.path.normcase(os.path.normpath(current))
        if key in seen:
            break
        seen.add(key)
        manifest_path = Path(current) / requested / "version.json"
        try:
            if not manifest_path.is_file():
                break
            payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            break
        if not isinstance(payload, dict):
            break
        moved_to = str(payload.get("moved_to", "") or "").strip()
        if not moved_to:
            break
        ok, _message = validate_karakal_update_root(moved_to)
        if not ok:
            # Accept a target that at least has the requested channel manifest.
            target_manifest = Path(moved_to) / requested / "version.json"
            if not target_manifest.is_file():
                break
        current = moved_to
        if persist:
            save_karakal_update_root(current)
    return current


def _channel_template(config: UpdateClientConfig, channel: str) -> str:
    template = str(config.get_manifest_url(channel) or "").strip()
    if template:
        return template
    return channel


def resolve_karakal_channel_manifest(
    channel: str | None = None,
    *,
    root: str | Path | None = None,
    follow_moves: bool = True,
    persist_moves: bool = True,
) -> str:
    config = load_karakal_update_client_config()
    selected = normalize_update_channel(channel or load_karakal_update_channel(config))
    update_root = str(root if root is not None else load_karakal_update_root()).strip()
    if not update_root:
        return ""
    if follow_moves:
        update_root = follow_update_root_moves(
            update_root,
            channel=selected,
            persist=persist_moves,
        )
    template = _channel_template(config, selected)
    if "://" in template:
        return template
    template_path = Path(template)
    if template_path.is_absolute():
        return str(template_path)
    joined = Path(update_root) / template_path
    if joined.is_dir() or joined.suffix.lower() != ".json":
        candidate = joined if joined.name.lower() == "version.json" else joined / "version.json"
        if candidate.is_file() or joined.is_dir():
            return str(joined if joined.is_dir() else candidate.parent if candidate.name == "version.json" else joined)
        return str(Path(update_root) / selected)
    return str(joined)


def load_karakal_update_client_config() -> UpdateClientConfig:
    config = load_update_client_config(
        app_id=KARAKAL_UPDATE_APP_ID,
        config_path=karakal_update_client_config_path(),
        env_prefix=KARAKAL_UPDATE_ENV_PREFIX,
    )
    default_channel = default_karakal_update_channel()
    if config.manifest_urls:
        return UpdateClientConfig(manifest_urls=config.manifest_urls, default_channel=default_channel)
    return UpdateClientConfig(
        manifest_urls=(("stable", "stable"), ("beta", "beta")),
        default_channel=default_channel,
    )


def load_karakal_update_channel(config: UpdateClientConfig | None = None) -> str:
    resolved_config = config or load_karakal_update_client_config()
    return load_selected_update_channel(
        resolved_config.default_channel,
        available_channels=resolved_config.available_channels,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
        settings_path=karakal_settings_ini_path(),
    )


def save_karakal_update_channel(channel: str) -> None:
    save_selected_update_channel(
        channel,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
        settings_path=karakal_settings_ini_path(),
    )


def _resolve_manifest_for_controller(channel: str, manual: bool) -> str | None:
    root = load_karakal_update_root()
    if not root:
        if not manual:
            return None
        return ""
    manifest = resolve_karakal_channel_manifest(channel, root=root, follow_moves=True, persist_moves=True)
    if not manifest:
        return ""
    # Touch the path early so auto-check fails fast on dead shares.
    path = Path(manifest)
    if path.exists() or path.parent.exists():
        return str(path if path.suffix.lower() == ".json" or path.is_file() else path)
    return str(Path(root) / normalize_update_channel(channel))


def karakal_channel_manifest_location(channel: str, *, manual: bool, window=None) -> str | None:
    """Folder of one channel in the update folder.

    ``""``: no update folder in the build or settings; ``None``: the folder is not reachable
    (a manual check says where the program looked).
    """

    root = load_karakal_update_root()
    if not root:
        return "" if manual else None
    resolved_root = follow_update_root_moves(root, channel=channel, persist=True)
    manifest_dir = Path(resolved_root) / normalize_update_channel(channel)
    if not manifest_dir.is_dir():
        # A missing folder is not "no updates": say where the program looked.
        if manual:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                window,
                KARAKAL_UPDATE_APP_NAME,
                "Папка обновлений недоступна:\n"
                f"{manifest_dir}\n\n"
                "Проверьте, что диск или сетевая папка подключены. Папку обновлений "
                "указывают при установке Каракала.",
            )
        return None
    return str(manifest_dir)


def create_karakal_update_controller(window) -> QtUpdateController:
    def resolve_manifest_url(channel: str, manual: bool) -> str | None:
        return karakal_channel_manifest_location(channel, manual=manual, window=window)

    controller = QtUpdateController(
        window,
        app_id=KARAKAL_UPDATE_APP_ID,
        app_name=KARAKAL_UPDATE_APP_NAME,
        current_version=display_version() or __version__,
        config_path=karakal_update_client_config_path(),
        env_prefix=KARAKAL_UPDATE_ENV_PREFIX,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
        settings_path=karakal_settings_ini_path(),
        check_timeout_seconds=KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS,
        resolve_manifest_url=resolve_manifest_url,
        status_callback=getattr(window, "show_status_message", None),
    )

    original_on_finished = controller._on_check_finished

    def _on_check_finished(update_info: object, *, manual: bool, channel: str) -> None:
        if update_info is not None:
            save_karakal_last_update_check()
        original_on_finished(update_info, manual=manual, channel=channel)

    controller._on_check_finished = _on_check_finished  # type: ignore[method-assign]
    return controller


def probe_karakal_update(channel: str | None = None) -> object | None:
    """Helper for tests/scripts: fetch update info for the selected channel."""

    selected = normalize_update_channel(channel or load_karakal_update_channel())
    manifest = resolve_karakal_channel_manifest(selected)
    if not manifest:
        return None
    return fetch_update_info(manifest, timeout_seconds=KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS, expected_channel=selected)

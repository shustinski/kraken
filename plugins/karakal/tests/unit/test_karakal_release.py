"""Publishing helpers: release notes from CHANGELOG.md, version numbers, update folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karakal import updater
from karakal.changelog import UNRELEASED_TITLE, notes_for_version, release_changelog, unreleased_notes
from karakal.release import next_publish_version, set_code_beta

CHANGELOG = f"""# Каракал — история изменений

Вступление.

## {UNRELEASED_TITLE}

- новая панель

## 0.2.9101-beta1 — 2026-10-04

- вторая бета

## 0.2.9101-beta0 — 2026-10-03

- первая бета

## 0.2.9100 — 2026-10-01

- прошлый релиз
"""


def _manifest(root: Path, channel: str, versions: list[str]) -> None:
    folder = root / channel
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": versions[0] if versions else "0.0.0",
        "channel": channel,
        "releases": [{"version": version} for version in versions],
    }
    (folder / "version.json").write_text(json.dumps(payload), encoding="utf-8")


def test_beta_release_moves_unreleased_under_the_version() -> None:
    text, notes = release_changelog(CHANGELOG, "0.2.9101-beta2", "2026-10-05")
    assert notes == "- новая панель"
    assert unreleased_notes(text) == ""
    assert notes_for_version(text, "0.2.9101-beta2") == "- новая панель"
    assert text.index(f"## {UNRELEASED_TITLE}") < text.index("## 0.2.9101-beta2 — 2026-10-05")
    assert text.startswith("# Каракал") and "Вступление." in text
    with pytest.raises(ValueError):
        release_changelog(text, "0.2.9101-beta3", "2026-10-06")  # nothing new to tell


def test_stable_release_gathers_the_betas_since_the_last_release() -> None:
    _text, notes = release_changelog(CHANGELOG, "0.2.9101", "2026-10-05")
    assert "- новая панель" in notes
    assert "### 0.2.9101-beta1" in notes and "### 0.2.9101-beta0" in notes
    assert "прошлый релиз" not in notes


def test_next_beta_number_comes_from_the_update_folder(tmp_path) -> None:
    assert next_publish_version("beta", tmp_path, base="0.2.9103") == "0.2.9103-beta0"
    _manifest(tmp_path, "beta", ["0.2.9103-beta1", "0.2.9103-beta0", "0.2.9102-beta4"])
    assert next_publish_version("beta", tmp_path, base="0.2.9103") == "0.2.9103-beta2"
    assert next_publish_version("beta", tmp_path, base="0.2.9104") == "0.2.9104-beta0"
    assert next_publish_version("stable", tmp_path, base="0.2.9103") == "0.2.9103"
    _manifest(tmp_path, "stable", ["0.2.9103"])
    with pytest.raises(ValueError):
        next_publish_version("beta", tmp_path, base="0.2.9103")  # would be older than the release
    with pytest.raises(ValueError):
        next_publish_version("stable", tmp_path, base="0.2.9103")


def test_set_code_beta_rewrites_only_the_beta_line(tmp_path) -> None:
    version_file = tmp_path / "version.py"
    version_file.write_text('RELEASE = "0.2"\nINTERFACE_NUMBER = 3\nBETA = 0\n', encoding="utf-8")
    assert set_code_beta(version_file, "0.2.9103-beta4")
    assert version_file.read_text(encoding="utf-8").endswith("BETA = 4\n")
    assert not set_code_beta(version_file, "0.2.9103")


def test_update_root_falls_back_to_update_client_json(tmp_path, monkeypatch) -> None:
    config = tmp_path / "update_client.json"
    config.write_text(json.dumps({"update_root": "\\\\server\\share\\karakal"}), encoding="utf-8")
    monkeypatch.setattr(updater, "karakal_update_client_config_path", lambda: config)
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv(updater.KARAKAL_UPDATE_ROOT_ENV, raising=False)
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT_DEFAULT", raising=False)
    assert updater.load_karakal_update_root() == "\\\\server\\share\\karakal"
    updater.save_karakal_update_root(str(tmp_path))
    assert updater.load_karakal_update_root() == str(tmp_path)  # the user's choice wins


def test_bundled_changelog_is_the_plugin_changelog() -> None:
    path = updater.karakal_changelog_path()
    assert path.is_file() and path.read_text(encoding="utf-8").startswith("# Каракал")

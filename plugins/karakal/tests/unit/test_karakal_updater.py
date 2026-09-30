"""Semver compare, sha256, update root, and frozen config path."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from updater.client import (
    compare_versions,
    file_sha256,
    is_newer_version,
    verify_installer_sha256,
)

from karakal.updater import (
    default_karakal_update_channel,
    follow_update_root_moves,
    karakal_update_client_config_path,
    load_karakal_update_root,
    resolve_karakal_channel_manifest,
    save_karakal_update_root,
    validate_karakal_update_root,
)
from karakal.version import APP_VERSION, numeric_version


def test_app_version_is_beta_prerelease() -> None:
    assert re.fullmatch(r"0\.1\.0-beta\d+", APP_VERSION)
    assert numeric_version() == "0.1.0.0"


def test_semver_prerelease_order() -> None:
    chain = [
        "0.1.0-beta0",
        "0.1.0-beta.1",
        "0.1.0-beta-2",
        "0.1.0-beta10",
        "0.1.0",
        "0.1.1-beta0",
        "0.1.1",
    ]
    for left, right in zip(chain, chain[1:]):
        assert compare_versions(left, right) < 0, (left, right)
        assert is_newer_version(right, left)
    assert compare_versions("1.2.3", "1.2.3") == 0
    assert compare_versions("1.2.3", "1.2.4") < 0
    assert compare_versions("0.1.0-BETA0", "0.1.0-beta0") == 0


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "Karakal-setup-0.1.0-beta0.exe"
    path.write_bytes(b"installer-bytes")
    digest = file_sha256(path)
    verify_installer_sha256(path, digest)
    with pytest.raises(ValueError):
        verify_installer_sha256(path, "0" * 64)
    verify_installer_sha256(path, "")  # missing → no-op


def test_channel_follows_build_profile(monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    assert default_karakal_update_channel() == "beta"
    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "dev")
    assert default_karakal_update_channel() == "stable"


def test_update_root_env_and_settings(tmp_path: Path, monkeypatch) -> None:
    share = tmp_path / "share"
    for channel in ("beta", "stable"):
        folder = share / channel
        folder.mkdir(parents=True)
        (folder / "version.json").write_text(
            json.dumps({"version": "0.1.0-beta0" if channel == "beta" else "0.1.0", "channel": channel}),
            encoding="utf-8",
        )
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "settings_home"))
    save_karakal_update_root("")
    assert load_karakal_update_root() == ""
    ok, _ = validate_karakal_update_root(share)
    assert ok
    monkeypatch.setenv("KARAKAL_UPDATE_ROOT", str(share))
    assert Path(load_karakal_update_root()) == share
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    save_karakal_update_root(str(share))
    assert Path(load_karakal_update_root()) == share
    manifest = resolve_karakal_channel_manifest("beta", root=share, follow_moves=False)
    assert Path(manifest).exists() or (share / "beta").exists()


def test_moved_to_chain(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old"
    mid = tmp_path / "mid"
    new = tmp_path / "new  кириллица"
    for root in (old, mid, new):
        for channel in ("beta", "stable"):
            folder = root / channel
            folder.mkdir(parents=True)
            payload: dict[str, object] = {"version": "0.1.0-beta1", "channel": channel}
            if root is old:
                payload["moved_to"] = str(mid)
            elif root is mid:
                payload["moved_to"] = str(new)
            (folder / "version.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "cfg"))
    resolved = follow_update_root_moves(old, channel="beta", persist=True)
    assert Path(resolved) == new
    assert Path(load_karakal_update_root()) == new


def test_frozen_update_client_path(tmp_path: Path, monkeypatch) -> None:
    exe_dir = tmp_path / "Karakal"
    resources = exe_dir / "_internal" / "resources"
    resources.mkdir(parents=True)
    config = resources / "update_client.json"
    config.write_text(json.dumps({"channels": {"beta": "beta", "stable": "stable"}}), encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "Karakal.exe"))
    assert karakal_update_client_config_path() == config

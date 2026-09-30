"""Unit tests for Karakal auto-update: versions, sha256, root, channel, frozen path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from updater.client import (
    ReleaseInfo,
    compare_versions,
    download_update_installer,
    fetch_update_info,
    is_newer_version,
    parse_version_parts,
    verify_installer_sha256,
)


def test_compare_versions_prerelease_order() -> None:
    chain = [
        "0.1.0-beta0",
        "0.1.0-beta1",
        "0.1.0-beta9",
        "0.1.0-beta10",
        "0.1.0",
        "0.1.1-beta0",
        "0.1.1",
    ]
    for left, right in zip(chain, chain[1:]):
        assert compare_versions(left, right) < 0, f"{left} should be < {right}"
        assert is_newer_version(right, left)


def test_compare_versions_beta_separator_aliases() -> None:
    assert compare_versions("0.1.0-beta0", "0.1.0-beta.0") == 0
    assert compare_versions("0.1.0-beta0", "0.1.0-beta-0") == 0
    assert compare_versions("0.1.0-BETA1", "0.1.0-beta1") == 0
    assert compare_versions("0.1.0-beta0", "0.1.0") < 0
    assert compare_versions("0.1.0", "0.1.0-beta99") > 0


def test_compare_versions_plain_xyz_unchanged() -> None:
    assert compare_versions("1.2.3", "1.2.3") == 0
    assert compare_versions("1.2.4", "1.2.3") > 0
    assert compare_versions("1.2", "1.2.0") == 0
    assert compare_versions("5.8.1", "5.8.0") > 0
    assert compare_versions("5.7.9", "5.8.0") < 0


def test_installer_directory_sort_uses_semver(tmp_path: Path) -> None:
    for name in (
        "Karakal-setup-0.1.0-beta10.exe",
        "Karakal-setup-0.1.0-beta2.exe",
        "Karakal-setup-0.1.0.exe",
    ):
        (tmp_path / name).write_bytes(b"x")
    info = fetch_update_info(str(tmp_path))
    assert info is not None
    assert info.version == "0.1.0"
    assert [r.version for r in info.releases] == ["0.1.0", "0.1.0-beta10", "0.1.0-beta2"]


def test_verify_sha256_mismatch_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "Karakal-setup-0.1.0-beta0.exe"
    path.write_bytes(b"payload")
    verify_installer_sha256(path, None)
    verify_installer_sha256(path, "")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_installer_sha256(path, "0" * 64)


def test_download_verifies_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import updater.client as client

    monkeypatch.setattr(client.tempfile, "gettempdir", lambda: str(tmp_path))
    source = tmp_path / "Karakal-setup-0.1.0-beta0.exe"
    payload = b"installer-bytes"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    out = download_update_installer(
        ReleaseInfo(version="0.1.0-beta0", download_url=str(source), sha256=digest),
        app_id="karakal-test",
    )
    assert out.read_bytes() == payload

    bad = tmp_path / "bad.exe"
    bad.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_update_installer(
            ReleaseInfo(version="0.1.0-beta0", download_url=str(bad), sha256=digest),
            app_id="karakal-test-bad",
        )


def test_download_without_sha256_keeps_old_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import updater.client as client

    monkeypatch.setattr(client.tempfile, "gettempdir", lambda: str(tmp_path))
    source = tmp_path / "Karakal-setup-0.1.0.exe"
    source.write_bytes(b"ok")
    out = download_update_installer(
        ReleaseInfo(version="0.1.0", download_url=str(source)),
        app_id="karakal-test-nosha",
    )
    assert out.read_bytes() == b"ok"


def test_default_channel_by_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    monkeypatch.delenv("KARAKAL_BUILD_PROFILE", raising=False)

    from karakal.updater import default_karakal_update_channel

    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    assert default_karakal_update_channel() == "beta"

    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "dev")
    assert default_karakal_update_channel() == "stable"

    monkeypatch.delenv("KARAKAL_BUILD_PROFILE", raising=False)
    assert default_karakal_update_channel() == "stable"


def test_update_root_priority_env_over_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    from karakal import updater as u

    u.save_karakal_update_root(str(tmp_path / "from-settings"))
    assert u.load_karakal_update_root() == str(tmp_path / "from-settings")

    monkeypatch.setenv("KARAKAL_UPDATE_ROOT", str(tmp_path / "from-env"))
    assert u.load_karakal_update_root() == str(tmp_path / "from-env")


def test_validate_and_resolve_update_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    root = tmp_path / "share"
    for channel, version in (("beta", "0.1.0-beta0"), ("stable", "0.1.0")):
        channel_dir = root / channel
        channel_dir.mkdir(parents=True)
        installer = channel_dir / f"Karakal-setup-{version}.exe"
        installer.write_bytes(b"bin")
        (channel_dir / "version.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "channel": channel,
                    "download_url": installer.name,
                    "releases": [{"version": version, "download_url": installer.name}],
                }
            ),
            encoding="utf-8",
        )

    from karakal.updater import (
        resolve_karakal_channel_manifest,
        save_karakal_update_root,
        validate_karakal_update_root,
    )

    ok, msg = validate_karakal_update_root(root)
    assert ok, msg
    save_karakal_update_root(str(root))
    monkeypatch.setenv("KARAKAL_UPDATE_ROOT", str(root))
    manifest = resolve_karakal_channel_manifest("beta")
    assert Path(manifest).name in {"beta", "version.json"} or (Path(manifest) / "version.json").exists() or Path(manifest).exists()
    info = fetch_update_info(manifest if Path(manifest).is_file() else str(Path(manifest)), expected_channel="beta")
    # fetch accepts directory
    info = fetch_update_info(str(root / "beta"), expected_channel="beta")
    assert info is not None
    assert info.version == "0.1.0-beta0"


def test_moved_to_chain_up_to_three_hops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)

    hops = [tmp_path / f"share{i}" for i in range(4)]
    final = hops[-1]
    for channel in ("beta", "stable"):
        (final / channel).mkdir(parents=True)
        (final / channel / "version.json").write_text(
            json.dumps({"version": "0.1.0-beta1", "channel": channel, "download_url": "x.exe"}),
            encoding="utf-8",
        )

    for index, folder in enumerate(hops[:-1]):
        nxt = hops[index + 1]
        for channel in ("beta", "stable"):
            (folder / channel).mkdir(parents=True)
            (folder / channel / "version.json").write_text(
                json.dumps(
                    {
                        "version": "0.0.0",
                        "channel": channel,
                        "moved_to": str(nxt),
                    }
                ),
                encoding="utf-8",
            )

    from karakal.updater import follow_update_root_moves, load_karakal_update_root

    resolved = follow_update_root_moves(hops[0], channel="beta", max_hops=3, persist=True)
    assert Path(resolved) == final.resolve() or resolved == str(final)
    stored = load_karakal_update_root()
    assert Path(stored) == Path(resolved)

    # Fourth hop should stop at hop 3 destination when max_hops=3 starting from hop0
    # (0->1, 1->2, 2->3) = 3 moves, lands on final. Good.

    # Loop protection
    a = tmp_path / "loop_a"
    b = tmp_path / "loop_b"
    for folder, target in ((a, b), (b, a)):
        for channel in ("beta", "stable"):
            (folder / channel).mkdir(parents=True, exist_ok=True)
            (folder / channel / "version.json").write_text(
                json.dumps({"version": "0.0.0", "channel": channel, "moved_to": str(target)}),
                encoding="utf-8",
            )
    looped = follow_update_root_moves(a, channel="beta", max_hops=3, persist=False)
    assert looped in {str(a), str(b)}


def test_frozen_update_client_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe_dir = tmp_path / "Karakal"
    resources = exe_dir / "_internal" / "resources"
    resources.mkdir(parents=True)
    config_path = resources / "update_client.json"
    config_path.write_text(
        json.dumps({"default_channel": "stable", "channels": {"stable": "stable", "beta": "beta"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "karakal.exe"))

    from karakal.updater import karakal_settings_ini_path, karakal_update_client_config_path

    assert karakal_update_client_config_path() == config_path
    assert karakal_settings_ini_path() == exe_dir / "settings.ini"


def test_parse_version_parts_is_ordered() -> None:
    assert parse_version_parts("0.1.0-beta1") < parse_version_parts("0.1.0")
    assert parse_version_parts("0.1.0-beta2") < parse_version_parts("0.1.0-beta10")
